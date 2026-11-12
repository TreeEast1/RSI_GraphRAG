"""Shared infrastructure for Question-Time Self-Evolving Answer & Retrieval Skills.

All experiments here reuse the frozen conv-30 artifacts from
``outputs/prompt_optimization_conv30``:

* the fixed COMP-P4_top5 (no LLM rerank) retrieval records,
* the P0/P7/P8 Reader prompts,
* the project + official LoCoMo scorers,
* the HY4 client / quota limiter / NV-Embed-v2 runtime.

Nothing in this file reads gold answers at inference time.  Gold is only used by
the adaptation/reflection code and by the scorers.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "question_evolution_conv30"
ART = OUT / "artifacts"
SNAP = OUT / "skill_snapshots"
LOGS = OUT / "logs"
for _p in (ART, SNAP, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# One shared quota bucket for every process spawned by this study.
os.environ.setdefault("HY4_RATE_FILE", "qe_main")
os.environ.setdefault("HY4_GLOBAL_QPS", "1.2")
os.environ.setdefault("HY4_CACHE_SHARD", "qe")

import numpy as np  # noqa: E402

from src.dataset.locomo import Conversation, load_locomo, num_tokens  # noqa: E402
from src.evaluation.metrics import exact_match, f1_score, select_by_budget  # noqa: E402
from src.prompt_optimization import reader_prompts as RP  # noqa: E402
from src.prompt_optimization.official_metrics import official_qa_score  # noqa: E402
from src.prompt_optimization.runner import (freeze_context, is_refusal,  # noqa: E402
                                            parse_answer, session_docs)
from src.retrieval.reader import format_context  # noqa: E402

BASE_VARIANT = "COMP-P4_top5"
RECORDS_PATH = (ROOT / "outputs" / "prompt_optimization_conv30" / "comprehension"
                / "retrieval_norerank" / BASE_VARIANT / "records.jsonl")
SPLIT_PATH = OUT / "DATA_SPLIT.json"
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
REFUSAL_MARKERS = ("no answer", "not mentioned", "no information", "cannot", "unknown",
                   "not stated", "not provided", "unanswerable")

_LOCK = threading.Lock()
_CACHE: Dict[str, object] = {}
_CLIENT = None


# --------------------------------------------------------------------------- data
def conv() -> Conversation:
    if "conv" not in _CACHE:
        convs = {c.sample_id: c for c in load_locomo(ROOT / "data" / "raw" / "locomo" / "locomo10.json")}
        _CACHE["conv"] = convs["conv-30"]
    return _CACHE["conv"]  # type: ignore[return-value]


def docs_by_id() -> Dict[str, Dict]:
    if "docs" not in _CACHE:
        _CACHE["docs"] = session_docs(conv())
    return _CACHE["docs"]  # type: ignore[return-value]


def qa_map() -> Dict[int, object]:
    if "qa" not in _CACHE:
        _CACHE["qa"] = {int(q.qid.split("q")[-1]): q for q in conv().qa}
    return _CACHE["qa"]  # type: ignore[return-value]


def records() -> Dict[int, Dict]:
    if "records" not in _CACHE:
        rows = [json.loads(l) for l in open(RECORDS_PATH, encoding="utf-8")]
        _CACHE["records"] = {int(r["qid"].split("q")[-1]): r for r in rows}
    return _CACHE["records"]  # type: ignore[return-value]


def split() -> Dict:
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def qidx_list(name: str) -> List[int]:
    return list(split()[name])


def gold_of(qidx: int) -> Optional[str]:
    return qa_map()[qidx].answer


def category_of(qidx: int) -> int:
    return int(qa_map()[qidx].category)


def gold_evidence_turns(qidx: int) -> List[str]:
    return list(qa_map()[qidx].evidence_turns)


def gold_sessions(qidx: int) -> List[int]:
    return list(qa_map()[qidx].evidence_sessions)


def gold_doc_ids(qidx: int) -> List[str]:
    return [f"conv-30::s{s:02d}" for s in gold_sessions(qidx)]


def session_header(doc_id: str) -> str:
    d = docs_by_id()[doc_id]
    return d["header"]


def base_ranked(qidx: int) -> List[str]:
    return list(records()[qidx]["ranked_docs"])


def record_budget_docs(qidx: int) -> Tuple[str, List[str], int]:
    """The exact frozen 4096-token context used by the P0/P7 baselines."""
    fz = freeze_context(records()[qidx], docs_by_id(), 4096)
    return fz["context"], fz["selected_docs"], fz["context_tokens"]


def build_context_from_ranked(ranked_ids: Sequence[str], budget: int = 4096) -> Tuple[str, List[str], int]:
    ranked = [docs_by_id()[d] for d in ranked_ids if d in docs_by_id()]
    selected = select_by_budget(ranked, budget, num_tokens)
    ctx = format_context(selected)
    return ctx, [d["document_id"] for d in selected], sum(num_tokens(d["content"]) for d in selected)


def gold_evidence_text(qidx: int, max_turns: int = 8) -> str:
    by_dia = {t.dia_id: t for s in conv().sessions for t in s.turns}
    parts = []
    for dia in gold_evidence_turns(qidx)[:max_turns]:
        t = by_dia.get(dia)
        if t:
            parts.append(f"[{dia} | Session {t.session_id} | "
                         f"{conv().session_map()[t.session_id].date_time_raw}] {t.render()}")
    return "\n".join(parts)


def compact_sessions(ranked_ids: Sequence[str], k: int = 5, limit: int = 420) -> str:
    out = []
    for did in ranked_ids[:k]:
        d = docs_by_id().get(did)
        if not d:
            continue
        text = re.sub(r"\s+", " ", d["content"])[:limit]
        out.append(f"[{d['document_id']} | {d['header']}] {text}")
    return "\n".join(out)


def entities_in(question: str) -> List[str]:
    names = list(conv().speakers)
    caps = re.findall(r"\b[A-Z][a-z]{2,}\b", question)
    for c in caps:
        if c.lower() not in {"when", "what", "which", "where", "who", "why", "how", "did",
                             "does", "was", "were", "the", "and", "jon", "gina"}:
            names.append(c)
    return list(dict.fromkeys(names))


# --------------------------------------------------------------------------- scoring
def score_answer(pred: str, qidx: int) -> Dict:
    cat = category_of(qidx)
    gold = gold_of(qidx)
    off = official_qa_score(pred, gold, cat)
    if cat == 5:
        refused = is_refusal(pred)
        return {"em": float(refused), "f1": float(refused), "refused": refused, **off}
    return {"em": exact_match(pred, [gold]), "f1": f1_score(pred, [gold]),
            "refused": is_refusal(pred), **off}


def mean(vals: Iterable[Optional[float]]) -> float:
    xs = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(xs)) if xs else float("nan")


def summarize(rows: List[Dict]) -> Dict:
    ans = [r for r in rows if int(r["category"]) != 5]
    adv = [r for r in rows if int(r["category"]) == 5]
    out = {
        "n": len(rows), "n_answerable": len(ans), "n_adversarial": len(adv),
        "project_em": mean(r.get("em") for r in ans),
        "project_f1": mean(r.get("f1") for r in ans),
        # official F1 is defined on answerable questions; category-5 is a refusal task
        "official_f1": mean(r.get("official_f1") for r in ans),
        "official_em": mean(r.get("official_em") for r in ans),
        "adversarial_refusal": mean(r.get("em") for r in adv),
        "adversarial_refusal_official_strict": mean(r.get("official_refused_strict") for r in adv),
        "adversarial_refusal_official_adapted": mean(r.get("official_refused_adapted") for r in adv),
        "refusal_rate_answerable": mean(float(r.get("refused", False)) for r in ans),
        "mean_calls": mean(r.get("n_llm_calls", 0) for r in rows),
        "mean_prompt_tokens": mean((r.get("usage") or {}).get("prompt_tokens") for r in rows),
        "mean_completion_tokens": mean((r.get("usage") or {}).get("completion_tokens") for r in rows),
        "mean_latency_s": mean(r.get("latency_s") for r in rows),
        "by_category": {},
    }
    for c in sorted({int(r["category"]) for r in rows}):
        sub = [r for r in rows if int(r["category"]) == c]
        out["by_category"][str(c)] = {
            "name": CATEGORY_NAMES.get(c, "?"), "n": len(sub),
            "em": mean(r.get("em") for r in sub), "f1": mean(r.get("f1") for r in sub),
            "official_f1": (mean(r.get("official_refused_adapted") for r in sub) if c == 5
                            else mean(r.get("official_f1") for r in sub)),
            "refused": mean(float(r.get("refused", False)) for r in sub),
        }
    return out


# --------------------------------------------------------------------------- LLM
def _shard() -> str:
    return os.environ.get("QE_CACHE_SHARD", "")


def client():
    global _CLIENT
    if _CLIENT is None:
        from src.adapters.hy4_client import HY4Client
        sh = _shard()
        _CLIENT = HY4Client(tag="qe",
                            cache_path=OUT / "artifacts" / f"hy4_cache{sh}.sqlite",
                            log_path=OUT / "logs" / f"hy4_calls{sh}.jsonl")
    return _CLIENT


def _cost_record(rec: Dict) -> None:
    with _LOCK:
        with open(LOGS / f"call_costs{_shard()}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _call(messages: List[Dict], *, stage: str, qid: Optional[str], purpose: str,
          max_tokens: int = 512, json_mode: bool = False, use_cache: bool = True,
          temperature: float = 0.0) -> Dict:
    t0 = time.time()
    out = client().chat(messages, max_tokens=max_tokens, temperature=temperature,
                        json_mode=json_mode, use_cache=use_cache, tag=f"qe_{stage}")
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage, "qid": qid,
        "purpose": purpose, "cached": bool(out.get("cached")),
        "usage": out.get("usage"), "latency_s": out.get("latency_s"),
        "finish_reason": out.get("finish_reason"),
    }
    _cost_record(rec)
    out["wall_s"] = round(time.time() - t0, 3)
    return out


def call_text(messages: List[Dict], *, stage: str, qid: Optional[str], purpose: str,
              max_tokens: int = 512, use_cache: bool = True) -> Dict:
    out = _call(messages, stage=stage, qid=qid, purpose=purpose, max_tokens=max_tokens,
                json_mode=False, use_cache=use_cache)
    return {"text": out["content"], "usage": out.get("usage"), "cached": out.get("cached"),
            "latency_s": out.get("latency_s"), "wall_s": out.get("wall_s"),
            "finish_reason": out.get("finish_reason")}


def call_json(messages: List[Dict], *, stage: str, qid: Optional[str], purpose: str,
              max_tokens: int = 900, use_cache: bool = True) -> Dict:
    out = _call(messages, stage=stage, qid=qid, purpose=purpose, max_tokens=max_tokens,
                json_mode=True, use_cache=use_cache)
    from src.adapters.hy4_client import parse_json_loose
    parsed = parse_json_loose(out["content"])
    repaired = False
    if parsed is None:
        repair = messages + [
            {"role": "assistant", "content": out["content"][:3000]},
            {"role": "user", "content": "Return ONLY valid JSON, no markdown and no commentary."},
        ]
        out2 = _call(repair, stage=stage, qid=qid, purpose=purpose + "_repair",
                     max_tokens=max_tokens, json_mode=True, use_cache=use_cache)
        parsed = parse_json_loose(out2["content"])
        repaired = True
        out = out2
    return {"json": parsed, "usage": out.get("usage"), "cached": out.get("cached"),
            "latency_s": out.get("latency_s"), "wall_s": out.get("wall_s"),
            "repaired": repaired, "raw": out["content"]}


# --------------------------------------------------------------------------- prompts
SKILL_BLOCK_HEADER = (
    "\nLearned answer strategies from previous questions (apply ONLY the strategies "
    "whose trigger conditions match this question; ignore the rest):\n"
)


def format_skill_block(skills: Sequence[Dict]) -> str:
    if not skills:
        return ""
    lines = []
    include_trigger = os.environ.get("QE_INCLUDE_TRIGGER") == "1"
    for s in skills:
        trig = str(s.get("trigger_conditions") or "").strip()
        action = str(s.get("action_instructions") or "").strip()
        if include_trigger and trig:
            lines.append(f"- [{s['skill_id']} v{s.get('version', 1)}] Apply only when: {trig} "
                         f"Rule: {action}")
        else:
            lines.append(f"- [{s['skill_id']} v{s.get('version', 1)}] {action}")
    return SKILL_BLOCK_HEADER + "\n".join(lines) + "\n"


def render_prompt(question: str, context: str, skills: Sequence[Dict] = (),
                  pid: str = "P7") -> Tuple[str, str]:
    spec = RP.get(pid)
    user = spec.template.format(context=context, question=question)
    if skills:
        block = format_skill_block(skills)
        marker = "\n\nRules:"
        if marker in user:
            user = user.replace(marker, block + marker, 1)
        else:
            user = user.replace("\n\nAnswer:", block + "\n\nAnswer:", 1)
    return spec.system, user


def answer(qidx: int, skills: Sequence[Dict] = (), *, pid: str = "P7",
           use_cache: bool = True, stage: str = "answer",
           context_override: Optional[Tuple[str, List[str], int]] = None,
           purpose: str = "reader") -> Dict:
    q = qa_map()[qidx]
    if context_override is None:
        context, sel, ntok = record_budget_docs(qidx)
    else:
        context, sel, ntok = context_override
    system, user = render_prompt(q.question, context, skills, pid=pid)
    out = call_text([{"role": "system", "content": system}, {"role": "user", "content": user}],
                    stage=stage, qid=q.qid, purpose=purpose, max_tokens=256, use_cache=use_cache)
    pred = parse_answer(pid, out["text"])
    rec = {
        "qid": q.qid, "qidx": qidx, "question": q.question, "category": int(q.category),
        "pred_answer": pred, "raw": out["text"], "usage": out["usage"],
        "cached": out["cached"], "latency_s": out["latency_s"], "finish_reason": out["finish_reason"],
        "context_docs": sel, "context_tokens": ntok, "skills_used": [s["skill_id"] for s in skills],
        "prompt_sha256": __import__("hashlib").sha256((system + "\n" + user).encode()).hexdigest(),
        "n_llm_calls": 1,
    }
    rec.update(score_answer(pred, qidx))
    return rec


def answer_context_cached(qidx: int, pid: str = "P7") -> Optional[Dict]:
    """Reuse the frozen P0/P7 rows produced by the prompt-optimization study."""
    path = (ROOT / "outputs" / "prompt_optimization_conv30" / "joint"
            / f"{BASE_VARIANT}_{pid}.jsonl")
    if "baseline_rows" not in _CACHE:
        _CACHE["baseline_rows"] = {}
    key = (pid,)
    if key not in _CACHE["baseline_rows"]:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        _CACHE["baseline_rows"][key] = {int(r["qidx"]): r for r in rows}  # type: ignore[index]
    row = _CACHE["baseline_rows"][key].get(qidx)  # type: ignore[index]
    if row is None:
        return None
    pred = row.get("pred_answer", "")
    rec = dict(row)
    rec["skills_used"] = []
    rec["n_llm_calls"] = 1
    rec["cached_reuse"] = True
    rec.update(score_answer(pred, qidx))
    return rec


# --------------------------------------------------------------------------- files
def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def evidence_metrics(qidx: int, ranked_ids: Sequence[str], budget: int = 4096) -> Dict:
    """Session/evidence metrics for one question; gold is only used for scoring."""
    if category_of(qidx) == 5:
        return {}
    gold = set(gold_doc_ids(qidx))
    top3 = set(ranked_ids[:3])
    top5 = set(ranked_ids[:5])
    ctx, sel, ntok = build_context_from_ranked(ranked_ids, budget)
    sel_set = set(sel)
    by_dia = {t.dia_id: t for s in conv().sessions for t in s.turns}
    turns = gold_evidence_turns(qidx)
    hit = sum(1 for d in turns if f"conv-30::s{by_dia[d].session_id:02d}" in sel_set) if turns else 0
    full = float(hit == len(turns)) if turns else float("nan")
    return {
        "recall@3": len(top3 & gold) / len(gold) if gold else float("nan"),
        "recall@5": len(top5 & gold) / len(gold) if gold else float("nan"),
        "evidence_full@4096": full,
        "evidence_partial@4096": (hit / len(turns)) if turns else float("nan"),
        "multi_evidence_complete": full if len(turns) >= 2 else float("nan"),
        "selected_docs": sel, "context_tokens": ntok, "n_evidence": len(turns),
    }


def answer_support(pred: str, context: str) -> float:
    """Cheap groundedness proxy: fraction of content tokens present in the context."""
    if not pred or is_refusal(pred):
        return 1.0
    toks = [t for t in re.findall(r"[a-z0-9]+", pred.lower()) if len(t) > 2]
    if not toks:
        return 1.0
    ctx = set(re.findall(r"[a-z0-9]+", context.lower()))
    return sum(1 for t in toks if t in ctx) / len(toks)
