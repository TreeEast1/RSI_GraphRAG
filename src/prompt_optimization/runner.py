"""Shared helpers for the prompt-optimization experiments.

Everything here is deterministic: contexts are reconstructed from frozen
retrieval records, prompts are rendered from ``reader_prompts``, and the reader
call is a single HY4 chat completion with the project's rate limiter/cache.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ..config import PATHS  # noqa: E402
from ..dataset.locomo import Conversation, load_locomo, num_tokens  # noqa: E402
from ..evaluation.metrics import exact_match, f1_score, select_by_budget  # noqa: E402
from ..prompt_optimization.official_metrics import official_qa_score  # noqa: E402
from ..retrieval.reader import READER_TEMPLATE, format_context  # noqa: E402

OUT_ROOT = ROOT / "outputs" / "prompt_optimization_conv30"
REFUSAL = ("no answer", "cannot", "not mentioned", "unknown", "unanswerable", "not stated",
           "no information available")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_conv(conv_id: str = "conv-30") -> Conversation:
    convs = {c.sample_id: c for c in load_locomo(ROOT / "data" / "raw" / "locomo" / "locomo10.json")}
    return convs[conv_id]


def session_docs(conv: Conversation) -> Dict[str, Dict]:
    return {
        s.document_id: {
            "document_id": s.document_id, "session_id": s.session_id,
            "header": f"Session {s.session_id} | {s.date_time_raw}", "content": s.text,
            "date_time": s.date_time_raw,
        }
        for s in conv.sessions
    }


def freeze_context(rec: Dict, docs: Dict[str, Dict], budget: int = 4096) -> Dict:
    """Reconstruct the exact Reader input for one frozen retrieval record."""
    ranked = [docs[d] for d in rec["ranked_docs"] if d in docs]
    selected = select_by_budget(ranked, budget, num_tokens)
    context = format_context(selected)
    return {
        "qid": rec["qid"], "qidx": int(rec["qid"].split("q")[-1]),
        "question": rec["question"], "category": rec["category"],
        "ranked_docs": rec["ranked_docs"],
        "selected_docs": [d["document_id"] for d in selected],
        "selected_sessions": [d["session_id"] for d in selected],
        "context": context,
        "context_tokens": sum(num_tokens(d["content"]) for d in selected),
        "n_docs": len(selected),
        "budget": budget,
    }


def parse_answer(pid: str, text: str) -> str:
    """Extract the answer span for every prompt style."""
    if not text:
        return ""
    pats = [r"Answer:\s*(.*)"]
    if pid == "P6":
        pats = [r"(?:Short answer|Answer)\s*:\s*(.*)", r"Answer:\s*(.*)"]
    for pat in pats:
        m = re.search(pat, text, flags=re.S | re.I)
        if m:
            return m.group(1).strip().split("\n")[0].strip()
    first = text.strip().split("\n")[0].strip()
    first = re.sub(r"^(?:Short answer|Answer)\s*:\s*", "", first, flags=re.I)
    return first


def render(spec, question: str, context: str) -> tuple[str, str]:
    user = spec.template.format(context=context, question=question)
    return spec.system, user


def is_refusal(ans: str) -> bool:
    low = str(ans or "").lower()
    return any(k in low for k in REFUSAL)


def evaluate(pred: str, gold: str, category: int) -> Dict:
    off = official_qa_score(pred, gold, category)
    if category == 5:
        refused = is_refusal(pred)
        return {"em": float(refused), "f1": float(refused), "refused": refused, **off}
    return {"em": exact_match(pred, [gold]), "f1": f1_score(pred, [gold]),
            "refused": is_refusal(pred), **off}


def mean(values: Iterable[Optional[float]]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def summarize_prompt(rows: List[Dict]) -> Dict:
    ans = [r for r in rows if r["category"] != 5]
    adv = [r for r in rows if r["category"] == 5]
    out = {
        "n": len(rows), "n_answerable": len(ans), "n_adversarial": len(adv),
        "project_em": mean(r["em"] for r in ans),
        "project_f1": mean(r["f1"] for r in ans),
        "official_f1": mean(r["official_f1"] for r in ans),
        "official_em": mean(r["official_em"] for r in ans),
        "refusal_rate_answerable": mean(float(r["refused"]) for r in ans),
        "adversarial_refusal_project": mean(r["em"] for r in adv),
        "adversarial_refusal_official": mean(r.get("official_refused_adapted") for r in adv),
        "mean_answer_words": mean(len(str(r.get("pred_answer") or "").split()) for r in rows),
        "mean_completion_tokens": mean((r.get("usage") or {}).get("completion_tokens") for r in rows),
        "mean_prompt_tokens": mean((r.get("usage") or {}).get("prompt_tokens") for r in rows),
        "truncated": sum(1 for r in rows if r.get("finish_reason") == "length"),
        "errors": sum(1 for r in rows if r.get("error")),
        "cache_hits": sum(1 for r in rows if r.get("cached")),
        "by_category": {},
    }
    for c in sorted({int(r["category"]) for r in ans}):
        sub = [r for r in ans if int(r["category"]) == c]
        out["by_category"][str(c)] = {
            "n": len(sub), "em": mean(r["em"] for r in sub),
            "f1": mean(r["f1"] for r in sub), "official_f1": mean(r["official_f1"] for r in sub),
            "refusal_rate": mean(float(r["refused"]) for r in sub),
        }
    return out


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
