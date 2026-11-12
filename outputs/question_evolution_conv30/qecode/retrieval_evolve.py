"""Retrieval-Skill self-evolution.

The base ranking is always the frozen COMP-P4_top5 HippoRAG2 ranking.  A
retrieval skill only adds *extra queries*; their dense session rankings are
merged with the base graph ranking by reciprocal rank fusion.  Gold evidence is
used only in adaptation/verification, never at inference.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import common
from . import skills as S
from .answer_evolve import load_embedder

QDM_SYSTEM = (
    "You decompose questions about a long conversation into independent retrieval sub-questions. "
    "Only decompose when the question genuinely requires chaining several facts. Output JSON."
)
QDM_PROMPT = """Question: {question}

Decide whether this question needs decomposition before retrieval.
- Single-hop factual questions do NOT need decomposition.
- Multi-hop questions that chain two or more facts DO need it.
- Temporal questions about a sequence of events may need it.

Return JSON:
{{"needs_decomposition": true/false,
  "sub_questions": ["..."],
  "reason": "<=20 words"}}
"""

SUFF_SYSTEM = (
    "You diagnose whether the retrieved sessions for a question still miss a relation chain. "
    "You never invent facts. Output JSON."
)
SUFF_PROMPT = """Question: {question}
Sub-questions answered so far: {subqs}

Retrieved session summaries (best first):
{evidence}

Task: identify at most {max_new} relation chains that are still missing to answer the question.
A missing chain means the retrieved sessions mention some entities of the question but not the
relation that connects them. If the evidence already covers the question, return an empty list.

Return JSON:
{{"missing": [{{"query": "<short retrieval query>", "why": "<=15 words"}}],
  "sufficient": true/false}}
"""

RET_REFLECT_SYSTEM = (
    "You are a reflective retrieval-skill learner. You see one training question whose gold "
    "evidence was not fully retrieved. Propose one reusable query-planning skill. Never store "
    "specific people, places, events, dates, or gold answers. Output JSON only."
)
RET_REFLECT_PROMPT = """Training question: {question}
Category: {category_name} ({category})

Base graph sessions (top 5):
{base_sessions}

Sessions after the current retrieval pipeline (top 5):
{final_sessions}

Evidence the reader needed (gold, training only):
{gold_evidence}

Gold sessions that are still missing from the final candidate list: {missing}

Current retrieval skills:
{current_skills}

Propose one general retrieval skill that would make this class of missing evidence retrievable.
The skill must add queries, not replace the base graph search. Query templates may use
{{question}} and {{entities}} placeholders. The categories list MUST include the source
question's category ({category}); otherwise the skill can never trigger on this question.

Return JSON:
{{"title": "...",
  "trigger_conditions": "...",
  "applicable_question_patterns": ["..."],
  "categories": [2],
  "action_instructions": "<=45 words",
  "query_templates": ["{{question}} ...", "{{entities}} ..."],
  "failure_modes": ["..."]}}
"""


def _tok(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if len(t) > 2]


def merge_rrf(lists: Sequence[Sequence[str]], k: int = 60, top: int = 20) -> List[str]:
    scores: Dict[str, float] = {}
    for lst in lists:
        for rank, did in enumerate(lst):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:top]]


class DenseIndex:
    def __init__(self, embedder=None):
        self.embedder = embedder or load_embedder()
        self.doc_ids = [s.document_id for s in common.conv().sessions]
        self.texts = [common.docs_by_id()[d]["content"] for d in self.doc_ids]
        self.matrix = self.embedder.encode_documents(self.texts)

    def retrieve(self, query: str, k: int = 10) -> List[str]:
        if not query.strip():
            return []
        qv = self.embedder.encode_queries([query])[0]
        sims = self.matrix @ qv
        order = np.argsort(-sims)[:k]
        return [self.doc_ids[i] for i in order]


_DENSE: Optional[DenseIndex] = None


def dense() -> DenseIndex:
    global _DENSE
    if _DENSE is None:
        _DENSE = DenseIndex()
    return _DENSE


def qdm(question: str, qid: str, use_cache: bool = True) -> Dict:
    out = common.call_json(
        [{"role": "system", "content": QDM_SYSTEM},
         {"role": "user", "content": QDM_PROMPT.format(question=question)}],
        stage="qdm", qid=qid, purpose="qdm", max_tokens=400, use_cache=use_cache)
    d = out.get("json") or {}
    subqs = [str(x).strip() for x in (d.get("sub_questions") or []) if str(x).strip()]
    needs = bool(d.get("needs_decomposition"))
    return {"needs_decomposition": needs, "sub_questions": subqs[:4] if needs else [],
            "reason": str(d.get("reason", ""))[:120], "usage": out.get("usage"),
            "cached": out.get("cached")}


def sufficiency(question: str, subqs: Sequence[str], ranked_ids: Sequence[str],
                qid: str, max_new: int = 2, use_cache: bool = True) -> Dict:
    prompt = SUFF_PROMPT.format(question=question,
                               subqs="; ".join(subqs) if subqs else "(none)",
                               evidence=common.compact_sessions(ranked_ids, k=5, limit=360),
                               max_new=max_new)
    out = common.call_json(
        [{"role": "system", "content": SUFF_SYSTEM}, {"role": "user", "content": prompt}],
        stage="suff", qid=qid, purpose="sufficiency", max_tokens=500, use_cache=use_cache)
    d = out.get("json") or {}
    missing = []
    for m in d.get("missing") or []:
        if isinstance(m, dict) and m.get("query"):
            missing.append({"query": str(m["query"]).strip(),
                            "why": str(m.get("why", ""))[:80]})
    return {"missing": missing[:max_new], "sufficient": bool(d.get("sufficient", not missing)),
            "usage": out.get("usage"), "cached": out.get("cached")}


def instantiate_templates(skill: Dict, question: str) -> List[str]:
    ents = " ".join(common.entities_in(question))
    out = []
    for t in (skill.get("query_templates") or [])[:4]:
        q = str(t).replace("{question}", question).replace("{entities}", ents).strip()
        if q and q != question:
            out.append(q)
    return out


def _skill_vecs(skills: Sequence[Dict], embedder) -> Dict[str, np.ndarray]:
    if embedder is None or not skills:
        return {}
    texts = [s.get("title", "") + ". " + s.get("trigger_conditions", "") + ". "
             + s.get("action_instructions", "") + ". " + " ".join(s.get("query_templates") or [])
             for s in skills]
    vecs = embedder.encode_queries(texts)
    return {s["skill_id"]: vecs[i] for i, s in enumerate(skills)}


def select_retrieval_skills(library: Sequence[Dict], question: str, category: int,
                            embedder, k: int = 2) -> List[Dict]:
    return S.select_skills(library, question, category, k=k, embedder=embedder,
                           skill_vecs=_skill_vecs(library, embedder))


def retrieval_pipeline(qidx: int, library: Sequence[Dict] = (), *, mode: str = "full",
                       use_cache: bool = True, embedder=None) -> Dict:
    """mode: base | qdm | qdm_suff | full (qdm + suff + skills)."""
    q = common.qa_map()[qidx]
    base = common.base_ranked(qidx)
    trace = {"qidx": qidx, "qid": q.qid, "category": int(q.category), "mode": mode,
             "base_ranked": base, "queries": [], "skills_used": [], "qdm": None,
             "sufficiency": None, "n_llm_calls": 0, "usage": {}}
    if mode == "base":
        trace["final_ranked"] = base
        trace.update(common.evidence_metrics(qidx, base))
        return trace

    ranked = list(base)
    qd = qdm(q.question, q.qid, use_cache=use_cache)
    trace["qdm"] = qd
    trace["n_llm_calls"] += 1
    trace["usage"] = _add_usage(trace["usage"], qd.get("usage"))
    lists = [base]
    if qd["needs_decomposition"] and qd["sub_questions"]:
        for sq in qd["sub_questions"][:3]:
            ids = dense().retrieve(sq, 10)
            lists.append(ids)
            trace["queries"].append({"query": sq, "source": "qdm", "docs": ids})
        ranked = merge_rrf(lists, top=20)

    if mode == "qdm":
        trace["final_ranked"] = ranked
        trace.update(common.evidence_metrics(qidx, ranked))
        return trace

    suf = sufficiency(q.question, qd["sub_questions"], ranked, q.qid, use_cache=use_cache)
    trace["sufficiency"] = suf
    trace["n_llm_calls"] += 1
    trace["usage"] = _add_usage(trace["usage"], suf.get("usage"))

    if (mode in ("full", "qdm_suff")) and not suf["sufficient"] and library:
        sel = select_retrieval_skills(library, q.question, int(q.category), embedder, k=2)
        trace["skills_used"] = [{"skill_id": s["skill_id"], "score": s.get("_select_score"),
                                 "reasons": s.get("_select_reasons")} for s in sel]
        lists2 = [ranked]
        for s in sel:
            for query in instantiate_templates(s, q.question)[:3]:
                ids = dense().retrieve(query, 10)
                lists2.append(ids)
                trace["queries"].append({"query": query, "source": s["skill_id"], "docs": ids})
        for m in suf["missing"][:2]:
            ids = dense().retrieve(m["query"], 10)
            lists2.append(ids)
            trace["queries"].append({"query": m["query"], "source": "missing_relation", "docs": ids})
        if len(lists2) > 1:
            ranked = merge_rrf(lists2, top=20)
    trace["final_ranked"] = ranked
    trace.update(common.evidence_metrics(qidx, ranked))
    return trace


def _add_usage(a: Dict, b: Optional[Dict]) -> Dict:
    a = dict(a or {})
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        a[k] = int(a.get(k, 0)) + int((b or {}).get(k, 0) or 0)
    return a


def verify_retrieval_candidate(cand: Dict, qidx: int, embedder,
                               history: Dict[int, Dict], order: Sequence[int]) -> Dict:
    """Local, LLM-free verification of a retrieval-skill candidate."""
    base = common.base_ranked(qidx)
    m_base = common.evidence_metrics(qidx, base)
    cats = [int(c) for c in (cand.get("categories") or [])]
    if cats and int(common.category_of(qidx)) not in cats:
        return {"accepted": False, "reason": "category_mismatch", "queries": []}
    queries = instantiate_templates(cand, common.qa_map()[qidx].question)
    if not queries:
        return {"accepted": False, "reason": "no_query_templates", "queries": []}
    lists = [base] + [dense().retrieve(q, 10) for q in queries]
    cand_ranked = merge_rrf(lists, top=20)
    m_cand = common.evidence_metrics(qidx, cand_ranked)

    def delta(a: Dict, b: Dict) -> Dict:
        return {"evidence_full": (b.get("evidence_full@4096", float("nan"))
                                  - a.get("evidence_full@4096", float("nan"))),
                "partial": (b.get("evidence_partial@4096", float("nan"))
                            - a.get("evidence_partial@4096", float("nan"))),
                "recall@5": b.get("recall@5", float("nan")) - a.get("recall@5", float("nan"))}

    src_delta = delta(m_base, m_cand)
    fix_pass = (m_base.get("evidence_full@4096", 0) == 0
                and m_cand.get("evidence_full@4096", 0) == 1) or \
               (src_delta["partial"] == src_delta["partial"] and src_delta["partial"] >= 0.25) or \
               (src_delta["recall@5"] == src_delta["recall@5"] and src_delta["recall@5"] >= 0.5)

    cat = common.category_of(qidx)
    seen = [q for q in order if q in history and q != qidx]
    same = [q for q in seen if common.category_of(q) == cat]
    same.sort(key=lambda x: -S._jaccard(common.qa_map()[qidx].question,
                                        common.qa_map()[x].question))
    neg = []
    for qq in same[:3]:
        b = common.evidence_metrics(qq, common.base_ranked(qq))
        r = merge_rrf([common.base_ranked(qq)]
                      + [dense().retrieve(x, 10)
                         for x in instantiate_templates(cand, common.qa_map()[qq].question)], top=20)
        c = common.evidence_metrics(qq, r)
        if b.get("evidence_full@4096") == 1 and c.get("evidence_full@4096") != 1:
            neg.append({"qidx": qq, "type": "full_to_not_full"})
        elif (b.get("recall@5", 1) - c.get("recall@5", 1)) > 0.30:
            neg.append({"qidx": qq, "type": "recall_drop", "before": b.get("recall@5"),
                        "after": c.get("recall@5")})
    checks = {"fix_pass": bool(fix_pass), "no_negative_transfer": len(neg) == 0}
    return {"accepted": bool(fix_pass and not neg), "checks": checks,
            "queries": queries, "source_before": m_base, "source_after": m_cand,
            "source_delta": src_delta, "negative_transfers": neg}


def reflect_retrieval(qidx: int, trace: Dict, library: Sequence[Dict], use_cache: bool = True) -> Dict:
    q = common.qa_map()[qidx]
    missing = [d for d in common.gold_doc_ids(qidx)
               if d not in list(trace.get("final_ranked", []))[:5]]
    cur = "\n".join(f"- [{s['skill_id']}] {s.get('action_instructions', '')} | "
                    f"{s.get('query_templates')}" for s in library[:6]) or "(none)"
    prompt = RET_REFLECT_PROMPT.format(
        question=q.question, category_name=common.CATEGORY_NAMES.get(int(q.category), "?"),
        category=int(q.category),
        base_sessions=common.compact_sessions(trace.get("base_ranked", []), 5, 360),
        final_sessions=common.compact_sessions(trace.get("final_ranked", []), 5, 360),
        gold_evidence=common.gold_evidence_text(qidx) or "(none)",
        missing=", ".join(missing) or "(none; all gold sessions are in the top 5)",
        current_skills=cur,
    )
    out = common.call_json(
        [{"role": "system", "content": RET_REFLECT_SYSTEM}, {"role": "user", "content": prompt}],
        stage="reflect_ret", qid=q.qid, purpose="reflect_retrieval", max_tokens=900,
        use_cache=use_cache)
    d = out.get("json") or {}
    d["_usage"] = out.get("usage")
    d["_raw"] = out.get("raw", "")[:1600]
    return d


def sanitize_retrieval(raw: Dict, qidx: int) -> Tuple[Optional[Dict], Dict]:
    info = {"ok": False, "reason": None, "leak_hits": []}
    if not isinstance(raw, dict):
        info["reason"] = "not_an_object"
        return None, info
    templates = []
    for t in (raw.get("query_templates") or [])[:4]:
        t = str(t).strip()
        if t and ("{question}" in t or "{entities}" in t):
            templates.append(t[:300])
    if not templates:
        info["reason"] = "no_placeholders"
        return None, info
    cats = []
    for c in raw.get("categories") or []:
        try:
            c = int(c)
        except (TypeError, ValueError):
            continue
        if c in (1, 2, 4, 5):
            cats.append(c)
    if not cats:
        cats = [int(common.category_of(qidx))]
    if int(common.category_of(qidx)) not in cats:
        info["reason"] = "category_mismatch"
        return None, info
    cand = {
        "title": str(raw.get("title") or "Learned retrieval rule")[:90],
        "trigger_conditions": str(raw.get("trigger_conditions") or "")[:220],
        "applicable_question_patterns": [str(x)[:120] for x in
                                         (raw.get("applicable_question_patterns") or [])][:4],
        "categories": sorted(set(cats)),
        "action_instructions": str(raw.get("action_instructions") or "")[:500],
        "query_templates": templates,
        "expected_output_format": "additional ranked sessions",
        "evidence_requirements": "do not use gold evidence at inference",
        "failure_modes": [str(x)[:120] for x in (raw.get("failure_modes") or [])][:4],
    }
    text = " ".join([cand["title"], cand["trigger_conditions"], cand["action_instructions"]] +
                    cand["query_templates"] + cand["applicable_question_patterns"])
    ok, hits = S.leak_check_retrieval(text, qidx)
    info["leak_hits"] = hits
    if not ok:
        info["reason"] = "leak_check"
        return None, info
    info["ok"] = True
    return cand, info


def main():
    import os
    embedder = load_embedder()
    memory = S.RetrievalSkillMemory(common.OUT / "artifacts" / "retrieval_skill_memory.json")
    order = common.qidx_list("adaptation_order")
    max_steps = int(os.environ.get("QE_MAX_STEPS", "0") or 0)
    if max_steps:
        order = order[:max_steps]
    trace_path = common.OUT / ("skill_evolution_trace.ret.debug.jsonl" if max_steps
                               else "skill_evolution_trace.jsonl")
    history: Dict[int, Dict] = {}
    snaps: Dict[int, Dict] = {}

    memory.snapshot(0, "empty")
    dev0 = []
    for qq in common.qidx_list("development"):
        m = common.evidence_metrics(qq, common.base_ranked(qq))
        if m:
            m["qidx"] = qq
            dev0.append(m)
    dev0_summary = {"step": 0, "n_active": 0,
                    "recall@5": common.mean(r.get("recall@5") for r in dev0),
                    "evidence_full@4096": common.mean(r.get("evidence_full@4096") for r in dev0),
                    "multi_evidence_complete": common.mean(r.get("multi_evidence_complete") for r in dev0)}
    common.write_json(common.SNAP / "retrieval_dev_eval_step000.json", dev0_summary)
    for i, qidx in enumerate(order, 1):
        tr = retrieval_pipeline(qidx, memory.active(), mode="full", embedder=embedder)
        history[qidx] = tr
        ev = {"event": "adapt_retrieval", "step": i, "qid": tr["qid"], "qidx": qidx,
              "category": tr["category"], "question": common.qa_map()[qidx].question,
              "evidence_full": tr.get("evidence_full@4096"),
              "evidence_partial": tr.get("evidence_partial@4096"),
              "recall@5": tr.get("recall@5"), "n_llm_calls": tr["n_llm_calls"],
              "skills_used": tr["skills_used"], "queries": len(tr["queries"]),
              "qdm": (tr.get("qdm") or {}).get("needs_decomposition"),
              "sufficient": (tr.get("sufficiency") or {}).get("sufficient")}
        needs_learning = (
            common.category_of(qidx) != 5 and common.gold_sessions(qidx) and (
                tr.get("evidence_full@4096") != 1
                or tr.get("recall@5") != 1
                or (tr.get("n_evidence", 0) >= 2
                    and tr.get("multi_evidence_complete") != 1)))
        if needs_learning:
            raw = reflect_retrieval(qidx, tr, memory.active())
            ev["reflection"] = {"title": raw.get("title"), "diagnosis": raw.get("action_instructions"),
                                "raw": raw}
            cand, info = sanitize_retrieval(raw, qidx)
            ev["sanitize"] = info
            if cand:
                ver = verify_retrieval_candidate(cand, qidx, embedder, history, order)
                ev["verification"] = ver
                if ver["accepted"]:
                    existing = next((s for s in memory.active()
                                     if S._jaccard(s.get("action_instructions", ""),
                                                   cand.get("action_instructions", "")) > 0.8), None)
                    if existing:
                        skill = memory.update(existing["skill_id"], cand, i, qidx, mode="append")
                        ev["decision"] = "updated"
                        ev["skill_id"] = existing["skill_id"]
                    else:
                        skill = memory.add(cand, i, qidx)
                        ev["decision"] = "added"
                        ev["skill_id"] = skill["skill_id"]
                    ev["skill_version"] = skill.get("version") if skill else None
                else:
                    ev["decision"] = "rejected"
            else:
                ev["decision"] = "rejected_sanitize"
        else:
            ev["decision"] = "sufficient"
        common.append_jsonl(trace_path, ev)

        if i % 10 == 0 or i == len(order):
            p = memory.snapshot(i, f"after_{i}_feedback")
            # retrieval-only development evaluation (no LLM; base + selected skill templates)
            dev_rows = []
            for qq in common.qidx_list("development"):
                base = common.base_ranked(qq)
                sel = select_retrieval_skills(memory.active(), common.qa_map()[qq].question,
                                              common.category_of(qq), embedder, k=2)
                lists = [base]
                for s in sel:
                    for query in instantiate_templates(s, common.qa_map()[qq].question)[:3]:
                        lists.append(dense().retrieve(query, 10))
                ranked = merge_rrf(lists, top=20) if len(lists) > 1 else base
                m = common.evidence_metrics(qq, ranked)
                if m:
                    m["qidx"] = qq
                    m["skills"] = [s["skill_id"] for s in sel]
                    dev_rows.append(m)
            dev = {
                "step": i, "n_active": len(memory.active()),
                "recall@5": common.mean(r.get("recall@5") for r in dev_rows),
                "evidence_full@4096": common.mean(r.get("evidence_full@4096") for r in dev_rows),
                "multi_evidence_complete": common.mean(r.get("multi_evidence_complete")
                                                       for r in dev_rows),
            }
            snaps[i] = dev
            common.write_json(common.SNAP / f"retrieval_dev_eval_step{i:03d}.json", dev)
            common.append_jsonl(trace_path, {"event": "retrieval_snapshot", **dev,
                                             "snapshot": str(p)})
            print(f"retrieval step {i}: dev R@5={dev['recall@5']:.3f} "
                  f"Ev={dev['evidence_full@4096']:.3f} active={dev['n_active']}", flush=True)

    merges = memory.merge_duplicates(embedder, threshold=0.90, feedback_step=len(order))
    if merges:
        common.append_jsonl(trace_path, {"event": "retrieval_merge", "merges": merges})
    memory.save()
    memory.snapshot(len(order), "final_after_consolidation")
    common.write_json(common.OUT / "artifacts" / "retrieval_evolution_state.json",
                      {"snapshots": {str(k): v for k, v in snaps.items()},
                       "final_n_active": len(memory.active()),
                       "n_skills_total": len(memory.skills), "merges": merges,
                       "meta": memory.meta})
    with open(common.OUT / "retrieval_skill_memory.jsonl", "w", encoding="utf-8") as f:
        for s in memory.skills:
            f.write(json.dumps({**s, "snapshot_step": len(order)}, ensure_ascii=False) + "\n")
    print("retrieval evolution done:", len(memory.active()), "active /", len(memory.skills), "total")


if __name__ == "__main__":
    main()
