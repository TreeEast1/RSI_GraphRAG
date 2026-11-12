"""QASPER QA evaluation helpers on top of the official scoring logic."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .dataset import Paper, QAItem
from .official_eval import evaluate, normalize_answer, paragraph_f1_score, token_f1_score


def gold_refs(q: QAItem) -> List[Dict]:
    refs = []
    for a in q.annotations:
        if a.unanswerable:
            refs.append({"answer": "Unanswerable", "evidence": [], "type": "none"})
            continue
        ev = [t for t in a.evidence if "FLOAT SELECTED" not in t]
        refs.append({"answer": a.answer, "evidence": ev, "type": a.answer_type})
    return refs


def score_one(pred: Dict, refs: List[Dict]) -> Dict:
    answer_f1 = max(token_f1_score(pred.get("answer", ""), r["answer"]) for r in refs)
    ev_f1 = max(paragraph_f1_score(pred.get("evidence", []), r["evidence"]) for r in refs)
    best_ans, best_type = max(
        ((token_f1_score(pred.get("answer", ""), r["answer"]), r["type"]) for r in refs),
        key=lambda x: x[0])
    em = float(any(normalize_answer(pred.get("answer", "")) == normalize_answer(r["answer"]) for r in refs))
    return {"answer_f1": answer_f1, "evidence_f1": ev_f1, "em": em, "answer_type": best_type}


def summarize_predictions(records: Sequence[Dict], subset: Optional[Sequence[str]] = None,
                           key: str = "qid") -> Dict:
    import numpy as np

    rows = list(records)
    if subset is not None:
        keep = set(subset)
        rows = [r for r in rows if r[key] in keep]
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for k in ("answer_f1", "evidence_f1", "em"):
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    for t in ("extractive", "abstractive", "boolean", "none"):
        vals = [r["answer_f1"] for r in rows if r.get("answer_type") == t]
        out[f"answer_f1_{t}"] = float(np.mean(vals)) if vals else None
        out[f"n_{t}"] = len(vals)
    return out


def scorer_for_papers(papers: Dict[str, Paper]) -> Dict[str, List[Dict]]:
    gold = {}
    for p in papers.values():
        for q in p.qas:
            gold[q.qid] = gold_refs(q)
    return gold
