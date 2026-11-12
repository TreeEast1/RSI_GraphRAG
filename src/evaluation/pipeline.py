"""Shared evaluation helpers: turn ranked documents into metrics + reader calls."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from ..dataset.locomo import CATEGORY_NAMES, Conversation, QAItem, num_tokens
from .metrics import (evidence_coverage, exact_match, f1_score, mrr,
                      ndcg_at_k, recall_at_k, select_by_budget)

K_LIST = [1, 3, 5, 10, 20]


def session_doc_id(conv_id: str, session_id: int) -> str:
    return f"{conv_id}::s{session_id:02d}"


def turn_to_session(conv: Conversation) -> Dict[str, str]:
    return {t.dia_id: s.document_id for s in conv.sessions for t in s.turns}


def ranked_metrics(ranked_doc_ids: List[str], gold_doc_ids: Set[str]) -> Dict[str, float]:
    out = {}
    for k in K_LIST:
        out[f"recall@{k}"] = recall_at_k(ranked_doc_ids, gold_doc_ids, k)
    out["mrr"] = mrr(ranked_doc_ids, gold_doc_ids)
    for k in (5, 10):
        out[f"ndcg@{k}"] = ndcg_at_k(ranked_doc_ids, gold_doc_ids, k)
    # evidence-in-candidate-list (any turn of a gold session counts, all docs listed)
    return out


def budget_metrics(
    ranked_docs: List[Dict],
    qa: QAItem,
    t2s: Dict[str, str],
    budgets: Sequence[int] = (2048, 4096, 8192),
) -> Dict[str, Dict]:
    """For each budget: which sessions/turns make it into the reader context."""
    gold_sessions = {session_doc_id(qa.conversation_id, s) for s in qa.evidence_sessions}
    out = {}
    for b in budgets:
        sel = select_by_budget(ranked_docs, b, num_tokens)
        sel_ids = {d["document_id"] for d in sel}
        cov = evidence_coverage(sel_ids, qa.evidence_turns, t2s)
        out[str(b)] = {
            "n_docs": len(sel),
            "tokens": sum(num_tokens(d["content"]) for d in sel),
            "session_recall_full": float(gold_sessions <= sel_ids) if gold_sessions else None,
            "session_recall_at_least_one": float(bool(gold_sessions & sel_ids)) if gold_sessions else None,
            "evidence_full": cov["full"],
            "evidence_partial": cov["partial"],
            "selected_ids": sorted(sel_ids),
        }
    return out


def question_record(
    method: str,
    conv: Conversation,
    qa: QAItem,
    ranked_docs: List[Dict],
    *,
    budgets=(2048, 4096, 8192),
    reader_answer: Optional[str] = None,
    reader_usage: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    t2s = turn_to_session(conv)
    ranked_ids = [d["document_id"] for d in ranked_docs]
    gold = {session_doc_id(conv.conversation_id, s) for s in qa.evidence_sessions}
    rec = {
        "method": method,
        "conv_id": conv.conversation_id,
        "qid": qa.qid,
        "question": qa.question,
        "gold_answer": qa.answer,
        "category": qa.category,
        "category_name": CATEGORY_NAMES.get(qa.category, "?"),
        "gold_sessions": qa.evidence_sessions,
        "gold_doc_ids": sorted(gold),
        "gold_turns": qa.evidence_turns,
        "n_evidence": len(qa.evidence_turns),
        "ranked_docs": ranked_ids,
        "ranked_scores": [round(float(d.get("score", 0.0)), 4) for d in ranked_docs],
        "retrieval": ranked_metrics(ranked_ids, gold) if gold else {},
        "budget": budget_metrics(ranked_docs, qa, t2s, budgets),
    }
    if reader_answer is not None:
        rec["pred_answer"] = reader_answer
        if qa.answerable:
            rec["em"] = exact_match(reader_answer, [qa.answer])
            rec["f1"] = f1_score(reader_answer, [qa.answer])
        else:
            # category 5: correct behaviour is to refuse / not hallucinate the adversarial answer
            refused = any(k in (reader_answer or "").lower() for k in ("no answer", "cannot", "not mentioned", "unknown", "unanswerable"))
            rec["em"] = float(refused)
            rec["f1"] = float(refused)
            rec["refused"] = refused
        rec["reader_usage"] = reader_usage or {}
    if extra:
        rec["extra"] = extra
    return rec


def write_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dig(row: Dict, path: Sequence[str]):
    cur = row
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _mean(rows: List[Dict], path: Sequence[str]) -> Optional[float]:
    import numpy as np

    vals = []
    for r in rows:
        v = _dig(r, path)
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def summarize(rows: List[Dict], method: str) -> Dict:
    out = {"method": method, "n": len(rows)}
    for k in K_LIST:
        out[f"recall@{k}"] = _mean(rows, ["retrieval", f"recall@{k}"])
    out["mrr"] = _mean(rows, ["retrieval", "mrr"])
    out["ndcg@10"] = _mean(rows, ["retrieval", "ndcg@10"])
    for b in ("2048", "4096", "8192"):
        out[f"evidence_full@{b}"] = _mean(rows, ["budget", b, "evidence_full"])
        out[f"evidence_partial@{b}"] = _mean(rows, ["budget", b, "evidence_partial"])
        out[f"session_full@{b}"] = _mean(rows, ["budget", b, "session_recall_full"])
    out["em"] = _mean(rows, ["em"])
    out["f1"] = _mean(rows, ["f1"])
    # main QA metrics are reported on answerable questions (categories 1-4);
    # category 5 (adversarial) is scored as refusal correctness.
    answerable = [r for r in rows if r.get("category") != 5]
    adversarial = [r for r in rows if r.get("category") == 5]
    out["em_answerable"] = _mean(answerable, ["em"])
    out["f1_answerable"] = _mean(answerable, ["f1"])
    out["n_answerable"] = len(answerable)
    out["refusal_rate"] = _mean(adversarial, ["em"])
    out["n_adversarial"] = len(adversarial)

    by_cat: Dict[str, List[Dict]] = {}
    for r in rows:
        by_cat.setdefault(str(r.get("category")), []).append(r)
    out["by_category"] = {
        cat: {
            "n": len(sub),
            "em": _mean(sub, ["em"]),
            "f1": _mean(sub, ["f1"]),
            "recall@5": _mean(sub, ["retrieval", "recall@5"]),
            "evidence_full@4096": _mean(sub, ["budget", "4096", "evidence_full"]),
        }
        for cat, sub in sorted(by_cat.items())
    }
    # per-conversation
    by_conv: Dict[str, List[Dict]] = {}
    for r in rows:
        by_conv.setdefault(r["conv_id"], []).append(r)
    out["by_conversation"] = {
        c: {"n": len(sub), "em": _mean(sub, ["em"]), "f1": _mean(sub, ["f1"]),
            "recall@5": _mean(sub, ["retrieval", "recall@5"])}
        for c, sub in sorted(by_conv.items())
    }
    return out
