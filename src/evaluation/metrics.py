"""Retrieval and QA metrics used by every experiment.

Two distinct retrieval views are always reported:
  * document (session) level  -- did we retrieve the session at all
  * evidence (turn) level     -- is the actual gold turn inside the context that
                                 would be handed to the reader under a fixed
                                 token budget
"""
from __future__ import annotations

import re
import string
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize_answer(s) -> str:
    s = str(s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred, golds) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def f1_score(pred, golds) -> float:
    p_toks = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        g_toks = normalize_answer(g).split()
        if not p_toks or not g_toks:
            best = max(best, float(p_toks == g_toks))
            continue
        common = {}
        for t in p_toks:
            common[t] = min(p_toks.count(t), g_toks.count(t))
        num_same = sum(common.values())
        if num_same == 0:
            continue
        prec = num_same / len(p_toks)
        rec = num_same / len(g_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def recall_at_k(ranked_session_ids: List[str], gold_session_ids: Set[str], k: int) -> float:
    if not gold_session_ids:
        return float("nan")
    topk = set(ranked_session_ids[:k])
    return len(topk & gold_session_ids) / len(gold_session_ids)


def mrr(ranked_session_ids: List[str], gold_session_ids: Set[str]) -> float:
    for i, sid in enumerate(ranked_session_ids, 1):
        if sid in gold_session_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_session_ids: List[str], gold_session_ids: Set[str], k: int) -> float:
    dcg = 0.0
    for i, sid in enumerate(ranked_session_ids[:k], 1):
        if sid in gold_session_ids:
            dcg += 1.0 / np.log2(i + 1)
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(gold_session_ids), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def select_by_budget(
    ranked_docs: List[Dict],
    budget_tokens: int,
    token_fn,
    all_or_nothing: bool = True,
) -> List[Dict]:
    """Greedily add documents in rank order until the token budget is exhausted.

    Each doc must be a dict with at least `document_id` and `content`.
    """
    selected, used = [], 0
    for doc in ranked_docs:
        n = token_fn(doc["content"])
        if used + n > budget_tokens:
            if all_or_nothing:
                continue
            break
        selected.append(doc)
        used += n
    return selected


def evidence_coverage(
    selected_document_ids: Set[str],
    gold_turns: Iterable[str],
    turn_to_session: Dict[str, str],
) -> Dict[str, float]:
    """Fraction of gold turns whose session made it into the reader context."""
    gold_turns = list(gold_turns)
    if not gold_turns:
        return {"full": float("nan"), "partial": float("nan"), "n": 0}
    hit = sum(1 for t in gold_turns if turn_to_session.get(t) in selected_document_ids)
    return {
        "full": float(hit == len(gold_turns)),
        "partial": hit / len(gold_turns),
        "n": len(gold_turns),
    }


def aggregate(rows: List[Dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_ci(values: List[float], n: int = 1000, seed: int = 0) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)]
    return {
        "mean": float(arr.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
    }
