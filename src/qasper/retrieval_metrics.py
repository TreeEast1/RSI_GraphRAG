"""Retrieval/evidence metrics with explicit multi-annotator handling.

QASPER gives each question several annotators, each with their own evidence set.
Those sets are **alternatives**: a system that retrieves annotator A's paragraph
and not annotator B's has fully answered the question. Therefore every metric is
computed per annotator set and averaged over sets (``*_per_annot``). Union-based
numbers are also reported, but only as a secondary view.

Conventions
-----------
* Any-Evidence@k  1 if at least one paragraph of the set is in the top-k.
* All-Evidence@k  1 if the whole set is in the top-k.
* Evidence Recall@k  |set ∩ top-k| / |set|.
* Evidence P@k  |set ∩ top-k| / k; F1@k from P/R with R = recall.
* MRR  reciprocal rank of the first set paragraph.
* nDCG@k  binary gains over the set.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

K_LIST = (1, 3, 5, 10)


def _first_hit(ranked: Sequence[str], s: Set[str]) -> int:
    for i, x in enumerate(ranked, 1):
        if x in s:
            return i
    return 0


def per_set_metrics(ranked: Sequence[str], s: Set[str], k_list: Sequence[int] = K_LIST) -> Dict[str, float]:
    out: Dict[str, float] = {}
    ranked = list(ranked)
    for k in k_list:
        top = set(ranked[:k])
        inter = len(top & s)
        out[f"any@{k}"] = float(inter > 0)
        out[f"all@{k}"] = float(s <= top) if s else float("nan")
        out[f"recall@{k}"] = inter / len(s) if s else float("nan")
        p = inter / k if k else 0.0
        r = inter / len(s) if s else 0.0
        out[f"precision@{k}"] = p
        out[f"f1@{k}"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    rank = _first_hit(ranked, s)
    out["mrr"] = 1.0 / rank if rank else 0.0
    for k in (10,):
        dcg = sum(1.0 / math.log2(i + 1) for i, x in enumerate(ranked[:k], 1) if x in s)
        ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(s), k) + 1))
        out[f"ndcg@{k}"] = dcg / ideal if ideal else float("nan")
    return out


def question_metrics(ranked: Sequence[str], annotator_sets: Sequence[Sequence[str]],
                     k_list: Sequence[int] = K_LIST) -> Dict[str, float]:
    """Average metrics over annotator sets. Empty sets are ignored."""
    sets = [set(s) for s in annotator_sets if s]
    if not sets:
        return {}
    per = [per_set_metrics(ranked, s, k_list) for s in sets]
    out: Dict[str, float] = {}
    for key in per[0]:
        vals = [p[key] for p in per if p.get(key) is not None and not (isinstance(p[key], float) and math.isnan(p[key]))]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    out["n_annotator_sets"] = len(sets)
    # secondary union view
    union = set().union(*sets)
    out["union_all@5"] = float(union <= set(ranked[:5]))
    out["union_recall@5"] = len(union & set(ranked[:5])) / len(union) if union else float("nan")
    return out


def coverage_at_budget(ranked: Sequence[str], annotator_sets: Sequence[Sequence[str]],
                       token_counts: Dict[str, int], budget: int) -> Dict[str, float]:
    """Greedy rank-order context under a token budget: evidence coverage.

    Returns the per-annotator mean of:
      full     whole evidence set fits in the selected context
      partial  fraction of the evidence set in the context
      any      at least one evidence paragraph in the context
    """
    sets = [set(s) for s in annotator_sets if s]
    if not sets:
        return {}
    selected: List[str] = []
    used = 0
    for pid in ranked:
        n = token_counts.get(pid, 0)
        if used + n > budget:
            continue
        selected.append(pid)
        used += n
    sel = set(selected)
    out = {"n_docs": len(selected), "tokens": used}
    for name, fn in (
        ("full", lambda s: float(s <= sel)),
        ("any", lambda s: float(bool(s & sel))),
        ("partial", lambda s: len(s & sel) / len(s)),
    ):
        out[name] = float(np.mean([fn(s) for s in sets]))
    return out


def aggregate_questions(rows: Iterable[Dict], prefix: str = "") -> Dict[str, float]:
    """Average every numeric metric present in question records."""
    vals: Dict[str, List[float]] = {}
    for r in rows:
        m = r.get("metrics", r)
        for k, v in m.items():
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                vals.setdefault(k, []).append(float(v))
    return {f"{prefix}{k}": float(np.mean(v)) for k, v in vals.items()}


def bootstrap_ci(values: Sequence[float], n: int = 2000, seed: int = 0) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)]
    return {"mean": float(arr.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5))}
