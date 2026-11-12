"""Metrics for reading groups (experiment B)."""
from __future__ import annotations

from typing import Dict, List, Sequence, Set

import numpy as np

from ..dataset.locomo import QAItem


def grouping_metrics(groups: List[List[str]], qa_items: Sequence[QAItem],
                     token_sizes: Dict[str, int], doc_to_session: Dict[str, int]) -> Dict:
    full = partial = 0
    cross_total = cross_ok = 0
    n_eval = 0
    for q in qa_items:
        gold = {f"{q.conversation_id}::s{sid:02d}" for sid in q.evidence_sessions}
        if not gold:
            continue
        n_eval += 1
        best = 0.0
        for g in groups:
            inter = len(gold & set(g))
            best = max(best, inter / len(gold))
        partial += best
        full += float(best == 1.0)
        if len(gold) > 1:
            cross_total += 1
            cross_ok += float(best == 1.0)

    sizes = [sum(token_sizes[d] for d in g) for g in groups]
    return {
        "n_qa": n_eval,
        "gold_full_coverage": full / n_eval if n_eval else float("nan"),
        "gold_partial_coverage": partial / n_eval if n_eval else float("nan"),
        "cross_session_qa": cross_total,
        "cross_session_colocation": cross_ok / cross_total if cross_total else float("nan"),
        "n_groups": len(groups),
        "avg_group_tokens": float(np.mean(sizes)) if sizes else 0.0,
        "max_group_tokens": int(max(sizes)) if sizes else 0,
        "min_group_tokens": int(min(sizes)) if sizes else 0,
        "total_reading_tokens": int(sum(sizes)),
        "duplicate_reading_tokens": int(sum(sizes) - sum(token_sizes.values())),
        "reading_cost_ratio": sum(sizes) / max(1, sum(token_sizes.values())),
        "avg_group_size": float(np.mean([len(g) for g in groups])) if groups else 0.0,
        "max_group_size": max((len(g) for g in groups), default=0),
    }
