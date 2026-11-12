"""Retrieval and QA metrics aligned with HippoRAG's Recall / EM / F1."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

HIPPO_SRC = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")
if str(HIPPO_SRC) not in sys.path:
    sys.path.insert(0, str(HIPPO_SRC))

from hipporag.utils.eval_utils import normalize_answer  # noqa: E402


def passage_recall(gold_ids: Sequence[str], ranked_ids: Sequence[str], ks: Sequence[int]) -> Dict[str, float]:
    gold = [item for item in gold_ids if item]
    out = {}
    for k in ks:
        if not gold:
            out[f"Recall@{k}"] = 0.0
            out[f"CompleteSupportRecall@{k}"] = 0.0
            continue
        top = set(ranked_ids[:k])
        hit = len(top & set(gold))
        out[f"Recall@{k}"] = hit / len(set(gold))
        out[f"CompleteSupportRecall@{k}"] = 1.0 if set(gold) <= top else 0.0
    return out


def exact_match(prediction: str, gold_answers: Iterable[str]) -> float:
    pred = normalize_answer(prediction or "")
    scores = [1.0 if pred == normalize_answer(gold) else 0.0 for gold in gold_answers]
    return max(scores) if scores else 0.0


def token_f1(prediction: str, gold_answers: Iterable[str]) -> float:
    pred_tokens = normalize_answer(prediction or "").split()

    def one(gold: str) -> float:
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            return 0.0
        common = Counter(pred_tokens) & Counter(gold_tokens)
        same = sum(common.values())
        if same == 0:
            return 0.0
        precision = same / len(pred_tokens)
        recall = same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    scores = [one(gold) for gold in gold_answers]
    return max(scores) if scores else 0.0


def aggregate_questions(rows: List[Dict], ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    if not rows:
        return {"n": 0}
    totals = defaultdict(float)
    hops = defaultdict(lambda: defaultdict(float))
    hop_n = defaultdict(int)
    for row in rows:
        totals["n"] += 1
        totals["em"] += row.get("em", 0.0)
        totals["f1"] += row.get("f1", 0.0)
        hop = str(row.get("hop") or "na")
        hop_n[hop] += 1
        hops[hop]["em"] += row.get("em", 0.0)
        hops[hop]["f1"] += row.get("f1", 0.0)
        for k in ks:
            totals[f"Recall@{k}"] += row.get(f"Recall@{k}", 0.0)
            totals[f"CompleteSupportRecall@{k}"] += row.get(f"CompleteSupportRecall@{k}", 0.0)
            hops[hop][f"Recall@{k}"] += row.get(f"Recall@{k}", 0.0)
    n = totals["n"]
    summary = {key: (value / n if key != "n" else n) for key, value in totals.items()}
    summary["by_hop"] = {
        hop: {key: (value / hop_n[hop]) for key, value in metrics.items()}
        for hop, metrics in hops.items()
    }
    return summary
