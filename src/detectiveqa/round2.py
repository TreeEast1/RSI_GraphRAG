"""Round-two DetectiveQA diagnostics.

The helpers keep evaluation-only signals separate from deployable inputs.  Gold
clues and gold answers are only used for metrics, stratification and post-hoc
judging; context builders never read them unless the condition is explicitly an
oracle diagnostic.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .data import Chunk, Question, gold_chunk_ids

LETTERS = ("A", "B", "C", "D")
CONTEXT_TOKEN_BUDGET = 4096


def render_chunk(chunk: Chunk) -> str:
    return f"[{chunk.chunk_id} | paragraphs {chunk.start_pid}-{chunk.end_pid}]\n{chunk.text}"


def tokens(text: str) -> int:
    from .reader import tokens as reader_tokens

    return reader_tokens(text)


def visible_chunks(chunks: Sequence[Chunk], cutoff: int) -> List[Chunk]:
    return [chunk for chunk in chunks if chunk.end_pid < cutoff]


def chunk_cost(chunk: Chunk) -> int:
    return tokens(render_chunk(chunk)) + 2


def recent_chunks(chunks: Sequence[Chunk], cutoff: int, budget: int) -> List[Chunk]:
    selected: List[Chunk] = []
    used = 0
    for chunk in reversed(visible_chunks(chunks, cutoff)):
        cost = chunk_cost(chunk)
        if selected and used + cost > budget:
            break
        if not selected and cost > budget:
            break
        selected.append(chunk)
        used += cost
    return sorted(selected, key=lambda c: c.index)


def pack_ranked(chunks: Sequence[Chunk], ranked_ids: Sequence[str], budget: int, *,
                exclude: Optional[Set[str]] = None) -> Tuple[List[Chunk], Dict]:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    exclude = exclude or set()
    chosen: List[Chunk] = []
    used = skipped = 0
    for chunk_id in dict.fromkeys(ranked_ids):
        chunk = by_id.get(chunk_id)
        if chunk is None or chunk_id in exclude:
            continue
        cost = chunk_cost(chunk)
        if used + cost > budget:
            skipped += 1
            continue
        chosen.append(chunk)
        used += cost
    return sorted(chosen, key=lambda c: c.index), {
        "n_selected": len(chosen), "context_tokens_est": used, "n_skipped_for_budget": skipped,
    }


def rrf(rankings: Sequence[Sequence[str]], k: int = 60, weights: Optional[Sequence[float]] = None) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    weights = weights or [1.0] * len(rankings)
    for weight, ranking in zip(weights, rankings):
        for rank, item in enumerate(dict.fromkeys(ranking), 1):
            scores[item] += float(weight) / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


def cutoff_ratio(question: Question, chunks: Sequence[Chunk]) -> float:
    last = max(chunk.end_pid for chunk in chunks) if chunks else 1
    return question.cutoff / max(1, last)


def cutoff_bucket(ratio: float) -> str:
    if ratio >= 0.90:
        return "late_ge_90"
    if ratio >= 0.70:
        return "middle_70_90"
    return "early_lt_70"


def gold_reach(context_ids: Iterable[str], gold_ids: Iterable[str]) -> Dict[str, float]:
    context = set(context_ids)
    gold = set(gold_ids)
    hit = len(context & gold)
    return {
        "gold_any_in_context": float(hit > 0),
        "gold_all_in_context": float(bool(gold) and gold <= context),
        "gold_recall_in_context": hit / len(gold) if gold else 0.0,
        "gold_hits_in_context": hit,
    }


def retrieval_metrics(ranked_ids: Sequence[str], gold_ids: Iterable[str], ks=(5, 10, 20, 50)) -> Dict[str, float]:
    gold = set(gold_ids)
    out: Dict[str, float] = {}
    for k in ks:
        top = set(ranked_ids[:k])
        hit = len(top & gold)
        out[f"any@{k}"] = float(hit > 0)
        out[f"recall@{k}"] = hit / len(gold) if gold else 0.0
        out[f"all@{k}"] = float(bool(gold) and gold <= top)
    first = next((rank for rank, item in enumerate(ranked_ids, 1) if item in gold), 0)
    out["mrr"] = 1.0 / first if first else 0.0
    out["first_gold_rank"] = first
    return out


def permute_options(question: Question, seed: int) -> Tuple[Question, Dict[str, str]]:
    """Return a question with shuffled option positions and displayed->original map."""
    if seed == 0:
        return question, {letter: letter for letter in question.options}
    original = list(question.options)
    rng = random.Random(f"{question.qid}|{seed}")
    shuffled = list(original)
    while shuffled == original:
        rng.shuffle(shuffled)
    displayed = {new: question.options[old] for new, old in zip(LETTERS, shuffled)}
    to_original = {new: old for new, old in zip(LETTERS, shuffled)}
    answer = next(new for new, old in to_original.items() if old == question.answer)
    return replace(question, options=displayed, answer=answer), to_original


def relation_text(relation: Dict) -> str:
    status = relation.get("fixed_status") or relation.get("status", "supported")
    return (f"{relation.get('subject', '')} {relation.get('predicate', '')} {relation.get('object', '')}. "
            f"Type: {relation.get('relation_type', '')}. Status: {status}. "
            f"Why: {relation.get('why_cross_chunk', '')}")


def relation_note(relation: Dict) -> str:
    ids = ", ".join(relation.get("source_chunk_ids") or [])
    return f"- [{relation.get('relation_id', '')}] {relation_text(relation)} Sources: {ids}"


def bootstrap_paired_by_novel(rows: Sequence[Dict], left: str, right: str, *, metric: str = "correct",
                              n: int = 5000, seed: int = 20261112) -> Dict:
    by_novel: Dict[str, List[float]] = defaultdict(list)
    wins = losses = 0
    for row in rows:
        a = float(row[left][metric])
        b = float(row[right][metric])
        by_novel[str(row["novel_id"])].append(b - a)
        wins += b > a
        losses += b < a
    novels = sorted(by_novel)
    observed = sum(sum(v) for v in by_novel.values()) / max(1, sum(len(v) for v in by_novel.values()))
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        draw = [by_novel[rng.choice(novels)] for _ in novels]
        samples.append(sum(sum(v) for v in draw) / max(1, sum(len(v) for v in draw)))
    samples.sort()
    return {
        "delta": observed, "ci_low": samples[int(0.025 * (n - 1))],
        "ci_high": samples[int(0.975 * (n - 1))], "wins": wins, "losses": losses,
        "mcnemar_exact_p": mcnemar_exact(wins, losses),
    }


def mcnemar_exact(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(0, min(wins, losses) + 1)) / (2 ** total)
    return min(1.0, 2 * tail)


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def gold_ids_for(question: Question, chunks: Sequence[Chunk]) -> List[str]:
    return gold_chunk_ids(question, chunks)


__all__ = [
    "CONTEXT_TOKEN_BUDGET", "bootstrap_paired_by_novel", "chunk_cost", "compact_json", "cutoff_bucket",
    "cutoff_ratio", "gold_ids_for", "gold_reach", "pack_ranked", "permute_options", "recent_chunks",
    "relation_note", "relation_text", "retrieval_metrics", "rrf", "visible_chunks",
]
