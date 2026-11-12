"""Unified DetectiveQA multiple-choice Reader and context conditions.

All conditions use the same prompt, the same model, temperature 0, the same
4096-token context budget and the same chronological rendering order.  Only the
way evidence is selected differs:

* options_only : no story text (parameter-memory / option-bias diagnostic)
* dense        : prefix-safe dense top-k, greedily packed by rank
* graph        : prefix-safe HippoRAG2 PPR top-k, greedily packed by rank
* oracle       : only chunks containing annotated gold clue paragraphs
* long_prefix  : the longest contiguous suffix of visible text before the cutoff

Selected chunks are always rendered in story order.  ``long_prefix`` drops
older text first (recency-preserving truncation), which is reported explicitly.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import tiktoken

from .data import Chunk, Question, gold_chunk_ids
from .prompts import MEMORY_SYSTEM, PROMPT_VERSION, READER, READER_SYSTEM

CONTEXT_TOKEN_BUDGET = 4096
_TOKENIZER = None
_ROOT = Path(__file__).resolve().parents[2]
# The GPU machine is offline; the only cached encoding is cl100k_base.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_ROOT / "cache" / "tiktoken"))


def tokens(text: str) -> int:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return len(_TOKENIZER.encode(text))


def render_chunk(chunk: Chunk) -> str:
    return f"[{chunk.chunk_id} | paragraphs {chunk.start_pid}-{chunk.end_pid}]\n{chunk.text}"


def pack_by_rank(chunks_in_rank_order: Sequence[Chunk], budget: int = CONTEXT_TOKEN_BUDGET,
                 truncate: str = "skip") -> Tuple[List[Chunk], Dict]:
    """Greedily pack chunks under a token budget; render later in story order.

    ``truncate='skip'`` keeps scanning after an oversized item (retrieval
    conditions); ``truncate='break'`` stops at the first item that would
    overflow (used for the recency suffix baseline).
    """
    chosen: List[Chunk] = []
    used = 0
    dropped = 0
    for chunk in chunks_in_rank_order:
        cost = tokens(render_chunk(chunk)) + 2
        if used + cost > budget:
            dropped += 1
            if truncate == "break":
                break
            continue
        chosen.append(chunk)
        used += cost
    return sorted(chosen, key=lambda c: c.index), {
        "n_selected": len(chosen), "context_tokens_est": used, "n_dropped_for_budget": dropped}


def build_context(condition: str, question: Question, chunks: Sequence[Chunk],
                  ranked_chunk_ids: Sequence[str], budget: int = CONTEXT_TOKEN_BUDGET) -> Dict:
    by_id = {c.chunk_id: c for c in chunks}
    visible = [c for c in chunks if c.end_pid < question.cutoff]
    visible_ids = {c.chunk_id for c in visible}
    if condition == "options_only" or condition == "options_only_memory":
        return {"context": "No story excerpts are provided.", "chunk_ids": [], "meta": {"n_selected": 0}}
    if condition == "dense" or condition == "graph":
        ranked = [by_id[cid] for cid in ranked_chunk_ids if cid in visible_ids]
        chosen, meta = pack_by_rank(ranked, budget)
    elif condition == "oracle":
        gold = set(gold_chunk_ids(question, chunks))
        chosen, meta = pack_by_rank([c for c in visible if c.chunk_id in gold], budget)
    elif condition == "long_prefix":
        chosen, meta = pack_by_rank(list(reversed(visible)), budget, truncate="break")
        meta["truncation"] = "keep_recency_drop_older"
    else:
        raise ValueError(f"unknown condition: {condition}")
    return {"context": "\n\n".join(render_chunk(c) for c in chosen),
            "chunk_ids": [c.chunk_id for c in chosen], "meta": meta}


def make_prompt(question: Question, context: str) -> str:
    options = "\n".join(f"{k}. {v}" for k, v in question.options.items())
    return READER.format(context=context, question=question.question, options=options)


def parse_answer(parsed: Dict, question: Question) -> Dict:
    raw = str(parsed.get("answer", "")).strip()
    match = re.search(r"\b([ABCD])\b", raw.upper())
    answer = match.group(1) if match else ""
    evidence = [str(x) for x in (parsed.get("evidence_ids") or [])]
    return {
        "answer": answer,
        "valid": answer in question.options,
        "evidence_ids": evidence,
        "insufficient_evidence": bool(parsed.get("insufficient_evidence", False)),
        "reasoning": str(parsed.get("reasoning", "")),
    }


def score_reader(pred: Dict, question: Question, gold_chunks: Sequence[str],
                 context_chunk_ids: Sequence[str]) -> Dict:
    gold = set(gold_chunks)
    supplied = set(context_chunk_ids)
    cited = set(pred["evidence_ids"]) & supplied
    return {
        "correct": float(pred["valid"] and pred["answer"] == question.answer),
        "invalid": float(not pred["valid"]),
        "insufficient": float(pred["insufficient_evidence"]),
        "evidence_any": float(bool(gold & cited)),
        "evidence_recall": len(gold & cited) / len(gold) if gold else 0.0,
        "citation_valid_ratio": len(cited) / len(pred["evidence_ids"]) if pred["evidence_ids"] else 0.0,
        "cited_gold_chunk_ids": sorted(gold & cited),
    }


def system_prompt(condition: str = "default") -> str:
    return MEMORY_SYSTEM if condition == "options_only_memory" else READER_SYSTEM


def prompt_version() -> str:
    return PROMPT_VERSION
