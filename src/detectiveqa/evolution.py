"""Leakage gates and acceptance rules for question-time DetectiveQA memory.

Feedback may decide whether a trace is worth reflecting on.  It must not become
memory.  Comprehension memory is accepted only when its evidence was retrieved
and every cited chunk was visible at the practice question cutoff.  Skills must
not contain novel names, option text, or answer letters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

from .data import Chunk, Question
from .readtime import norm

_NAME = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b")
_STOP_NAMES = {"The", "What", "Who", "Why", "How", "When", "Where", "Which", "According", "Based"}


@dataclass
class VerificationResult:
    accepted: bool
    reason: str
    local_gain: float = 0.0
    dev_delta: float = 0.0
    harmed: int = 0


def _names(text: str) -> Set[str]:
    return {x for x in _NAME.findall(text) if x not in _STOP_NAMES}


def _contains_any(text: str, needles: Iterable[str], min_words: int = 2) -> bool:
    haystack = f" {norm(text)} "
    for needle in needles:
        n = norm(needle)
        if len(n.split()) >= min_words and f" {n} " in haystack:
            return True
    return False


def memory_gate(memory: Dict, question: Question, chunks: Sequence[Chunk],
                retrieved_chunk_ids: Sequence[str]) -> Optional[str]:
    """Return a rejection reason, or None if the memory may enter local verification."""
    by_id = {c.chunk_id: c for c in chunks}
    cited = list(dict.fromkeys(memory.get("source_chunk_ids") or []))
    if len(cited) < 2:
        return "needs_two_chunks"
    if any(cid not in by_id for cid in cited):
        return "unknown_chunk"
    if any(by_id[cid].available_at > question.cutoff for cid in cited):
        return "future_evidence"
    if not set(cited) <= set(retrieved_chunk_ids):
        return "evidence_not_retrieved"
    triple_text = " ".join(str(memory.get(k, "")) for k in ("subject", "predicate", "object"))
    if _contains_any(triple_text, question.options.values()):
        return "option_text_leak"
    quotes = {str(e.get("chunk_id")): str(e.get("quote", "")) for e in memory.get("evidence") or []
              if isinstance(e, dict)}
    for cid in cited:
        q = norm(quotes.get(cid, ""))
        if not q or q not in norm(by_id[cid].text):
            return "quote_not_found"
    return None


def skill_gate(skill: Dict, question: Question) -> Optional[str]:
    text = " ".join(str(skill.get(k, "")) for k in ("title", "trigger", "procedure"))
    text += " " + " ".join(str(x) for x in skill.get("checks") or [])
    if len(text.split()) < 8:
        return "too_short"
    if len(str(skill.get("procedure", "")).split()) > 70:
        return "too_long"
    protected = _names(question.question) | set().union(*(_names(v) for v in question.options.values()))
    if any(re.search(rf"\b{re.escape(name)}\b", text) for name in protected):
        return "specific_name_leak"
    if _contains_any(text, question.options.values()):
        return "option_text_leak"
    if re.search(r"\b(?:option|answer)\s+[ABCD]\b|\b[ABCD]\s+is\s+correct\b", text, re.I):
        return "answer_letter_leak"
    return None


def memory_is_applicable(memory: Dict, question: Question) -> bool:
    return memory.get("novel_id") == question.novel_id and int(memory.get("available_at", 10 ** 12)) <= question.cutoff


def score_attempt(answer: str, question: Question, evidence_ids: Sequence[str],
                  gold_chunk_ids: Sequence[str]) -> Dict:
    gold = set(gold_chunk_ids)
    hit = len(gold & set(evidence_ids))
    return {"correct": float(answer == question.answer),
            "evidence_recall": hit / len(gold) if gold else 0.0,
            "evidence_any": float(hit > 0)}


def accept_update(before: Dict, after: Dict, dev_before: Sequence[float], dev_after: Sequence[float], *,
                  max_dev_drop: float = 0.0, max_harm: int = 0) -> VerificationResult:
    """Accept only local improvement without measurable development harm.

    Development lists must be paired outcomes from the same frozen questions,
    reader prompt, retrieval budget and random seed.
    """
    if len(dev_before) != len(dev_after):
        raise ValueError("paired dev outcomes are required")
    local_gain = (after["correct"] - before["correct"]) + 0.25 * (
        after.get("evidence_recall", 0.0) - before.get("evidence_recall", 0.0))
    harmed = sum(a < b for b, a in zip(dev_before, dev_after))
    delta = (sum(dev_after) - sum(dev_before)) / max(1, len(dev_before))
    if local_gain <= 0:
        return VerificationResult(False, "no_local_gain", local_gain, delta, harmed)
    if delta < -max_dev_drop or harmed > max_harm:
        return VerificationResult(False, "dev_regression", local_gain, delta, harmed)
    return VerificationResult(True, "accepted", local_gain, delta, harmed)
