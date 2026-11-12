"""Evidence, duplication, and memorization checks shared by both evolution stages."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

HIPPO_SRC = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")
if str(HIPPO_SRC) not in sys.path:
    sys.path.insert(0, str(HIPPO_SRC))

from hipporag.utils.eval_utils import normalize_answer  # noqa: E402


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def token_jaccard(left: str, right: str) -> float:
    a = set(squash(left).split())
    b = set(squash(right).split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def span_supported(span: str, passage_text: str, min_chars: int = 12) -> bool:
    needle = squash(span)
    haystack = squash(passage_text)
    return len(needle) >= min_chars and needle in haystack


def leaks_gold_answer(text: str, answers: Iterable[str]) -> bool:
    haystack = normalize_answer(text or "")
    if not haystack:
        return False
    for answer in answers:
        norm = normalize_answer(answer or "")
        if len(norm) >= 3 and norm in haystack:
            return True
    return False


def near_duplicate(candidate: str, existing_texts: Sequence[str], threshold: float = 0.8) -> bool:
    return any(token_jaccard(candidate, other) >= threshold for other in existing_texts if other)
