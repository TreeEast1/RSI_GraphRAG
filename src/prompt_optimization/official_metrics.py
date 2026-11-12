"""Exact re-implementation of the official LoCoMo task_eval scorer.

Source: snap-research/locomo, ``task_eval/evaluation.py`` (fetched 2026-11-09).
Only ``nltk`` / ``numpy`` are required; ``bert_score`` is not used.

Differences from ``src/evaluation/metrics.py`` (the project's SQuAD-style
normaliser) that matter for our numbers:

* ``normalize_answer`` removes **commas first** (no space inserted), then
  removes punctuation, lowercases, and strips articles **and the word "and"**.
* ``exact_match_score`` is **set equality** of tokens, not string equality.
* ``f1_score`` applies a **Porter stemmer** to every token.
* Multi-hop (category 1) uses ``f1()``: split both sides on commas and average,
  over gold sub-answers, the best stemmed-token F1 against any predicted part.
* Category 3 uses only the part of the gold answer before ``;``.
* Adversarial (category 5) is scored 1 iff the output contains
  ``"no information available"`` or ``"not mentioned"`` (official prompt asks a
  two-option multiple choice, so this keyword set matches the official setup).

Nothing here overwrites the project's original metric: both are reported.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from nltk.stem import PorterStemmer

_PS = PorterStemmer()
_PUNCT = set(string.punctuation)


def normalize_answer(s: Any) -> str:
    s = str(s or "")
    s = s.replace(",", "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def exact_match_score(prediction: Any, ground_truth: Any) -> bool:
    return set(normalize_answer(prediction).split()) == set(normalize_answer(ground_truth).split())


def f1_score(prediction: Any, ground_truth: Any) -> float:
    pred_toks = [_PS.stem(w) for w in normalize_answer(prediction).split()]
    gold_toks = [_PS.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def f1_multi(prediction: Any, ground_truth: Any) -> float:
    """Official multi-hop F1: comma-split gold parts, best match per part, mean."""
    predictions = [p.strip() for p in str(prediction or "").split(",")]
    ground_truths = [g.strip() for g in str(ground_truth or "").split(",")]
    if not ground_truths:
        return 0.0
    return float(
        sum(max(f1_score(p, g) for p in predictions) for g in ground_truths) / len(ground_truths)
    )


REFUSAL_KEYWORDS_OFFICIAL = ("no information available", "not mentioned")
REFUSAL_KEYWORDS_ADAPTED = REFUSAL_KEYWORDS_OFFICIAL + (
    "no answer", "cannot", "unanswerable", "unknown", "not stated", "not provided",
)


def official_qa_score(pred: Any, gold: Any, category: int) -> dict:
    """Return the official per-question scores for one QA item."""
    pred = str(pred if pred is not None else "")
    if category == 3:
        gold = str(gold or "").split(";")[0].strip()
    if category == 5:
        low = pred.lower()
        strict = float(any(k in low for k in REFUSAL_KEYWORDS_OFFICIAL))
        adapted = float(any(k in low for k in REFUSAL_KEYWORDS_ADAPTED))
        return {
            "official_f1": strict, "official_f1_adapted": adapted,
            "official_em": strict, "official_refused_strict": strict,
            "official_refused_adapted": adapted,
        }
    if category == 1:
        return {
            "official_f1": f1_multi(pred, gold),
            "official_f1_adapted": f1_multi(pred, gold),
            "official_em": float(exact_match_score(pred, gold)),
        }
    f = f1_score(pred, gold)
    return {
        "official_f1": f, "official_f1_adapted": f,
        "official_em": float(exact_match_score(pred, gold)),
    }
