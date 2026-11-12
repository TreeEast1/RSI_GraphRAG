"""Official QASPER answer/evidence evaluation logic (re-implemented).

Mirrors allenai's qasper_evaluator.py: token F1 against each annotator answer,
taking the max over annotators; paragraph-level evidence F1 against each
annotator's evidence set, also taking the max. Multiple annotators are therefore
treated as alternatives, which is the official QASPER convention.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    p = normalize_answer(prediction).split()
    g = normalize_answer(ground_truth).split()
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


def paragraph_f1_score(prediction, ground_truth) -> float:
    if not ground_truth and not prediction:
        return 1.0
    num_same = len(set(ground_truth) & set(prediction))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction) if prediction else 0.0
    recall = num_same / len(ground_truth) if ground_truth else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(gold: Dict[str, List[Dict]], predicted: Dict[str, Dict]) -> Dict:
    answer_f1s, evidence_f1s = [], []
    by_type: Dict[str, List[float]] = {"extractive": [], "abstractive": [], "boolean": [], "none": []}
    missing = 0
    for qid, refs in gold.items():
        pred = predicted.get(qid)
        if pred is None:
            missing += 1
            answer_f1s.append(0.0)
            evidence_f1s.append(0.0)
            continue
        scored = [(token_f1_score(pred.get("answer", ""), r["answer"]), r["type"]) for r in refs]
        best, typ = sorted(scored, key=lambda x: -x[0])[0]
        answer_f1s.append(best)
        by_type.setdefault(typ, []).append(best)
        evidence_f1s.append(max(paragraph_f1_score(pred.get("evidence", []), r["evidence"]) for r in refs))

    mean = lambda x: sum(x) / len(x) if x else 0.0  # noqa: E731
    return {
        "Answer F1": mean(answer_f1s),
        "Answer F1 by type": {k: mean(v) for k, v in by_type.items()},
        "Evidence F1": mean(evidence_f1s),
        "n": len(answer_f1s),
        "Missing predictions": missing,
    }
