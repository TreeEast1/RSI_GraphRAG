"""MuSiQue (HippoRAG2 benchmark release) loading, gold mapping and frozen splits.

Corpus formatting follows the official HippoRAG2 ``main.py``: one passage =
``f"{title}\\n{text}"``.  Gold passages are the ``is_supporting`` paragraphs of
each question, matched to corpus rows by exact (title, text).  Every question
also carries its official decomposition (single-hop sub-questions, their
answers and the supporting paragraph of each hop); this is only used for
*evaluation-side* diagnosis, never as input to any method.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "third_party" / "HippoRAG" / "reproduce" / "dataset"
OUT = ROOT / "outputs" / "musique"


@dataclass
class Question:
    qid: str
    question: str
    answers: List[str]                 # answer + aliases (official get_gold_answers order)
    hop_type: str                      # 2hop / 3hop1 / ...
    gold_pids: List[int]               # corpus row ids of supporting paragraphs
    hops: List[Dict] = field(default_factory=list)  # [{q, a, pid}] official decomposition


def passage_text(p: Dict) -> str:
    return f"{p['title']}\n{p['text']}"


def load_corpus() -> List[Dict]:
    return json.load(open(RAW / "musique_corpus.json", encoding="utf-8"))


def load_questions() -> List[Question]:
    corpus = load_corpus()
    index = {(p["title"].strip(), p["text"].strip()): i for i, p in enumerate(corpus)}
    out = []
    for s in json.load(open(RAW / "musique.json", encoding="utf-8")):
        local2pid = {}
        gold = []
        for p in s["paragraphs"]:
            pid = index[(p["title"].strip(), p["paragraph_text"].strip())]
            local2pid[p["idx"]] = pid
            if p["is_supporting"]:
                gold.append(pid)
        hops = [{"q": h["question"], "a": h["answer"],
                 "pid": local2pid.get(h["paragraph_support_idx"])}
                for h in s["question_decomposition"]]
        answers = list(dict.fromkeys([s["answer"]] + list(s.get("answer_aliases", []))))
        out.append(Question(qid=s["id"], question=s["question"], answers=answers,
                            hop_type=s["id"].split("__")[0], gold_pids=list(dict.fromkeys(gold)),
                            hops=hops))
    return out


def make_splits(questions: List[Question], seed: int = 20261111,
                n_practice: int = 400, n_dev: int = 100) -> Dict[str, List[str]]:
    """Stratified by hop type; the remainder is the held-out evaluation split."""
    rng = random.Random(seed)
    by_type: Dict[str, List[str]] = {}
    for q in sorted(questions, key=lambda q: q.qid):
        by_type.setdefault(q.hop_type, []).append(q.qid)
    n = len(questions)
    splits = {"practice": [], "dev": [], "eval": []}
    for t, ids in sorted(by_type.items()):
        rng.shuffle(ids)
        a = round(len(ids) * n_practice / n)
        d = round(len(ids) * n_dev / n)
        splits["practice"] += ids[:a]
        splits["dev"] += ids[a:a + d]
        splits["eval"] += ids[a + d:]
    order = list(splits["practice"])
    rng.shuffle(order)                       # learning order is independent of membership
    splits["practice_order"] = order
    return splits


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap_stats(questions: List[Question], splits: Dict[str, List[str]]) -> Dict:
    byid = {q.qid: q for q in questions}
    prac_pids = set(p for qid in splits["practice"] for p in byid[qid].gold_pids)
    res = {}
    for name in ("dev", "eval"):
        ids = splits[name]
        share = [qid for qid in ids if set(byid[qid].gold_pids) & prac_pids]
        res[name] = {"n": len(ids), "shares_gold_passage_with_practice": len(share),
                     "hop_types": dict(Counter(byid[q].hop_type for q in ids))}
    return res
