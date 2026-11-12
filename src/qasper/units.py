"""Retrieval over comprehension units and single-paragraph summaries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "outputs" / "qasper_readtime"


def load_units(paper_id: str, grouping: str = "section") -> List[Dict]:
    path = OUT / "comprehension" / grouping / paper_id / "units.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_summaries(paper_id: str, grouping: str = "section") -> List[Dict]:
    path = OUT / "comprehension" / grouping / paper_id / "summaries.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    out = []
    for i, s in enumerate(data.get("summaries", [])):
        out.append({
            "unit_id": f"{paper_id}::sum{i:03d}",
            "concise_text": s["summary"],
            "source_passage_ids": [s["passage_id"]],
            "kind": "summary",
        })
    return out


class KnowledgeIndex:
    """Dense index over arbitrary knowledge records with passage provenance.

    Each record must have ``unit_id``, ``concise_text`` and ``source_passage_ids``.
    """

    def __init__(self, records: Sequence[Dict], embedder, kind: str = "unit"):
        self.records = list(records)
        self.kind = kind
        self.embedder = embedder
        self.matrix = (embedder.encode_documents([r["concise_text"] for r in self.records])
                       if self.records else np.zeros((0, 4096), dtype=np.float32))

    def __len__(self):
        return len(self.records)

    def rank_records(self, question: str, k: int = 50) -> List[Dict]:
        if not self.records:
            return []
        qv = self.embedder.encode_queries([question])[0]
        scores = self.matrix @ qv
        order = np.argsort(-scores)[:k]
        return [{**self.records[i], "score": float(scores[i]), "rank": r}
                for r, i in enumerate(order, 1)]

    def rank_passages(self, question: str, k: int = 20, max_units: int = 100) -> List[Dict]:
        """Aggregate unit evidence to source paragraphs (best unit per paragraph)."""
        best: Dict[str, Dict] = {}
        for rec in self.rank_records(question, max_units):
            srcs = rec.get("source_passage_ids") or []
            for j, pid in enumerate(srcs):
                s = rec["score"] * (1.0 if j == 0 else 0.98)
                if pid not in best or s > best[pid]["score"]:
                    best[pid] = {"passage_id": pid, "score": s, "unit_id": rec["unit_id"],
                                 "unit_text": rec["concise_text"], "unit_rank": rec["rank"]}
        ranked = sorted(best.values(), key=lambda x: -x["score"])
        return ranked[:k]


def rrf_merge(rankings: Sequence[Sequence[Dict]], k: int = 60, top_k: int = 20,
              weights: Optional[Sequence[float]] = None) -> List[Dict]:
    weights = weights or [1.0] * len(rankings)
    acc: Dict[str, float] = {}
    src: Dict[str, List[str]] = {}
    for lst, w in zip(rankings, weights):
        for pos, item in enumerate(lst, 1):
            pid = item["passage_id"]
            acc[pid] = acc.get(pid, 0.0) + w / (k + pos)
            src.setdefault(pid, []).append(item.get("unit_id") or item.get("source", "list"))
    ordered = sorted(acc.items(), key=lambda x: -x[1])[:top_k]
    return [{"passage_id": pid, "score": float(s), "sources": src[pid]} for pid, s in ordered]
