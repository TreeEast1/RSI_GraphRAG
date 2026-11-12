"""Paragraph retrieval baselines for QASPER: BM25, dense (NV-Embed-v2), hybrid.

The retrieval unit is always the original paragraph. One index is built per
paper because the official QASPER task restricts retrieval to the question's
own paper.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(str(text or ""))]


class BM25:
    """Okapi BM25 over a small per-paper paragraph pool."""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75, doc_ids: Optional[Sequence[str]] = None):
        self.doc_ids = list(doc_ids) if doc_ids is not None else [str(i) for i in range(len(docs))]
        self.toks = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.N = len(self.toks)
        self.avgdl = (sum(len(t) for t in self.toks) / self.N) if self.N else 0.0
        self.tf: List[Counter] = [Counter(t) for t in self.toks]
        self.dl = [len(t) for t in self.toks]
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self._avg_idf = (sum(self.idf.values()) / len(self.idf)) if self.idf else 0.0

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(self.N, dtype=np.float64)
        if not q or not self.N:
            return out
        for w in q:
            idf = self.idf.get(w)
            if idf is None:
                # unseen term: small weight so long junk queries do not swamp valid terms
                idf = 0.1 * self._avg_idf
            for i, c in enumerate(self.tf):
                f = c.get(w, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avgdl or 1))
                out[i] += idf * f * (self.k1 + 1) / denom
        return out

    def top_k(self, query: str, k: int = 20) -> List[Dict]:
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:k]
        return [{"passage_id": self.doc_ids[i], "score": float(s[i])} for i in order if s[i] > 0]


def rrf_fuse(rankings: Sequence[Sequence[Dict]], k: int = 60, top_k: int = 20,
             weights: Optional[Sequence[float]] = None) -> List[Dict]:
    """Reciprocal rank fusion of several ranked lists of {passage_id, score}."""
    weights = weights or [1.0] * len(rankings)
    acc: Dict[str, float] = {}
    for lst, w in zip(rankings, weights):
        for pos, item in enumerate(lst, 1):
            acc[item["passage_id"]] = acc.get(item["passage_id"], 0.0) + w / (k + pos)
    ordered = sorted(acc.items(), key=lambda x: -x[1])[:top_k]
    return [{"passage_id": pid, "score": float(s)} for pid, s in ordered]


class DenseParagraphIndex:
    """NV-Embed-v2 paragraph index for one or more papers (shared embedder)."""

    def __init__(self, passages: Sequence[Dict], embedder):
        self.passages = list(passages)
        self.embedder = embedder
        self.ids = [p["passage_id"] for p in self.passages]
        self.matrix = embedder.encode_documents([p["text"] for p in self.passages]) if self.passages else np.zeros((0, 4096), dtype=np.float32)

    def top_k(self, query: str, k: int = 20) -> List[Dict]:
        if not self.passages:
            return []
        qv = self.embedder.encode_queries([query])[0]
        scores = self.matrix @ qv
        order = np.argsort(-scores)[:k]
        return [{"passage_id": self.ids[i], "score": float(scores[i])} for i in order]


class HybridRetriever:
    def __init__(self, passages: Sequence[Dict], embedder=None, dense_index=None):
        self.passages = list(passages)
        self.ids = [p["passage_id"] for p in self.passages]
        self.bm25 = BM25([p["text"] for p in self.passages], doc_ids=self.ids)
        self.dense = dense_index or (DenseParagraphIndex(self.passages, embedder) if embedder is not None else None)

    def retrieve(self, query: str, method: str = "hybrid", k: int = 20) -> List[Dict]:
        if method == "bm25":
            return self.bm25.top_k(query, k)
        if method == "dense":
            return self.dense.top_k(query, k)
        if method == "hybrid":
            b = self.bm25.top_k(query, max(k, 50))
            d = self.dense.top_k(query, max(k, 50))
            return rrf_fuse([d, b], top_k=k)
        raise ValueError(method)
