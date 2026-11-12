"""Dense passage retrieval baseline over session documents (no graph)."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..embedding.nvembed import NVEmbedder


class DenseRetriever:
    def __init__(self, embedder: NVEmbedder, documents: List[Dict]):
        self.embedder = embedder
        self.documents = documents
        self.doc_matrix = embedder.encode_documents([d["content"] for d in documents])

    def retrieve(self, query: str, k: int = 20) -> List[Dict]:
        qv = self.embedder.encode_queries([query])[0]
        scores = self.doc_matrix @ qv
        order = np.argsort(-scores)[:k]
        return [
            {**self.documents[i], "score": float(scores[i])}
            for i in order
        ]
