"""Thin wrapper around HippoRAG's official NV-Embed-v2 embedding model.

Adds a content-addressed on-disk cache so that the same text is never encoded
twice across experiments (document embeddings in particular are shared by every
grouping method and every graph variant).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

QUERY_INSTRUCTION_PASSAGE = "Given a question, retrieve passages that answer the question"
QUERY_INSTRUCTION_FACT = "Given a question, retrieve relevant triplet facts that matches this question."


class EmbeddingCacheStore:
    """Simple pickle-backed cache: sha1(model|instruction|text) -> np.ndarray."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {}
        if self.path.exists():
            try:
                with open(self.path, "rb") as f:
                    self.data = pickle.load(f)
            except Exception:
                self.data = {}

    @staticmethod
    def key(model: str, instruction: str, text: str) -> str:
        h = hashlib.sha1()
        h.update(model.encode("utf-8"))
        h.update(b"\x00")
        h.update((instruction or "").encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def get_many(self, model: str, instruction: str, texts: List[str]):
        keys = [self.key(model, instruction, t) for t in texts]
        return keys, [self.data.get(k) for k in keys]

    def put_many(self, keys: List[str], vecs) -> None:
        for k, v in zip(keys, vecs):
            self.data[k] = np.asarray(v, dtype=np.float32)

    def save(self) -> None:
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.path)


class NVEmbedder:
    """Wraps a HippoRAG embedding model with caching.

    `model` may be None until `attach()` is called, which lets callers reuse a
    single GPU-resident model across many experiment objects.
    """

    def __init__(self, model, cache_path: str | Path = ROOT / "cache" / "doc_embeddings.pkl"):
        self.model = model
        self.name = getattr(model, "embedding_model_name", "unknown")
        self.cache = EmbeddingCacheStore(cache_path)

    @classmethod
    def from_hipporag_config(cls, global_config, cache_path=None):
        from hipporag.embedding_model import _get_embedding_model_class

        cls_ = _get_embedding_model_class(
            embedding_model_name=global_config.embedding_model_name,
            provider=global_config.embedding_provider,
        )
        model = cls_(global_config=global_config,
                     embedding_model_name=global_config.embedding_model_name)
        return cls(model, cache_path or ROOT / "cache" / "doc_embeddings.pkl")

    def encode(self, texts: Iterable[str], instruction: str = "") -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        keys, cached = self.cache.get_many(self.name, instruction, texts)
        missing_idx = [i for i, v in enumerate(cached) if v is None]
        if missing_idx:
            to_encode = [texts[i] for i in missing_idx]
            new_vecs = self.model.batch_encode(to_encode, instruction=instruction, norm=True)
            new_vecs = np.asarray(new_vecs, dtype=np.float32)
            self.cache.put_many([keys[i] for i in missing_idx], new_vecs)
            self.cache.save()
            for i, v in zip(missing_idx, new_vecs):
                cached[i] = v
        return np.vstack(cached)

    def encode_documents(self, texts: Iterable[str]) -> np.ndarray:
        return self.encode(texts, instruction="")

    def encode_queries(self, texts: Iterable[str], kind: str = "passage") -> np.ndarray:
        instr = QUERY_INSTRUCTION_FACT if kind == "fact" else QUERY_INSTRUCTION_PASSAGE
        return self.encode(texts, instruction=instr)

    @property
    def dim(self) -> int:
        return int(getattr(self.model, "embedding_dim", 0)) or 4096
