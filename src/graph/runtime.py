"""Shared runtime: one GPU-resident embedding model, many HippoRAG instances."""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

from ..config import INDEX_IDENTITY, NVEMBED_PATH, PATHS

_HIPPO_SRC = str(PATHS["third_party"] / "src")
if _HIPPO_SRC not in sys.path:
    sys.path.insert(0, _HIPPO_SRC)

from hipporag.embedding_model import _get_embedding_model_class  # noqa: E402
from hipporag.utils.config_utils import BaseConfig  # noqa: E402

from .hipporag_adapter import HY4CacheOpenAI, build_hipporag, make_base_config  # noqa: E402


def working_dir(save_dir: str | Path, llm_name: str, embedding_name: str) -> Path:
    return Path(save_dir) / f"{llm_name.replace('/', '_')}_{embedding_name.replace('/', '_')}"


class EmbeddingProvider:
    """Loads NV-Embed-v2 once and hands the same instance to every HippoRAG."""

    _instance = None

    def __init__(self, base_config: BaseConfig):
        self.base_config = base_config
        cls_ = _get_embedding_model_class(
            embedding_model_name=base_config.embedding_model_name,
            provider=base_config.embedding_provider,
        )
        self.model = cls_(global_config=base_config,
                          embedding_model_name=base_config.embedding_model_name)

    @classmethod
    def get(cls, base_config: BaseConfig) -> "EmbeddingProvider":
        if cls._instance is None:
            cls._instance = EmbeddingProvider(base_config)
        return cls._instance


@contextmanager
def open_hipporag(
    save_dir: str | Path,
    *,
    conv_id: str = "unknown",
    embedding_provider: Optional[EmbeddingProvider] = None,
    llm: Optional[HY4CacheOpenAI] = None,
    force_index_from_scratch: bool = False,
    force_openie_from_scratch: bool = False,
    **cfg_kwargs,
):
    cfg = make_base_config(save_dir, conv_id=conv_id, **cfg_kwargs)
    cfg.force_index_from_scratch = force_index_from_scratch
    cfg.force_openie_from_scratch = force_openie_from_scratch
    provider = embedding_provider or EmbeddingProvider.get(cfg)
    if llm is None:
        llm = HY4CacheOpenAI.from_experiment_config(cfg)
    hippo = build_hipporag(cfg, embedding_model=provider.model, extraction_llm=llm, qa_llm=llm)
    try:
        yield hippo
    finally:
        # We do not own the embedding model or LLM; closing keep them alive.
        hippo.close()


def make_llm(cfg: BaseConfig) -> HY4CacheOpenAI:
    return HY4CacheOpenAI.from_experiment_config(cfg)
