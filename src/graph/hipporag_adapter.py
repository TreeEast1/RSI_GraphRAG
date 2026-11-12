"""Adapters that bind the official HippoRAG2 machinery to HY4 + NV-Embed-v2.

We deliberately keep the official `HippoRAG` class, graph construction, PPR and
fact reranking untouched. Everything new lives in this package and is injected
through the official extension points (`extraction_llm`, `qa_llm`,
`embedding_model`, `index_identity`, `text_preprocessor`).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..config import (HY4_REASONING_EFFORT, INDEX_IDENTITY, NVEMBED_PATH,
                      PATHS)

# sys.path must contain third_party/HippoRAG/src for `import hipporag`
import sys

_HIPPO_SRC = str(PATHS["third_party"] / "src")
if _HIPPO_SRC not in sys.path:
    sys.path.insert(0, _HIPPO_SRC)

from hipporag.HippoRAG import HippoRAG  # noqa: E402
from hipporag.llm.openai_gpt import CacheOpenAI  # noqa: E402
from hipporag.utils.config_utils import BaseConfig  # noqa: E402


class HY4CacheOpenAI(CacheOpenAI):
    """Official OpenAI-compatible adapter, pinned to HY4 with thinking disabled.

    Adds an outer retry loop on top of the official sqlite cache. The TokenLink
    endpoint returns 429 above roughly 32-64 simultaneous requests per key, and
    a burst from several parallel indexing processes can trip it; the official
    SDK backoff occasionally gives up before the window clears.
    """

    def _init_llm_config(self) -> None:
        super()._init_llm_config()
        if HY4_REASONING_EFFORT:
            self.llm_config.generate_params["reasoning_effort"] = HY4_REASONING_EFFORT

    def infer(self, messages, **kwargs):  # type: ignore[override]
        import random
        import time

        from ..adapters.ratelimit import acquire as _acquire

        last_exc = None
        for attempt in range(14):
            try:
                _acquire()  # shared cross-process quota, ~1 QPS by default
                return super().infer(messages, **kwargs)
            except Exception as exc:  # noqa: BLE001
                text = str(exc)
                retryable = any(tok in text for tok in ("429", "rate", "timed out", "Timeout", "502", "503", "504"))
                last_exc = exc
                if not retryable or attempt == 13:
                    raise
                delay = min(120.0, (2 ** attempt) * 1.5) * (0.6 + random.random() * 0.8)
                time.sleep(delay)
        raise last_exc


def make_base_config(
    save_dir: str | Path,
    *,
    conv_id: str = "unknown",
    embedding_model_name: str = NVEMBED_PATH,
    embedding_provider: str = "nvembed",
    embedding_batch_size: int = 16,
    embedding_dtype: str = "float16",
    llm_name: str = "hy4",
    llm_base_url: Optional[str] = None,
    max_new_tokens: int = 8192,
    openie_max_workers: int = int(os.environ.get("HY4_OPENIE_WORKERS", "5")),
    openie_ner_max_tokens: int = 2048,
    openie_triple_max_tokens: int = 8192,
    temperature: float = 0.0,
    retrieval_top_k: int = 200,
    linking_top_k: int = 5,
    qa_top_k: int = 5,
    dataset: str = "locomo",
) -> BaseConfig:
    """Build a BaseConfig with the knobs that matter for our experiments."""
    cfg = BaseConfig(
        save_dir=str(save_dir),
        llm_name=llm_name,
        llm_base_url=llm_base_url or os.environ.get("HY4_BASE_URL", "https://tokenlink.woa.com/openapi/v1"),
        llm_api_key_env="HY4_API_KEY",
        llm_supports_max_completion_tokens=True,
        embedding_model_name=embedding_model_name,
        embedding_provider=embedding_provider,
        embedding_batch_size=embedding_batch_size,
        embedding_model_dtype=embedding_dtype,
        embedding_max_seq_len=2048,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        openie_max_workers=openie_max_workers,
        openie_ner_max_tokens=openie_ner_max_tokens,
        openie_triple_max_tokens=openie_triple_max_tokens,
        retrieval_top_k=retrieval_top_k,
        linking_top_k=linking_top_k,
        qa_top_k=qa_top_k,
        dataset=dataset,
        # We rely on our own HY4 cache for non-HippoRAG calls; HippoRAG keeps its
        # own sqlite cache for OpenIE/QA calls under save_dir/llm_cache.
        save_openie=True,
        max_retry_attempts=8,
        synonymy_edge_topk=2047,
        synonymy_edge_sim_threshold=0.8,
    )
    return cfg


def build_hipporag(
    base_config: BaseConfig,
    *,
    embedding_model=None,
    extraction_llm=None,
    qa_llm=None,
    index_identity: str = INDEX_IDENTITY,
) -> HippoRAG:
    """Instantiate HippoRAG2 with our injected models (no monkey-patching)."""
    if extraction_llm is None:
        extraction_llm = HY4CacheOpenAI.from_experiment_config(base_config)
    if qa_llm is None:
        qa_llm = extraction_llm
    return HippoRAG(
        global_config=base_config,
        extraction_llm=extraction_llm,
        qa_llm=qa_llm,
        embedding_model=embedding_model,
        index_identity=index_identity,
    )
