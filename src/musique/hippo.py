"""HippoRAG2 construction for MuSiQue (official settings, HY4 + NV-Embed-v2)."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from ..adapters.hy4_client import load_env
from ..config import PATHS
from ..graph.runtime import EmbeddingProvider
from ..graph.hipporag_adapter import HY4CacheOpenAI, build_hipporag, make_base_config

load_env()
WORK = Path(os.environ.get("MUSIQUE_WORK", "/root/work/musique"))
RERANK_FILE = str(PATHS["third_party"] / "src" / "hipporag" / "prompts" / "dspy_prompts"
                  / "filter_llama3.3-70B-Instruct.json")
INDEX_IDENTITY = "rsigraph-musique|hy4-no_think|nvembed-v2|v1"


def musique_config(save_dir, **kw):
    cfg = make_base_config(save_dir, dataset="musique",
                           openie_max_workers=int(os.environ.get("HY4_OPENIE_WORKERS", "10")),
                           embedding_batch_size=int(os.environ.get("EMB_BS", "32")), **kw)
    cfg.rerank_dspy_file_path = RERANK_FILE      # official main.py sets this explicitly
    cfg.qa_top_k = 5
    cfg.retrieval_top_k = 200
    return cfg


@contextmanager
def open_musique_hippo(save_dir, *, force_index_from_scratch=False, llm=None, provider=None, **kw):
    cfg = musique_config(str(save_dir), **kw)
    cfg.force_index_from_scratch = force_index_from_scratch
    provider = provider or EmbeddingProvider.get(cfg)
    llm = llm or HY4CacheOpenAI.from_experiment_config(cfg)
    hippo = build_hipporag(cfg, embedding_model=provider.model, extraction_llm=llm, qa_llm=llm,
                           index_identity=INDEX_IDENTITY)
    try:
        yield hippo
    finally:
        hippo.close()
