"""Central configuration for the RSI-GraphRAG experiments."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATHS = {
    "root": ROOT,
    "third_party": ROOT / "third_party" / "HippoRAG",
    "data_raw": ROOT / "data" / "raw",
    "data_processed": ROOT / "data" / "processed",
    "splits": ROOT / "data" / "splits",
    "mappings": ROOT / "data" / "mappings",
    "outputs": ROOT / "outputs",
    "raw_graphs": ROOT / "outputs" / "raw_graphs",
    "groupings": ROOT / "outputs" / "groupings",
    "comprehension": ROOT / "outputs" / "comprehension",
    "retrieval": ROOT / "outputs" / "retrieval",
    "qa": ROOT / "outputs" / "qa",
    "skills": ROOT / "outputs" / "skills",
    "reports": ROOT / "outputs" / "reports",
    "logs": ROOT / "outputs" / "logs",
    "cache": ROOT / "cache",
    "models": ROOT / "cache" / "models",
}

for p in PATHS.values():
    if isinstance(p, Path):
        p.mkdir(parents=True, exist_ok=True)

_NVEMBED_CANDIDATES = [
    os.environ.get("NVEMBED_PATH"),
    "/root/models/NV-Embed-v2",                       # local-disk copy (fast I/O)
    str(PATHS["models"] / "NV-Embed-v2"),             # project cache copy
]
NVEMBED_PATH = next((p for p in _NVEMBED_CANDIDATES if p and Path(p, "config.json").exists()),
                    str(PATHS["models"] / "NV-Embed-v2"))
NVEMBED_ALIAS = "nvidia/NV-Embed-v2"

HY4_MODEL = os.environ.get("HY4_MODEL", "hy4")
HY4_REASONING_EFFORT = os.environ.get("HY4_REASONING_EFFORT", "no_think")

# Stable identity string for everything that influences an index. Changing the
# extraction prompt or the LLM invalidates caches by design.
INDEX_IDENTITY = "rsigraph|hy4-no_think|nvembed-v2|v1"


@dataclass
class RunConfig:
    """Per-conversation experiment configuration."""

    conversation_id: str
    save_dir: Path
    embedding_model_name: str = NVEMBED_PATH
    embedding_provider: str = "nvembed"
    embedding_batch_size: int = 16
    embedding_dtype: str = "float16"
    llm_name: str = HY4_MODEL
    llm_base_url: str = os.environ.get("HY4_BASE_URL", "https://tokenlink.woa.com/openapi/v1")
    openie_max_workers: int = 8
    openie_ner_max_tokens: int = 2048
    openie_triple_max_tokens: int = 8192
    temperature: float = 0.0
    retrieval_top_k: int = 200
    linking_top_k: int = 5
    qa_top_k: int = 5
    seed: int = 42
    extra: dict = field(default_factory=dict)
