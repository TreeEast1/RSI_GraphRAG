from pathlib import Path

V5 = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/experiments/rsi_graphrag_v5_mechanism_search")
V4 = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/experiments/rsi_graphrag_v4")
HIPPO = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG")
OPENIE = V4 / "artifacts" / "openie" / "luna"
V4_CACHE = V4 / "artifacts" / "llm_cache.sqlite"
ART = V5 / "artifacts"
PROMPTS = V5 / "prompts"
REPORTS = V5 / "reports"
V4_PROMPTS = V4 / "prompts"

MAX_FRESH_CALLS = 60
MAX_INPUT_TOKENS = 750_000
MAX_OUTPUT_TOKENS = 150_000
SEED = 20261030
