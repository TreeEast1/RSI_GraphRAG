#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken HF_HUB_OFFLINE=1 HY4_GLOBAL_QPS=${HY4_GLOBAL_QPS:-1.0}
. .venv/bin/activate
SPLIT="${1:-dev}"
python scripts/run_read_time_extract.py --split "$SPLIT" \
  --scopes cluster_hybrid_0.5_b8192,full_conv --modes triple,comp --max-tokens 16384
if [ "$SPLIT" = "dev" ]; then
  CONVS=conv-30,conv-44,conv-48
else
  CONVS=conv-26,conv-41,conv-42,conv-43,conv-47,conv-49,conv-50
fi
python scripts/run_read_time_eval.py --convs "$CONVS" \
  --variants cluster_hybrid_0.5_b8192_comp:comp,cluster_hybrid_0.5_b8192_triple:triple,full_conv_comp:comp,full_conv_triple:triple \
  --out-prefix readtime
