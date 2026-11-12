#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken HF_HUB_OFFLINE=1 HY4_GLOBAL_QPS=${HY4_GLOBAL_QPS:-1.0}
. .venv/bin/activate
python scripts/run_question_time.py --convs "${1:-conv-30}" --strategies q3 \
  --graph cluster_hybrid_0.5_b8192_comp --out-prefix qtime_joint
