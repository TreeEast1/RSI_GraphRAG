#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken HF_HUB_OFFLINE=1 HY4_GLOBAL_QPS=${HY4_GLOBAL_QPS:-1.0}
. .venv/bin/activate
python scripts/run_baselines.py --split "${1:-dev}" --out-prefix baselines
