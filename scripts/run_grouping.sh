#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken HF_HUB_OFFLINE=1
. .venv/bin/activate
python scripts/run_grouping.py --split "${1:-dev}" --budgets 4096,8192
