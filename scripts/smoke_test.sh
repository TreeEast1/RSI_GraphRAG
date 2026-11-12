#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken HF_HUB_OFFLINE=1
. .venv/bin/activate
python scripts/smoke_test.py --conv conv-30 --n-sessions 3
