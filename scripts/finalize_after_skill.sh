#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
while pgrep -f "python.*run_skill_evolution.py" >/dev/null 2>&1; do sleep 60; done
. .venv/bin/activate
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken
python scripts/aggregate.py        > outputs/logs/finalize.log 2>&1
python scripts/build_final_tables.py >> outputs/logs/finalize.log 2>&1
python scripts/make_manifest.py     >> outputs/logs/finalize.log 2>&1
echo "FINALIZE_DONE"
