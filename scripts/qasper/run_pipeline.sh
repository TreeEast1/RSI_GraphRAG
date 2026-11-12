#!/usr/bin/env bash
# Full QASPER read-time pipeline. Run from the project root.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HY4_GLOBAL_QPS=${HY4_GLOBAL_QPS:-1.0}
export HY4_RATE_FILE=${HY4_RATE_FILE:-qasper}
export HY4_OPENIE_WORKERS=${HY4_OPENIE_WORKERS:-12}
SCOPE=${SCOPE:-sample100}
GROUPING=${GROUPING:-section}

python -I scripts/qasper/prepare_data.py
python -I scripts/qasper/run_baselines.py --scope "$SCOPE" --methods bm25,dense,hybrid
python -I scripts/qasper/build_graphs.py --scope "$SCOPE" --stage raw
python -I scripts/qasper/run_readtime.py --scope "$SCOPE" --grouping "$GROUPING"
python -I scripts/qasper/build_variants.py --scope "$SCOPE" --grouping "$GROUPING"
python -I scripts/qasper/run_retrieval.py --scope "$SCOPE" --grouping "$GROUPING"
python -I scripts/qasper/run_qa.py --scope "$SCOPE" --grouping "$GROUPING"
python -I scripts/qasper/track_units.py --scope "$SCOPE" --grouping "$GROUPING"
python -I scripts/qasper/make_reports.py --scope "$SCOPE" --grouping "$GROUPING"
