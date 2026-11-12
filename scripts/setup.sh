#!/usr/bin/env bash
# Environment bootstrap (already applied on this host; kept for reproducibility).
set -euo pipefail
cd "$(dirname "$0")/.."
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken
export HF_HUB_OFFLINE=1
python3 -m venv .venv            # on this host: symlink to /root/venvs/rsigraph
. .venv/bin/activate
pip install -e third_party/HippoRAG "sentence-transformers>=3.0" datasets rouge_score accelerate pytest
python scripts/prepare_data.py
python -m pytest tests -q
