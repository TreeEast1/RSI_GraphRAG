#!/usr/bin/env bash
# Shared paths for the local Qwen pilot. Weights stay on the data disk.
export EXP_ROOT="/data2-HDD-SATA-20T/nzq/TAG-experiment/experiments/rsi_local_qwen_pilot"
export PILOT_ROOT="/data2-HDD-SATA-20T/nzq/TAG-experiment/rsi_local_qwen_pilot"
export HF_HOME="/data2-HDD-SATA-20T/nzq/huggingface_model"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export TMPDIR="/data2-HDD-SATA-20T/nzq/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export UV_CACHE_DIR="/data2-HDD-SATA-20T/nzq/.cache/uv"
export PIP_CACHE_DIR="/data2-HDD-SATA-20T/nzq/pip_cache"
export XDG_CACHE_HOME="/data2-HDD-SATA-20T/nzq/.cache"
export TOKENIZERS_PARALLELISM=false
export VENV="${EXP_ROOT}/.venv"
mkdir -p "${EXP_ROOT}/logs" "${TMPDIR}"
