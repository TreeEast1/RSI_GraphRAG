#!/usr/bin/env bash
# Serve Qwen3.5-9B as an OpenAI-compatible text endpoint on one free 3090.
# Does not kill foreign GPU processes. Refuses a GPU that is already occupied.
set -euo pipefail
source "$(dirname "$0")/env.sh"

MODEL="${MODEL:-/data2-HDD-SATA-20T/nzq/huggingface_model/Qwen3.5-9B}"
NAME="${NAME:-qwen3.5-9b}"
PORT="${PORT:-18181}"
GPU="${CUDA_VISIBLE_DEVICES:-3}"
MAX_LEN="${MAX_LEN:-8192}"
MEM_UTIL="${MEM_UTIL:-0.90}"
LOG="${EXP_ROOT}/logs/vllm_${NAME}.log"

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
if [ "${used}" -gt 1500 ]; then
  echo "GPU ${GPU} is not free (${used} MiB). Refusing to start." | tee "$LOG"
  exit 1
fi
if [ ! -x "${VENV}/bin/vllm" ]; then
  echo "vllm is not installed in ${VENV}" | tee "$LOG"
  exit 1
fi
if [ ! -f "${MODEL}/config.json" ]; then
  echo "model config missing: ${MODEL}" | tee "$LOG"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
echo "model=${MODEL} name=${NAME} gpu=${GPU} port=${PORT} len=${MAX_LEN} util=${MEM_UTIL}" | tee "$LOG"
# Text-only. MTP / speculative decoding stays off.
exec "${VENV}/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name "$NAME" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$MEM_UTIL" \
  --max-num-seqs 4 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --trust-remote-code \
  >> "$LOG" 2>&1
