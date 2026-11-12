#!/usr/bin/env bash
# OpenAI-compatible Qwen server. Does not kill other GPU processes.
# Override MODEL, NAME, PORT, CUDA_VISIBLE_DEVICES, TP, MAX_LEN, MEM_UTIL.
set -euo pipefail
ROOT="/data2-HDD-SATA-20T/nzq/TAG-experiment/rsi_local_qwen_pilot"
VENV="$ROOT/.venv"
MODEL="${MODEL:-/data2-HDD-SATA-20T/nzq/huggingface_model/Qwen3.5-9B}"
NAME="${NAME:-qwen3.5-9b}"
PORT="${PORT:-18181}"
TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-8192}"
MEM_UTIL="${MEM_UTIL:-0.90}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
# System nvcc is CUDA 9.1, so FlashInfer's JIT sampler cannot compile.
# The PyTorch sampler is enough for temperature-0 graph experiments.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# vLLM 0.21's _C extension is linked to libcudart.so.13. Torch in this venv is cu128.
# Prepend a CUDA 13 runtime that is already on this machine. Do not upgrade the driver.
CU13_ROOT="${CU13_ROOT:-/data2-HDD-SATA-20T/nzq/TAG-experiment/experiments/rsi_local_qwen_pilot/.venv/lib/python3.12/site-packages/nvidia}"
if [ -d "$CU13_ROOT" ]; then
  CU13_LIBS="$(find "$CU13_ROOT" -type d -name lib | paste -sd:)"
  export LD_LIBRARY_PATH="${CU13_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export HF_HOME="${HF_HOME:-/data2-HDD-SATA-20T/nzq/huggingface_model}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/data2-HDD-SATA-20T/nzq/huggingface_model/hub}"
export TMPDIR="${TMPDIR:-/data2-HDD-SATA-20T/nzq/tmp}"
mkdir -p "$ROOT/logs" "$TMPDIR"
LOG="$ROOT/logs/vllm_${NAME}.log"
echo "model=$MODEL name=$NAME gpus=$CUDA_VISIBLE_DEVICES tp=$TP len=$MAX_LEN util=$MEM_UTIL" | tee "$LOG"
exec "$VENV/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name "$NAME" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$MEM_UTIL" \
  --max-num-seqs 4 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --trust-remote-code \
  ${ENFORCE_EAGER:+--enforce-eager} \
  >> "$LOG" 2>&1
