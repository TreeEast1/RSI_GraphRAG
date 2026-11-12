#!/usr/bin/env bash
# Run capability, traditional extraction, read-time patches, NV-Embed updates, and paired QA.
# The OpenAI server must already be healthy. Embedding uses a different free GPU.
set -euo pipefail
source "$(dirname "$0")/env.sh"
# A local forward proxy is set in this shell and would swallow 127.0.0.1.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
PY="${PY:-/home/nzq/anaconda3/bin/python}"
EMBED_PY="${EMBED_PY:-/data2-HDD-SATA-20T/nzq/conda_envs/echorag/bin/python}"
EMBED_GPU="${EMBED_GPU:-0}"
LOG="${EXP_ROOT}/logs/pilot_run.log"
cd "$PILOT_ROOT"
mkdir -p "$PILOT_ROOT/logs" "$EXP_ROOT/logs"

echo "START $(date -Is)" | tee -a "$LOG"
curl -fsS --noproxy '*' --max-time 10 "http://127.0.0.1:18181/v1/models" | tee -a "$LOG"
echo | tee -a "$LOG"

echo "==== capability ====" | tee -a "$LOG"
"$PY" src/capability_eval.py 2>&1 | tee -a "$LOG"

echo "==== extract ====" | tee -a "$LOG"
"$PY" src/run_readtime.py extract 2>&1 | tee -a "$LOG"

echo "==== readtime ====" | tee -a "$LOG"
"$PY" src/run_readtime.py readtime 2>&1 | tee -a "$LOG"

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$EMBED_GPU" | tr -d ' ')
if [ "$used" -gt 1500 ]; then
  echo "embed GPU ${EMBED_GPU} busy (${used} MiB); trying GPU 2" | tee -a "$LOG"
  EMBED_GPU=2
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$EMBED_GPU" | tr -d ' ')
fi
if [ "$used" -gt 1500 ]; then
  echo "no free embed GPU (${used} MiB). Stop before QA." | tee -a "$LOG"
  exit 2
fi
echo "==== embed triples/chunks on GPU ${EMBED_GPU} ====" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="$EMBED_GPU" "$EMBED_PY" src/embed_nv.py --what triples --batch-size 8 2>&1 | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="$EMBED_GPU" "$EMBED_PY" src/embed_nv.py --what chunks --batch-size 8 2>&1 | tee -a "$LOG"

echo "==== baseline qa ====" | tee -a "$LOG"
"$PY" src/run_readtime.py baseline 2>&1 | tee -a "$LOG"
echo "==== evolved qa ====" | tee -a "$LOG"
"$PY" src/run_readtime.py evolved 2>&1 | tee -a "$LOG"
echo "==== summary ====" | tee -a "$LOG"
"$PY" src/run_readtime.py summary 2>&1 | tee -a "$LOG"
echo "DONE $(date -Is)" | tee -a "$LOG"
