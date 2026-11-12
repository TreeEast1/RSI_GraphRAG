#!/usr/bin/env bash
# Sequential pilot. One local server, so stages must not overlap.
set -u
ROOT="/data2-HDD-SATA-20T/nzq/TAG-experiment/rsi_local_qwen_pilot"
PY="/home/nzq/anaconda3/bin/python"
EMB="/data2-HDD-SATA-20T/nzq/conda_envs/echorag/bin/python"
cd "$ROOT"
mkdir -p logs
echo "pipeline start $(date -Is)"
"$PY" -u src/prepare_musique.py
"$PY" -u src/capability_eval.py
"$PY" -u src/run_readtime.py extract
CUDA_VISIBLE_DEVICES=0 "$EMB" -u src/embed_nv.py --what triples --batch-size 8
CUDA_VISIBLE_DEVICES=0 "$EMB" -u src/embed_nv.py --what queries --batch-size 8
"$PY" -u src/run_readtime.py baseline
"$PY" -u src/run_readtime.py readtime
CUDA_VISIBLE_DEVICES=0 "$EMB" -u src/embed_nv.py --what chunks --batch-size 8
"$PY" -u src/run_readtime.py evolved
"$PY" -u src/run_readtime.py summary
echo "pipeline done $(date -Is)"
