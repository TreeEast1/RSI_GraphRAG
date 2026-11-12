#!/usr/bin/env bash
# Keep extracting until every read-set document has a row, then rerun the paired eval.
ROOT="/data2-HDD-SATA-20T/nzq/TAG-experiment/rsi_local_qwen_pilot"
PY="/home/nzq/anaconda3/bin/python"
EMB="/data2-HDD-SATA-20T/nzq/conda_envs/echorag/bin/python"
cd "$ROOT"
while true; do
  echo "extract attempt $(date -Is)"
  "$PY" -u src/run_readtime.py extract || echo "extract exited $?"
  missing=$("$PY" - << 'PY'
import json
root="/data2-HDD-SATA-20T/nzq/TAG-experiment/rsi_local_qwen_pilot"
ids=set(json.load(open(root+"/data/read_doc_ids.json")))
have=set(json.loads(l)["doc_id"] for l in open(root+"/predictions/triples.jsonl") if l.strip())
print(len(ids-have))
PY
)
  echo "missing $missing"
  if [ "$missing" = "0" ]; then
    break
  fi
  sleep 2
done
rm -f results/baseline.jsonl results/evolved.jsonl results/summary.json \
  graph_patches/readtime.jsonl graph_patches/accepted_triples.jsonl graph_patches/accepted_chunks.jsonl
echo "rerun retrieval $(date -Is)"
CUDA_VISIBLE_DEVICES=0 "$EMB" -u src/embed_nv.py --what triples --batch-size 8
"$PY" -u src/run_readtime.py baseline
"$PY" -u src/run_readtime.py readtime
CUDA_VISIBLE_DEVICES=0 "$EMB" -u src/embed_nv.py --what chunks --batch-size 8
"$PY" -u src/run_readtime.py evolved
"$PY" -u src/run_readtime.py summary
echo "rerun done $(date -Is)"
