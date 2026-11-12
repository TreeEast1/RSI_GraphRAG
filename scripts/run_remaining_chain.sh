#!/usr/bin/env bash
# Chained overnight orchestration: run every remaining experiment back-to-back,
# then regenerate all aggregated artefacts and reports.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken
export HY4_RATE_FILE=poolA
export HY4_MAX_CONCURRENCY=2
export HY4_OPENIE_WORKERS=4
. .venv/bin/activate
K1=$(grep '^HY4_API_KEY=' .env | cut -d= -f2)
LOG=outputs/logs/chain.log

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

wait_for() {  # wait until no python process matches the pattern
  local pat="$1"
  while pgrep -f "python.*${pat}" >/dev/null 2>&1; do sleep 60; done
}

set_rate() { echo "$1" > cache/rate_qps_poolA.txt; say "rate -> $1 QPS"; }

say "=== chain start ==="

# ---------------------------------------------------------------- stage 0
say "waiting for skill-evolution apply to finish"
wait_for "run_skill_evolution.py"
say "skill evolution finished"

# ---------------------------------------------------------------- stage 1
run_strategy() {  # $1 = strategy prefix (q1|q2), $2 = out prefix
  local strat="$1" prefix="$2"
  say "stage: $strat on C6 graph (test)"
  set_rate 2.5
  CUDA_VISIBLE_DEVICES=0 HY4_API_KEY=$K1 HY4_CACHE_SHARD=_${strat}a HY4_GLOBAL_QPS=2.5 \
    nohup python -u scripts/run_question_time.py --convs "conv-26,conv-41,conv-42" \
      --strategies "$strat" --graph full_conv_comp --budget 4096 --max-workers 2 \
      --out-prefix "$prefix" > outputs/logs/${prefix}_g0.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 HY4_API_KEY=$K1 HY4_CACHE_SHARD=_${strat}b HY4_GLOBAL_QPS=2.5 \
    nohup python -u scripts/run_question_time.py --convs "conv-43,conv-47" \
      --strategies "$strat" --graph full_conv_comp --budget 4096 --max-workers 2 \
      --out-prefix "$prefix" > outputs/logs/${prefix}_g1.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 HY4_API_KEY=$K1 HY4_CACHE_SHARD=_${strat}c HY4_GLOBAL_QPS=2.5 \
    nohup python -u scripts/run_question_time.py --convs "conv-49,conv-50" \
      --strategies "$strat" --graph full_conv_comp --budget 4096 --max-workers 2 \
      --out-prefix "$prefix" > outputs/logs/${prefix}_g2.log 2>&1 &
  wait_for "run_question_time.py"
  say "stage $strat done: $(ls outputs/qa/${prefix}_*.jsonl 2>/dev/null | wc -l) files"
}

run_strategy q1 qtime_test_r1q1
run_strategy q2 qtime_test_r1q2

# ---------------------------------------------------------------- finalize
set_rate 1.0
say "regenerating aggregation + reports"
python scripts/aggregate.py >/dev/null 2>&1 || say "aggregate failed"
python scripts/build_final_tables.py >/dev/null 2>&1 || say "tables failed"
python scripts/make_manifest.py >/dev/null 2>&1 || say "manifest failed"
say "=== chain complete ==="
