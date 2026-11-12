#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_baselines.sh dev
bash scripts/run_grouping.sh dev
bash scripts/run_read_time.sh dev
bash scripts/run_question_time.sh conv-30
bash scripts/run_joint_eval.sh conv-30
. .venv/bin/activate
python scripts/aggregate.py
python scripts/build_final_tables.py
python scripts/make_manifest.py
