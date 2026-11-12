# QASPER Read-Time Comprehension — reproducibility

All artifacts live under `outputs/qasper_readtime/`. Nothing here overwrites the
LoCoMo experiment outputs.

## Environment

```bash
cd /group/40150/licochen/Memory/RSI_GraphRAG
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken
export CUDA_VISIBLE_DEVICES=0
# shared HY4 quota: keep the cross-process limiter at ~1 QPS
export HY4_GLOBAL_QPS=1
export HY4_RATE_FILE=qasper
export HY4_OPENIE_WORKERS=12
```

## Pipeline (in order)

```bash
# 1. parse + evidence mapping + frozen 100-paper sample (seed 20261110)
python -I scripts/qasper/prepare_data.py

# 2. paragraph retrieval baselines (no LLM)
python -I scripts/qasper/run_baselines.py --scope sample100 --methods bm25,dense,hybrid

# 3. frozen HippoRAG2 G_raw, one graph per paper
python -I scripts/qasper/build_graphs.py --scope sample100 --stage raw

# 4. read-time: atomic facts -> cross-paragraph insights -> verifier -> summaries
python -I scripts/qasper/run_readtime.py --scope sample100 --grouping section

# 5. graph variants M2 (atom) and M3 (atom+comp)
python -I scripts/qasper/build_variants.py --scope sample100 --grouping section

# 6. M0-M6 retrieval comparison
python -I scripts/qasper/run_retrieval.py --scope sample100 --grouping section

# 7. reader QA under a fixed 4096-token budget (+ top-5 sensitivity, oracle, full paper)
python -I scripts/qasper/run_qa.py --scope sample100 --grouping section

# 8. per-unit mechanism tracking
python -I scripts/qasper/track_units.py --scope sample100 --grouping section

# 9. tables + report
python -I scripts/qasper/make_reports.py --scope sample100
```

Semantic grouping (exploratory) replaces `--grouping section` with
`--grouping semantic` in steps 4-8; it is reported separately.

## Fixed design decisions

* Retrieval unit: original paragraph; retrieval is always within the question's paper.
* Gold evidence: per-annotator sets are alternatives; metrics average over sets.
* `G_raw` is built before any read-time stage and never edited; M2/M3 are
  separate cloned variants built by injecting triples into the OpenIE rows of
  the source paragraphs and re-running the official `index()`.
* No prompt ever sees a question, gold answer or gold evidence.
* All methods share one reader prompt and one 4096-token context budget.
* HY4 temperature 0, `reasoning_effort=no_think`, content-addressed caches.
