# QASPER Read-Time Comprehension — Experiment Configuration

Frozen before the first formal retrieval/QA run.

| item | value |
|---|---|
| dataset | QASPER v0.3 validation (`qasper-dev-v0.3.json`, sha256 prefix `2ae7ee62a65b1c42`) |
| official evaluator | QASPER v0.3 answer/evidence F1 logic re-implemented in `src/qasper/official_eval.py` |
| papers parsed | 281 validation papers, 1,005 questions |
| passage unit | original paragraph; id `paper_id::s{section:02d}::p{para:03d}` |

## Frozen first-round sample

| item | value |
|---|---|
| seed | 20261110 |
| rule | uniform random sample without replacement over validation papers with ≥1 answerable question whose text evidence maps to a paragraph; order by `sha1(f"{seed}:{paper_id}")`, take the first 100 |
| sample | 100 papers, 372 questions, 333 answerable with mapped text evidence, 141 multi-evidence, 56 cross-section |
| recorded in | `dataset/sample.json` (written before any retrieval evaluation) |

## Models and decoding

| item | value |
|---|---|
| LLM | HY4 (Tencent TokenLink), `HY4_MODEL=hy4` |
| decoding | temperature 0, `reasoning_effort=no_think`, JSON mode for structured extraction |
| embedding | `nvidia/NV-Embed-v2`, fp16, dim 4096, local path `/root/models/NV-Embed-v2` |
| HippoRAG | OSU-NLP-Group/HippoRAG @ `d5c8329` (v2.0.0a5), official index/retrieve |
| torch / transformers | 2.5.1+cu124 / 4.45.2 |
| HY4 rate | cross-process file limiter, 1 QPS shared bucket (`HY4_RATE_FILE=qasper`) |
| caches | content-addressed HY4 sqlite caches; HippoRAG `llm_cache` under each `save_dir` |

## Read-time

| item | value |
|---|---|
| grouping (primary) | section-based, 4,096–8,192-token reading groups |
| atomic facts | ≤4 facts/paragraph, ≤24/block; blocks of ≤6 paragraphs / 2,500 tokens |
| comprehension | ≤10 cross-paragraph insights per group; triples + one-sentence units from the same call |
| verifier | one batched grounded-verification call per paper (support, novelty, causality, source existence) |
| summaries (M6) | one ≤30-word summary per paragraph |
| leakage | prompts receive only paper title, paragraph text, and previously extracted facts; no question/answer/evidence |

## Retrieval and QA

| item | value |
|---|---|
| candidate pool | the question's own paper only |
| reader context | greedy, 4,096 tokens for every method |
| mode B | comprehension units consume part of the same 4,096-token budget |
| oracle | gold evidence paragraphs under 8,192 tokens |
| reader | one fixed QASPER prompt (`src/qasper/reader.py`) for every method |
| evidence scoring | text evidence only (`FLOAT SELECTED` excluded); per-annotator sets treated as alternatives |
| bootstrap | 2,000 paired resamples, question level |

## Codes and artifacts

- Parsing / mapping: `src/qasper/dataset.py`, `scripts/qasper/prepare_data.py`
- Baselines: `src/qasper/retrieval.py`, `scripts/qasper/run_baselines.py`
- Graphs: `src/qasper/graph.py`, `scripts/qasper/build_graphs.py`, `scripts/qasper/build_variants.py`
- Read-time: `src/qasper/prompts.py`, `src/qasper/readtime.py`, `scripts/qasper/run_readtime.py`
- Units: `src/qasper/units.py`
- Retrieval / QA: `scripts/qasper/run_retrieval.py`, `scripts/qasper/run_qa.py`, `src/qasper/reader.py`, `src/qasper/official_eval.py`
- Mechanism: `scripts/qasper/track_units.py`, `scripts/qasper/fusion_sweep.py`
- Reports: `scripts/qasper/make_reports.py`
