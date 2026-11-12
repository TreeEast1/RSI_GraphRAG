# QASPER Read-Time Comprehension — Artifact Index

Snapshot: 2026-11-10. Everything here is produced by `scripts/qasper/*`; the
LoCoMo experiment outputs are untouched.

## Read this first

| file | content |
|---|---|
| `FINAL_REPORT.md` | method comparison, six research questions, mechanism analysis |
| `dataset_audit.md` | QASPER parsing, passage ids, evidence mapping, pool statistics |
| `baseline_retrieval.md` | BM25 / dense / hybrid on all 100 sampled papers |
| `comprehension_quality.md` | generation volume, verifier retention, relation types, source distance |
| `cases.md` | positive / negative / no-effect / overreach cases |
| `retrieval_comparison.csv` | M0-M6 retrieval metrics by subset |
| `qa_comparison.csv` | reader Answer F1 / Evidence F1 / EM by method |
| `experiment_config.md` | frozen configuration, sample seed, model versions, rate policy |
| `README_PIPELINE.md` | reproducible command order |

## Data

- `dataset/sample.json` — frozen 100-paper validation sample (seed 20261110, rule recorded).
- `dataset/papers/{paper_id}/paragraphs.jsonl` — stable paragraph ids + section mapping.
- `dataset/papers/{paper_id}/qa.jsonl` — per-annotator answers and mapped evidence.
- `dataset/summary.json` — corpus / pool / evidence-mapping statistics.
- `dataset/unmatched_evidence.jsonl` — zero text-evidence misses; file kept as a check.

## Graphs

- `graphs/raw/{paper_id}/` — frozen HippoRAG2 G_raw (`qasper_graph_info.json` has counts).
- `graphs/raw/{paper_id}/variants/atom/` — M2: G_raw + atomic facts.
- `graphs/raw/{paper_id}/variants/atom_comp/` — M3: M2 + comprehension triples.
- `graphs/build_log_*.json` — per-paper build status, patched chunks, timings.

## Read-time knowledge

- `comprehension/section/{paper_id}/groups.json` — reading groups (no QA labels used).
- `atomic_facts.json` — per-paragraph atomic facts + call records.
- `insights_raw.json`, `insights_verified.json` — candidate and verified insights, verifier verdicts.
- `units.jsonl` — verified Comprehension Units (the M4 index).
- `summaries.json` — per-paragraph summaries (M6).
- `triples_atomic_by_passage.json`, `triples_comp_by_passage.json` — graph injection inputs.
- `source_provenance.jsonl` — unit → source passages → sections/excerpts.

## Retrieval and QA

- `retrieval/comparison_sample100.jsonl` — per-question ranked lists and metrics for all methods.
- `retrieval/baselines_sample100.jsonl` — M0 on all 100 papers.
- `retrieval/fusion_sweep_sample100.json` — unit/graph RRF weight sensitivity.
- `qa/predictions_sample100.jsonl` — per-question reader answers, contexts, scores.
- `qa/topk_sensitivity_sample100.*` — Top-3 / Top-5 reader sensitivity.
- `mechanism/unit_usage_sample100.jsonl` — per-unit usage trajectory.
- `mechanism/question_chains_sample100.jsonl` — per-question retrieval → unit → QA chain.

## Logs and caches

- `logs/hy4_calls*.jsonl` — every adapter call with status, latency, tokens (no secrets).
- `logs/build_*.log`, `logs/readtime_*.log`, `logs/qa_*.log` — process logs.
- `cache/hy4_*.sqlite` — content-addressed response caches (resumable).
- Each HippoRAG `save_dir/llm_cache/` — official OpenIE/retrieval cache.

## Scope of this snapshot

The validation split has 281 papers / 1,005 questions. The first-round sample is
frozen at 100 papers / 372 questions. **All 100 sampled papers are complete**:
G_raw, M2/M3 variants, read-time knowledge, M0-M6 retrieval (333 answerable
questions) and reader QA (372 questions, including full-paper and oracle
references). M0 baselines additionally cover the same 100 papers.

A 20-paper exploratory semantic-grouping run is also included
(`comprehension/semantic/`, `retrieval/grouping_comparison.json`); it is not part
of the primary comparison.
