# EXPERIMENT_PLAN — actually executed matrix

This file records the experiments that were really run, with their parameters.
Anything not listed here was not executed.

## 0. Fixed configuration

| item | value |
|---|---|
| dataset | LoCoMo `locomo10.json` (official, sha256 in `run_manifest.json`) |
| documents | 1 Session = 1 document (272 documents, 5882 turns) |
| split | seed 20240501, dev = conv-30/44/48, test = the other 7 |
| LLM | HY4 via TokenLink, `reasoning_effort=no_think`, temperature 0 |
| embedding | nvidia/NV-Embed-v2, fp16, dim 4096, query instructions from HippoRAG |
| HippoRAG | OSU-NLP-Group/HippoRAG @ `d5c8329` (v2.0.0a5), official index/retrieve |
| reader | one shared prompt (`src/retrieval/reader.py`), 2048/4096/8192-token context budgets |
| retrieval cut | top-20 documents; main QA budget 4096 tokens |

## A. P0 baselines

| id | method | description |
|---|---|---|
| A1 | `dense` | NV-Embed-v2 session retrieval → reader |
| A2 | `hipporag` | official HippoRAG2 on G_raw (OpenIE → graph → PPR) |
| A3 | `fullcontext` | whole conversation in one prompt (reported separately; not a fixed-budget baseline) |

## B. P1 grouping (experiment B)

Budgets: 4096 and 8192 tokens per reading group.
Methods: `random` (5 seeds), `temporal`, `entity_jaccard`, `entity_idf`,
`embedding`, `hybrid_0.25/0.5/0.75`, `community` (Louvain on the doc–entity
bipartite graph with IDF hub suppression).

Entity sets come from the official OpenIE NER output of G_raw (no extra LLM
calls).  Document embeddings come from NV-Embed-v2 with an empty instruction.

Metrics: gold-coverage (full/partial), cross-session co-location, group size and
token distribution, duplicate reading cost.

## C. P1 read-time (experiment C)

| id | scope | graph |
|---|---|---|
| C0/C1 | official per-session OpenIE | G_raw (frozen) |
| C2 | joint extraction over reading groups (`cluster_*_triple`) | G_raw + joint triples |
| C3 | C2 + cross-session comprehension (`cluster_*_comp`) | G_raw + triples + relations |
| C3v | C3 + LLM verification (`cluster_*_comp_ver`) | as C3, verified subset |
| C5 | whole-conversation extraction (`full_conv_triple`) | G_raw + full-conv triples |
| C6 | whole-conversation + comprehension (`full_conv_comp`) | G_raw + full-conv triples + relations |

C1 is documented as equivalent to C0 because both are the official
NER→triple OpenIE prompt on one session at a time with the same HY4 model; we do
not report them as independent results.

Graph variants are built by seeding `openie_state.json` and re-running the
official `index()` in a separate directory (`src/graph/variants.py`).

## D. P1 retrieval / QA evaluation

Every graph variant is evaluated with the same reader, the same top-20 cut and
the 4096-token context budget.  Metrics: Recall@1/3/5/10/20, MRR, nDCG@10,
session-level and turn-level evidence coverage at 2048/4096/8192 tokens, EM, F1,
and per-category breakdowns.

## E. P2 question-time (experiment E)

| id | strategy |
|---|---|
| Q0 | raw question |
| Q1 | QDM decomposition (only when the model judges it necessary) |
| Q2 | Q1 + retrieval skill (missing-relation diagnosis + extra queries, RRF merge) |
| Q3 | Q2 + answer skill (evidence-checked revision / refusal) |

Run on dev for tuning; skill library frozen before any test run.

## F. P2 joint

R0Q0, R1Q0, R0Q1, R1Q1 (best dev read-time variant × best dev question-time
strategy), reported with absolute metrics and percentage-point deltas.

## Cost accounting

Per-stage HY4 token counts are available in `outputs/logs/hy4_calls*.jsonl`
(this covers our adapters) and in each HippoRAG `save_dir/llm_cache/*.sqlite`
for OpenIE/QA calls made through the official adapter.  Indexing (OpenIE) tokens
and question-time tokens are reported separately.
