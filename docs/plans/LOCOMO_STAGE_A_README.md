# RSI GraphRAG / CogitoRAG — LoCoMo full-chain experiments

Read-Time Comprehension Evolution × Question-Time Retrieval Skill, built on the
official HippoRAG2 implementation, with HY4 (Tencent TokenLink) as the only LLM
and NV-Embed-v2 as the only embedding model.

## Quick start

```bash
cd /group/40150/licochen/Memory/RSI_GraphRAG
source .venv/bin/activate          # python 3.10, torch 2.5.1, hipporag 2.0.0a5
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken
export HF_HUB_OFFLINE=1

python scripts/prepare_data.py                       # LoCoMo -> processed + audit
python scripts/run_baselines.py --split dev          # P0 baselines
python scripts/run_grouping.py --split dev           # P1 experiment B
python scripts/run_read_time_extract.py --split dev  # P1 experiment C (LLM stage)
python scripts/run_read_time_eval.py --split dev --variants <scope>:<suffix>
python scripts/run_question_time.py --split dev      # P2 experiment E
python scripts/aggregate.py                          # metrics.csv
```

`outputs/reports/START_HERE.md` is the entry point for reviewing results.

## Environment notes (important)

* **The project lives on a Ceph mount whose small-file and sequential-read
  throughput is very low (~9 MB/s warm-cold).** Two large artefacts therefore
  live on local disk:
  * `.venv` → symlink to `/root/venvs/rsigraph`
  * NV-Embed-v2 → `/root/models/NV-Embed-v2` (config.json `_name_or_path` patched
    to the local path because the model's remote code reloads the tokenizer by
    repo id). `src/config.py` picks it up automatically via `NVEMBED_PATH`.
* `cache/models/NV-Embed-v2` under the project keeps an identical copy as a
  fallback; `src/config.py` selects the first existing candidate.
* GitHub / HuggingFace are only reachable through `http://star-proxy.oa.com:3128`.
* tiktoken's BPE file is cached in `cache/tiktoken` (`TIKTOKEN_CACHE_DIR`).

## Layout

```
configs/            experiment configs
third_party/HippoRAG  official repo @ d5c8329 (HippoRAG2, v2.0.0a5)
data/raw/locomo/    locomo10.json (official)
data/processed/     per-conversation documents / turns / qa (evidence mapped)
data/splits/        fixed dev(3) / test(7) split
src/
  adapters/         HY4 client (cache, retry, concurrency, token accounting)
  dataset/          LoCoMo parsing, session & turn modelling
  embedding/        NV-Embed-v2 wrapper + content-addressed cache
  entity/           (entity sets come from official OpenIE NER)
  grouping/         B0-B5 document grouping + group metrics
  graph/            HippoRAG adapters, graph variants, runtime
  comprehension/    read-time extraction + comprehension prompts
  retrieval/        dense retriever, unified reader
  question_skill/   QDM, retrieval skill, answer skill, skill library
  evaluation/       metrics, per-question records, aggregation
scripts/            executable pipeline stages
outputs/            graphs, groupings, retrieval, qa, reports, logs
```

## What is official vs. what is ours

**Official HippoRAG2 (unchanged):** `index()` (chunking, NER→triples OpenIE,
entity/fact/passage graph construction, synonymy edges, PPR), `retrieve()`
(fact reranking + Personalized PageRank), embedding stores, caches, manifests.

**Ours (injected through official extension points):**

| file | role |
|---|---|
| `src/adapters/hy4_client.py` | HY4 OpenAI-compatible client (`reasoning_effort=no_think`) |
| `src/graph/hipporag_adapter.py` | `HY4CacheOpenAI` subclass + config builder |
| `src/graph/variants.py` | builds G_comp variants by seeding OpenIE state and re-running official `index()` |
| `src/grouping/*` | unsupervised grouping strategies |
| `src/comprehension/*` | joint extraction / comprehension prompts + parsing |
| `src/question_skill/*` | QDM / retrieval skill / answer skill |
| `src/retrieval/reader.py` | single reader shared by every method |

Graph variants are **never** produced by editing `G_raw`: a variant directory
copies the raw embedding stores + OpenIE state (graph pickle and manifest are
deliberately omitted) and then re-runs the official `index()` with
`force_index_from_scratch=True`. See `src/graph/variants.py` for details.

## Reproducibility

* Fixed split seed 20240501: dev = `conv-30, conv-44, conv-48`, test = the
  remaining seven conversations.
* Every HY4 call is content-addressed in a sqlite cache and logged to
  `outputs/logs/hy4_calls*.jsonl` with token usage, latency and status.
* HippoRAG keeps its own LLM cache under each `save_dir/llm_cache`.
* `outputs/reports/run_manifest.json` records configs, commits and costs.

## Results (2026-11-06 run)

| experiment | status | headline |
|---|---|---|
| P0 baselines (10 conv, 1,986 QA) | complete | HippoRAG2 R@5 .863 / evidence .828 vs Dense .781/.732; Full-Context F1 .491 ≈ HippoRAG .483 |
| Grouping (dev+test, 2 budgets) | complete | random grouping already reaches .883 gold coverage; clustering adds ~0 |
| Read-Time (dev, 4 variants) | complete | best C5/C6: EM .271/.268, evidence .845/.847 (raw .253/.819) |
| Read-Time (test, C3, retrieval) | complete (7/7) | R@5 .881 vs .864, evidence .842 vs .831 |
| Question-Time (dev conv-30) | complete | Q2 hurts QA (.502 F1), Q3 recovers to .539 |
| Joint 2×2 (dev conv-30) | complete | R1Q3 best retrieval (.941) but not best QA → H6 not supported |

Full narrative: `outputs/reports/FINAL_EXPERIMENT_REPORT.md`.
Entry point for review: `outputs/reports/START_HERE.md`.

## API quota behaviour (important for reruns)

The TokenLink pool is shared company-wide. Measured on 2026-11-06: 1 QPS gives ~90% success,
2 QPS gives ~7%, a 32-request burst succeeds but a sustained 64 fails. `src/adapters/ratelimit.py`
implements a **cross-process file-based limiter** so any number of worker processes together stay
under the configured rate (`HY4_GLOBAL_QPS`). Multiple keys from the same pool share one bucket
(`HY4_RATE_FILE=poolA`); an independent account uses a different bucket (`poolB`).
Without the limiter, retry storms waste most of the quota.
