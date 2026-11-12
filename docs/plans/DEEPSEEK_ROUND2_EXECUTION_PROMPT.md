# DeepSeek execution prompt: DetectiveQA Round 2

You are running the DetectiveQA Round 2 experiment on the GPU machine.

Project:
`/group/40150/licochen/Memory/RSI_GraphRAG`

Read first:
- `DETECTIVEQA_ROUND2_PLAN.md`
- `outputs/detectiveqa/PILOT_REPORT.md`
- `src/detectiveqa/{data,graph,reader,readtime,round2,prompts}.py`
- `scripts/detectiveqa/run_round2_*.py`, `run_readtime_v3.py`, `run_relation_channel.py`,
  `analyze_round2.py`, `judge_round2_cases.py`, `build_relation_variants.py`

Objective: explain why round 1 graph/comprehension did not work. Do not optimize for a headline score.

## Non-negotiable rules

1. Do not modify official HippoRAG `index()`, PPR, OpenIE prompts or raw graph directories.
2. Primary deployable retrieval is question-only (`q_*`). Question+options (`qo_*`) is diagnostic only.
3. Gold answers/clues can be used only for metrics, oracle and post-hoc case judging.
4. Read-time v3 packets must be built only from OpenIE entity statistics; no question/options/answers/clues.
5. Keep the 4096-token reader budget. Relation notes must use part of that same budget.
6. Do not tune prompts on the reported eval results. If code is broken, fix the bug, rerun the stage, and log the fix.
7. Report absolute values, deltas, n, novel-clustered CIs, McNemar p values, and case evidence.
8. Save all commands and errors in `outputs/detectiveqa/round2/RUN_LOG.md`.

## Setup and smoke tests

```bash
cd /group/40150/licochen/Memory/RSI_GraphRAG
source .venv/bin/activate || source /root/venvs/rsigraph/bin/activate
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken HF_HUB_OFFLINE=1
export HY4_RATE_FILE=detectiveqa HY4_GLOBAL_QPS=1
mkdir -p outputs/detectiveqa/round2
python -m unittest discover -s tests -p 'test_detectiveqa*.py' -v
python scripts/detectiveqa/check_prefix_safety.py --split eval_human --novels 27,79,117,118,121,124 --k 20 --out prefix_safety_round2
python scripts/detectiveqa/run_round2_retrieval.py --limit 2 --out round2_retrieval_smoke
python scripts/detectiveqa/run_round2_reader.py --retrieval round2_retrieval_smoke.jsonl --limit 1 --option-seeds 0 --conditions q_dense,recent --out round2_reader_smoke
python scripts/detectiveqa/run_readtime_v3.py --novels 27 --packets-per-novel 2 --max-total-packets 2 --out readtime_v3_smoke
```

Inspect the smoke outputs. Stop and fix only if outputs are malformed, empty, or leak future chunks.

## Stage A: retrieval diagnosis

```bash
python scripts/detectiveqa/run_round2_retrieval.py --modes q,qo --top-k 50 --fact-top-k 5,10,20 --out round2_retrieval
```

Compare dense, BM25, graph f5/f10/f20, synonym on/off and RRF. Analyze per novel and cutoff bucket.
For each `dense_better` and `graph_better` gold-rank case, inspect top facts, fact provenance and gold ranks.

## Stage B: reader causality and robustness

```bash
python scripts/detectiveqa/run_round2_reader.py \
  --retrieval round2_retrieval.jsonl \
  --conditions options_only,options_only_memory,q_dense,q_graph,q_rrf,qo_dense,recent,nonrecent_q_dense,recent_plus_q_dense,oracle \
  --option-seeds 0,1 --out round2_reader
python scripts/detectiveqa/analyze_round2.py --retrieval round2_retrieval.jsonl --reader round2_reader.jsonl --out round2_analysis
python scripts/detectiveqa/judge_round2_cases.py --reader round2_reader.jsonl --out round2_case_judge
```

Key questions:
- Does q-only graph/RRF beat q-only dense?
- Is `recent` still strong outside late cutoffs?
- Does `nonrecent_q_dense` collapse, proving locality/reveal dependence?
- Are errors all-gold, partial-gold or no-gold?
- Are answers stable after option permutation?

## Stage C: read-time comprehension v3

```bash
python scripts/detectiveqa/run_readtime_v3.py --packets-per-novel 20 --chunks-per-packet 5 --max-words 1500 --out readtime_v3
python scripts/detectiveqa/run_relation_channel.py --readtime readtime_v3.jsonl --retrieval round2_retrieval.jsonl --out relation_channel_kept
python scripts/detectiveqa/run_relation_channel.py --readtime readtime_v3.jsonl --retrieval round2_retrieval.jsonl --include-dropped --out relation_channel_validated
```

Manually audit at least 60 relations: 30 kept, 20 verifier-dropped, 10 rule-rejected, covering all 6 novels.
Labels: correct_cross_chunk, correct_but_fact_rewrite, insufficient_evidence, entity_alignment_error,
claim_as_fact, too_generic. Write `outputs/detectiveqa/round2/readtime_v3_manual_audit.jsonl`.

## Stage D: relation QA and graph injection gate

Always run relation notes QA if relation_channel_kept touches gold for any question:

```bash
python scripts/detectiveqa/run_round2_reader.py \
  --retrieval round2_retrieval.jsonl --relations outputs/detectiveqa/relation_channel_kept.jsonl \
  --conditions q_dense,relations_q_dense,q_rrf,relations_q_rrf --option-seeds 0 --out round2_reader_relations
```

Run graph injection only if read-time v3 has >=100 kept relations and manual kept precision >=75%:

```bash
python scripts/detectiveqa/build_relation_variants.py --readtime readtime_v3.jsonl --variants r2_joint,r3_validated,r4_kept,r5_joint_count
for v in r2_joint r3_validated r4_kept r5_joint_count; do
  python scripts/detectiveqa/run_round2_retrieval.py --variant $v --modes q --top-k 50 --fact-top-k 5 --out round2_retrieval_$v
done
```

If the gate fails, do not build variants; explain why using audit counts and relation-channel cases.

## Required case study

Read every question in `round2_analysis.json.case_queues`. For each case write:
qid, novel, cutoff bucket, conditions correct/wrong, gold recall in context, evidence location,
LLM judge label, your corrected label, and concrete method implication. Pay special attention to:
27:00, 27:06, 117:01, 117:149:01, 118:06, 124:05 and all cases where recent is correct without gold.

## Final report

Write `outputs/detectiveqa/round2/ROUND2_REPORT.md` with:
1. Executive verdict: graph, recency, reader, comprehension; what works and what does not.
2. Retrieval table: q/qo dense, BM25, graph variants, RRF, with R@10/R@20/All@20/All@50/MRR.
3. Reader table: all conditions, seed 0 and 1, gold-any/gold-all, by cutoff bucket.
4. Paired tests with novel-clustered CI and McNemar; label all results exploratory.
5. Failure taxonomy and case table.
6. Read-time v3 funnel, audit precision, relation-channel gold touch and relation QA.
7. Graph variants or a precise reason for not running them.
8. Cost table from HY4 logs.
9. Next decision: scale, change benchmark split, change retrieval, or stop graph direction.

Return to the user: report path, the 5 most important numbers, the 5 most informative cases, and the
decision with evidence. Do not claim improvement unless it survives the stated gates.
