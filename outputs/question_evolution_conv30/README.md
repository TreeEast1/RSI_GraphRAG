# Question-Time Self-Evolving Answer & Retrieval Skills (conv-30)

This directory contains the code and all measured artifacts for the RSI GraphRAG
Question-Time self-evolution study on LoCoMo conv-30.

## Quick start

```bash
cd /group/40150/licochen/Memory/RSI_GraphRAG/outputs/question_evolution_conv30
PY=/group/40150/licochen/Memory/RSI_GraphRAG/.venv/bin/python

$PY -m qecode.s0_split                 # fixed stratified split + DATA_SPLIT.md
$PY -m qecode.s1_baselines             # P0 / P7 / P8 baselines
$PY -m qecode.answer_evolve            # Answer Skill loop + snapshots
$PY -m qecode.retrieval_evolve         # Retrieval Skill loop + snapshots
$PY -m qecode.s3_joint                 # E0-E6 + equal-budget static control
$PY -m qecode.s4_react                 # ReAct + grounded verifier
$PY -m qecode.s5_repeats               # 3 fresh runs of final candidates
$PY -m qecode.s6_analysis              # error analysis, cost, qa_predictions
$PY -m qecode.s7_reports               # narrative reports
```

The endpoint is the shared HY4 TokenLink pool.  `qecode/common.py` sets one
cross-process rate bucket (`HY4_RATE_FILE=qe_main`, `HY4_GLOBAL_QPS=1.2`).
Every HY4 call is cached in `artifacts/hy4_cache.sqlite` by model + params +
messages; `logs/call_costs.jsonl` records usage and latency for every call.

## Design

**Fixed base.** The retrieval base is the frozen `COMP-P4_top5` HippoRAG2
ranking (no LLM fact reranker) from `outputs/prompt_optimization_conv30`,
4,096-token Reader context.  This isolates skill effects from graph changes.

**Answer Skills.** A skill has `skill_id, title, trigger_conditions,
applicable_question_patterns, categories, action_instructions,
expected_output_format, evidence_requirements, failure_modes,
source_training_qids, success_count, failure_count, confidence, version,
status`.  Memory retrieval is category + pattern + embedding similarity.
A candidate is accepted only if it (a) fixes its source question, (b) does not
flip similar/dissimilar training questions from correct to wrong, and (c) does
not make adversarial questions stop refusing.  A sanitizer rejects any
candidate containing a distinctive answer token, proper noun, or explicit year.

**Retrieval Skills.** Base graph ranking + QDM sub-queries + an evidence
sufficiency check + at most one extra skill/missing-relation retrieval round.
Skill queries are dense session retrievals merged with the base ranking by RRF.
Verification is local and LLM-free (source fix, similar replay, negative
transfer).

**Modes.** Adaptation uses gold answers/evidence only for reflection and
verification.  Evaluation never sees gold and never writes to memory.  The
final snapshot is selected on the 19-question development split; the 43-question
evaluation split is untouched until that choice is frozen.

## Artifact map

| path | contents |
|---|---|
| `DATA_SPLIT.md`, `DATA_SPLIT.json` | fixed stratified split (seed 20261110) |
| `baseline_results.md` | P0/P7/P8 |
| `answer_skill_evolution.md` | answer loop, snapshots, per-skill reuse |
| `retrieval_skill_evolution.md` | retrieval loop, query templates, evidence |
| `joint_experiments.md` | E0-E6 + EB |
| `react_and_verification.md` | R0-R5 |
| `error_analysis.md` | fixed/worsened cases, skill hits |
| `cost_latency.md` | per-stage calls/tokens/latency |
| `repeat_stability.md` | 3 fresh runs |
| `FINAL_REPORT.md` | answers to the study questions |
| `qa_predictions.jsonl` | eval predictions for all main methods |
| `skill_memory.jsonl` | final Answer Skill memory (active + disabled) |
| `retrieval_skill_memory.jsonl` | final Retrieval Skill memory |
| `skill_evolution_trace.jsonl` | every adaptation/reflection/verification decision |
| `skill_snapshots/` | memory + dev evaluation at 0/10/20/30/43 feedback |
| `artifacts/` | caches, summaries, per-question JSONL, costs, error cases |
| `logs/` | HY4 call log and stage logs |

## Important caveat

conv-30 has been used in earlier prompt-optimization rounds.  The evaluation
split is fixed before learning, but it is an internal exploratory test, not a
fully independent final benchmark.  Category-level estimates are noisy (five
multi-hop questions in the evaluation split).
