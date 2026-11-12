# DetectiveQA Round 2 — Diagnosis Report

Date: 2026-11-12. All numbers come from files under `outputs/detectiveqa/round2/`; commands and errors are in `RUN_LOG.md`. Official HippoRAG `index()`, PPR, OpenIE prompts and raw graph directories were not modified.

## 1. Executive verdict

**Graph retrieval does not work on this benchmark split, and the reason is not the graph.**

- **Graph fails the pre-registered gate.** q-only `q_graph_f5` R@10 .425 vs dense .418 (delta +0.007, novel-clustered CI [-0.035,+0.043]); All@20 .244 vs .220 (delta +0.024, CI [+0.000,+0.081]); MRR .497 vs .544 (delta -0.048, CI [-0.128,+0.032], McNemar p=.012, 4 wins / 16 losses). RRF does not meet the gate either (R@10 .429, +1.1pp, CI crosses 0; All@20 .220, no gain). **No positive clustered-CI lower bound on R@10 or All@20.**
- **Recency dominates.** `recent` (last 4096 tokens before cutoff) scores **.805 on both option seeds**, equal to oracle seed 0 and above q_dense (.658/.585), q_graph (.683/.658) and q_rrf (.634/.634). Removing recent chunks (`nonrecent_q_dense`) collapses accuracy to **.439/.390**, near the memory-only baseline (.463/.439). Paired recent→nonrecent: -36.6pp (CI [-51.2,-17.1], p=.0015) at seed 0 and -41.5pp (CI [-54.5,-22.2], p=.0005) at seed 1.
- **This is not only a late-cutoff artifact.** `recent` is .833/.833 (early), .75/.75 (middle), .80/.80 (late) across seeds; `nonrecent_q_dense` is .50/.39, .375/.375, .40/.40. The benchmark often places decisive evidence near the cutoff in all buckets.
- **Gold clues are not a reliable upper bound.** Oracle is .805/.707; recent is ≥ oracle. The LLM judge calls 16/41 gold sets insufficient, 23/41 ambiguous/questionable, and 30/41 have explicit reveal before cutoff. 27:00, 27:06, 117:01, 118:06 are concrete cases.
- **Reader remains a bottleneck, but evidence/locality comes first.** Dense seed-0 errors: 1 all-gold, 10 partial-gold, 3 no-gold. Oracle still errs on 6 all-gold questions. However, the dominant loss is evidence/locality and label ambiguity, not a missing graph.
- **Comprehension v3 is higher-yield but not precise enough.** 601 kept relations (gate quantity passes), but manual kept precision is **20/30 = 66.7%** genuine cross-chunk (24/30 = 80% if fact rewrites count as correct). Relation channel touches gold for 17/41 questions, yet equal-weight RRF lowers R@10 (.312 vs dense .418) and relation notes give no QA gain (q_dense .658→.658; q_rrf .634→.658, CI crosses 0). **Graph-injection gate fails; R2–R5 were not built.**
- **Decision: do not scale, and stop the graph direction on the current split.** Change the benchmark split/question construction (anti-reveal, re-audited gold) before any scale-up; see §9.

## 2. Retrieval table (q-only primary, qo diagnostic)

| condition | R@10 | R@20 | All@20 | All@50 | MRR |
|---|---:|---:|---:|---:|---:|
| q_dense | 0.418 | 0.515 | 0.220 | 0.463 | 0.544 |
| q_bm25 | 0.287 | 0.426 | 0.122 | 0.390 | 0.455 |
| q_graph_f5 | 0.425 | 0.538 | 0.244 | 0.463 | 0.497 |
| q_graph_f10 | 0.420 | 0.540 | 0.268 | 0.463 | 0.462 |
| q_graph_f20 | 0.415 | 0.534 | 0.244 | 0.463 | 0.452 |
| q_graph_f5_nosyn | 0.412 | 0.534 | 0.220 | 0.463 | 0.501 |
| q_rrf_dense_graph | 0.429 | 0.523 | 0.220 | 0.463 | 0.516 |
| q_rrf_dense_bm25 | 0.391 | 0.499 | 0.195 | 0.390 | 0.530 |
| q_rrf_all | 0.402 | 0.508 | 0.220 | 0.488 | 0.541 |
| qo_dense | 0.436 | 0.527 | 0.244 | 0.439 | 0.604 |
| qo_graph_f5 | 0.407 | 0.537 | 0.244 | 0.463 | 0.486 |
| qo_rrf_dense_graph | 0.436 | 0.535 | 0.244 | 0.463 | 0.563 |

MRR counts a missed gold chunk as 0; all metrics are over all 41 questions. `qo_*` is diagnostic only (options are not available at retrieval time in the paper setting).

**Novel-clustered paired tests (bootstrap 5000):**

| comparison | metric | delta | 95% CI | wins/losses | McNemar p |
|---|---|---:|---|---:|---:|
| q_dense→q_graph_f5 | recall@10 | +0.0072 | [-0.0347,+0.0434] | 8/9 | 1 |
| q_dense→q_graph_f5 | all@20 | +0.0244 | [+0.0000,+0.0811] | 1/0 | 1 |
| q_dense→q_graph_f5 | mrr | -0.0478 | [-0.1276,+0.0320] | 4/16 | 0.01182 |
| q_dense→q_graph_f10 | recall@10 | +0.0016 | [-0.0471,+0.0563] | 5/7 | 0.7744 |
| q_dense→q_graph_f10 | all@20 | +0.0488 | [+0.0000,+0.1111] | 2/0 | 0.5 |
| q_dense→q_graph_f10 | mrr | -0.0820 | [-0.1872,+0.0313] | 4/20 | 0.001544 |
| q_dense→q_graph_f20 | recall@10 | -0.0034 | [-0.0527,+0.0529] | 7/7 | 1 |
| q_dense→q_graph_f20 | all@20 | +0.0244 | [-0.0556,+0.1053] | 2/1 | 1 |
| q_dense→q_graph_f20 | mrr | -0.0926 | [-0.2053,+0.0441] | 6/20 | 0.009355 |
| q_dense→q_graph_f5_nosyn | recall@10 | -0.0059 | [-0.0518,+0.0389] | 7/9 | 0.8036 |
| q_dense→q_graph_f5_nosyn | all@20 | +0.0000 | [+0.0000,+0.0000] | 0/0 | 1 |
| q_dense→q_graph_f5_nosyn | mrr | -0.0430 | [-0.1381,+0.0392] | 4/14 | 0.03088 |
| q_dense→q_bm25 | recall@10 | -0.1308 | [-0.1540,-0.1147] | 5/19 | 0.006611 |
| q_dense→q_bm25 | all@20 | -0.0976 | [-0.1892,-0.0208] | 0/4 | 0.125 |
| q_dense→q_bm25 | mrr | -0.0894 | [-0.1500,-0.0023] | 11/16 | 0.4421 |
| q_dense→q_rrf_dense_graph | recall@10 | +0.0112 | [-0.0165,+0.0417] | 4/3 | 1 |
| q_dense→q_rrf_dense_graph | all@20 | +0.0000 | [-0.0698,+0.0769] | 1/1 | 1 |
| q_dense→q_rrf_dense_graph | mrr | -0.0287 | [-0.0650,+0.0018] | 5/8 | 0.5811 |
| q_dense→q_rrf_dense_bm25 | recall@10 | -0.0269 | [-0.0630,+0.0207] | 4/7 | 0.5488 |
| q_dense→q_rrf_dense_bm25 | all@20 | -0.0244 | [-0.0769,+0.0000] | 0/1 | 1 |
| q_dense→q_rrf_dense_bm25 | mrr | -0.0143 | [-0.0507,+0.0263] | 10/8 | 0.8145 |
| q_dense→q_rrf_all | recall@10 | -0.0159 | [-0.0548,+0.0358] | 4/6 | 0.7539 |
| q_dense→q_rrf_all | all@20 | +0.0000 | [-0.0698,+0.0769] | 1/1 | 1 |
| q_dense→q_rrf_all | mrr | -0.0038 | [-0.0479,+0.0370] | 12/7 | 0.3593 |
| q_dense→qo_dense | recall@10 | +0.0179 | [-0.0454,+0.0862] | 9/9 | 1 |
| q_graph_f5→qo_graph_f5 | recall@10 | -0.0178 | [-0.0850,+0.0446] | 4/11 | 0.1185 |

**By cutoff bucket (q-only):**

| bucket | n | q_dense R@10 / All@20 / MRR | q_graph_f5 | q_graph_f20 | q_rrf | q_bm25 |
|---|---:|---|---|---|---|---|
| early_lt_70 | 18 | 0.578 / 0.389 / 0.667 | 0.588 / 0.444 / 0.572 | 0.615 / 0.444 / 0.558 | 0.592 / 0.389 / 0.624 | 0.436 / 0.222 / 0.568 |
| late_ge_90 | 15 | 0.239 / 0.000 / 0.395 | 0.212 / 0.000 / 0.429 | 0.174 / 0.000 / 0.369 | 0.253 / 0.000 / 0.375 | 0.142 / 0.000 / 0.337 |
| middle_70_90 | 8 | 0.396 / 0.250 / 0.550 | 0.458 / 0.250 / 0.454 | 0.417 / 0.250 / 0.368 | 0.396 / 0.250 / 0.537 | 0.225 / 0.125 / 0.423 |

The graph is slightly better on early/middle R@10 but worse on late R@10; All@20 is 0 for every method in the late bucket. The only stable graph signal is a small early/middle coverage gain, not a deployable overall gain.

**Per novel (q-only, R@10 / All@20 / MRR):**

| novel | n | q_dense | q_graph_f5 | q_rrf |
|---|---:|---|---|---|
| 27 | 7 | 0.327 / 0.286 / 0.438 | 0.387 / 0.286 / 0.404 | 0.363 / 0.143 / 0.440 |
| 79 | 5 | 0.233 / 0.000 / 0.285 | 0.233 / 0.000 / 0.297 | 0.233 / 0.000 / 0.282 |
| 117 | 10 | 0.312 / 0.000 / 0.633 | 0.311 / 0.000 / 0.592 | 0.295 / 0.000 / 0.624 |
| 118 | 7 | 0.373 / 0.286 / 0.392 | 0.429 / 0.286 / 0.508 | 0.444 / 0.286 / 0.404 |
| 121 | 6 | 0.465 / 0.167 / 0.584 | 0.479 / 0.333 / 0.415 | 0.486 / 0.333 / 0.487 |
| 124 | 6 | 0.861 / 0.667 / 0.875 | 0.764 / 0.667 / 0.681 | 0.819 / 0.667 / 0.778 |

**Fact-seed mechanism:** q_graph_f5 has at least one gold-touching fact for 27/41 questions (mean 1.32 of 5 facts); f10 30/41, f20 33/41. Expanding seeds raises coverage but lowers MRR (f5 .497, f10 .462, f20 .452), matching the round-1 'top20 is worse' result. Synonym edges add little (q_graph_f5_nosyn: R@10 .412, All@20 .220, MRR .501 vs .425/.244/.497).

## 3. Reader table

| condition | seed0 acc | seed0 gold-any | seed0 gold-all | seed1 acc | seed1 gold-any | seed1 gold-all |
|---|---:|---:|---:|---:|---:|---:|
| nonrecent_q_dense | 0.439 | 0.585 | 0.000 | 0.390 | 0.585 | 0.000 |
| options_only | 0.293 | 0.000 | 0.000 | 0.439 | 0.000 | 0.000 |
| options_only_memory | 0.463 | 0.000 | 0.000 | 0.439 | 0.000 | 0.000 |
| oracle | 0.805 | 1.000 | 0.927 | 0.707 | 1.000 | 0.927 |
| q_dense | 0.658 | 0.756 | 0.122 | 0.585 | 0.756 | 0.122 |
| q_graph | 0.683 | 0.780 | 0.146 | 0.658 | 0.780 | 0.146 |
| q_rrf | 0.634 | 0.780 | 0.122 | 0.634 | 0.780 | 0.122 |
| qo_dense | 0.707 | 0.732 | 0.220 | 0.707 | 0.732 | 0.220 |
| recent | 0.805 | 0.512 | 0.122 | 0.805 | 0.512 | 0.122 |
| recent_plus_q_dense | 0.805 | 0.683 | 0.195 | 0.732 | 0.683 | 0.195 |

`nonrecent_q_dense` removes the most recent 4096-token suffix and then packs q_dense; `recent_plus_q_dense` spends half the budget on recent text and half on q_dense. `oracle` gets only annotated gold chunks. `options_only_memory` sees no text but may use prior knowledge.

**By cutoff bucket (accuracy, seed0/seed1):**

| condition | early_lt_70 | middle_70_90 | late_ge_90 |
|---|---:|---:|---:|
| q_dense | 0.72/0.72 | 0.50/0.38 | 0.67/0.53 |
| q_graph | 0.72/0.78 | 0.50/0.50 | 0.73/0.60 |
| q_rrf | 0.72/0.72 | 0.50/0.50 | 0.60/0.60 |
| recent | 0.83/0.83 | 0.75/0.75 | 0.80/0.80 |
| nonrecent_q_dense | 0.50/0.39 | 0.38/0.38 | 0.40/0.40 |
| recent_plus_q_dense | 0.94/0.83 | 0.50/0.50 | 0.80/0.73 |
| oracle | 0.83/0.78 | 1.00/0.88 | 0.67/0.53 |
| qo_dense | 0.67/0.78 | 0.50/0.38 | 0.87/0.80 |
| options_only_memory | 0.56/0.50 | 0.38/0.50 | 0.40/0.33 |

(Each cell is seed0/seed1.)

**Error decomposition, seed 0 (wrong answers):**

| condition | all gold present | partial gold | no gold |
|---|---:|---:|---:|
| q_dense | 1 | 10 | 3 |
| q_graph | 2 | 8 | 3 |
| q_rrf | 2 | 10 | 3 |
| qo_dense | 2 | 7 | 3 |
| recent | 0 | 3 | 5 |
| recent_plus_q_dense | 0 | 5 | 3 |
| oracle | 6 | 2 | 0 |
| nonrecent_q_dense | 0 | 12 | 11 |
| options_only_memory | 0 | 0 | 22 |

Dense's 14 errors are mostly partial-gold (10), not no-gold (3). Oracle errs on 6 all-gold questions. Removing recent text increases no-gold errors to 11. This is a reader/locality problem, not a missing graph.

**Option-order stability (same original answer, seeds 0 vs 1):**

| condition | agreement |
|---|---:|
| nonrecent_q_dense | 0.951 |
| options_only | 0.488 |
| options_only_memory | 0.732 |
| oracle | 0.854 |
| q_dense | 0.902 |
| q_graph | 0.902 |
| q_rrf | 0.902 |
| qo_dense | 0.805 |
| recent | 0.976 |
| recent_plus_q_dense | 0.902 |

`recent` is the most stable condition (.976); `recent_plus_q_dense` .902; q_dense/graph/rrf .902; oracle .854; options-only memory .732; strict options-only .488. Retrieval and oracle answers are option-order sensitive, so single-seed deltas must be treated as exploratory.

## 4. Paired tests (novel-clustered, McNemar)

Seed 0 reader comparisons (delta = right − left; positive means right is better):

| comparison | delta | 95% CI | wins | losses | McNemar p |
|---|---:|---|---:|---:|---:|
| q_dense->q_graph | +0.0244 | [+0.0000,+0.0811] | 2 | 1 | 1 |
| q_dense->q_rrf | -0.0244 | [-0.0750,+0.0000] | 1 | 2 | 1 |
| q_dense->recent | +0.1463 | [+0.0571,+0.2143] | 8 | 2 | 0.1094 |
| q_dense->nonrecent_q_dense | -0.2195 | [-0.3333,-0.0833] | 1 | 10 | 0.01172 |
| q_dense->recent_plus_q_dense | +0.1463 | [+0.0426,+0.2821] | 7 | 1 | 0.07031 |
| q_dense->qo_dense | +0.0488 | [+0.0000,+0.1176] | 3 | 1 | 0.625 |
| q_dense->oracle | +0.1463 | [+0.0426,+0.2727] | 9 | 3 | 0.146 |
| q_dense->options_only_memory | -0.1951 | [-0.3409,+0.0000] | 4 | 12 | 0.07681 |
| recent->nonrecent_q_dense | -0.3659 | [-0.5116,-0.1714] | 3 | 18 | 0.00149 |
| recent->recent_plus_q_dense | +0.0000 | [-0.1064,+0.1081] | 2 | 2 | 1 |

Seed 1:

| comparison | delta | 95% CI | wins | losses | McNemar p |
|---|---:|---|---:|---:|---:|
| q_dense->q_graph | +0.0732 | [+0.0227,+0.1389] | 4 | 1 | 0.375 |
| q_dense->q_rrf | +0.0488 | [-0.0455,+0.1351] | 3 | 1 | 0.625 |
| q_dense->recent | +0.2195 | [+0.1081,+0.3200] | 10 | 1 | 0.01172 |
| q_dense->nonrecent_q_dense | -0.1951 | [-0.3243,-0.0769] | 2 | 10 | 0.03857 |
| q_dense->recent_plus_q_dense | +0.1463 | [+0.1224,+0.1765] | 6 | 0 | 0.03125 |
| q_dense->qo_dense | +0.1220 | [+0.0488,+0.2286] | 5 | 0 | 0.0625 |
| q_dense->oracle | +0.1220 | [+0.0000,+0.2353] | 9 | 4 | 0.2668 |
| q_dense->options_only_memory | -0.1463 | [-0.3256,+0.0263] | 6 | 12 | 0.2379 |
| recent->nonrecent_q_dense | -0.4146 | [-0.5455,-0.2222] | 3 | 20 | 0.0004883 |
| recent->recent_plus_q_dense | -0.0732 | [-0.1957,+0.0571] | 1 | 4 | 0.375 |

Key readings: q_dense→recent +14.6pp (seed0, CI [+5.7,+21.4], p=.109) and +22.0pp (seed1, CI [+10.8,+32.0], p=.0117); q_dense→nonrecent −22.0pp (seed0, p=.0117) and −19.5pp (seed1, p=.0386); recent→nonrecent −36.6pp / −41.5pp (p=.0015 / .0005). q_dense→q_graph is +2.4pp seed0 (CI [0,+8.1], 2/1, p=1) and +7.3pp seed1 (CI [+2.3,+13.9], 4/1, p=.375) — directionally positive but not robust across seeds and not supported by retrieval R@10/All@20. All results are exploratory: 6 novels, 41 questions.

## 5. Failure taxonomy and case table

**LLM case judge (41 items, 0 errors).** Item labels:

| label | count |
|---|---:|
| reveal_before_cutoff | 28 |
| gold_ambiguous_or_wrong | 23 |
| entity_alias_error | 22 |
| gold_incomplete | 12 |
| option_bias_or_memory | 9 |
| correct_by_shortcut | 6 |
| partial_evidence | 6 |
| correct_supported | 5 |
| reader_reasoning_error | 4 |

Explicit reveal before cutoff: **30/41**; gold set judged insufficient: **16/41**.

Condition-level judge labels (correct_supported / correct_by_shortcut / retrieval_miss / reader_reasoning_error / partial_evidence):

| condition | correct_supported | correct_by_shortcut | retrieval_miss | reader_reasoning_error | partial_evidence |
|---|---:|---:|---:|---:|---:|
| q_dense | 19 | 7 | 6 | 4 | 5 |
| q_graph | 19 | 8 | 5 | 4 | 5 |
| q_rrf | 19 | 7 | 6 | 5 | 4 |
| recent | 24 | 4 | 1 | 4 | 4 |
| oracle | 24 | 2 | 0 | 4 | 6 |

The judge marks 40/41 memory-only conditions as `option_bias_or_memory`; memorization is not a marginal caveat. It also flags 22/41 item-level `entity_alias_error`.

**Corrected case labels (34 unique questions across all case queues).** Full row-by-row table: `round2_case_table.md`; machine-readable: `round2_case_table.jsonl`.

| qid | bucket | dense/graph/rrf/recent/nonrecent/recent+dense/oracle/mem | corrected label | implication |
|---|---|---|---|---|
| human_anno:27:00 | early_lt_70 | Y/Y/Y/Y/Y/Y/N/Y | gold_incomplete_reveal | oracle wrong, all else correct incl. memory; gold omits caviar/wealth evidence |
| human_anno:27:06 | middle_70_90 | Y/Y/Y/Y/Y/Y/Y/Y | nondiscriminative_alias | all methods incl. memory correct; alias-heavy gold, reveal |
| human_anno:117:01 | late_ge_90 | Y/Y/Y/Y/N/Y/Y/N | reveal_gold_incomplete | late acrophobia reveal; gold clue set incomplete/ambiguous |
| human_anno:117:149:01 | late_ge_90 | Y/Y/Y/Y/N/Y/Y/N | gold_ambiguous_reveal | A vs B both defensible; late reveal; gold questionable |
| human_anno:118:06 | late_ge_90 | N/Y/N/Y/N/Y/N/N | gold_incomplete_locality | recent/graph correct; dense/rrf/oracle wrong; gold incomplete |
| human_anno:124:05 | late_ge_90 | N/N/N/N/Y/N/N/N | gold_ambiguous_or_wrong | all conditions fail; final evidence implicates Mrs. Brand/Martin; gold likely wrong |
| human_anno:27:04 | middle_70_90 | N/N/N/Y/N/Y/Y/Y | locality_reveal_memory | recent/recent_plus/oracle correct; retrieval misses; memory correct |
| human_anno:79:04 | late_ge_90 | N/N/N/Y/N/Y/Y/N | locality_reveal | recent/recent_plus/oracle correct; retrieval misses; reveal |
| human_anno:124:00 | early_lt_70 | N/N/N/Y/N/Y/Y/Y | reader_error_all_gold | dense has all gold but wrong; oracle/recent correct; reader bottleneck |
| human_anno:118:01 | early_lt_70 | Y/N/N/Y/N/Y/Y/N | dense_graph_disagree | dense/recent/oracle correct, graph/rrf wrong; reader variability |
| human_anno:121:00 | early_lt_70 | N/Y/Y/Y/N/Y/Y/Y | dense_graph_disagree_reveal | graph/rrf/recent/oracle correct, dense wrong; memory also correct |
| human_anno:79:02 | middle_70_90 | Y/Y/Y/N/Y/N/Y/Y | recency_misleads | dense/graph/oracle correct, recent wrong; recency can displace correct evidence |

The full 34-row table adds the remaining recent-only / memory-correct / oracle-wrong cases. The dominant pattern is `recent` or `recent_plus` correct while dense/graph miss; a second pattern is all methods correct including memory (non-discriminative); a third is gold ambiguous/wrong (124:05, 79:03, 27:01, 79:01). The only dense-vs-graph disagreement cases (118:01, 118:06, 121:00) all have `recent` correct too, so they are not graph-specific wins.

## 6. Read-time v3, manual audit, relation channel and relation QA

**Funnel:** 120 packets → 5,114 joint facts → 1,435 comprehension candidates → 1,122 rule-validated → **601 verifier-kept**. Rule rejects: too_long 110, needs_two_chunks 180, quote_not_found 21, generic_predicate 1, duplicate_fact 1. Errors: 0.

**Manual audit (60; 30 kept / 20 verifier-dropped / 10 rule-rejected; all 6 novels).**

| label | count | share |
|---|---:|---:|
| correct_cross_chunk | 26 | 43% |
| correct_but_fact_rewrite | 19 | 32% |
| insufficient_evidence | 10 | 17% |
| entity_alignment_error | 3 | 5% |
| too_generic | 2 | 3% |

- Kept genuine cross-chunk: **20/30 (66.7%)**; including fact rewrites as 'correct': 24/30 (80.0%).
- Verifier false rejects: 3/20 were genuine cross-chunk (a21-like alias/status cases are still an issue).
- Rule false rejects: 3/10 genuine cross-chunk were rejected, mostly by the `too_long` rule (a54/a55/a59).

**Audit precision by relation type:** alias_or_identity 11/17 (65%); alibi_or_opportunity 3/3; contradiction_or_deception 3/9; method_or_capability 1/4; motive_or_interest 2/6; state_change 0/3; timeline_or_causal 6/18. The types most useful for graph injection (timeline, motive, contradiction) are mostly fact rewrites or insufficient; alias relations are precise but mostly entity canonicalization, not new cross-chunk understanding.

**Graph-injection gate:** quantity 601 ≥ 100 passes; manual kept precision 66.7% < 75% fails. Even the 80% ccc+cbfr count would pass only by counting fact rewrites, which defeats the comprehension objective. **R2–R5 variants were not built.**

**Relation channel (kept relations):** mean visible relations/question 59.95; gold-touching top relation 17/41 (41.5%, passes the 25% coverage gate). But equal-weight RRF displaces good chunks: relation_chunks alone R@10 .137; RRF (dense+graph+relation) R@10 .312 / All@20 .195 / MRR .352 vs q_dense .418/.220/.544. Validated (include-dropped) channel is similar (R@10 .340, All@20 .195, MRR .387). 

**Relation-note QA (Stage D, 4 conditions × 41):** q_dense .658, relations_q_dense .658; q_rrf .634, relations_q_rrf .658. Paired q_dense→relations_q_dense delta 0.0 (CI [-7.0,+7.3]pp, 2/2, p=1); q_rrf→relations_q_rrf +2.4pp (CI [-5.9,+10.5], 2/1, p=1). Relation notes add no measurable QA gain.

## 7. Graph variants

**Not run.** The pre-registered gate requires ≥100 kept relations (passes: 601) and manual kept precision ≥75% (fails: 20/30 = 66.7% genuine cross-chunk). The relation channel also lowers retrieval R@10 and relation notes do not improve QA. Building R2–R5 would test a representation that the audit shows is mostly alias variants and fact rewrites; it would not answer a live question. This is a gate failure, not a missing run.

## 8. Cost table

| stage | calls | prompt tok | completion tok | total tok | note |
|---|---:|---:|---:|---:|---|
| Read-time v3 (120 packets) | 350 | 1,344,190 | 591,686 | 1,935,876 | six parallel per-novel processes |
| Reader Stage B (10 cond × 2 seeds × 41) | 534 | 1,844,218 | 51,413 | 1,895,631 | 287 cache hits |
| Relation QA Stage D (4 cond × 41) | 96 | 401,409 | 9,725 | 411,134 | 68 cache hits |
| Case judge (41 questions) | 41 | 226,055 | 12,891 | 238,946 | post-hoc only |
| **Total** | **1,021** | **3,815,872** | **665,715** | **4,481,587** | |

Stage A retrieval, prefix-safety checks, relation channel and graph inspection use no LLM calls (NV-Embed only). R2–R5 were not built, so no additional embedding/indexing cost.

## 9. Next decision

**Do not scale to 23 novels. Change the benchmark split/question construction, and stop the graph direction on the current split.**

Evidence for the decision:

1. The graph stop/go gate fails (R@10 +0.7pp, CI crosses 0; All@20 +2.4pp, CI lower bound 0; MRR −4.8pp, p=.012). RRF and fact-top-k/synonym variants do not rescue it.
2. Recency explains the benchmark: recent .805 on both seeds; nonrecent collapses to .439/.390; the effect holds in early, middle and late buckets. A graph that reads the story differently cannot beat a 4096-token recency window on this split.
3. Gold labels are unreliable: 16/41 insufficient, 23/41 ambiguous/questionable, 30/41 explicit reveal; oracle is below or equal to recent on both seeds.
4. Comprehension v3 is higher-yield but not precise enough (kept ccc 66.7%), and the relation channel/notes do not improve retrieval or QA.
5. Memory-only is .463/.439; 40/41 memory conditions are judged option-bias/memory. Any claim must be reported over this baseline.

Recommended next steps, in order:

1. **Build an anti-reveal evaluation subset:** keep only questions where the decisive answer evidence is not in the last 4096 tokens, and re-audit gold clues/answers (especially 124:05, 79:03, 27:01, 79:01). Re-run Stage A/B there before any graph claim.
2. **Make `recent` and `options_only_memory` mandatory baselines** in every retrieval/reader table; report gains over both.
3. **Fix the relation layer before injecting:** filter alias-only/fact-rewrite relations, use `fixed_status`, relax `too_long`, and target timeline/motive/contradiction relations with explicit cross-chunk necessity. Re-audit 60 relations; require kept ccc ≥75%.
4. **Only then test R3 on the anti-reveal subset** with paired novel-clustered CIs; require R@10 or All@20 positive lower bound and reader gain outside late cutoffs.
5. Do not start the Question-Time loop until the anti-reveal split and relation gate pass.

## Appendix A. Case table

# Round 2 case table (34 unique case-queue questions)

| qid | novel | bucket | queues | correct dense/graph/rrf/recent/nonrecent/recent+dense/oracle/mem | gold any/all: dense / recent / oracle | LLM labels | corrected label | implication |
|---|---|---|---|---|---|---|---|---|
| human_anno:117:00 | 117 | early_lt_70 | memory_correct | Y/Y/Y/Y/N/Y/Y/Y | 1.0/0.0/1.0/0.0/1.0/1.0 | reveal_before_cutoff, correct_supported | nondiscriminative | all methods incl. memory correct; reveal/memory, no graph signal |
| human_anno:117:01 | 117 | late_ge_90 | recent_correct_without_gold | Y/Y/Y/Y/N/Y/Y/N | 0.0/0.0/0.0/0.0/1.0/1.0 | reveal_before_cutoff, gold_incomplete, gold_ambiguous_or_wrong, entity_alias_error | reveal_gold_incomplete | late acrophobia reveal; gold clue set incomplete/ambiguous |
| human_anno:117:02 | 117 | middle_70_90 | recent_correct_without_gold,recent_only | N/N/N/Y/N/N/Y/N | 1.0/0.0/0.0/0.0/1.0/1.0 | reveal_before_cutoff, entity_alias_error, gold_ambiguous_or_wrong | locality_reveal | recent only (plus oracle); retrieval misses; answer near cutoff |
| human_anno:117:03 | 117 | late_ge_90 | recent_correct_without_gold,memory_correct | Y/Y/Y/Y/Y/Y/Y/Y | 1.0/0.0/0.0/0.0/1.0/1.0 | reveal_before_cutoff, correct_supported | nondiscriminative | all methods incl. memory correct; no graph signal |
| human_anno:117:04 | 117 | late_ge_90 | oracle_wrong,memory_correct | Y/Y/Y/Y/Y/Y/N/Y | 1.0/0.0/1.0/0.0/1.0/0.0 | reveal_before_cutoff, entity_alias_error | gold_incomplete_alias | oracle wrong; reveal + alias variants; gold insufficient |
| human_anno:117:149:00 | 117 | early_lt_70 | memory_correct | Y/Y/Y/Y/Y/Y/Y/Y | 1.0/0.0/1.0/0.0/1.0/1.0 | reveal_before_cutoff, correct_supported | nondiscriminative | all methods incl. memory correct; no graph signal |
| human_anno:117:149:01 | 117 | late_ge_90 | recent_correct_without_gold | Y/Y/Y/Y/N/Y/Y/N | 0.0/0.0/0.0/0.0/1.0/1.0 | gold_ambiguous_or_wrong, reveal_before_cutoff, correct_by_shortcut | gold_ambiguous_reveal | A vs B both defensible; late reveal; gold questionable |
| human_anno:117:149:02 | 117 | middle_70_90 | recent_correct_without_gold,recent_only | N/N/N/Y/N/N/Y/N | 1.0/0.0/0.0/0.0/1.0/1.0 | gold_ambiguous_or_wrong, entity_alias_error, option_bias_or_memory | gold_ambiguous | oracle/recent correct, dense/graph wrong; gold and entity alias unclear |
| human_anno:117:149:03 | 117 | late_ge_90 | recent_correct_without_gold | Y/Y/Y/Y/Y/Y/Y/N | 1.0/0.0/0.0/0.0/1.0/1.0 | reveal_before_cutoff, gold_incomplete | gold_incomplete_reveal | all retrieval/oracle correct; gold incomplete; reveal |
| human_anno:117:149:04 | 117 | late_ge_90 | oracle_wrong | Y/Y/Y/Y/Y/Y/N/N | 1.0/0.0/1.0/0.0/1.0/0.0 | reveal_before_cutoff, entity_alias_error, gold_ambiguous_or_wrong | gold_incomplete_alias | oracle wrong; reveal + alias; gold insufficient |
| human_anno:118:00 | 118 | early_lt_70 | memory_correct | Y/Y/Y/Y/Y/Y/Y/Y | 1.0/0.0/1.0/0.0/1.0/1.0 | gold_ambiguous_or_wrong, correct_by_shortcut, entity_alias_error | nondiscriminative_memory | all methods incl. memory correct; gold ambiguous |
| human_anno:118:01 | 118 | early_lt_70 | dense_graph_disagree | Y/N/N/Y/N/Y/Y/N | 1.0/1.0/1.0/1.0/1.0/1.0 | gold_ambiguous_or_wrong, partial_evidence, reader_reasoning_error | dense_graph_disagree | dense/recent/oracle correct, graph/rrf wrong; reader variability |
| human_anno:118:02 | 118 | early_lt_70 | memory_correct | Y/Y/Y/Y/Y/Y/Y/Y | 1.0/0.0/1.0/0.0/1.0/1.0 | reveal_before_cutoff, correct_supported | nondiscriminative | all methods incl. memory correct; reveal |
| human_anno:118:04 | 118 | late_ge_90 | recent_correct_without_gold | Y/Y/Y/Y/N/Y/Y/N | 0.0/0.0/0.0/0.0/1.0/1.0 | reveal_before_cutoff, gold_incomplete, option_bias_or_memory | gold_incomplete_reveal | all retrieval correct, memory wrong; gold incomplete |
| human_anno:118:05 | 118 | late_ge_90 | memory_correct | Y/Y/Y/Y/N/Y/Y/Y | 1.0/0.0/1.0/0.0/1.0/1.0 | reveal_before_cutoff, gold_ambiguous_or_wrong, entity_alias_error | nondiscriminative_memory | all methods incl. memory correct; gold ambiguous |
| human_anno:118:06 | 118 | late_ge_90 | recent_correct_without_gold,oracle_wrong,recent_only,dense_graph_disagree | N/Y/N/Y/N/Y/N/N | 1.0/0.0/0.0/0.0/1.0/1.0 | reveal_before_cutoff, gold_incomplete, entity_alias_error | gold_incomplete_locality | recent/graph correct; dense/rrf/oracle wrong; gold incomplete |
| human_anno:121:00 | 121 | early_lt_70 | memory_correct,recent_only,dense_graph_disagree | N/Y/Y/Y/N/Y/Y/Y | 1.0/0.0/1.0/0.0/1.0/1.0 | reveal_before_cutoff, partial_evidence | dense_graph_disagree_reveal | graph/rrf/recent/oracle correct, dense wrong; memory also correct |
| human_anno:121:01 | 121 | early_lt_70 | memory_correct | Y/Y/Y/Y/Y/Y/Y/Y | 1.0/0.0/1.0/0.0/1.0/1.0 | reveal_before_cutoff, correct_supported | nondiscriminative | all methods incl. memory correct; reveal |
| human_anno:121:02 | 121 | late_ge_90 | memory_correct | Y/Y/N/Y/N/Y/Y/Y | 1.0/0.0/1.0/0.0/1.0/1.0 | reveal_before_cutoff, gold_incomplete | gold_incomplete_memory | most correct incl. memory; gold incomplete |
| human_anno:121:05 | 121 | late_ge_90 | recent_correct_without_gold,memory_correct | Y/Y/Y/Y/Y/Y/Y/Y | 0.0/0.0/0.0/0.0/1.0/1.0 | gold_incomplete, gold_ambiguous_or_wrong, correct_by_shortcut, option_bias_or_memory, entity_alias_error | memory_gold_ambiguous | all correct incl. memory; gold ambiguous; location elsewhere |
| human_anno:124:00 | 124 | early_lt_70 | memory_correct,recent_only,dense_wrong_all_gold | N/N/N/Y/N/Y/Y/Y | 1.0/1.0/1.0/0.0/1.0/1.0 | gold_ambiguous_or_wrong, entity_alias_error, partial_evidence | reader_error_all_gold | dense has all gold but wrong; oracle/recent correct; reader bottleneck |
| human_anno:124:02 | 124 | early_lt_70 | memory_correct | Y/Y/Y/Y/Y/Y/Y/Y | 1.0/1.0/1.0/1.0/1.0/1.0 | reveal_before_cutoff, gold_ambiguous_or_wrong | nondiscriminative_memory | all correct incl. memory; gold ambiguous |
| human_anno:124:05 | 124 | late_ge_90 | oracle_wrong | N/N/N/N/Y/N/N/N | 1.0/0.0/1.0/0.0/1.0/1.0 | gold_ambiguous_or_wrong, entity_alias_error, gold_incomplete | gold_ambiguous_or_wrong | all conditions fail; final evidence implicates Mrs. Brand/Martin; gold likely wrong |
| human_anno:27:00 | 27 | early_lt_70 | recent_correct_without_gold,oracle_wrong,memory_correct | Y/Y/Y/Y/Y/Y/N/Y | 1.0/0.0/0.0/0.0/1.0/1.0 | reveal_before_cutoff, gold_incomplete, gold_ambiguous_or_wrong, correct_by_shortcut, reader_reasoning_error | gold_incomplete_reveal | oracle wrong, all else correct incl. memory; gold omits caviar/wealth evidence |
| human_anno:27:01 | 27 | early_lt_70 | oracle_wrong | N/N/N/N/N/Y/N/N | 1.0/0.0/0.0/0.0/1.0/1.0 | gold_ambiguous_or_wrong, reader_reasoning_error, option_bias_or_memory | all_fail_gold_ambiguous | only recent_plus correct; gold ambiguous; reader/option noise |
| human_anno:27:03 | 27 | middle_70_90 | recent_correct_without_gold | Y/Y/Y/Y/Y/Y/Y/N | 1.0/0.0/0.0/0.0/1.0/1.0 | reveal_before_cutoff, gold_ambiguous_or_wrong | nondiscriminative | all retrieval/oracle correct, memory wrong; gold ambiguous |
| human_anno:27:04 | 27 | middle_70_90 | recent_correct_without_gold,memory_correct,recent_only | N/N/N/Y/N/Y/Y/Y | 0.0/0.0/0.0/0.0/1.0/1.0 | reveal_before_cutoff, entity_alias_error | locality_reveal_memory | recent/recent_plus/oracle correct; retrieval misses; memory correct |
| human_anno:27:05 | 27 | early_lt_70 | recent_only | N/N/N/Y/N/Y/Y/N | 1.0/0.0/1.0/0.0/1.0/1.0 | reveal_before_cutoff, gold_ambiguous_or_wrong, entity_alias_error | locality_reveal | recent/recent_plus/oracle correct; retrieval misses; reveal |
| human_anno:27:06 | 27 | middle_70_90 | recent_correct_without_gold,memory_correct | Y/Y/Y/Y/Y/Y/Y/Y | 0.0/0.0/0.0/0.0/1.0/0.0 | reveal_before_cutoff, entity_alias_error, correct_by_shortcut, gold_incomplete | nondiscriminative_alias | all methods incl. memory correct; alias-heavy gold, reveal |
| human_anno:79:00 | 79 | early_lt_70 | recent_correct_without_gold,memory_correct | Y/Y/Y/Y/Y/Y/Y/Y | 0.0/0.0/0.0/0.0/1.0/1.0 | reveal_before_cutoff, gold_incomplete, entity_alias_error, correct_by_shortcut | nondiscriminative | all methods incl. memory correct; gold incomplete |
| human_anno:79:01 | 79 | early_lt_70 | oracle_wrong | N/N/N/N/N/N/N/N | 1.0/0.0/1.0/0.0/1.0/1.0 | gold_ambiguous_or_wrong, gold_incomplete, entity_alias_error, option_bias_or_memory | all_fail_gold_ambiguous | all conditions fail; gold ambiguous/incomplete |
| human_anno:79:02 | 79 | middle_70_90 | memory_correct | Y/Y/Y/N/Y/N/Y/Y | 1.0/0.0/0.0/0.0/1.0/1.0 | gold_ambiguous_or_wrong, entity_alias_error, option_bias_or_memory, partial_evidence | recency_misleads | dense/graph/oracle correct, recent wrong; recency can displace correct evidence |
| human_anno:79:03 | 79 | late_ge_90 | oracle_wrong,memory_correct | N/N/N/N/N/N/N/Y | 0.0/0.0/0.0/0.0/1.0/1.0 | gold_ambiguous_or_wrong, entity_alias_error, option_bias_or_memory | memory_only_gold_ambiguous | only memory correct; gold ambiguous; not evidence-based |
| human_anno:79:04 | 79 | late_ge_90 | recent_correct_without_gold,recent_only | N/N/N/Y/N/Y/Y/N | 0.0/0.0/0.0/0.0/1.0/1.0 | reveal_before_cutoff, entity_alias_error, gold_ambiguous_or_wrong | locality_reveal | recent/recent_plus/oracle correct; retrieval misses; reveal |


## Appendix B. Stage A rank-disagreement cases

# Stage A rank-disagreement cases

| queue | qid | bucket | n_gold | dense rank | graph rank | dense gold ranks | graph gold ranks | facts (top 3) | fact prov touches gold |
|---|---|---|---:|---:|---:|---|---|---|---|
| graph_better | human_anno:27:06 | middle_70_90 | 15 | 15 | 13 | 0,27,0,0,0,0,15,0 (hits 4/15) | 0,37,0,0,0,0,13,0 (hits 5/15) | ellery / referred to / the killer; crosett / realized he had killed the wrong man / true; ellery / identified killer as / vilga crocetta | 0/5 |
| graph_better | human_anno:79:01 | early_lt_70 | 3 | 4 | 1 | 8,0,4 (hits 2/3) | 1,0,6 (hits 2/3) | mukayama seisakusho / had chairman / mukayama tadashi; mukayama tadashi / was / second generation president and chairman of mukayama seisakusho; mukizuki / thought asking / inaba would help understand where the head is | 2/5 |
| graph_better | human_anno:117:149:01 | late_ge_90 | 4 | 0 | 5 | 0,0,0,0 (hits 0/4) | 5,0,0,0 (hits 1/4) | christine / has / alibi; christine / lied about / acrophobia; christine / laughed at / poirot s statement | 0/5 |
| graph_better | human_anno:118:04 | late_ge_90 | 4 | 14 | 1 | 28,0,14,0 (hits 2/4) | 26,0,1,0 (hits 2/4) | david / devised plan to divert suspicion from / david; david hunter / gave account of / his whereabouts; david / had motive to kill / jard | 1/5 |
| dense_better | human_anno:27:00 | early_lt_70 | 4 | 2 | 4 | 2,20,3,12 (hits 4/4) | 4,19,5,7 (hits 4/4) | andrew / is not married to / tom; andrew van / is / well educated man; andrew van / was / victim | 1/5 |
| dense_better | human_anno:79:02 | middle_70_90 | 4 | 1 | 3 | 3,18,41,1 (hits 4/4) | 7,18,37,3 (hits 4/4) | yuki / has feelings for / kyousuke; xiang yue / is also called / mr xiangyue; jinma / loves / yuyan | 1/5 |
| dense_better | human_anno:79:03 | late_ge_90 | 4 | 11 | 12 | 11,0,41,0 (hits 2/4) | 12,0,46,0 (hits 2/4) | kanai san / would not be suspected in / arahama case; kumagai san / was not suspected as killer by / shima; mcarthur / was listed as / suspect | 0/5 |
| dense_better | human_anno:79:04 | late_ge_90 | 5 | 12 | 15 | 0,12,0,13,0 (hits 2/5) | 0,15,0,19,0 (hits 2/5) | daisuke inagaki / is / perpetrator of this serial murder case; mutsu / concluded / daisuke inagaki is the perpetrator of this serial murder case; detective jinguji / suggested / killer is fanatical logic enthusiast | 0/5 |
| dense_better | human_anno:117:149:00 | early_lt_70 | 7 | 1 | 2 | 41,7,4,19,33,25,1 (hits 7/7) | 40,12,3,14,32,23,2 (hits 7/7) | eleanor marshall / asked poirot not to tell anyone / where eleanor marshall is; linda / thinks she can have fun if / alicia is not there; christine / wants to leave / here | 2/5 |
| dense_better | human_anno:117:02 | middle_70_90 | 6 | 6 | 9 | 6,13,0,0,0,0 (hits 2/6) | 9,10,0,0,0,0 (hits 2/6) | linda / said she could not think of anyone who would want to kill / elena; elena / is described as / strong woman; elena / was / dead woman | 1/5 |
| dense_better | human_anno:117:149:02 | middle_70_90 | 6 | 6 | 9 | 6,13,0,0,0,0 (hits 2/6) | 9,10,0,0,0,0 (hits 2/6) | linda / said she could not think of anyone who would want to kill / elena; elena / is described as / strong woman; elena / was / dead woman | 1/5 |
| dense_better | human_anno:118:00 | early_lt_70 | 4 | 6 | 8 | 7,6,13,11 (hits 4/4) | 10,9,14,8 (hits 4/4) | david / went to see / rolly; rolly / asked / who enock yadon is; enoch yate / asked david to visit / stagg | 0/5 |
| dense_better | human_anno:118:06 | late_ge_90 | 9 | 9 | 14 | 0,0,29,41,47,0,0,9 (hits 4/9) | 0,0,36,47,0,0,0,14 (hits 3/9) | inspector / said the man was murdered at / stagg inn; david hunter / was accused of / premeditated murder; david hunter / may have killed / andré | 0/5 |
| dense_better | human_anno:118:05 | late_ge_90 | 2 | 7 | 9 | 34,7 (hits 2/2) | 33,9 (hits 2/2) | rosaline / is also referred to as / rosalind; rosaline / is also known as / mrs claude; rosaline / is now / twenty six | 0/5 |
| dense_better | human_anno:121:00 | early_lt_70 | 6 | 1 | 2 | 1,4,12,11,6,33 (hits 6/6) | 2,8,17,18,12,34 (hits 6/6) | mrs habberd / interjected / stealing is stealing; mrs haberdash / took out / small notebook; miss lemon / confirmed / things had been stolen | 1/5 |
| dense_better | human_anno:121:03 | late_ge_90 | 8 | 1 | 2 | 11,43,0,0,0,1,3,5 (hits 5/8) | 2,0,0,0,0,5,8,12 (hits 4/8) | patricia / was involved in the discussion about / murder; patricia / discussed / methods of murder; patricia lane / might be resentful enough to kill / sylvia austin | 1/5 |
| dense_better | human_anno:121:02 | late_ge_90 | 6 | 8 | 9 | 0,8,0,0,0,0 (hits 1/6) | 0,9,0,0,0,0 (hits 1/6) | celia / was killed by / unknown person; valerie / asked about who killed / celia; celia / was killed / true | 0/5 |
| dense_better | human_anno:121:05 | late_ge_90 | 3 | 21 | 23 | 21,43,35 (hits 3/3) | 23,41,32 (hits 3/3) | mrs nikoleitis / is referred to as / mrs nicolaitis; mrs nicolaitis / claimed it was / suicide; mrs nicoletti / is also known as / mrs nik | 0/5 |
| dense_better | human_anno:124:00 | early_lt_70 | 3 | 1 | 3 | 5,4,1 (hits 3/3) | 5,3,6 (hits 3/3) | hakaste / reported missing / leather travel clock; miss pemshaw / said small travel clock was not in room when / miss pemshaw left at half past one; detective hakast / placed small travel clock in hand of / miss pemshaw | 1/5 |
| dense_better | human_anno:124:02 | early_lt_70 | 2 | 1 | 2 | 4,1 (hits 2/2) | 2,6 (hits 2/2) | mrs rotton / said miss webber is / mrs rotton s sister s child; mrs rotten / was / aunt; mrs routon / confirmed / hila webber is her sister s illegitimate child | 2/5 |


## Appendix C. Reproduction

All commands and errors are in `outputs/detectiveqa/round2/RUN_LOG.md`. Key artifacts: `round2_retrieval.jsonl`, `round2_reader.jsonl`, `readtime_v3.jsonl`, `readtime_v3_manual_audit.jsonl`, `relation_channel_kept.jsonl`, `round2_reader_relations.jsonl`, `round2_case_judge.jsonl`, `round2_case_table.md`.

