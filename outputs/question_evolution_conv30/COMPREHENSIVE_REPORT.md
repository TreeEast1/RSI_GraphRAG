# LoCoMo conv-30 Question-Time Self-Evolving Answer & Retrieval Skills — 完整实验报告

**日期**: 2026-11-10  
**项目**: RSI GraphRAG / CogitoRAG  
**底座**: HippoRAG2 + HY4 (`no_think`) + NV-Embed-v2  
**固定检索**: COMP-P4_top5（关闭 LLM fact reranker），4096-token Reader context  
**代码**: `qecode/`  
**所有原始产物**: `artifacts/`, `skill_snapshots/`, `logs/`

## 目录

1. [执行摘要与核心结论](#part-1)
2. [固定数据划分](#part-2)
3. [Baseline：P0 / P7 / P8](#part-3)
4. [Answer Skill 自进化](#part-4)
5. [Retrieval Skill 自进化](#part-5)
6. [E0–E6 联合实验与等预算静态对照](#part-6)
7. [Question-Time ReAct 与 Grounded Verifier](#part-7)
8. [错误分析](#part-8)
9. [成本与延迟](#part-9)
10. [重复运行稳定性](#part-10)
11. [运行清单与交付物索引](#part-11)


---

# Part 1 — 执行摘要与核心结论

# FINAL REPORT — LoCoMo conv-30 Question-Time Self-Evolving Answer & Retrieval Skills

Date: 2026-11-10.  Project: RSI GraphRAG.  Base graph/retrieval: frozen **COMP-P4_top5** HippoRAG2 (no LLM fact reranker), 4096-token context.  LLM: HY4 (`no_think`), embedding: NV-Embed-v2.

## 1. Executive summary

A closed loop was implemented and executed: adaptation feedback → error reflection → candidate Answer/Retrieval Skill → replay/adversarial verification → versioned Skill Memory → snapshot selection on development → frozen evaluation. The evaluation split (43 questions) was never used for skill selection.

**Headline:** the loop is mechanically functional, but on the 43-question evaluation split no skill layer beats the fixed P7 Reader.  The evolved Answer Skills beat the hand-written static rules and the long static prompt, but still lose to P7; the retrieval layer degrades evidence recall.  This is a negative generalisation result, not a positive one.

- Answer Skill evolution: 3 skills added, 0 updated, 8 candidates rejected, 3 active at the selected snapshot.
- Retrieval Skill evolution: 1 learned skill; development snapshot selection deployed 0 (the learned skill lowered dev evidence coverage). Snapshots exist at 0/10/20/30/43 feedback questions.
- Evaluation matrix E0–E6 and a ReAct/verifier loop were run on the same frozen retrieval records.
- Total LLM call records: 2582 (cache hits 1023; endpoint attempts 2186, success rate 0.713).

## 2. Main results

### 2.1 E0–E6 matrix (evaluation split, n=43)

| Exp | Answer layer | Retrieval layer | EM | F1 | official F1 | adv refusal | Ev@4096 | multi-ev | calls |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | P7 | base | 0.303 | 0.563 | 0.606 | 0.900 | 0.909 | 0.714 | 1.00 |
| E1 | static answer | base | 0.273 | 0.423 | 0.436 | 0.900 | 0.909 | 0.714 | 1.00 |
| E2 | evolved answer | base | 0.273 | 0.508 | 0.574 | 0.900 | 0.909 | 0.714 | 1.00 |
| E3 | P7 | static retrieval | 0.303 | 0.545 | 0.576 | 0.900 | 0.818 | 0.429 | 3.00 |
| E4 | P7 | evolved retrieval | 0.273 | 0.546 | 0.602 | 0.900 | 0.848 | 0.429 | 3.00 |
| E5 | evolved answer | static retrieval | 0.303 | 0.537 | 0.556 | 0.900 | 0.818 | 0.429 | 3.00 |
| E6 | evolved answer | evolved retrieval | 0.273 | 0.506 | 0.589 | 0.900 | 0.848 | 0.429 | 3.00 |
| EB | long static prompt | base | 0.303 | 0.488 | 0.523 | 0.900 | 0.909 | 0.714 | 1.00 |

Note: E4/E6 deploy 0 learned retrieval skills because development snapshot selection rejected the one learned skill; they therefore measure QDM + sufficiency without a learned skill.  E3/E5 use the hand-written static retrieval library.

### 2.2 ReAct / verifier (evaluation)

| Variant | EM | F1 | official F1 | adv refusal | answer support | revise rate | calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | 0.273 | 0.562 | 0.631 | 0.900 | 0.999 | 0.000 | 1.00 |
| R1 | 0.273 | 0.549 | 0.617 | 0.900 | 0.999 | 0.395 | 2.14 |
| R2 | 0.303 | 0.520 | 0.544 | 0.900 | 0.999 | 0.535 | 4.26 |
| R3 | 0.273 | 0.484 | 0.553 | 0.900 | 0.999 | 0.000 | 1.00 |
| R4 | 0.273 | 0.490 | 0.558 | 0.900 | 0.999 | 0.326 | 2.07 |
| R5 | 0.242 | 0.425 | 0.445 | 0.900 | 1.000 | 0.442 | 4.12 |

## 3. Answers to the required questions

### Q1. Can Answer Skills self-evolve without changing model weights?

Yes, mechanically: the loop generated and versioned 6 snapshots and accepted 3 skill writes.  The selected memory contains 3 active skills.  No weights were modified; the full policy is prompt text in `skill_memory.jsonl`.

### Q2. Does it beat static P7?

E2 − E0: EM -0.030, F1 -0.056, official F1 -0.032, adversarial refusal +0.000.  E2 vs the long static prompt (EB): F1 +0.020, official F1 +0.052.  On 43 evaluation questions a change of one question is about 2.3 points of EM; the repeated-run spread in Section 4 bounds the noise.  **E2 therefore does not beat static P7 on this evaluation split.**

### Q3. Did it fix P7's single-hop / multi-hop regression?

| category | E0 F1 | E2 F1 | delta | E0 official F1 | E2 official F1 |
|---|---:|---:|---:|---:|---:|
| multi-hop | 0.567 | 0.382 | -0.186 | 0.543 | 0.543 |
| temporal | 0.940 | 0.680 | -0.260 | 0.940 | 0.680 |
| single-hop | 0.353 | 0.447 | +0.094 | 0.437 | 0.524 |

**Interpretation:** E2 improves single-hop F1 (+0.094) and official F1 (+0.087) but regresses temporal (-0.260 F1) and multi-hop (-0.186 F1).  The learned temporal-granularity rule over-applies to temporal questions, and the single-hop phrase rules do not transfer cleanly to multi-hop.  It therefore does not fix both P7 regressions; it trades temporal/multi-hop accuracy for single-hop accuracy.

Read this together with the fixed 43-question size: multi-hop has only five evaluation questions, so one flip is ±0.2 EM in that stratum.

### Q4. Can Retrieval Skills improve multi-evidence recall?

E4 − E0: R@5 -0.045, evidence full@4096 -0.061, multi-evidence complete -0.286.  E4 is evaluated with QDM + sufficiency + at most one extra retrieval round.  The loop learned 1 retrieval skill, but development snapshot selection chose step 0 (0 deployed skills), so E4 is QDM without a learned retrieval skill.  **It does not improve multi-evidence recall; the QDM dense merge is the main cause of the drop.**

### Q5. Does Skill Memory accumulate, update and reuse?

Snapshots were written at 0/10/20/30/43 feedback questions.  Answer memory: 3 active skills, 0 merges, and reuse counts are listed in `answer_skill_evolution.md`.  Retrieval memory: 1 learned skill with 0 merges, but the development-selected snapshot deploys 0 skills.  The number of skills a later question actually selected is the operational test of reuse, not the number of skills stored.

### Q6. Is ReAct + verification worth the extra cost?

R1 − R0: F1 -0.013 at +1.14 extra calls/question.  R5 − R3: F1 -0.060 at +3.12 extra calls.  (R5 was run with the raw learned retrieval skill before development snapshot selection rejected it; it is an upper-bound exploratory variant, not the frozen system.)  The verifier is a grounded check, not a gold-answer oracle; it can suppress hallucinations but cannot create missing evidence.  **On this split the verifier is not worth the extra calls: it revises a third to a half of answers without an F1 gain.**

### Q7. Which skills generalize and which memorize the training questions?

The sanitizer rejects any candidate containing a distinctive gold-answer token, a proper noun from the question, or an explicit year.  Skills that were never selected after their creation are local fixes; skills with reuse_count > 0 are the generalisation candidates.  Per-skill reuse and success are in `answer_skill_evolution.md` and `retrieval_skill_evolution.md`.  On the evaluation split, 28/43 questions retrieved at least one Answer Skill, but only 1 was helpful, 1 was harmful and 26 were neutral; the harmful hit (a temporal answer changed to a relative phrase) is a concrete over-generalisation.  The skill-hit table in `error_analysis.md` lists every case.

### Q8. Can this be an independent component of a full RSI GraphRAG?

Yes as a policy layer: it is independent of graph construction and model weights, it has persistent versioned memory, a strict adaptation/evaluation split, and a verification gate.  Its ceiling is set by the frozen retrieval evidence: with P7 on gold-evidence oracle reaching ~0.60 F1 in the prior study, answer-side skills can only recover the errors whose evidence is already present.

## 4. Repeated-run stability

| configuration | EM A | EM B | F1 A | F1 B | adv refusal A/B | EM-unstable |
|---|---:|---:|---:|---:|---|---:|
| E0 vs R0 | 0.303 | 0.273 | 0.563 | 0.562 | 0.900 / 0.900 | 1 |
| E2 vs R3 | 0.273 | 0.273 | 0.508 | 0.484 | 0.900 / 0.900 | 0 |

Two independent runs were available per configuration (the endpoint quota did not allow three fresh runs).  Per-question flips are in `repeat_stability.md`.

## 5. Cost

- LLM call records 2582; prompt tokens 7431528; completion tokens 98445.
- Endpoint latency P50/P95: 2.05s / 3.79s.
- Adaptation reflection/verification is a one-off training cost; evaluation methods E0–E6 use 1–3 Reader-side calls per question.

## 6. Limitations

- conv-30 has been studied in prior rounds.  The 43-question evaluation split is fixed before learning, but it is an internal exploratory test, not an independent final benchmark.
- The retrieval base is frozen; the study isolates skill effects but does not rebuild the graph.
- Category-5 questions are entity-swapped adversarial mirrors; the official strict keyword protocol and the project adapted-refusal protocol are both reported.
- HY4 at temperature=0 is not deterministic; repeated runs are reported for the final candidates.
- The Answer Skill memory is prompt text.  It improves policy, not knowledge; it cannot recover evidence the retrieval layer never returned.
- The deployed Reader shows only each skill's action text.  A trigger-condition-aware rendering was tested on development and was neutral/slightly worse (official F1 0.683 vs 0.687), so it was not deployed; this is a remaining design choice, not evidence that trigger rendering would fix over-application.
- Small strata (5 multi-hop evaluation questions) make category-level claims directional.

## 7. Artifact index

| file | contents |
|---|---|
| `DATA_SPLIT.md` | fixed stratified split and seed |
| `baseline_results.md` | P0/P7/P8 baselines |
| `answer_skill_evolution.md` | answer-skill loop, snapshots, reuse |
| `retrieval_skill_evolution.md` | retrieval-skill loop, query templates, evidence effects |
| `joint_experiments.md` | E0–E6 + equal-budget static control |
| `react_and_verification.md` | ReAct variants and grounded verifier |
| `error_analysis.md` | fixed/worsened questions, skill hits, overgeneralisation |
| `cost_latency.md` | calls, tokens, latency by stage |
| `repeat_stability.md` | independent repeated runs of the key configurations |
| `qa_predictions.jsonl` | per-question predictions for all main methods |
| `skill_memory.jsonl` / `retrieval_skill_memory.jsonl` | final memories |
| `skill_evolution_trace.jsonl` | every adaptation/verification decision |
| `skill_snapshots/` | memory snapshots and dev evaluations at 0/10/20/30/43 |

---

# Part 2 — 固定数据划分

## DATA_SPLIT - conv-30 Question-Time Self-Evolution

Seed: `20261110`. Split fixed before any skill was learned.

| subset | n | multi-hop (1) | temporal (2) | single-hop (4) | adversarial (5) |
|---|---:|---:|---:|---:|---:|
| adaptation | 43 | 4 | 11 | 18 | 10 |
| development | 19 | 2 | 5 | 8 | 4 |
| evaluation | 43 | 5 | 10 | 18 | 10 |

### Question IDs

#### Adaptation (43)

- **multi-hop**: q17, q24, q3, q27
- **temporal**: q11, q35, q0, q6, q1, q32, q33, q34, q20, q7, q22
- **single-hop**: q53, q60, q81, q58, q68, q47, q55, q57, q75, q72, q51, q70, q69, q63, q44, q4, q49, q54
- **adversarial**: q88, q103, q92, q99, q85, q84, q104, q82, q83, q93

#### Development (19)

- **multi-hop**: q31, q5
- **temporal**: q13, q26, q15, q14, q10
- **single-hop**: q59, q46, q76, q43, q65, q74, q45, q39
- **adversarial**: q100, q96, q98, q101

#### Evaluation (43)

- **multi-hop**: q29, q23, q25, q9, q18
- **temporal**: q12, q36, q21, q30, q16, q8, q37, q19, q38, q28
- **single-hop**: q2, q62, q41, q77, q56, q73, q64, q66, q71, q50, q80, q78, q67, q52, q61, q48, q42, q40
- **adversarial**: q97, q86, q94, q91, q90, q102, q89, q79, q95, q87

### Adaptation order

q53, q11, q60, q17, q35, q0, q81, q58, q68, q6, q88, q47, q103, q1, q55, q57, q75, q72, q92, q51, q70, q24, q99, q85, q69, q84, q63, q32, q44, q33, q34, q20, q7, q4, q3, q49, q104, q22, q82, q83, q54, q27, q93

### Caveat

conv-30 has been studied in earlier rounds. This split is fixed before skill learning, but the evaluation set remains an internal exploratory test, not a fully independent final benchmark. Only development questions select the final snapshot; evaluation is untouched until the end.

---

# Part 3 — Baseline：P0 / P7 / P8

## Baseline Results (conv-30)

Retrieval: frozen **COMP-P4_top5** ranked sessions, 4096-token context. P0/P7 reused from `outputs/prompt_optimization_conv30/joint`; P8 run here.

### All 105 questions

| Reader | EM | F1 | official F1 | adversarial refusal | answer support |
|---|---:|---:|---:|---:|---:|
| P0 | 0.222 | 0.417 | 0.445 | 0.958 | 0.967 |
| P7 | 0.259 | 0.539 | 0.569 | 0.917 | 0.970 |
| P8 | 0.222 | 0.504 | 0.529 | 0.958 | 0.985 |

### By subset (project EM / F1 / official F1 / adversarial refusal)

| Reader | subset | n | EM | F1 | official F1 | adv refusal |
|---|---|---:|---:|---:|---:|---:|
| P0 | adaptation | 43 | 0.152 | 0.369 | 0.378 | 1.000 |
| P0 | development | 19 | 0.267 | 0.416 | 0.466 | 1.000 |
| P0 | evaluation | 43 | 0.273 | 0.465 | 0.503 | 0.900 |
| P7 | adaptation | 43 | 0.182 | 0.463 | 0.484 | 0.900 |
| P7 | development | 19 | 0.333 | 0.654 | 0.677 | 1.000 |
| P7 | evaluation | 43 | 0.303 | 0.563 | 0.606 | 0.900 |
| P8 | adaptation | 43 | 0.152 | 0.433 | 0.449 | 1.000 |
| P8 | development | 19 | 0.333 | 0.632 | 0.651 | 1.000 |
| P8 | evaluation | 43 | 0.242 | 0.516 | 0.555 | 0.900 |

### By category (all 105; project EM / F1)

| Reader | multi-hop | temporal | single-hop |
|---|---:|---:|---:|
| P0 | 0.273 / 0.546 | 0.269 / 0.331 | 0.182 / 0.436 |
| P7 | 0.273 / 0.511 | 0.500 / 0.869 | 0.114 / 0.352 |
| P8 | 0.182 / 0.408 | 0.423 / 0.805 | 0.114 / 0.349 |

Notes:
- Project EM/F1 use the SQuAD-style normalizer; official F1 is the LoCoMo scorer on answerable questions.
- Adversarial refusal is the fraction of category-5 questions answered with a refusal.
- The adaptation and development columns are the fixed selection signals; the evaluation column is untouched until the final snapshot is chosen.

---

# Part 4 — Answer Skill 自进化

## Answer Skill Evolution

Adaptation split: 43 questions.  The Reader is P7 plus at most two retrieved Answer Skills.  Gold answers are visible only in adaptation/reflection and in the scorer; the evaluation split is untouched until the final snapshot is frozen.

### Adaptation outcome

- Adaptation questions: **43**
- Correct without a new skill: **16**
- Skills added: **3**; updated: **0**
- Candidates rejected by verification: **8**; accepted: **3**; verification pass rate: **3/11**
- Sanitizer rejections: **6** (leak-check: 6)
- Self-repair attempts / accepted: **5 / 0**
- Active skills in the selected snapshot (43): **3**
- Total skills ever created (including disabled/merged): **3**
- Merges: **0**; rollbacks: **0**

#### Error types driving reflection

| error_type | count |
|---|---:|
| format_mismatch | 5 |
| over_verbose | 4 |
| evidence_not_retrieved | 4 |
| temporal_granularity | 4 |
| multi_fact_omission | 4 |
| temporal_arithmetic | 3 |
| evidence_not_used | 1 |
| wrong_target_or_premise | 1 |
| wrong_refusal | 1 |

### Development snapshots (snapshot selection)

| feedback questions | active skills | dev EM | dev F1 | dev official F1 | adv refusal |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.333 | 0.654 | 0.677 | 1.000 |
| 10 | 3 | 0.400 | 0.662 | 0.687 | 1.000 |
| 20 | 3 | 0.400 | 0.662 | 0.687 | 1.000 |
| 30 | 3 | 0.400 | 0.662 | 0.687 | 1.000 |
| 40 | 3 | 0.400 | 0.662 | 0.687 | 1.000 |
| 43 | 3 | 0.400 | 0.662 | 0.687 | 1.000 |

Selected snapshot: **step 43** (highest development F1; evaluation never participated in selection).

Ablation: rendering each skill's trigger conditions in the Reader prompt (instead of action text only) scored dev official F1 0.683 / project F1 0.658, versus 0.687 / 0.662 for the deployed action-only rendering.  It was not deployed because dev selection preferred the action-only format.

### Skills in the selected memory

| skill_id | title | cats | v | status | success/failure | confidence | created | reused later (success) |
|---|---|---|---:|---|---:|---:|---:|---:|
| ans_use_explicit_caused_state_wording | Use explicit caused-state wording | [4] | 1 | active | 1/0 | 0.67 | 3 | 16 (2) |
| ans_return_concise_explicit_purpose_for_why_ | Return concise explicit purpose for why-questions | [4] | 1 | active | 1/0 | 0.67 | 8 | 14 (2) |
| ans_preserve_temporal_granularity_for_relati | Preserve temporal granularity for relative expressions | [2] | 1 | active | 0/0 | 0.55 | 10 | 7 (2) |

### Reuse across questions (RSI evidence)

- `ans_use_explicit_caused_state_wording` was learned at feedback step 3 and selected on 16 later adaptation questions (first reuse step 7), 2 correct.
- `ans_return_concise_explicit_purpose_for_why_` was learned at feedback step 8 and selected on 14 later adaptation questions (first reuse step 9), 2 correct.
- `ans_preserve_temporal_granularity_for_relati` was learned at feedback step 10 and selected on 7 later adaptation questions (first reuse step 14), 2 correct.

### Selected skill texts

#### ans_use_explicit_caused_state_wording (v1)

**Use explicit caused-state wording** — created at step 3, source qids [60].

> Prefer the exact adjective or state in an explicit causative phrase over nearby metaphors, summaries, or paraphrases. Quote or minimally return that stated state, and avoid substituting broader interpretations.

#### ans_return_concise_explicit_purpose_for_why_ (v1)

**Return concise explicit purpose for why-questions** — created at step 8, source qids [58].

> Use the shortest explicit purpose phrase tied directly to the asked action. Omit later explanations, expected benefits, pronouns, and paraphrases unless needed to preserve meaning.

#### ans_preserve_temporal_granularity_for_relati (v1)

**Preserve temporal granularity for relative expressions** — created at step 10, source qids [6].

> Resolve relative time expressions from the evidence timestamp, but answer only at the granularity explicitly supported. Do not convert month-level expressions into exact calendar days or add unsupported precision.

### Reading

- Acceptance requires the candidate to fix its source question, pass similar-question replay, pass dissimilar-question replay, and preserve adversarial refusal.
- The sanitizer rejects candidates that mention any distinctive answer token, proper noun, or explicit year from the source question.
- A skill that is created and never reused is a local fix, not evidence of RSI generalisation.  The reuse table is the direct test.
- Reuse is frequent (16/14/7 later adaptation questions) but successful reuse is rare (2 correct answers per skill).  Selection is category-driven, so the skills are applied to many questions whose trigger they do not really match.
- On the evaluation split, 28/43 questions retrieved at least one skill, but only 1 hit was helpful, 1 harmful and 26 neutral (`error_analysis.md`).  Frequent retrieval is therefore not evidence of benefit.

---

# Part 5 — Retrieval Skill 自进化

## Retrieval Skill Evolution

The base ranking is the frozen COMP-P4_top5 HippoRAG2 ranking.  Retrieval skills only add queries; their dense session rankings are merged by RRF.  QDM and the evidence-sufficiency check are shared across static/evolved libraries; only the skill library differs.

### Adaptation outcome

- Adaptation questions: **43**
- Evidence already complete after the base + QDM/sufficiency pipeline: **38**
- Retrieval skills added: **1**; updated: **0**; rejected: **2**; candidate pass rate: **1/3**
- Questions whose gold evidence was recovered by the skill layer: **1**
- Questions whose gold evidence was lost relative to the base ranking: **1**
- Active retrieval skills in the raw final memory: **1**
- Snapshot chosen on development: **step 0**, deploying **0** active skills.

### Development snapshots (retrieval-only, no LLM)

| feedback | active skills | R@5 | evidence full@4096 | multi-evidence complete |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.967 | 0.933 | 0.667 |
| 10 | 1 | 0.933 | 0.867 | 0.333 |
| 20 | 1 | 0.933 | 0.867 | 0.333 |
| 30 | 1 | 0.933 | 0.867 | 0.333 |
| 40 | 1 | 0.933 | 0.867 | 0.333 |
| 43 | 1 | 0.933 | 0.867 | 0.333 |

### Retrieval skills

| skill_id | title | cats | v | status | queries | source qids |
|---|---|---|---:|---|---|---|
| ret_motivation_and_trigger_expansion_for_why | Motivation and Trigger Expansion for Why-Decisions | [1] | 1 | rejected_by_dev | {question} reason motivation because wanted to; {entities} passionate about love for unique interests blend; {entities} lost job unemployment before starting bu | [17] |

### Independent retrieval contribution (eval)

| Exp | R@3 | R@5 | evidence full@4096 | multi-evidence complete |
|---|---:|---:|---:|---:|
| E0 | 0.811 | 0.932 | 0.909 | 0.714 |
| E3 | 0.614 | 0.803 | 0.818 | 0.429 |
| E4 | 0.811 | 0.886 | 0.848 | 0.429 |

### Auxiliary: raw learned retrieval skill applied without the sufficiency gate

This is the skill *before* development snapshot selection; it is shown only as an ablation, not as the deployed E4/E6 memory.

- Active skills applied: 1
- Base R@5 -> applied R@5: 0.932 -> 0.879
- Base evidence full -> applied: 0.909 -> 0.848
- Recovered questions: 0; lost: 2


### Reading

- The learned skill passed adaptation verification but lowered development evidence coverage (step 0 vs later snapshots).  Snapshot selection therefore chose **step 0**; the skill is retained in the trace as a verified-but-rejected negative-transfer case, not deployed in E4/E6.
- The pipeline is capped at two retrieval rounds (QDM round, then one skill/missing-relation round).
- At evaluation time the sufficiency check sees only the question and retrieved sessions; gold evidence is never used to decide whether to retrieve again.
- Retrieval-skill verification is local and LLM-free: replay the source, similar and dissimilar training questions with dense retrieval only.

---

# Part 6 — E0–E6 联合实验与等预算静态对照

## Joint Experiments (evaluation split, n=43)

The base retrieval is the frozen COMP-P4_top5 HippoRAG2 ranking.  Every method uses the same P7 Reader unless stated.  E3-E6 use QDM + the evidence-sufficiency check + at most one extra retrieval round.

| Exp | Answer | Retrieval | EM | F1 | official F1 | adv refusal | Ev@4096 | multi-ev | R@5 | LLM calls |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | P7 | base | 0.303 | 0.563 | 0.606 | 0.900 | 0.909 | 0.714 | 0.932 | 1.00 |
| E1 | static answer | base | 0.273 | 0.423 | 0.436 | 0.900 | 0.909 | 0.714 | 0.932 | 1.00 |
| E2 | evolved answer | base | 0.273 | 0.508 | 0.574 | 0.900 | 0.909 | 0.714 | 0.932 | 1.00 |
| E3 | P7 | static retrieval | 0.303 | 0.545 | 0.576 | 0.900 | 0.818 | 0.429 | 0.803 | 3.00 |
| E4 | P7 | evolved retrieval | 0.273 | 0.546 | 0.602 | 0.900 | 0.848 | 0.429 | 0.886 | 3.00 |
| E5 | evolved answer | static retrieval | 0.303 | 0.537 | 0.556 | 0.900 | 0.818 | 0.429 | 0.803 | 3.00 |
| E6 | evolved answer | evolved retrieval | 0.273 | 0.506 | 0.589 | 0.900 | 0.848 | 0.429 | 0.886 | 3.00 |
| EB | long static prompt | base | 0.303 | 0.488 | 0.523 | 0.900 | 0.909 | 0.714 | 0.932 | 1.00 |

### By category (F1 / official F1 / n)

| Exp | multi-hop | temporal | single-hop |
|---|---|---|---|
| E0 | 0.567 / 0.543 (5) | 0.940 / 0.940 (10) | 0.353 / 0.437 (18) |
| E1 | 0.567 / 0.543 (5) | 0.600 / 0.600 (10) | 0.284 / 0.314 (18) |
| E2 | 0.382 / 0.543 (5) | 0.680 / 0.680 (10) | 0.447 / 0.524 (18) |
| E3 | 0.476 / 0.455 (5) | 0.907 / 0.907 (10) | 0.364 / 0.426 (18) |
| E4 | 0.377 / 0.543 (5) | 0.940 / 0.940 (10) | 0.373 / 0.431 (18) |
| E5 | 0.476 / 0.455 (5) | 0.860 / 0.860 (10) | 0.374 / 0.415 (18) |
| E6 | 0.377 / 0.543 (5) | 0.680 / 0.680 (10) | 0.445 / 0.551 (18) |
| EB | 0.567 / 0.543 (5) | 0.600 / 0.600 (10) | 0.403 / 0.474 (18) |

### Reading

- E0 is the strongest method on EM/F1/official F1 and on evidence recall.  Every skill layer lowers at least one of these on the 43-question evaluation split.
- E2 (evolved answer) beats E1 (retrieved static answer rules) and EB (one long static prompt), so the evolved rules are less damaging than hand-written rules, but neither exceeds P7.
- E4/E6 deploy 0 learned retrieval skills after development selection; they are QDM + sufficiency without a learned skill.  E3/E5 use the static retrieval library.
- E3-E6 show that QDM + dense-query merging degrades the strong graph ranking: R@5 falls from 0.932 to about 0.80-0.89 and evidence-full from 0.909 to 0.82-0.85.
- Within the retrieval layer, the static 4-rule library (E3, R@5 0.803) is worse than QDM-only (E4, R@5 0.886), so adding hand-written retrieval queries creates ranking noise rather than recovering evidence.
- One question is 2.3 EM points on this split; the joint differences are larger than that for E1/E4/E6, but the overall conclusion is still negative.

---

# Part 7 — Question-Time ReAct 与 Grounded Verifier

## Question-Time ReAct + Grounded Verification (evaluation split)

| Variant | Description | EM | F1 | official F1 | adv refusal | answer support | revision rate | LLM calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | direct P7 | 0.273 | 0.562 | 0.631 | 0.900 | 0.999 | 0.000 | 1.00 |
| R1 | P7 + verifier | 0.273 | 0.549 | 0.617 | 0.900 | 0.999 | 0.395 | 2.14 |
| R2 | P7 + static ReAct + verifier | 0.303 | 0.520 | 0.544 | 0.900 | 0.999 | 0.535 | 4.26 |
| R3 | evolved Answer Skills | 0.273 | 0.484 | 0.553 | 0.900 | 0.999 | 0.000 | 1.00 |
| R4 | evolved Answer + verifier | 0.273 | 0.490 | 0.558 | 0.900 | 0.999 | 0.326 | 2.07 |
| R5 | evolved Answer + evolved Retrieval + verifier | 0.242 | 0.425 | 0.445 | 0.900 | 1.000 | 0.442 | 4.12 |

The verifier sees only the question, retrieved evidence and candidate answer.  It never receives gold answers, gold evidence or evaluation labels.  Revision is capped at one extra Reader call; retrieval is capped at two rounds.

Reading: the verifier revises 33-54% of answers but does not improve F1; it mainly adds calls.  The full evolved system (R5) is the worst variant because both the answer and retrieval skill layers over-apply.

---

# Part 8 — 错误分析

## Error Analysis (evaluation split)

### E0 -> E1

Fixed: **0**; worsened: **1**; still wrong: 24.

#### Worsened

- **q38** (temporal) When did Gina go to a dance class with a group of friends?
  - gold: `21 July 2023`
  - E1: `Last Friday` (skills: static_ans_1_temporal_anchor, static_ans_2_temporal_granularity)

### E0 -> E2

Fixed: **1**; worsened: **2**; still wrong: 23.

#### Fixed

- **q71** (single-hop) How does Gina describe the studio that Jon has opened?
  - gold: `amazing`
  - E2: `amazing` (skills: ans_use_explicit_caused_state_wording, ans_return_concise_explicit_purpose_for_why_)

#### Worsened

- **q18** (multi-hop) Do Jon and Gina start businesses out of what they love?
  - gold: `Yes`
  - E2: `Yes, "I'm passionate about dancing and it'd be great to share it with others", "I'm passionate about fashion trends and finding unique pieces. Plus, I wanted to` (skills: none)
- **q38** (temporal) When did Gina go to a dance class with a group of friends?
  - gold: `21 July 2023`
  - E2: `Last Friday` (skills: ans_preserve_temporal_granularity_for_relati)

### E0 -> E6

Fixed: **1**; worsened: **2**; still wrong: 23.

#### Fixed

- **q71** (single-hop) How does Gina describe the studio that Jon has opened?
  - gold: `amazing`
  - E6: `amazing` (skills: ans_use_explicit_caused_state_wording, ans_return_concise_explicit_purpose_for_why_)

#### Worsened

- **q18** (multi-hop) Do Jon and Gina start businesses out of what they love?
  - gold: `Yes`
  - E6: `Yes, "I'm starting a dance studio 'cause I'm passionate about dancing and it'd be great to share it with others", "I'm passionate about dance and fashion so com` (skills: none)
- **q38** (temporal) When did Gina go to a dance class with a group of friends?
  - gold: `21 July 2023`
  - E6: `Last Friday` (skills: ans_preserve_temporal_granularity_for_relati)

### E2 -> E6

Fixed: **0**; worsened: **0**; still wrong: 25.

### Skill hit analysis (E2)

- Questions where at least one skill was retrieved: **28**
- Helpful (E0 wrong -> E2 right): **1**
- Harmful (E0 right -> E2 wrong): **1**
- Neutral: **26**
- Wrong without any skill hit: **5**

| skill | uses | helpful | harmful | neutral |
|---|---:|---:|---:|---:|
| ans_use_explicit_caused_state_wording | 18 | 1 | 0 | 17 |
| ans_return_concise_explicit_purpose_for_why_ | 18 | 1 | 0 | 17 |
| ans_preserve_temporal_granularity_for_relati | 10 | 0 | 1 | 9 |

### Skill hit analysis (E6, full system)

Hit/helpful/harmful/neutral/no-hit-wrong: 28 / 1 / 1 / 26 / 5

### Evidence present but the Reader still refused

- E2: **2** questions
  - q9 (multi-hop): Which city have both Jean and John visited? -> `No answer`
  - q67 (single-hop): How does Jon use the clipboard with a notepad attached to it? -> `No answer`
- E6: **2** questions
  - q9 (multi-hop): Which city have both Jean and John visited? -> `No answer`
  - q67 (single-hop): How does Jon use the clipboard with a notepad attached to it? -> `No answer`

### Interpretation

- A skill hit does not mean the skill caused the change; compare the same question under E0.  The per-skill table separates helpful from harmful hits.
- Skills that are retrieved but never change an answer are neutral, not evidence of generalisation.
- Repeated-run instability bounds how much of any single-question flip can be attributed to the skill layer.

---

# Part 9 — 成本与延迟

## Cost and Latency

### Total endpoint usage (this study, cached + uncached)

- Calls logged: **2582** (cache hits: 1023)
- Prompt tokens: **7,431,528**
- Completion tokens: **98,445**
- Call latency P50/P95: **2.05s / 3.79s**

- Endpoint success rate (all logged attempts): 0.713; rows with at least one retry: 619.
### By stage

| stage | calls | cache hits | prompt tok | completion tok | latency P50 | latency P95 |
|---|---:|---:|---:|---:|---:|---:|
| adapt | 41 | 7 | 186,049 | 1,023 | 1.79 | 2.58 |
| baseline | 105 | 1 | 472,606 | 1,682 | 1.88 | 4.15 |
| dev | 141 | 83 | 646,215 | 1,612 | 1.73 | 2.45 |
| joint_E1 | 86 | 2 | 398,240 | 833 | 2.32 | 2.63 |
| joint_E2 | 48 | 5 | 217,729 | 490 | 2.50 | 2.73 |
| joint_E3 | 68 | 26 | 302,825 | 1,083 | 1.86 | 2.87 |
| joint_E4 | 86 | 10 | 384,583 | 1,229 | 1.71 | 2.85 |
| joint_E5 | 43 | 15 | 194,737 | 403 | 1.72 | 2.09 |
| joint_E6 | 86 | 37 | 390,511 | 924 | 1.71 | 2.04 |
| joint_EB | 43 | 1 | 209,180 | 389 | 2.48 | 2.75 |
| qdm | 488 | 319 | 68,784 | 16,732 | 1.90 | 2.49 |
| react_R0 | 43 | 1 | 191,851 | 630 | 2.45 | 2.80 |
| react_R1 | 43 | 43 | 191,851 | 630 | 2.45 | 2.80 |
| react_R2 | 43 | 3 | 192,026 | 591 | 1.72 | 2.26 |
| react_R3 | 43 | 0 | 194,815 | 436 | 2.41 | 2.87 |
| react_R4 | 43 | 43 | 194,815 | 436 | 2.41 | 2.87 |
| react_R5 | 43 | 3 | 195,434 | 419 | 2.02 | 2.54 |
| reflect | 84 | 26 | 124,519 | 20,027 | 4.16 | 4.69 |
| reflect_ret | 17 | 3 | 29,767 | 4,861 | 4.86 | 5.12 |
| repeat_E0 | 35 | 0 | 156,272 | 616 | 4.67 | 5.14 |
| revise | 25 | 2 | 115,533 | 480 | 1.70 | 2.03 |
| suff | 483 | 314 | 387,341 | 21,439 | 2.35 | 2.93 |
| verify | 273 | 70 | 1,225,379 | 3,689 | 1.71 | 1.99 |
| verify_answer | 172 | 9 | 760,466 | 17,791 | 2.64 | 3.78 |

### Per-question cost on the evaluation split

| Exp | LLM calls | prompt tokens | completion tokens | reader latency s |
|---|---:|---:|---:|---:|
| E0 | 1.00 | 4462 | 10.9 | 1.76 |
| E1 | 1.00 | 4584 | 9.4 | 2.40 |
| E2 | 1.00 | 4531 | 10.8 | 2.49 |
| E3 | 3.00 | 5403 | 93.7 | 1.97 |
| E4 | 3.00 | 5405 | 93.8 | 1.83 |
| E5 | 3.00 | 5472 | 89.0 | 1.77 |
| E6 | 3.00 | 5474 | 90.4 | 1.79 |
| EB | 1.00 | 4865 | 9.0 | 2.43 |

### ReAct variants

| Variant | LLM calls | prompt tokens | completion tokens |
|---|---:|---:|---:|
| R0 | 1.00 | 4462 | 14.7 |
| R1 | 2.14 | 9528 | 119.1 |
| R2 | 4.26 | 10052 | 132.0 |
| R3 | 1.00 | 4531 | 10.1 |
| R4 | 2.07 | 9275 | 106.7 |
| R5 | 4.12 | 9521 | 115.4 |

Costs are measured from the HY4 response usage fields.  Cached calls are content-addressed repeats and cost no endpoint quota; adaptation reflection and verification are one-off training costs, while the E-matrix columns are per evaluation question.

---

# Part 10 — 重复运行稳定性

## Repeated-run stability

The endpoint quota allowed two independent runs of the two key configurations instead of the planned three fresh runs.  E0/E2 are the main cached runs; R0/R3 are independent fresh calls of the same configuration.

| configuration | EM A | EM B | F1 A | F1 B | adv refusal A/B | EM-unstable | F1 change > 0.25 |
|---|---:|---:|---:|---:|---|---:|---:|
| E0 vs R0 | 0.303 | 0.273 | 0.563 | 0.562 | 0.900 / 0.900 | 1 | 2 |
| E2 vs R3 | 0.273 | 0.273 | 0.508 | 0.484 | 0.900 / 0.900 | 0 | 1 |

### Per-question changes (answerable questions)

#### E0 vs R0

- q18: EM A=1.0 B=0.0; F1 A=1.000 B=0.050
- q64: EM A=0.0 B=0.0; F1 A=0.000 B=0.909

#### E2 vs R3

- q64: EM A=0.0 B=0.0; F1 A=0.909 B=0.000

A difference of two EM points on 33 answerable questions is about 0.7 question.  The per-question flips show that several apparent skill gains/losses are within HY4 nondeterminism.

---

# Part 11 — 运行清单与交付物索引

## 11.1 Run manifest

```json
{
  "created": "2026-11-10T04:23:32",
  "project": "RSI GraphRAG / LoCoMo conv-30 Question-Time self-evolution",
  "base_retrieval_variant": "COMP-P4_top5",
  "context_budget": 4096,
  "split": {
    "seed": 20261110,
    "n_adaptation": 43,
    "n_development": 19,
    "n_evaluation": 43
  },
  "answer_skill": {
    "selected_dev_step": 43,
    "n_active": 3
  },
  "retrieval_skill": {
    "selected_dev_step": 0,
    "n_deployed": 0,
    "note": "learned skill rejected by dev snapshot selection"
  },
  "methods": [
    "E0",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "EB",
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5"
  ],
  "endpoint": {
    "api_rows": 2173,
    "status": {
      "ok": 1546,
      "error:RateLimitError": 627
    },
    "retried_rows": 619,
    "success_rate": 0.7114588127013346
  },
  "total_calls": 2569,
  "cache_hits": 1023,
  "prompt_tokens": 7371223,
  "completion_tokens": 98330,
  "llm": "hy4 (reasoning_effort=no_think), temperature=0",
  "embedding": "NV-Embed-v2",
  "caveat": "conv-30 previously studied; eval split is internal exploratory, not independent final test."
}
```


---

## 11.2 Adaptation 决策统计


---

| 事件 | 数量 |
|---|---:|
| Answer adaptation questions | 43 |
| Answer decisions | {'rejected': 8, 'correct': 16, 'added': 3, 'no_candidate': 10, 'rejected_sanitize': 6} |
| Retrieval adaptation questions | 43 |
| Retrieval decisions | {'sufficient': 38, 'added': 1, 'rejected': 2, 'rejected_sanitize': 2} |
| Trace events total | 96 |


---

## 11.3 最终交付物

| 文件/目录 | 内容 |
|---|---|
| `COMPREHENSIVE_REPORT.md` | 本单文件完整报告 |
| `FINAL_REPORT.md` | 核心报告 |
| `DATA_SPLIT.md` / `DATA_SPLIT.json` | 固定分层划分与 seed |
| `baseline_results.md` | P0/P7/P8 |
| `answer_skill_evolution.md` | Answer Skill 循环、快照、复用 |
| `retrieval_skill_evolution.md` | Retrieval Skill 循环、query 模板、证据影响 |
| `joint_experiments.md` | E0–E6 + EB |
| `react_and_verification.md` | R0–R5 |
| `error_analysis.md` | fixed/worsened、skill hit、过度泛化 |
| `cost_latency.md` | 调用/token/延迟 |
| `repeat_stability.md` | 独立重复运行 |
| `qa_predictions.jsonl` | 14 个方法的逐题预测 |
| `skill_memory.jsonl` | 最终 Answer Skill memory |
| `retrieval_skill_memory.jsonl` | 最终 Retrieval Skill memory（含 dev 否决标记） |
| `skill_evolution_trace.jsonl` | 43+43 条 adaptation 及快照事件 |
| `skill_snapshots/` | 0/10/20/30/43 快照与 dev 评测 |
| `artifacts/` | 缓存、summary、逐题 JSONL、error cases、cost、run manifest |
| `logs/` | HY4 调用日志与各 stage 日志 |

