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
