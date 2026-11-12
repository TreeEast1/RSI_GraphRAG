# Answer Skill Evolution

Adaptation split: 43 questions.  The Reader is P7 plus at most two retrieved Answer Skills.  Gold answers are visible only in adaptation/reflection and in the scorer; the evaluation split is untouched until the final snapshot is frozen.

## Adaptation outcome

- Adaptation questions: **43**
- Correct without a new skill: **16**
- Skills added: **3**; updated: **0**
- Candidates rejected by verification: **8**; accepted: **3**; verification pass rate: **3/11**
- Sanitizer rejections: **6** (leak-check: 6)
- Self-repair attempts / accepted: **5 / 0**
- Active skills in the selected snapshot (43): **3**
- Total skills ever created (including disabled/merged): **3**
- Merges: **0**; rollbacks: **0**

### Error types driving reflection

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

## Development snapshots (snapshot selection)

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

## Skills in the selected memory

| skill_id | title | cats | v | status | success/failure | confidence | created | reused later (success) |
|---|---|---|---:|---|---:|---:|---:|---:|
| ans_use_explicit_caused_state_wording | Use explicit caused-state wording | [4] | 1 | active | 1/0 | 0.67 | 3 | 16 (2) |
| ans_return_concise_explicit_purpose_for_why_ | Return concise explicit purpose for why-questions | [4] | 1 | active | 1/0 | 0.67 | 8 | 14 (2) |
| ans_preserve_temporal_granularity_for_relati | Preserve temporal granularity for relative expressions | [2] | 1 | active | 0/0 | 0.55 | 10 | 7 (2) |

## Reuse across questions (RSI evidence)

- `ans_use_explicit_caused_state_wording` was learned at feedback step 3 and selected on 16 later adaptation questions (first reuse step 7), 2 correct.
- `ans_return_concise_explicit_purpose_for_why_` was learned at feedback step 8 and selected on 14 later adaptation questions (first reuse step 9), 2 correct.
- `ans_preserve_temporal_granularity_for_relati` was learned at feedback step 10 and selected on 7 later adaptation questions (first reuse step 14), 2 correct.

## Selected skill texts

### ans_use_explicit_caused_state_wording (v1)

**Use explicit caused-state wording** — created at step 3, source qids [60].

> Prefer the exact adjective or state in an explicit causative phrase over nearby metaphors, summaries, or paraphrases. Quote or minimally return that stated state, and avoid substituting broader interpretations.

### ans_return_concise_explicit_purpose_for_why_ (v1)

**Return concise explicit purpose for why-questions** — created at step 8, source qids [58].

> Use the shortest explicit purpose phrase tied directly to the asked action. Omit later explanations, expected benefits, pronouns, and paraphrases unless needed to preserve meaning.

### ans_preserve_temporal_granularity_for_relati (v1)

**Preserve temporal granularity for relative expressions** — created at step 10, source qids [6].

> Resolve relative time expressions from the evidence timestamp, but answer only at the granularity explicitly supported. Do not convert month-level expressions into exact calendar days or add unsupported precision.

## Reading

- Acceptance requires the candidate to fix its source question, pass similar-question replay, pass dissimilar-question replay, and preserve adversarial refusal.
- The sanitizer rejects candidates that mention any distinctive answer token, proper noun, or explicit year from the source question.
- A skill that is created and never reused is a local fix, not evidence of RSI generalisation.  The reuse table is the direct test.
- Reuse is frequent (16/14/7 later adaptation questions) but successful reuse is rare (2 correct answers per skill).  Selection is category-driven, so the skills are applied to many questions whose trigger they do not really match.
- On the evaluation split, 28/43 questions retrieved at least one skill, but only 1 hit was helpful, 1 harmful and 26 neutral (`error_analysis.md`).  Frequent retrieval is therefore not evidence of benefit.
