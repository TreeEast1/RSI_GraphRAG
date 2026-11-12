# Question-Time ReAct + Grounded Verification (evaluation split)

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
