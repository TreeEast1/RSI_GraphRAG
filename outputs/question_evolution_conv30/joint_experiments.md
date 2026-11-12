# Joint Experiments (evaluation split, n=43)

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

## By category (F1 / official F1 / n)

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

## Reading

- E0 is the strongest method on EM/F1/official F1 and on evidence recall.  Every skill layer lowers at least one of these on the 43-question evaluation split.
- E2 (evolved answer) beats E1 (retrieved static answer rules) and EB (one long static prompt), so the evolved rules are less damaging than hand-written rules, but neither exceeds P7.
- E4/E6 deploy 0 learned retrieval skills after development selection; they are QDM + sufficiency without a learned skill.  E3/E5 use the static retrieval library.
- E3-E6 show that QDM + dense-query merging degrades the strong graph ranking: R@5 falls from 0.932 to about 0.80-0.89 and evidence-full from 0.909 to 0.82-0.85.
- Within the retrieval layer, the static 4-rule library (E3, R@5 0.803) is worse than QDM-only (E4, R@5 0.886), so adding hand-written retrieval queries creates ranking noise rather than recovering evidence.
- One question is 2.3 EM points on this split; the joint differences are larger than that for E1/E4/E6, but the overall conclusion is still negative.
