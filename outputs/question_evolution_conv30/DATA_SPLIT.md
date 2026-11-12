# DATA_SPLIT - conv-30 Question-Time Self-Evolution

Seed: `20261110`. Split fixed before any skill was learned.

| subset | n | multi-hop (1) | temporal (2) | single-hop (4) | adversarial (5) |
|---|---:|---:|---:|---:|---:|
| adaptation | 43 | 4 | 11 | 18 | 10 |
| development | 19 | 2 | 5 | 8 | 4 |
| evaluation | 43 | 5 | 10 | 18 | 10 |

## Question IDs

### Adaptation (43)

- **multi-hop**: q17, q24, q3, q27
- **temporal**: q11, q35, q0, q6, q1, q32, q33, q34, q20, q7, q22
- **single-hop**: q53, q60, q81, q58, q68, q47, q55, q57, q75, q72, q51, q70, q69, q63, q44, q4, q49, q54
- **adversarial**: q88, q103, q92, q99, q85, q84, q104, q82, q83, q93

### Development (19)

- **multi-hop**: q31, q5
- **temporal**: q13, q26, q15, q14, q10
- **single-hop**: q59, q46, q76, q43, q65, q74, q45, q39
- **adversarial**: q100, q96, q98, q101

### Evaluation (43)

- **multi-hop**: q29, q23, q25, q9, q18
- **temporal**: q12, q36, q21, q30, q16, q8, q37, q19, q38, q28
- **single-hop**: q2, q62, q41, q77, q56, q73, q64, q66, q71, q50, q80, q78, q67, q52, q61, q48, q42, q40
- **adversarial**: q97, q86, q94, q91, q90, q102, q89, q79, q95, q87

## Adaptation order

q53, q11, q60, q17, q35, q0, q81, q58, q68, q6, q88, q47, q103, q1, q55, q57, q75, q72, q92, q51, q70, q24, q99, q85, q69, q84, q63, q32, q44, q33, q34, q20, q7, q4, q3, q49, q104, q22, q82, q83, q54, q27, q93

## Caveat

conv-30 has been studied in earlier rounds. This split is fixed before skill learning, but the evaluation set remains an internal exploratory test, not a fully independent final benchmark. Only development questions select the final snapshot; evaluation is untouched until the end.
