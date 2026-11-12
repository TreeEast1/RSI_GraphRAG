# Baseline Results (conv-30)

Retrieval: frozen **COMP-P4_top5** ranked sessions, 4096-token context. P0/P7 reused from `outputs/prompt_optimization_conv30/joint`; P8 run here.

## All 105 questions

| Reader | EM | F1 | official F1 | adversarial refusal | answer support |
|---|---:|---:|---:|---:|---:|
| P0 | 0.222 | 0.417 | 0.445 | 0.958 | 0.967 |
| P7 | 0.259 | 0.539 | 0.569 | 0.917 | 0.970 |
| P8 | 0.222 | 0.504 | 0.529 | 0.958 | 0.985 |

## By subset (project EM / F1 / official F1 / adversarial refusal)

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

## By category (all 105; project EM / F1)

| Reader | multi-hop | temporal | single-hop |
|---|---:|---:|---:|
| P0 | 0.273 / 0.546 | 0.269 / 0.331 | 0.182 / 0.436 |
| P7 | 0.273 / 0.511 | 0.500 / 0.869 | 0.114 / 0.352 |
| P8 | 0.182 / 0.408 | 0.423 / 0.805 | 0.114 / 0.349 |

Notes:
- Project EM/F1 use the SQuAD-style normalizer; official F1 is the LoCoMo scorer on answerable questions.
- Adversarial refusal is the fraction of category-5 questions answered with a refusal.
- The adaptation and development columns are the fixed selection signals; the evaluation column is untouched until the final snapshot is chosen.
