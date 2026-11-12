# Repeated-run stability

The endpoint quota allowed two independent runs of the two key configurations instead of the planned three fresh runs.  E0/E2 are the main cached runs; R0/R3 are independent fresh calls of the same configuration.

| configuration | EM A | EM B | F1 A | F1 B | adv refusal A/B | EM-unstable | F1 change > 0.25 |
|---|---:|---:|---:|---:|---|---:|---:|
| E0 vs R0 | 0.303 | 0.273 | 0.563 | 0.562 | 0.900 / 0.900 | 1 | 2 |
| E2 vs R3 | 0.273 | 0.273 | 0.508 | 0.484 | 0.900 / 0.900 | 0 | 1 |

## Per-question changes (answerable questions)

### E0 vs R0

- q18: EM A=1.0 B=0.0; F1 A=1.000 B=0.050
- q64: EM A=0.0 B=0.0; F1 A=0.000 B=0.909

### E2 vs R3

- q64: EM A=0.0 B=0.0; F1 A=0.909 B=0.000

A difference of two EM points on 33 answerable questions is about 0.7 question.  The per-question flips show that several apparent skill gains/losses are within HY4 nondeterminism.
