# Cost and Latency

## Total endpoint usage (this study, cached + uncached)

- Calls logged: **2582** (cache hits: 1023)
- Prompt tokens: **7,431,528**
- Completion tokens: **98,445**
- Call latency P50/P95: **2.05s / 3.79s**

- Endpoint success rate (all logged attempts): 0.713; rows with at least one retry: 619.
## By stage

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

## Per-question cost on the evaluation split

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

## ReAct variants

| Variant | LLM calls | prompt tokens | completion tokens |
|---|---:|---:|---:|
| R0 | 1.00 | 4462 | 14.7 |
| R1 | 2.14 | 9528 | 119.1 |
| R2 | 4.26 | 10052 | 132.0 |
| R3 | 1.00 | 4531 | 10.1 |
| R4 | 2.07 | 9275 | 106.7 |
| R5 | 4.12 | 9521 | 115.4 |

Costs are measured from the HY4 response usage fields.  Cached calls are content-addressed repeats and cost no endpoint quota; adaptation reflection and verification are one-off training costs, while the E-matrix columns are per evaluation question.
