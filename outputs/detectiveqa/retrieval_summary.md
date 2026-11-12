# DetectiveQA retrieval summary

Input: `retrieval_eval_human_raw_both.jsonl`

All metrics are over all questions; MRR counts a missed gold chunk as 0.

### All pilot human questions (n=41)

| method | Any@5 | Any@10 | Any@20 | R@5 | R@10 | R@20 | All@5 | All@10 | All@20 | MRR | Hit@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q_dense | 0.610 | 0.756 | 0.854 | 0.307 | 0.418 | 0.515 | 0.098 | 0.146 | 0.220 | 0.543 | 0.854 |
| q_graph | 0.658 | 0.780 | 0.878 | 0.262 | 0.425 | 0.538 | 0.024 | 0.146 | 0.244 | 0.496 | 0.878 |
| qo_dense | 0.707 | 0.732 | 0.829 | 0.292 | 0.436 | 0.527 | 0.024 | 0.220 | 0.244 | 0.599 | 0.829 |
| qo_graph | 0.658 | 0.780 | 0.878 | 0.311 | 0.407 | 0.537 | 0.098 | 0.195 | 0.244 | 0.484 | 0.878 |

### Sensitivity: drop the five duplicate novel-117 annotations (n=36)

| method | Any@5 | Any@10 | Any@20 | R@5 | R@10 | R@20 | All@5 | All@10 | All@20 | MRR | Hit@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q_dense | 0.611 | 0.750 | 0.861 | 0.325 | 0.433 | 0.534 | 0.111 | 0.167 | 0.250 | 0.531 | 0.861 |
| q_graph | 0.639 | 0.750 | 0.861 | 0.273 | 0.440 | 0.554 | 0.028 | 0.167 | 0.278 | 0.486 | 0.861 |
| qo_dense | 0.722 | 0.750 | 0.833 | 0.314 | 0.467 | 0.554 | 0.028 | 0.250 | 0.278 | 0.597 | 0.833 |
| qo_graph | 0.639 | 0.778 | 0.861 | 0.337 | 0.439 | 0.559 | 0.111 | 0.222 | 0.278 | 0.479 | 0.861 |

### Dense fallback (graph mode)

| condition | fallback ratio |
|---|---:|
| q_graph | 0.000 |
| qo_graph | 0.000 |

### Per-novel Recall@10 (qo)

| novel | n | dense R@10 | graph R@10 | dense MRR | graph MRR |
|---|---:|---:|---:|---:|---:|
| 27 | 7 | 0.399 | 0.363 | 0.429 | 0.437 |
| 79 | 5 | 0.230 | 0.247 | 0.443 | 0.385 |
| 117 | 10 | 0.219 | 0.173 | 0.633 | 0.509 |
| 118 | 7 | 0.524 | 0.524 | 0.576 | 0.517 |
| 121 | 6 | 0.486 | 0.479 | 0.498 | 0.361 |
| 124 | 6 | 0.861 | 0.778 | 1.000 | 0.667 |

### Rank cases (qo): graph better 6, dense better 17

**graph_better**
- human_anno:27:00: dense rank 2, graph rank 1, gold 4
- human_anno:27:06: dense rank 0, graph rank 17, gold 15
- human_anno:79:01: dense rank 11, graph rank 1, gold 3
- human_anno:117:149:01: dense rank 0, graph rank 5, gold 4
- human_anno:118:03: dense rank 2, graph rank 1, gold 6
- human_anno:121:02: dense rank 11, graph rank 9, gold 6

**dense_better**
- human_anno:27:05: dense rank 1, graph rank 2, gold 8
- human_anno:79:02: dense rank 1, graph rank 3, gold 4
- human_anno:79:03: dense rank 8, graph rank 11, gold 4
- human_anno:79:04: dense rank 1, graph rank 2, gold 5
- human_anno:117:02: dense rank 4, graph rank 7, gold 6
- human_anno:117:04: dense rank 1, graph rank 3, gold 12
- human_anno:117:149:04: dense rank 1, graph rank 3, gold 12
- human_anno:118:00: dense rank 3, graph rank 7, gold 4
