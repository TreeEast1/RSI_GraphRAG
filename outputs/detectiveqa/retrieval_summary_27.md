# DetectiveQA retrieval summary

Input: `retrieval_27_both.jsonl`

### All pilot human questions (n=7)

| method | Any@5 | Any@10 | Any@20 | R@5 | R@10 | R@20 | All@5 | All@10 | All@20 | MRR | Hit@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q_dense | 0.571 | 0.571 | 0.714 | 0.244 | 0.327 | 0.456 | 0.000 | 0.000 | 0.286 | 0.613 | 0.714 |
| q_graph | 0.571 | 0.571 | 0.714 | 0.274 | 0.387 | 0.474 | 0.000 | 0.000 | 0.286 | 0.565 | 0.714 |
| qo_dense | 0.571 | 0.571 | 0.571 | 0.268 | 0.399 | 0.423 | 0.000 | 0.143 | 0.143 | 0.750 | 0.571 |
| qo_graph | 0.571 | 0.571 | 0.714 | 0.292 | 0.363 | 0.450 | 0.000 | 0.143 | 0.143 | 0.612 | 0.714 |

### Sensitivity: drop the five duplicate novel-117 annotations (n=7)

| method | Any@5 | Any@10 | Any@20 | R@5 | R@10 | R@20 | All@5 | All@10 | All@20 | MRR | Hit@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q_dense | 0.571 | 0.571 | 0.714 | 0.244 | 0.327 | 0.456 | 0.000 | 0.000 | 0.286 | 0.613 | 0.714 |
| q_graph | 0.571 | 0.571 | 0.714 | 0.274 | 0.387 | 0.474 | 0.000 | 0.000 | 0.286 | 0.565 | 0.714 |
| qo_dense | 0.571 | 0.571 | 0.571 | 0.268 | 0.399 | 0.423 | 0.000 | 0.143 | 0.143 | 0.750 | 0.571 |
| qo_graph | 0.571 | 0.571 | 0.714 | 0.292 | 0.363 | 0.450 | 0.000 | 0.143 | 0.143 | 0.612 | 0.714 |

### Dense fallback (graph mode)

| condition | fallback ratio |
|---|---:|
| q_graph | 0.000 |
| qo_graph | 0.000 |

### Per-novel Recall@10 (qo)

| novel | n | dense R@10 | graph R@10 | dense MRR | graph MRR |
|---|---:|---:|---:|---:|---:|
| 27 | 7 | 0.399 | 0.363 | 0.750 | 0.612 |

### Rank cases (qo): graph better 2, dense better 1

**graph_better**
- human_anno:27:00: dense rank 2, graph rank 1, gold 4, graph mode ``
- human_anno:27:06: dense rank 0, graph rank 17, gold 15, graph mode ``

**dense_better**
- human_anno:27:05: dense rank 1, graph rank 2, gold 8, graph mode ``
