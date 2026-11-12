# DetectiveQA Round 2 deterministic analysis

## Retrieval

| condition | R@10 | R@20 | All@20 | All@50 | MRR |
|---|---:|---:|---:|---:|---:|
| q_dense | 0.418 | 0.515 | 0.220 | 0.463 | 0.544 |
| q_bm25 | 0.287 | 0.426 | 0.122 | 0.390 | 0.455 |
| q_graph_f5 | 0.425 | 0.538 | 0.244 | 0.463 | 0.497 |
| q_graph_f5_nosyn | 0.412 | 0.534 | 0.220 | 0.463 | 0.501 |
| q_graph_f20 | 0.415 | 0.534 | 0.244 | 0.463 | 0.452 |
| q_rrf_dense_graph | 0.429 | 0.523 | 0.220 | 0.463 | 0.516 |
| qo_dense | 0.436 | 0.527 | 0.244 | 0.439 | 0.604 |
| qo_graph_f5 | 0.407 | 0.537 | 0.244 | 0.463 | 0.486 |

## Reader seed 0

| condition | Acc | gold any | gold all | wrong all-gold | wrong partial | wrong no-gold |
|---|---:|---:|---:|---:|---:|---:|
| nonrecent_q_dense | 0.439 | 0.585 | 0.000 | 0 | 12 | 11 |
| options_only | 0.293 | 0.000 | 0.000 | 0 | 0 | 29 |
| options_only_memory | 0.463 | 0.000 | 0.000 | 0 | 0 | 22 |
| oracle | 0.805 | 1.000 | 0.927 | 6 | 2 | 0 |
| q_dense | 0.658 | 0.756 | 0.122 | 1 | 10 | 3 |
| q_graph | 0.683 | 0.780 | 0.146 | 2 | 8 | 3 |
| q_rrf | 0.634 | 0.780 | 0.122 | 2 | 10 | 3 |
| qo_dense | 0.707 | 0.732 | 0.220 | 2 | 7 | 3 |
| recent | 0.805 | 0.512 | 0.122 | 0 | 3 | 5 |
| recent_plus_q_dense | 0.805 | 0.683 | 0.195 | 0 | 5 | 3 |

## Case queues

- recent_correct_without_gold (15): human_anno:117:01, human_anno:117:02, human_anno:117:03, human_anno:117:149:01, human_anno:117:149:02, human_anno:117:149:03, human_anno:118:04, human_anno:118:06, human_anno:121:05, human_anno:27:00, human_anno:27:03, human_anno:27:04, human_anno:27:06, human_anno:79:00, human_anno:79:04
- oracle_wrong (8): human_anno:117:04, human_anno:117:149:04, human_anno:118:06, human_anno:124:05, human_anno:27:00, human_anno:27:01, human_anno:79:01, human_anno:79:03
- memory_correct (19): human_anno:117:00, human_anno:117:03, human_anno:117:04, human_anno:117:149:00, human_anno:118:00, human_anno:118:02, human_anno:118:05, human_anno:121:00, human_anno:121:01, human_anno:121:02, human_anno:121:05, human_anno:124:00, human_anno:124:02, human_anno:27:00, human_anno:27:04, human_anno:27:06, human_anno:79:00, human_anno:79:02, human_anno:79:03
- recent_only (8): human_anno:117:02, human_anno:117:149:02, human_anno:118:06, human_anno:121:00, human_anno:124:00, human_anno:27:04, human_anno:27:05, human_anno:79:04
- dense_graph_disagree (3): human_anno:118:01, human_anno:118:06, human_anno:121:00
- dense_wrong_all_gold (1): human_anno:124:00
