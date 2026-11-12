# Stage A retrieval diagnosis

n=41

## Condition means

| condition | R@10 | R@20 | All@20 | All@50 | MRR |
|---|---:|---:|---:|---:|---:|
| q_dense | 0.418 | 0.515 | 0.220 | 0.463 | 0.544 |
| q_bm25 | 0.287 | 0.426 | 0.122 | 0.390 | 0.455 |
| q_graph_f5 | 0.425 | 0.538 | 0.244 | 0.463 | 0.497 |
| q_graph_f10 | 0.420 | 0.540 | 0.268 | 0.463 | 0.462 |
| q_graph_f20 | 0.415 | 0.534 | 0.244 | 0.463 | 0.452 |
| q_graph_f5_nosyn | 0.412 | 0.534 | 0.220 | 0.463 | 0.501 |
| q_rrf_dense_graph | 0.429 | 0.523 | 0.220 | 0.463 | 0.516 |
| q_rrf_dense_bm25 | 0.391 | 0.499 | 0.195 | 0.390 | 0.530 |
| q_rrf_all | 0.402 | 0.508 | 0.220 | 0.488 | 0.541 |
| qo_dense | 0.436 | 0.527 | 0.244 | 0.439 | 0.604 |
| qo_graph_f5 | 0.407 | 0.537 | 0.244 | 0.463 | 0.486 |
| qo_rrf_dense_graph | 0.436 | 0.535 | 0.244 | 0.463 | 0.563 |

## Paired (novel-clustered bootstrap, delta, 95% CI, wins/losses, McNemar p)

- q_dense->q_graph_f5:recall@10: delta +0.0072, CI [-0.0347,+0.0434], wins 8, losses 9, p=1
- q_dense->q_graph_f5:all@20: delta +0.0244, CI [+0.0000,+0.0811], wins 1, losses 0, p=1
- q_dense->q_graph_f5:mrr: delta -0.0478, CI [-0.1276,+0.0320], wins 4, losses 16, p=0.01182
- q_dense->q_graph_f10:recall@10: delta +0.0016, CI [-0.0471,+0.0563], wins 5, losses 7, p=0.7744
- q_dense->q_graph_f10:all@20: delta +0.0488, CI [+0.0000,+0.1111], wins 2, losses 0, p=0.5
- q_dense->q_graph_f10:mrr: delta -0.0820, CI [-0.1872,+0.0313], wins 4, losses 20, p=0.001544
- q_dense->q_graph_f20:recall@10: delta -0.0034, CI [-0.0527,+0.0529], wins 7, losses 7, p=1
- q_dense->q_graph_f20:all@20: delta +0.0244, CI [-0.0556,+0.1053], wins 2, losses 1, p=1
- q_dense->q_graph_f20:mrr: delta -0.0926, CI [-0.2053,+0.0441], wins 6, losses 20, p=0.009355
- q_dense->q_graph_f5_nosyn:recall@10: delta -0.0059, CI [-0.0518,+0.0389], wins 7, losses 9, p=0.8036
- q_dense->q_graph_f5_nosyn:all@20: delta +0.0000, CI [+0.0000,+0.0000], wins 0, losses 0, p=1
- q_dense->q_graph_f5_nosyn:mrr: delta -0.0430, CI [-0.1381,+0.0392], wins 4, losses 14, p=0.03088
- q_dense->q_bm25:recall@10: delta -0.1308, CI [-0.1540,-0.1147], wins 5, losses 19, p=0.006611
- q_dense->q_bm25:all@20: delta -0.0976, CI [-0.1892,-0.0208], wins 0, losses 4, p=0.125
- q_dense->q_bm25:mrr: delta -0.0894, CI [-0.1500,-0.0023], wins 11, losses 16, p=0.4421
- q_dense->q_rrf_dense_graph:recall@10: delta +0.0112, CI [-0.0165,+0.0417], wins 4, losses 3, p=1
- q_dense->q_rrf_dense_graph:all@20: delta +0.0000, CI [-0.0698,+0.0769], wins 1, losses 1, p=1
- q_dense->q_rrf_dense_graph:mrr: delta -0.0287, CI [-0.0650,+0.0018], wins 5, losses 8, p=0.5811
- q_dense->q_rrf_dense_bm25:recall@10: delta -0.0269, CI [-0.0630,+0.0207], wins 4, losses 7, p=0.5488
- q_dense->q_rrf_dense_bm25:all@20: delta -0.0244, CI [-0.0769,+0.0000], wins 0, losses 1, p=1
- q_dense->q_rrf_dense_bm25:mrr: delta -0.0143, CI [-0.0507,+0.0263], wins 10, losses 8, p=0.8145
- q_dense->q_rrf_all:recall@10: delta -0.0159, CI [-0.0548,+0.0358], wins 4, losses 6, p=0.7539
- q_dense->q_rrf_all:all@20: delta +0.0000, CI [-0.0698,+0.0769], wins 1, losses 1, p=1
- q_dense->q_rrf_all:mrr: delta -0.0038, CI [-0.0479,+0.0370], wins 12, losses 7, p=0.3593
- q_dense->qo_dense:recall@10: delta +0.0179, CI [-0.0454,+0.0862], wins 9, losses 9, p=1
- q_graph_f5->qo_graph_f5:recall@10: delta -0.0178, CI [-0.0850,+0.0446], wins 4, losses 11, p=0.1185

## Case counts

- dense_better 16 ({'early_lt_70': 6, 'middle_70_90': 3, 'late_ge_90': 7})
- graph_better 4 ({'middle_70_90': 1, 'early_lt_70': 1, 'late_ge_90': 2})
