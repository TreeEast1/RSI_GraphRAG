# 官方 LoCoMo 评分口径 vs 项目口径（conv-30，无 LLM 重跑）

官方实现：`snap-research/locomo task_eval/evaluation.py`（2026-11-09 拉取）。

| 方法 | n(答) | 项目EM | 项目F1 | 官方F1 | 官方EM | 官方F1(含cat5) | 拒答率(项目) | 拒答率(官方严格) | 拒答率(官方+No answer) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G_raw | 81 | 0.2099 | 0.3900 | 0.4262 | 0.2099 | 0.3288 | 0.9583 | 0.0000 | 0.9583 |
| C2 | 81 | 0.2222 | 0.4036 | 0.4329 | 0.2222 | 0.3339 | 0.9167 | 0.0000 | 0.9167 |
| C3 | 81 | 0.2469 | 0.4151 | 0.4446 | 0.2469 | 0.3430 | 0.9167 | 0.0000 | 0.9167 |
| C5 | 81 | 0.2099 | 0.4048 | 0.4415 | 0.2099 | 0.3406 | 0.9583 | 0.0000 | 0.9583 |
| C6 | 81 | 0.1975 | 0.4010 | 0.4369 | 0.1975 | 0.3371 | 0.9583 | 0.0000 | 0.9583 |
| C3_composed | 81 | 0.2099 | 0.4056 | 0.4350 | 0.2099 | 0.3356 | 0.9583 | 0.0000 | 0.9583 |
| C3_inferred | 81 | 0.1852 | 0.3682 | 0.3962 | 0.1852 | 0.3057 | 0.9167 | 0.0000 | 0.9167 |

## 分题型（官方F1 / 项目F1）

| 方法 | cat1 multi-hop | cat2 temporal | cat3 open | cat4 single |
|---|---:|---:|---:|---:|
| G_raw | 0.487 / 0.489 | 0.292 / 0.292 | — | 0.490 / 0.423 |
| C2 | 0.501 / 0.511 | 0.292 / 0.292 | — | 0.499 / 0.443 |
| C3 | 0.489 / 0.496 | 0.369 / 0.369 | — | 0.478 / 0.422 |
| C5 | 0.425 / 0.458 | 0.262 / 0.262 | — | 0.552 / 0.476 |
| C6 | 0.422 / 0.459 | 0.300 / 0.300 | — | 0.522 / 0.446 |
| C3_composed | 0.536 / 0.542 | 0.292 / 0.292 | — | 0.494 / 0.438 |
| C3_inferred | 0.498 / 0.509 | 0.254 / 0.254 | — | 0.455 / 0.400 |

## 差异最大的 15 题（官方F1 − 项目F1；G_raw + C3）

| 方法 | qid | cat | question | gold | pred | 项目F1 | 官方F1 | Δ |
|---|---|---|---|---|---|---:|---:|---:|
| G_raw | q77 | 4 | What did Gina make a limited edition line of? | Hoodies | hoodie | 0.000 | 1.000 | +1.000 |
| C3 | q77 | 4 | What did Gina make a limited edition line of? | Hoodies | hoodie | 0.000 | 1.000 | +1.000 |
| G_raw | q2 | 4 | How do Jon and Gina both like to destress? | by dancing | dance | 0.000 | 0.667 | +0.667 |
| G_raw | q76 | 4 | What is the general sentiment about the upcoming grand opening? | excitement | excited and positive | 0.000 | 0.667 | +0.667 |
| C3 | q76 | 4 | What is the general sentiment about the upcoming grand opening? | excitement | excited and positive | 0.000 | 0.667 | +0.667 |
| G_raw | q61 | 4 | What did Gina receive from a dance contest? | a trophy | one of my trophies | 0.000 | 0.400 | +0.400 |
| C3 | q61 | 4 | What did Gina receive from a dance contest? | a trophy | one of my trophies | 0.000 | 0.400 | +0.400 |
| G_raw | q55 | 4 | What does Gina's tattoo symbolize? | Freedom and expressing herself through dance | freedom - dancing without worrying what people think | 0.154 | 0.333 | +0.179 |
| C3 | q55 | 4 | What does Gina's tattoo symbolize? | Freedom and expressing herself through dance | freedom - dancing without worrying what people think | 0.154 | 0.333 | +0.179 |
| G_raw | q49 | 4 | What did Gina want her customers to feel in her store? | cozy and comfortable | cozy and inviting | 0.667 | 0.500 | -0.167 |
| C3 | q49 | 4 | What did Gina want her customers to feel in her store? | cozy and comfortable | cozy and inviting | 0.667 | 0.500 | -0.167 |
| G_raw | q81 | 4 | What offer does Gina make to Jon regarding social media? | Helping with making content and managing his social media ac | help with making content or even managing your accounts | 0.526 | 0.667 | +0.140 |
| C3 | q81 | 4 | What offer does Gina make to Jon regarding social media? | Helping with making content and managing his social media ac | help with making content or even managing your accounts | 0.526 | 0.667 | +0.140 |
| G_raw | q53 | 4 | What did Gina say about creating an experience for her customers? | making them want to come back | Just creating an experience that'll make 'em wanna come back | 0.267 | 0.400 | +0.133 |
| C3 | q53 | 4 | What did Gina say about creating an experience for her customers? | making them want to come back | Just creating an experience that'll make 'em wanna come back | 0.267 | 0.400 | +0.133 |
