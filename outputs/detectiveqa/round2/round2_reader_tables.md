# Round 2 reader tables

n_rows=820

## Accuracy by condition and seed

| condition | seed0 acc | seed0 gold-any | seed0 gold-all | seed1 acc | seed1 gold-any | seed1 gold-all |
|---|---:|---:|---:|---:|---:|---:|
| nonrecent_q_dense | 0.439 | 0.585 | 0.000 | 0.390 | 0.585 | 0.000 |
| options_only | 0.293 | 0.000 | 0.000 | 0.439 | 0.000 | 0.000 |
| options_only_memory | 0.463 | 0.000 | 0.000 | 0.439 | 0.000 | 0.000 |
| oracle | 0.805 | 1.000 | 0.927 | 0.707 | 1.000 | 0.927 |
| q_dense | 0.658 | 0.756 | 0.122 | 0.585 | 0.756 | 0.122 |
| q_graph | 0.683 | 0.780 | 0.146 | 0.658 | 0.780 | 0.146 |
| q_rrf | 0.634 | 0.780 | 0.122 | 0.634 | 0.780 | 0.122 |
| qo_dense | 0.707 | 0.732 | 0.220 | 0.707 | 0.732 | 0.220 |
| recent | 0.805 | 0.512 | 0.122 | 0.805 | 0.512 | 0.122 |
| recent_plus_q_dense | 0.805 | 0.683 | 0.195 | 0.732 | 0.683 | 0.195 |

## Paired tests (novel-clustered bootstrap; seed 0)

| comparison | delta | 95% CI | wins | losses | McNemar p |
|---|---:|---|---:|---:|---:|
| q_dense->q_graph | +0.0244 | [+0.0000,+0.0811] | 2 | 1 | 1 |
| q_dense->q_rrf | -0.0244 | [-0.0750,+0.0000] | 1 | 2 | 1 |
| q_dense->recent | +0.1463 | [+0.0571,+0.2143] | 8 | 2 | 0.1094 |
| q_dense->nonrecent_q_dense | -0.2195 | [-0.3333,-0.0833] | 1 | 10 | 0.01172 |
| q_dense->recent_plus_q_dense | +0.1463 | [+0.0426,+0.2821] | 7 | 1 | 0.07031 |
| q_dense->qo_dense | +0.0488 | [+0.0000,+0.1176] | 3 | 1 | 0.625 |
| q_dense->oracle | +0.1463 | [+0.0426,+0.2727] | 9 | 3 | 0.146 |
| q_dense->options_only_memory | -0.1951 | [-0.3409,+0.0000] | 4 | 12 | 0.07681 |
| recent->nonrecent_q_dense | -0.3659 | [-0.5116,-0.1714] | 3 | 18 | 0.00149 |
| recent->recent_plus_q_dense | +0.0000 | [-0.1064,+0.1081] | 2 | 2 | 1 |

## Option-order agreement

- nonrecent_q_dense: 0.951 (41 pairs)
- options_only: 0.488 (41 pairs)
- options_only_memory: 0.732 (41 pairs)
- oracle: 0.854 (41 pairs)
- q_dense: 0.902 (41 pairs)
- q_graph: 0.902 (41 pairs)
- q_rrf: 0.902 (41 pairs)
- qo_dense: 0.805 (41 pairs)
- recent: 0.976 (41 pairs)
- recent_plus_q_dense: 0.902 (41 pairs)

## Relation QA

| condition | n | accuracy | gold any |
|---|---:|---:|---:|
| q_dense | 41 | 0.658 | 0.756 |
| q_rrf | 41 | 0.634 | 0.780 |
| relations_q_dense | 41 | 0.658 | 0.756 |
| relations_q_rrf | 41 | 0.658 | 0.707 |

| comparison | delta | 95% CI | wins | losses | McNemar p |
|---|---:|---|---:|---:|---:|
| q_dense->relations_q_dense | +0.0000 | [-0.0698,+0.0732] | 2 | 2 | 1 |
| q_rrf->relations_q_rrf | +0.0244 | [-0.0588,+0.1053] | 2 | 1 | 1 |
| q_dense->q_rrf | -0.0244 | [-0.0750,+0.0000] | 1 | 2 | 1 |
| relations_q_dense->relations_q_rrf | +0.0000 | [-0.0769,+0.0682] | 2 | 2 | 1 |
