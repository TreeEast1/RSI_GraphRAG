# Comprehension quality audit

Classes: A = duplicate of G_raw, B = paraphrase of G_raw, C = single-session, D = cross-session supported, E = low-confidence inference.

| conv | scope | relations | A | B | C | D | E | %D |
|---|---|---|---|---|---|---|---|---|
| conv-26 | cluster_hybrid_0.5_b8192_comp | 28 | 0 | 0 | 0 | 28 | 0 | 1.000 |
| conv-30 | cluster_hybrid_0.5_b8192_comp | 26 | 0 | 0 | 0 | 26 | 0 | 1.000 |
| conv-30 | cluster_temporal_b8192_comp | 22 | 0 | 0 | 0 | 22 | 0 | 1.000 |
| conv-30 | full_conv_comp | 18 | 0 | 0 | 0 | 17 | 1 | 0.944 |
| conv-41 | cluster_hybrid_0.5_b8192_comp | 82 | 0 | 0 | 0 | 80 | 2 | 0.976 |
| conv-42 | cluster_hybrid_0.5_b8192_comp | 51 | 0 | 0 | 0 | 51 | 0 | 1.000 |
| conv-43 | cluster_hybrid_0.5_b8192_comp | 155 | 0 | 1 | 0 | 140 | 14 | 0.903 |
| conv-44 | cluster_hybrid_0.5_b8192_comp | 27 | 0 | 0 | 0 | 27 | 0 | 1.000 |
| conv-44 | cluster_temporal_b8192_comp | 35 | 0 | 0 | 0 | 34 | 1 | 0.971 |
| conv-44 | full_conv_comp | 66 | 0 | 0 | 0 | 64 | 2 | 0.970 |
| conv-47 | cluster_hybrid_0.5_b8192_comp | 52 | 0 | 0 | 0 | 52 | 0 | 1.000 |
| conv-48 | cluster_hybrid_0.5_b8192_comp | 67 | 0 | 0 | 0 | 63 | 4 | 0.940 |
| conv-49 | cluster_hybrid_0.5_b8192_comp | 46 | 0 | 0 | 0 | 46 | 0 | 1.000 |
| conv-50 | cluster_hybrid_0.5_b8192_comp | 42 | 0 | 0 | 0 | 38 | 4 | 0.905 |