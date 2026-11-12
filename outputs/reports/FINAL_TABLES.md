# Final result tables (auto-generated from run artefacts)

Main QA columns exclude category-5 (adversarial) questions; those are scored as refusal.

## DEV

### Baselines — dev

| method | n | EM (ans.) | F1 (ans.) | R@5 | R@20 | Evidence full@4096 | Refusal (cat5) |
|---|---|---|---|---|---|---|---|
| dense | 502 | 0.241 | 0.451 | 0.822 | 0.985 | 0.771 | 0.963 |
| hipporag | 502 | 0.253 | 0.481 | 0.858 | 0.986 | 0.819 | 0.981 |
| fullcontext | 502 | 0.261 | 0.486 | N/A | N/A | N/A | 0.935 |

### Read-time — dev

| graph variant | n | EM (ans.) | F1 (ans.) | R@5 | R@20 | Evidence full@4096 | Refusal |
|---|---|---|---|---|---|---|---|
| cluster_hybrid_0.5_b8192_comp | 502 | 0.261 | 0.492 | 0.873 | 0.990 | 0.849 | 0.963 |
| cluster_hybrid_0.5_b8192_comp_composed | 502 | 0.266 | 0.501 | 0.869 | 0.991 | 0.845 | 0.972 |
| cluster_hybrid_0.5_b8192_comp_inferred | 502 | 0.253 | 0.494 | 0.875 | 0.990 | 0.841 | 0.963 |
| cluster_hybrid_0.5_b8192_triple | 502 | 0.253 | 0.488 | 0.868 | 0.990 | 0.837 | 0.944 |
| full_conv_comp | 502 | 0.268 | 0.501 | 0.883 | 0.988 | 0.847 | 0.963 |
| full_conv_triple | 502 | 0.271 | 0.500 | 0.881 | 0.988 | 0.845 | 0.963 |

### Question-time — dev

| strategy | n | EM (ans.) | F1 (ans.) | R@5 | R@20 | Evidence full@4096 |
|---|---|---|---|---|---|---|
| cluster_hybrid_0.5_b8192_comp_q3 | 105 | 0.235 | 0.411 | 0.941 | 1.000 | 0.933 |
| raw_q1 | 502 | 0.258 | 0.487 | 0.863 | 0.986 | 0.827 |
| raw_q2 | 502 | 0.258 | 0.487 | 0.875 | 0.991 | 0.835 |
| raw_q3 | 502 | 0.258 | 0.483 | 0.873 | 0.991 | 0.829 |

## TEST

### Baselines — test

| method | n | EM (ans.) | F1 (ans.) | R@5 | R@20 | Evidence full@4096 | Refusal (cat5) |
|---|---|---|---|---|---|---|---|
| dense | 1484 | 0.214 | 0.431 | 0.781 | 0.978 | 0.732 | 0.941 |
| hipporag | 1484 | 0.240 | 0.484 | 0.864 | 0.987 | 0.831 | 0.944 |
| fullcontext | 1484 | 0.239 | 0.493 | N/A | N/A | N/A | 0.909 |

### Read-time — test

| graph variant | n | EM (ans.) | F1 (ans.) | R@5 | R@20 | Evidence full@4096 | Refusal |
|---|---|---|---|---|---|---|---|
| cluster_hybrid_0.5_b8192_comp | 1484 | 0.238 | 0.484 | 0.881 | 0.986 | 0.842 | 0.941 |
| cluster_hybrid_0.5_b8192_comp_triple | 1484 | 0.241 | 0.490 | 0.875 | 0.985 | 0.838 | 0.938 |
| full_conv_comp | 1484 | 0.243 | 0.492 | 0.881 | 0.988 | 0.849 | 0.941 |
| full_conv_comp_triple | 1484 | 0.245 | 0.489 | 0.874 | 0.987 | 0.841 | 0.944 |

### Question-time — test

| strategy | n | EM (ans.) | F1 (ans.) | R@5 | R@20 | Evidence full@4096 |
|---|---|---|---|---|---|---|
| full_conv_comp_q1 | 1484 | 0.244 | 0.494 | 0.876 | 0.989 | 0.843 |
| full_conv_comp_q2 | 1484 | 0.239 | 0.491 | 0.885 | 0.990 | 0.851 |
| full_conv_comp_q3 | 1484 | 0.238 | 0.479 | 0.884 | 0.988 | 0.852 |
| raw_q1 | 1484 | 0.237 | 0.483 | 0.860 | 0.987 | 0.826 |
| raw_q2 | 1484 | 0.236 | 0.489 | 0.870 | 0.987 | 0.831 |
| raw_q3 | 1484 | 0.238 | 0.472 | 0.868 | 0.986 | 0.833 |

## Grouping

### Grouping (mean over dev+test conversations, all budgets)

| method | budget | convs | gold full | gold partial | cross-session co-location | avg group tokens | max group tokens |
|---|---|---|---|---|---|---|---|
| community | 4096 | 3 | 0.854 | 0.918 | 0.086 | 1430 | 3778 |
| community | 8192 | 3 | 0.854 | 0.918 | 0.086 | 1430 | 3778 |
| embedding | 4096 | 3 | 0.867 | 0.933 | 0.180 | 3528 | 4086 |
| embedding | 8192 | 3 | 0.884 | 0.946 | 0.259 | 6421 | 8137 |
| entity_idf | 4096 | 3 | 0.870 | 0.935 | 0.195 | 3210 | 3956 |
| entity_idf | 8192 | 3 | 0.905 | 0.958 | 0.431 | 6421 | 8173 |
| entity_jaccard | 4096 | 3 | 0.869 | 0.936 | 0.163 | 3528 | 3989 |
| entity_jaccard | 8192 | 3 | 0.886 | 0.948 | 0.249 | 6421 | 7978 |
| hybrid_0.25 | 4096 | 3 | 0.867 | 0.933 | 0.180 | 3528 | 4086 |
| hybrid_0.25 | 8192 | 3 | 0.884 | 0.946 | 0.259 | 6421 | 8137 |
| hybrid_0.5 | 4096 | 3 | 0.867 | 0.933 | 0.180 | 3528 | 4086 |
| hybrid_0.5 | 8192 | 10 | 0.881 | 0.946 | 0.281 | 6644 | 8184 |
| hybrid_0.75 | 4096 | 3 | 0.867 | 0.933 | 0.180 | 3528 | 4086 |
| hybrid_0.75 | 8192 | 3 | 0.884 | 0.946 | 0.259 | 6421 | 8137 |
| random | 4096 | 15 | 0.858 | 0.927 | 0.113 | 3528 | 4096 |
| random | 8192 | 15 | 0.883 | 0.947 | 0.304 | 6421 | 8107 |
| temporal | 4096 | 3 | 0.871 | 0.932 | 0.226 | 3528 | 3998 |
| temporal | 8192 | 3 | 0.883 | 0.948 | 0.305 | 6421 | 8099 |

### Comprehension audit

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
