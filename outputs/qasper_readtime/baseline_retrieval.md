# Baseline Retrieval on QASPER (paragraph units)

Scope: frozen 100-paper validation sample (seed 20261110), answerable questions with mapped text evidence. Passage = original paragraph; retrieval restricted to the question's paper. Multiple annotators are alternatives: every metric is the mean over annotator evidence sets.

## Overall

| method | n | Any@1 | Any@3 | Any@5 | All@1 | All@3 | All@5 | Recall@5 | MRR | nDCG@10 | Cov@4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 BM25 | 333 | 0.204 | 0.362 | 0.521 | 0.135 | 0.251 | 0.378 | 0.435 | 0.341 | 0.369 | 0.716 |
| M0 Dense | 333 | 0.363 | 0.628 | 0.740 | 0.251 | 0.470 | 0.593 | 0.658 | 0.530 | 0.563 | 0.889 |
| M0 Hybrid | 333 | 0.312 | 0.566 | 0.680 | 0.216 | 0.420 | 0.517 | 0.582 | 0.476 | 0.501 | 0.841 |

## Subsets

| subset | n | Any@5 | All@5 | Recall@5 | Cov@4096 |
|---|---|---|---|---|---|
| all | 333 | 0.740 | 0.593 | 0.658 | 0.889 |
| single | 219 | 0.733 | 0.733 | 0.733 | 0.970 |
| multi | 141 | 0.741 | 0.394 | 0.547 | 0.773 |
| cross | 56 | 0.750 | 0.295 | 0.497 | 0.786 |
| multi_annot | 98 | 0.709 | 0.444 | 0.563 | 0.827 |

## Difficulty reading

- paragraph pool per paper: mean 46.95, P10 22.0, P50 43.0, P90 70.0; top-5 covers 14.0% of the pool on average.
- dense Any@5 = 0.740, All@5 = 0.593; the gap between Any and All is the part that is not saturated.
- BM25 is far below dense, so the hybrid RRF with equal weights is diluted by BM25; dense is the M0 reference.

No window-level retrieval was added: the paragraph pool already has a median of 43.0 units and the multi-evidence / cross-section subsets provide the discriminating signal; splitting paragraphs would inflate recall artificially.
