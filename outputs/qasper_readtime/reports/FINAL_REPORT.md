# FINAL REPORT — QASPER Read-Time Comprehension

## 0. What was run

- validation split parsed: 281 papers, 1005 questions; frozen 100-paper sample (seed 20261110) with 372 questions, 333 answerable with mapped gold, 141 multi-evidence, 56 cross-section.
- read-time: 100 papers, 10539 atomic facts, 1313 candidate insights, 733 verified units, 580 rejected.
- graphs: G_raw per paper + M2 (atomic) and M3 (atomic+comprehension triples) variants.
- retrieval: M0-M6 on paragraphs; QA: fixed 4096-token reader budget with an oracle-context reference.
- **scope**: 100 papers with complete G_raw + variants + read-time outputs (333 answerable questions; 372 QA questions). M0 baselines cover the same 100 sampled papers.

### Headline
- M3 comprehension triples are neutral-to-slightly-positive overall (All@5 0.556 vs M2 0.545, +1.1pp, 95% CI [-0.2,+2.4]) and clearly positive on the questions their sources cover.
- M4/M5 comprehension-unit fusion is strongly negative overall (All@5 0.401 vs M2 0.545) because units cover only part of the evidence space and equal-weight RRF displaces correct graph evidence on uncovered questions.
- M6 full per-paragraph summaries recover dense-level retrieval (All@5 0.575), while M6 with the same number of units as insights does not (0.360): the driver is coverage, not comprehension.
- Graph retrieval itself does not beat dense on this task: M1 All@5 0.544 vs dense 0.593; the graph layers change recall marginally, not the ranking ceiling.
- Oracle-context QA is only 0.487 vs dense 0.464: on this reader, retrieval differences have little room to change Answer F1.

## 1. Does QASPER fix the LoCoMo small-pool problem?

Partially. The paragraph pool is mean 46.95, median 43.0, P90 70.0 per paper — roughly 2-3× LoCoMo's 19-session pool, and top-5 is 14.0% of the pool. On the full 100-paper sample, dense Any@5 is 74.0%, so single-evidence retrieval is close to saturated; All@5 = 59.3% and the multi-evidence subset (39.4% All@5) are the discriminating conditions. This is a harder retrieval task than LoCoMo, but not a large-corpus task.

## 2. Does read-time comprehension improve gold-evidence recall?

| method | Any@5 | All@5 | Recall@5 | MRR | nDCG@10 | Cov@4096 |
|---|---|---|---|---|---|---|
| M1 G_raw | 0.692 | 0.544 | 0.606 | 0.461 | 0.517 | 0.883 |
| M2 +Atomic | 0.697 | 0.545 | 0.608 | 0.477 | 0.527 | 0.884 |
| M3 +Triples | 0.707 | 0.556 | 0.620 | 0.493 | 0.540 | 0.886 |
| M4 +Units | 0.545 | 0.401 | 0.461 | 0.375 | 0.414 | 0.842 |
| M5 Triples+Units | 0.535 | 0.395 | 0.454 | 0.375 | 0.413 | 0.836 |
| M6 +Summaries | 0.709 | 0.575 | 0.635 | 0.521 | 0.555 | 0.884 |
| M6 matched count | 0.518 | 0.360 | 0.425 | 0.337 | 0.395 | 0.853 |

Paired question-level deltas (95% bootstrap CI):

- m3_atom_comp_vs_m2_atom: Any@5 +0.011 [-0.003,+0.026], All@5 +0.011 [-0.002,+0.024], Recall@5 +0.012
- m4_units_vs_m2_atom: Any@5 -0.152 [-0.209,-0.092], All@5 -0.144 [-0.197,-0.090], Recall@5 -0.148
- m5_units_graph_vs_m3_atom_comp: Any@5 -0.173 [-0.227,-0.116], All@5 -0.161 [-0.213,-0.107], Recall@5 -0.166
- m6_matched_vs_m2_atom: Any@5 -0.179 [-0.228,-0.129], All@5 -0.185 [-0.233,-0.135], Recall@5 -0.183
- m5_units_graph_vs_m2_atom: Any@5 -0.162 [-0.218,-0.102], All@5 -0.150 [-0.203,-0.096], Recall@5 -0.155

## 3. Which representation is most effective: triples, units, or both?

- M3 +Triples: All@5 0.556, Any@5 0.707, Recall@5 0.620, Cov@4096 0.886.
- M4 +Units: All@5 0.401, Any@5 0.545, Recall@5 0.461, Cov@4096 0.842.
- M5 Triples+Units: All@5 0.395, Any@5 0.535, Recall@5 0.454, Cov@4096 0.836.
- M6 matched count: All@5 0.360, Any@5 0.518, Recall@5 0.425, Cov@4096 0.853.

Fusion-weight sensitivity (RRF weight on the unit list; graph weight fixed at 1.0):

| base | unit weight | n | Any@5 | All@5 | Recall@5 |
|---|---|---|---|---|---|
| m2_atom | 0.10 | 333 | 0.691 | 0.529 | 0.596 |
| m3_atom_comp | 0.10 | 333 | 0.694 | 0.539 | 0.605 |
| m2_atom | 0.20 | 333 | 0.596 | 0.438 | 0.503 |
| m3_atom_comp | 0.20 | 333 | 0.590 | 0.432 | 0.496 |
| m2_atom | 0.30 | 333 | 0.559 | 0.413 | 0.472 |
| m3_atom_comp | 0.30 | 333 | 0.553 | 0.411 | 0.470 |
| m2_atom | 0.50 | 333 | 0.557 | 0.411 | 0.471 |
| m3_atom_comp | 0.50 | 333 | 0.545 | 0.405 | 0.463 |
| m2_atom | 0.75 | 333 | 0.557 | 0.413 | 0.472 |
| m3_atom_comp | 0.75 | 333 | 0.544 | 0.404 | 0.462 |
| m2_atom | 1.00 | 333 | 0.545 | 0.401 | 0.461 |
| m3_atom_comp | 1.00 | 333 | 0.535 | 0.395 | 0.454 |

Even a small fixed weight (0.1) is below graph-only on the full evaluated subset (Any@5 0.691 vs 0.697; All@5 0.529 vs 0.545); at 0.2+ the unit list floods the top ranks and performance collapses. Fixed weighting cannot recover the signal - the unit channel has to be gated by relevance. The primary M4/M5 definition uses equal weights, which is the pre-registered comparison.

### Reading-group comparison (exploratory, 20 papers / 72 questions)

| grouping | papers | units | unit gold coverage | Any@5 | All@5 | Recall@5 |
|---|---|---|---|---|---|---|
| section | 20 | 164 | 0.722 | 0.694 | 0.521 | 0.596 |
| semantic | 20 | 144 | 0.694 | 0.639 | 0.465 | 0.531 |

Section-based groups produce more units and slightly better M2+unit fusion than semantic k-means groups on this subset. This is exploratory: no QA was run for the semantic grouping.

## 4. How much of the gain is independent of ordinary index expansion?

- M6 with the same number of summary units as verified insights: All@5 0.360 vs M4 0.401 vs M3 0.556. The M6-matched condition controls both the number of indexed units and the reader budget, so a gain that survives it is not just index expansion.

## 5. Cross-paragraph / multi-evidence questions

| subset | method | Any@5 | All@5 | Recall@5 | Cov@4096 |
|---|---|---|---|---|---|
| multi | M1 G_raw | 0.691 | 0.340 | 0.488 | 0.759 |
| multi | M2 +Atomic | 0.716 | 0.358 | 0.508 | 0.762 |
| multi | M3 +Triples | 0.720 | 0.362 | 0.514 | 0.766 |
| multi | M4 +Units | 0.557 | 0.216 | 0.358 | 0.713 |
| multi | M5 Triples+Units | 0.539 | 0.209 | 0.348 | 0.713 |
| multi | M6 matched count | 0.571 | 0.199 | 0.352 | 0.716 |
| cross | M1 G_raw | 0.679 | 0.250 | 0.436 | 0.750 |
| cross | M2 +Atomic | 0.741 | 0.312 | 0.499 | 0.759 |
| cross | M3 +Triples | 0.741 | 0.312 | 0.499 | 0.768 |
| cross | M4 +Units | 0.571 | 0.161 | 0.333 | 0.670 |
| cross | M5 Triples+Units | 0.554 | 0.152 | 0.320 | 0.670 |
| cross | M6 matched count | 0.554 | 0.161 | 0.324 | 0.679 |

## 6. QA results

| method | n | AnswerF1 | EvidenceF1 | EM | F1 ans | F1 multi | Gold all |
|---|---|---|---|---|---|---|---|
| M0 Dense | 372 | 0.464 | 0.119 | 0.210 | 0.436 | 0.428 | 0.763 |
| M1 G_raw | 372 | 0.455 | 0.119 | 0.202 | 0.427 | 0.431 | 0.758 |
| M2 +Atomic | 372 | 0.457 | 0.119 | 0.199 | 0.429 | 0.426 | 0.761 |
| M3 +Triples | 372 | 0.454 | 0.119 | 0.204 | 0.425 | 0.422 | 0.763 |
| M4 +Units | 372 | 0.435 | 0.113 | 0.191 | 0.405 | 0.402 | 0.707 |
| M5 Triples+Units | 372 | 0.443 | 0.112 | 0.194 | 0.414 | 0.411 | 0.702 |
| m5_units_graph_B | 372 | 0.443 | 0.113 | 0.191 | 0.414 | 0.402 | 0.702 |
| oracle | 372 | 0.487 | 0.955 | 0.242 | 0.455 | 0.455 | 0.895 |
| full | 372 | 0.476 | 0.073 | 0.220 | 0.448 | 0.455 | 0.890 |

QA by evidence type:

| group | method | n | AnswerF1 |
|---|---|---|---|
| text_evidence | M0 Dense | 339 | 0.438 |
| figure_table_only | M0 Dense | 11 | 0.369 |
| unanswerable | M0 Dense | 22 | 0.909 |
| text_evidence | M1 G_raw | 339 | 0.429 |
| figure_table_only | M1 G_raw | 11 | 0.369 |
| unanswerable | M1 G_raw | 22 | 0.909 |
| text_evidence | M2 +Atomic | 339 | 0.431 |
| figure_table_only | M2 +Atomic | 11 | 0.369 |
| unanswerable | M2 +Atomic | 22 | 0.909 |
| text_evidence | M3 +Triples | 339 | 0.427 |
| figure_table_only | M3 +Triples | 11 | 0.369 |
| unanswerable | M3 +Triples | 22 | 0.909 |
| text_evidence | M4 +Units | 339 | 0.407 |
| figure_table_only | M4 +Units | 11 | 0.369 |
| unanswerable | M4 +Units | 22 | 0.909 |
| text_evidence | M5 Triples+Units | 339 | 0.415 |
| figure_table_only | M5 Triples+Units | 11 | 0.369 |
| unanswerable | M5 Triples+Units | 22 | 0.909 |
| text_evidence | oracle | 339 | 0.458 |
| figure_table_only | oracle | 11 | 0.364 |
| unanswerable | oracle | 22 | 1.000 |

QA deltas (paired by question):

- m3_atom_comp_vs_m2_atom: Answer F1 -0.003 [-0.015,+0.008], Evidence F1 +0.001
- m4_units_vs_m2_atom: Answer F1 -0.022 [-0.039,-0.006], Evidence F1 -0.006
- m5_units_graph_vs_m3_atom_comp: Answer F1 -0.011 [-0.028,+0.007], Evidence F1 -0.007
- m5_units_graph_vs_m2_atom: Answer F1 -0.014 [-0.031,+0.002], Evidence F1 -0.006
- m5_units_graph_B_vs_m5_units_graph: Answer F1 -0.000 [-0.018,+0.019], Evidence F1 +0.001

Top-k reader sensitivity (same question set, M5 rankings, no token padding):

| k | n | AnswerF1 | EvidenceF1 | EM |
|---|---|---|---|---|
| 3 | 333 | 0.311 | 0.232 | 0.132 |
| 5 | 333 | 0.327 | 0.197 | 0.129 |

Answer F1 falls sharply at k=3/5, while Evidence F1 *rises*: the 4096-token context predicts many paragraphs, which depresses official paragraph-F1 precision. Retrieval-side metrics, not QA Evidence F1, are the reliable evidence measure.

## 7. Mechanism

- verified units: 733; never retrieved as a candidate: 0; never in top-5: 136; touch gold: 229; touch gold but never top-5: 21.
- questions with at least one gold-touching unit: 193 of 372.
- unit-layer gold coverage upper bound: 58.0% of questions.

### Coverage-limited mechanism check

| subset | method | n | Any@5 | All@5 | Recall@5 |
|---|---|---|---|---|---|
| unit_covered | M2 +Atomic | 193 | 0.746 | 0.567 | 0.640 |
| unit_covered | M3 +Triples | 193 | 0.759 | 0.585 | 0.659 |
| unit_covered | M4 +Units | 193 | 0.837 | 0.627 | 0.716 |
| unit_covered | M5 Triples+Units | 193 | 0.829 | 0.619 | 0.709 |
| unit_covered | M5 gated units | 193 | 0.798 | 0.583 | 0.675 |
| unit_covered | M6 matched count | 193 | 0.492 | 0.324 | 0.396 |
| unit_uncovered | M2 +Atomic | 140 | 0.629 | 0.514 | 0.564 |
| unit_uncovered | M3 +Triples | 140 | 0.636 | 0.514 | 0.566 |
| unit_uncovered | M4 +Units | 140 | 0.143 | 0.089 | 0.109 |
| unit_uncovered | M5 Triples+Units | 140 | 0.129 | 0.086 | 0.102 |
| unit_uncovered | M5 gated units | 140 | 0.243 | 0.179 | 0.201 |
| unit_uncovered | M6 matched count | 140 | 0.554 | 0.411 | 0.466 |

`unit_covered` contains questions whose gold evidence is touched by at least one verified unit's source paragraph (upper bound, before ranking); `unit_uncovered` contains the rest. If M4/M5 only move the covered subset, the limiting factor is coverage of the generated knowledge rather than fusion.

### QA on unit-covered vs uncovered questions

| subset | method | n | AnswerF1 | EvidenceF1 |
|---|---|---|---|---|
| unit_covered | M2 +Atomic | 193 | 0.461 | 0.141 |
| unit_uncovered | M2 +Atomic | 140 | 0.400 | 0.121 |
| unit_covered | M3 +Triples | 193 | 0.467 | 0.142 |
| unit_uncovered | M3 +Triples | 140 | 0.384 | 0.121 |
| unit_covered | M4 +Units | 193 | 0.440 | 0.139 |
| unit_uncovered | M4 +Units | 140 | 0.371 | 0.108 |
| unit_covered | M5 Triples+Units | 193 | 0.448 | 0.139 |
| unit_uncovered | M5 Triples+Units | 140 | 0.381 | 0.107 |
| unit_covered | m5_units_graph_B | 193 | 0.454 | 0.141 |
| unit_uncovered | m5_units_graph_B | 140 | 0.372 | 0.107 |

## 8. Cost

- read-time tokens: prompt 2,883,843, completion 1,620,654.
- graph OpenIE and reader token usage are logged under `outputs/qasper_readtime/logs/` and each HippoRAG `save_dir/llm_cache`.

## 9. Honest limitations

- Paragraph pool is small; Any-Evidence@5 saturation is a real ceiling.
- Table/figure evidence is excluded from text-evidence metrics and reported separately.
- Hyperedges are represented as triples injected into each source paragraph; this is an approximation of a true hypergraph.
- Section-based reading groups are the primary condition. The semantic k-means grouping with a small overlap band was run on 20 papers (`comprehension/semantic/`) for an exploratory comparison only; no semantic QA was run.
- Table/figure evidence: 253 `FLOAT SELECTED` evidence items across the validation split; questions whose only evidence is a figure/table (3.5%) are excluded from text-evidence retrieval metrics but remain in QA.
- Top-3/Top-5 reader sensitivity was run on the evaluated question set; it is not a fixed-budget comparison because no token padding is added.
- HY4 is a shared company endpoint; all reported runs use a cross-process limiter at ~1 QPS. Generating the full 100-paper sample was quota-bound (many hours) and latency varies; the pipeline is resumable and every call is cached.

## 10. Next mechanism worth studying

The mechanism table answers this directly. 229 of 733 verified units touch gold evidence at least once, and only 21 touch gold but never reach the unit top-5: unit **ranking** is not the main failure. The failures are (a) **coverage**: only 58.0% of questions have any unit whose source is gold, so most questions cannot benefit, and (b) **displacement**: equal-weight RRF still injects unit passages on those uncovered questions and pushes correct graph evidence out. The next experiment should therefore (1) generate units that cover every section, not ~7 per paper, and (2) gate the unit channel by a calibrated relevance score instead of a fixed RRF weight; the fusion sweep shows that even 0.1 is already below graph-only and 0.2+ collapses. Reader-side unit display (mode B) adds nothing, so units should be used to route to paragraphs, not shown as evidence.