# Comprehension Quality

Grouping: **section**. Read-time calls see only the paper title, paragraphs and atomic facts.

## Volume and cost

| item | value |
|---|---|
| papers processed | 100 |
| reading groups | 147 |
| atomic facts | 10539 |
| candidate insights | 1313 |
| verified insights / units | 733 |
| rejected insights | 580 |
| verifier retention | 55.8% |
| units per paper | 7.3 |
| paragraphs that are a unit source (per paper, mean) | 30.4% |
| unit length (words, mean / P90) | 24.6 / 29.0 |
| units sourced from >1 section | 71.2% |
| units sourced from adjacent paragraphs | 18.3% |
| source paragraph gap (mean / P90) | 9.9 / 25.0 |
| paragraph summaries (M6) | 4561 |
| read-time prompt tokens | 2,883,843 |
| read-time completion tokens | 1,620,654 |

## Relation types (verified units)

| relation_type | count |
|---|---|
| claim_evidence | 150 |
| method_result | 135 |
| problem_solution | 119 |
| setting_outcome | 108 |
| combined_fact | 95 |
| assumption_limitation | 65 |
| entity_resolution | 61 |

## Relation-type effectiveness (joined with mechanism tracking)

| relation_type | units | touches gold | ever in unit top-5 | touches gold but never top-5 | QA helped | QA harmed |
|---|---|---|---|---|---|---|
| claim_evidence | 150 | 46 | 120 | 4 | 44 | 49 |
| method_result | 135 | 44 | 117 | 5 | 37 | 50 |
| problem_solution | 119 | 35 | 92 | 2 | 25 | 45 |
| setting_outcome | 108 | 44 | 97 | 2 | 36 | 40 |
| combined_fact | 95 | 25 | 75 | 3 | 23 | 36 |
| assumption_limitation | 65 | 11 | 47 | 2 | 16 | 15 |
| entity_resolution | 61 | 24 | 49 | 3 | 16 | 24 |

## Verifier rejections

| category | count |
|---|---|
| restatement_of_one_paragraph | 207 |
| unsupported_causality | 143 |
| other | 116 |
| unsupported_content | 114 |

## Does the unit layer cover gold evidence at all?

- questions whose gold evidence touches the source of at least one verified unit: **193/333 = 58.0%** (upper bound for M4/M5, before any retrieval ranking).
- units generated: 733; never a retrieval candidate: 0; never in top-5: 136; touch gold: 229; touch gold but never top-5: 21.
- units on questions whose M5 QA improved: 197; units on questions whose M5 QA got worse: 259.

## Examples of verified units

- `1503.00841::u000` [problem_solution] The bias problem in prior knowledge motivates three auxiliary regularization terms in the Generalized Expectation Criteria framework to make GE-FL more robust. (sources: 1503.00841::s00::p002, 1503.00841::s00::p003, 1503.00841::s00::p005, 1503.00841::s01::p000)
- `1503.00841::u002` [setting_outcome] With unbalanced labeled features on balanced and unbalanced movie datasets, incorporating the KL divergence of class distribution performs much better than GE-FL. (sources: 1503.00841::s07::p001, 1503.00841::s07::p002, 1503.00841::s07::p005)
- `1503.00841::u003` [setting_outcome] When balanced labeled features are used on the unbalanced movie dataset, incorporating KL divergence is much better than GE-FL because it provides the true class distribution. (sources: 1503.00841::s08::p001, 1503.00841::s08::p002, 1503.00841::s08::p003)
- `1503.00841::u006` [claim_evidence] LDA selects predictive labeled features without considering class balance, and GE-FL does not control this issue, so auxiliary regularization terms promote the model significantly. (sources: 1503.00841::s11::p001, 1503.00841::s11::p002)
- `1503.00841::u007` [entity_resolution] The top 10 frequent words after removing stop words are used as neutral features in all experiments, implementing the proposed approach without manual annotation. (sources: 1503.00841::s00::p004, 1503.00841::s04::p004, 1503.00841::s06::p004)

## Examples of rejected candidates

- Because neutral features are assumed uniformly distributed over class labels, maximum entropy and neutral features both mislead the model on the unbalanced movie dataset.  
  reason: Sources say neutral features assume uniform class labels and that maximum entropy/neutral features are worse on unbalanced data because forcing uniform distribution misleads the model, but they do not say maximum entropy is based on the neutral-feature uniform assumption or that both mislead due to that assumption.
- On movie datasets with class distributions 1:2, 1:3, and 1:4, incorporating KL divergence increases remarkably while neutral features and maximum entropy decrease significantly.  
  reason: Sources support KL divergence increasing remarkably while neutral features and maximum entropy decrease as class distribution becomes more unbalanced, but the concise_text/triple claim direct outperformance on 1:2, 1:3, and 1:4 datasets, which is not explicitly stated.
- Using LDA to select labeled features without instance labels, the three methods significantly outperform GE-FL on all the 9 datasets.  
  reason: The cited passages say the three methods significantly outperform GE-FL overall, but then specify each method is better on only 7, 8, or 7 of the 9 datasets; 'on all the 9 datasets' is unsupported.
- GE-FL is the baseline that leverages labeled features as prior knowledge, and the paper addresses the robustness problem on top of GE-FL.  
  reason: The concise_text is a restatement of s01:p000, and the triple's predicate 'requires' is not supported: GE-FL is the baseline/method on top of which the robustness problem is addressed, not something requiring that problem.
- The multilingual model uses word alignments between sentences in a parallel corpus to couple monolingual models, but only 8% of English arguments and 17% of German arguments are aligned.  
  reason: Passages state the model uses alignments and that only 8%/17% of arguments are aligned, but do not support that these percentages 'limit' the model's use of alignments as a causal relation; this is inferred.