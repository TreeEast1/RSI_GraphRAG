# DetectiveQA Round 2: diagnosis-first experiment plan

Date: 2026-11-12

## Why round 1 did not work

1. The main retrieval query used question + options.  Recent DetectiveQA RAG papers expose options only at answer time.
2. "Reader bottleneck" was overstated because `gold reached` meant any gold chunk, not all gold or answer-sufficient evidence.
3. Many cutoffs are late (15/41 >= 90% of the novel).  Recent text can contain the detective's explanation, so `long_prefix` measures reveal reading, not only cross-chunk reasoning.
4. Graph and dense were compared only at one HippoRAG setting.  We do not know whether facts, synonym edges or rank fusion explain the loss.
5. Read-time relations were too sparse (39 kept / 30 packets), the verifier rechecked quotes despite deterministic validation, `fixed_status` was ignored, and some predicates were relation-type labels.
6. Statistics were per-question bootstrap, while questions are clustered by novel.

## Isolated questions

| ID | Question | Main comparison |
|---|---|---|
| E1 | Does graph help under the paper-style q-only query? | q_dense vs q_graph vs q_rrf |
| E2 | Is graph loss from HippoRAG settings? | fact top-k 5/10/20, synonym on/off |
| E3 | Does recency explain QA? | recent, nonrecent_q_dense, recent_plus_q_dense |
| E4 | Are reader conclusions stable? | option order seed 0 vs seed 1 |
| E5 | Where do failures come from? | all-gold/partial/no-gold + LLM case judge + manual audit |
| E6 | Does read-time comprehension retrieve evidence? | v3 relation channel and RRF with q_dense/q_graph |
| E7 | Does injection into HippoRAG help? | R2 joint, R3 validated, R4 kept, R5 count-matched joint |

## Rules

- Primary deployable retrieval uses question only.  `qo_*` is diagnostic.
- Reader always gets 4096 estimated tokens and the same prompt.  Relation notes consume that same budget.
- Gold clues and gold answers are used only for metrics, oracle and post-hoc case judging.
- Read-time packets use OpenIE entity statistics only.  No questions, options, answers or clues.
- Report all 41 questions, dedup-36 sensitivity, and cutoff buckets: early < 70%, middle 70-90%, late >= 90%.
- Use novel-clustered bootstrap and McNemar.  With six novels, all results are exploratory.

## Stop / go gates

- Graph route: q_graph or q_rrf must beat q_dense on Recall@10 or All@20 with a positive clustered CI lower bound.  If only MRR improves, do not claim graph help.
- Relation route: relation channel must touch gold for at least 25% of questions and improve q_rrf Recall@10 by >= 3pp, or produce clear case-level wins in early/middle buckets.
- Scale-up: graph/relation gains must appear outside late-cutoff questions and survive option-order seed 1.

## Required case study

Manually inspect every item in these queues: dense_graph_disagree, recent_only, recent_correct_without_gold, dense_wrong_all_gold, oracle_wrong, memory_correct.  For each case record evidence location, failure label, whether the case is a benchmark artifact, and the concrete method change it suggests.
