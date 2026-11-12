# Retrieval Skill Evolution

The base ranking is the frozen COMP-P4_top5 HippoRAG2 ranking.  Retrieval skills only add queries; their dense session rankings are merged by RRF.  QDM and the evidence-sufficiency check are shared across static/evolved libraries; only the skill library differs.

## Adaptation outcome

- Adaptation questions: **43**
- Evidence already complete after the base + QDM/sufficiency pipeline: **38**
- Retrieval skills added: **1**; updated: **0**; rejected: **2**; candidate pass rate: **1/3**
- Questions whose gold evidence was recovered by the skill layer: **1**
- Questions whose gold evidence was lost relative to the base ranking: **1**
- Active retrieval skills in the raw final memory: **1**
- Snapshot chosen on development: **step 0**, deploying **0** active skills.

## Development snapshots (retrieval-only, no LLM)

| feedback | active skills | R@5 | evidence full@4096 | multi-evidence complete |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.967 | 0.933 | 0.667 |
| 10 | 1 | 0.933 | 0.867 | 0.333 |
| 20 | 1 | 0.933 | 0.867 | 0.333 |
| 30 | 1 | 0.933 | 0.867 | 0.333 |
| 40 | 1 | 0.933 | 0.867 | 0.333 |
| 43 | 1 | 0.933 | 0.867 | 0.333 |

## Retrieval skills

| skill_id | title | cats | v | status | queries | source qids |
|---|---|---|---:|---|---|---|
| ret_motivation_and_trigger_expansion_for_why | Motivation and Trigger Expansion for Why-Decisions | [1] | 1 | rejected_by_dev | {question} reason motivation because wanted to; {entities} passionate about love for unique interests blend; {entities} lost job unemployment before starting bu | [17] |

## Independent retrieval contribution (eval)

| Exp | R@3 | R@5 | evidence full@4096 | multi-evidence complete |
|---|---:|---:|---:|---:|
| E0 | 0.811 | 0.932 | 0.909 | 0.714 |
| E3 | 0.614 | 0.803 | 0.818 | 0.429 |
| E4 | 0.811 | 0.886 | 0.848 | 0.429 |

## Auxiliary: raw learned retrieval skill applied without the sufficiency gate

This is the skill *before* development snapshot selection; it is shown only as an ablation, not as the deployed E4/E6 memory.

- Active skills applied: 1
- Base R@5 -> applied R@5: 0.932 -> 0.879
- Base evidence full -> applied: 0.909 -> 0.848
- Recovered questions: 0; lost: 2


## Reading

- The learned skill passed adaptation verification but lowered development evidence coverage (step 0 vs later snapshots).  Snapshot selection therefore chose **step 0**; the skill is retained in the trace as a verified-but-rejected negative-transfer case, not deployed in E4/E6.
- The pipeline is capped at two retrieval rounds (QDM round, then one skill/missing-relation round).
- At evaluation time the sufficiency check sees only the question and retrieved sessions; gold evidence is never used to decide whether to retrieve again.
- Retrieval-skill verification is local and LLM-free: replay the source, similar and dissimilar training questions with dense retrieval only.
