# LoCoMo Dataset Audit

Source: `/group/40150/licochen/Memory/RSI_GraphRAG/data/raw/locomo/locomo10.json`
Split seed: 20240501 — dev=['conv-30', 'conv-44', 'conv-48'], test=['conv-26', 'conv-41', 'conv-42', 'conv-43', 'conv-47', 'conv-49', 'conv-50']

## Conversation summary

| conversation | split | sessions | turns | QA | avg session tokens | max | min |
|---|---|---|---|---|---|---|---|
| conv-26 | test | 19 | 419 | 199 | 771.1 | 1168 | 434 |
| conv-30 | dev | 19 | 369 | 105 | 601.8 | 960 | 373 |
| conv-41 | test | 32 | 663 | 193 | 686.8 | 1157 | 432 |
| conv-42 | test | 29 | 629 | 260 | 655.7 | 1231 | 318 |
| conv-43 | test | 29 | 680 | 242 | 735.3 | 1257 | 379 |
| conv-44 | dev | 28 | 675 | 158 | 750.2 | 1180 | 451 |
| conv-47 | test | 31 | 689 | 190 | 657.7 | 1082 | 417 |
| conv-48 | dev | 30 | 681 | 239 | 654.2 | 1397 | 367 |
| conv-49 | test | 25 | 509 | 196 | 653.0 | 921 | 469 |
| conv-50 | test | 30 | 568 | 204 | 683.8 | 1294 | 366 |

## Totals

- Conversations: 10
- Sessions (documents): 272
- Turns: 5882
- QA items: 1986
- Session tokens: mean=684.8, median=642.0, max=1397, total=186262 (cl100k proxy)

## QA categories

| category | name | all | dev | test |
|---|---|---|---|---|
| 1 | multi-hop | 282 | 62 | 220 |
| 2 | temporal | 321 | 92 | 229 |
| 3 | open-domain | 96 | 17 | 79 |
| 4 | single-hop | 841 | 224 | 617 |
| 5 | adversarial | 446 | 107 | 339 |

## Evidence

- QA rows with evidence: 1982 / 1986
- QA rows without evidence: 4
- Unmapped evidence dia_ids: 8
- QA spanning >1 session: 328 (16.5%)
- Mean evidence items per QA: 1.42

| #evidence | QA count |
|---|---|
| 0 | 4 |
| 1 | 1559 |
| 2 | 239 |
| 3 | 83 |
| 4 | 58 |
| 5 | 20 |
| 6 | 6 |
| 7 | 7 |
| 8 | 2 |
| 9 | 2 |
| 10 | 2 |
| 11 | 2 |
| 17 | 1 |
| 19 | 1 |

## Notes

- Category 5 (adversarial) rows have no gold `answer`; they carry `adversarial_answer` and must be scored as refusal/robustness, not EM/F1.
- `session_summary`, `observation` and `event_summary` fields are withheld from all indexing paths to avoid leakage; they may only be used for analysis.
- Token counts use cl100k_base as a budget proxy; actual HY4 tokenisation may differ.
