# QASPER Dataset Audit

Source: `qasper-dev-v0.3.json` (official validation split, QASPER v0.3).

Passage unit: one **paragraph** (`paper_id::s{section:02d}::p{para:03d}`); retrieval is always restricted to the question's own paper.

## Corpus

- papers: **281**, questions: **1005**, answerable: **945**, unanswerable: **60**
- evidence-bearing answerable questions: **888**
- gold paragraphs per question/annotator: mean 1.75, P10 1.0, P50 1.0, P90 3.0, max 19
- gold-paragraph count distribution: 1→777, 2→223, 3→61, 4→49, 5→23, 6→9, 7→11, 8→8, 9→7, 10→1, 11→1, 12→1, 14→1, 16→1, 19→1
- single-paragraph gold: **63.2%**; multi-paragraph: **36.8%**; cross-section: **19.0%**
- questions whose evidence is figures/tables only: **3.5%**
- annotations per question: {1: 261, 2: 729, 3: 15}

## Candidate pool (paragraphs per paper)

mean **46.95**, P10 **22.0**, P50 **43.0**, P90 **70.0**, max **281**
Paper length: mean 4680.71 tokens, P50 4438.0, P90 6636.0.
Paragraph length: mean 99.7 tokens, P50 86.0, P90 193.0, max 1490.
Gold evidence paragraph length: mean 120.48 tokens, P50 106.0, P90 221.0, max 1182.

### Questions by paper-length quintile

| quintile | paper tokens | papers | questions | single | multi | cross-section | mean gold |
|---|---|---|---|---|---|---|---|
| 1 | [796, 2851] | 56 | 158 | 63.3% | 36.7% | 18.4% | 1.89 |
| 2 | [2858, 3752] | 56 | 191 | 64.9% | 35.1% | 17.8% | 1.73 |
| 3 | [3783, 4974] | 56 | 184 | 65.8% | 34.2% | 22.3% | 1.58 |
| 4 | [4985, 5860] | 56 | 191 | 63.9% | 36.1% | 18.3% | 1.71 |
| 5 | [5866, 20902] | 57 | 164 | 57.3% | 42.7% | 18.3% | 2.23 |

Top-3/5/10 as a fraction of the pool: top-3 8.4%, top-5 14.0%, top-10 27.8%.

## Evidence mapping

- evidence items by method: {'exact': 2426, 'float': 253, 'heading': 129}
- mapped rate: **100.00%**; unmatched items: 0 across 0 questions
- **multiple annotators are treated as alternative evidence sets.** All-Evidence@k is computed per annotator set and averaged over sets, never against the union. Any-Evidence@k uses the union. Identical annotator sets are de-duplicated.

## Frozen first-round sample

- seed **20261110**, rule: uniform random sample without replacement over validation papers that have at least one answerable question with mapped gold evidence; ordering by sha1(f'{seed}:{paper_id}'), take the first 100. Rule frozen before any retrieval/QA evaluation.
- 100 papers, 372 questions, 333 answerable with gold
- multi-evidence subset: 141 questions; cross-section subset: 56
- sample pool: mean 48.09, P10 22.9, P90 68.2

## Decision-gate reading

The paragraph pool is small relative to a standard RAG corpus (P90 = 70.0 paragraphs/paper; top-5 covers 14.0% of the pool on average). This is flagged as a saturation risk. Difficulty must therefore be judged on **All-Evidence@k** and on the multi-paragraph / cross-section subsets rather than on Any-Evidence@5 alone.

Unmatched evidence records are in `dataset/unmatched_evidence.jsonl`; figure/table evidence (`FLOAT SELECTED` paragraphs) is never mapped to a text paragraph and is excluded from text-evidence metrics, but reported separately.
