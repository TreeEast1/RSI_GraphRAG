# DetectiveQA Round 2 — RUN LOG

Date started: 2026-11-12. Environment: `.venv` -> `/root/venvs/rsigraph`; HY4 `no_think`,
`HY4_RATE_FILE=detectiveqa`, `HY4_GLOBAL_QPS=1`; NV-Embed-v2 local; 4x H20.

## Pre-run code audit (before any Stage A call)

Two correctness bugs were fixed before running the diagnostic stages:

1. **Prefix leak in BM25 baseline.** `run_round2_retrieval.py` built one BM25 over *all* chunks,
   so IDF/statistics used future text. Fixed: BM25 is rebuilt per question over visible chunks only
   (`visible_docs`), and the returned relative indices are mapped back to passage indices.
   Reason: Stage A must compare prefix-safe methods; a full-corpus IDF is a future-information leak.
2. **Relation-note budget used words, not tokens.** `run_round2_reader.py` compared
   `len(note_text.split())` against a token budget, so relation notes could exceed the 4096-token
   reader budget. Fixed: use `round2.tokens(note_text)`, the same cl100k estimator as the reader.
3. **Fixed status was ignored in relation notes.** `round2.relation_text` used `status`, not
   `fixed_status`. Fixed: prefer `fixed_status` (round-1 flaw explicitly listed in the plan).

All three fixes are adapter/diagnostic fixes; official HippoRAG `index()`, PPR, OpenIE prompts and
raw graph directories were not modified.

## Commands

4. **Legacy test fixture used a now-generic predicate.** `tests/test_detectiveqa.py` used
   `predicate="contradicts"`, which v3 validation correctly rejects as a relation-type/generic
   predicate. Updated the fixture to `"lied about"`; the test still checks two-chunk, quote and
   attach-late behavior. No experimental method changed.

5. **Legacy test fixture lacked `relation_type`.** v3 validation requires one of the seven allowed
   relation types. Added `relation_type="contradiction_or_deception"` to the same fixture.
   No experimental method changed.

### Smoke tests (required)

```bash
python -m unittest discover -s tests -p 'test_detectiveqa*.py' -v
python scripts/detectiveqa/check_prefix_safety.py --split eval_human \
  --novels 27,79,117,118,121,124 --k 20 --out prefix_safety_round2
python scripts/detectiveqa/run_round2_retrieval.py --limit 2 --out round2_retrieval_smoke
python scripts/detectiveqa/run_round2_reader.py --retrieval round2_retrieval_smoke.jsonl \
  --limit 1 --option-seeds 0 --conditions q_dense,recent --out round2_reader_smoke
python scripts/detectiveqa/run_readtime_v3.py --novels 27 --packets-per-novel 2 \
  --max-total-packets 2 --out readtime_v3_smoke
```

Smoke results (2026-11-12):
- Unit tests: 12/12 pass (after two fixture updates above).
- `prefix_safety_round2`: 41/41 questions, 0 failures; modes all `prefix_ppr`.
- `round2_retrieval_smoke`: 2 questions x 22 conditions, no empty ranking, no future chunk in any
  ranked list or fact provenance; BM25 now rebuilt on visible chunks only.
- `round2_reader_smoke`: 2 rows, no errors; q_dense and recent both correct on the first question.
- `readtime_v3_smoke`: 2 packets, 84 joint facts, 24 raw -> 16 validated -> 14 kept, 0 errors;
  kept relations have specific predicates and fixed_status.

### Stage A

```bash
python scripts/detectiveqa/run_round2_retrieval.py --modes q,qo --top-k 50 \
  --fact-top-k 5,10,20 --out round2_retrieval
```

**Path/layout fix (not a method change):** the shared `OUT` constant is
`outputs/detectiveqa/`, so `--out round2_retrieval` wrote to the parent directory. To keep the
required layout, Stage A outputs were moved to `outputs/detectiveqa/round2/`, smoke outputs to
`outputs/detectiveqa/round2/smoke/`, and remaining commands pass `round2/...` path prefixes
(e.g. `--out round2/round2_reader`). No data or result was altered.

Stage A result (deterministic, no API): q_dense R@10 .418 / All@20 .220 / MRR .544;
q_graph_f5 .425 / .244 / .497. q_dense->q_graph_f5 R@10 delta +0.007, clustered CI
[-0.035,+0.043]; All@20 +0.024, CI [+0.000,+0.081]; MRR -0.048, CI [-0.128,+0.032],
McNemar p=.012 (4 wins / 16 losses). BM25 R@10 .287. Graph route fails the stop/go gate.

### Stage B + Stage C (launched concurrently, shared 1 QPS)

```bash
python scripts/detectiveqa/run_round2_reader.py \
  --retrieval round2/round2_retrieval.jsonl \
  --conditions options_only,options_only_memory,q_dense,q_graph,q_rrf,qo_dense,recent,nonrecent_q_dense,recent_plus_q_dense,oracle \
  --option-seeds 0,1 --out round2/round2_reader
python scripts/detectiveqa/run_readtime_v3.py --packets-per-novel 20 \
  --chunks-per-packet 5 --max-words 1500 --out round2/readtime_v3
```

**Wall-clock fix (not a method change):** sequential `run_readtime_v3.py` was latency-bound
(~1.2 packets/min). Stopped it after 5 packets (responses cached) and relaunched one process per
novel (`--novels N --out round2/readtime_v3_N`), sharing the same `HY4_RATE_FILE` and 1 QPS.
Prompts, packets, verifier and budgets are unchanged; outputs will be merged by packet id.

Stage C result (120 packets, merged from six per-novel runs):
- 5,114 joint facts; 1,435 raw comprehension relations; 1,122 rule-validated; **601 verifier-kept**.
- Rule rejections: too_long 110, needs_two_chunks 180, quote_not_found 21, generic_predicate 1,
  duplicate_fact 1. Errors: 0.
- Kept by novel: 27:94, 79:43, 117:126, 118:116, 121:107, 124:115.

### Relation channel

```bash
python scripts/detectiveqa/run_relation_channel.py --readtime round2/readtime_v3.jsonl \
  --retrieval round2/round2_retrieval.jsonl --out round2/relation_channel_kept
python scripts/detectiveqa/run_relation_channel.py --readtime round2/readtime_v3.jsonl \
  --retrieval round2/round2_retrieval.jsonl --include-dropped --out round2/relation_channel_validated
```

### Manual audit (60 relations)

Sample: 30 kept, 20 verifier-dropped, 10 rule-rejected; all six novels represented
(27:11, 79:11, 117:10, 118:10, 121:9, 124:9). File:
`outputs/detectiveqa/round2/readtime_v3_manual_audit.jsonl`.

Labels: correct_cross_chunk 26, correct_but_fact_rewrite 19, insufficient_evidence 10,
entity_alignment_error 3, too_generic 2, claim_as_fact 0.
Kept: 20/30 ccc (66.7%); 24/30 ccc+cbfr (80%). Verifier-dropped: 3/20 ccc (false rejects).
Rule-rejected: 3/10 ccc (too_long rejected valid relations a54/a55/a59).

Graph-injection gate: 601 kept >= 100 (passes), but kept ccc precision 66.7% < 75% (fails).
Even counting cbfr as "correct" gives 80%, but cbfr is not genuine cross-chunk comprehension;
the conservative decision is **do not build R2-R5 variants**.

### Stage D relation-note QA

```bash
python scripts/detectiveqa/run_round2_reader.py --retrieval round2/round2_retrieval.jsonl \
  --relations round2/relation_channel_kept.jsonl \
  --conditions q_dense,relations_q_dense,q_rrf,relations_q_rrf \
  --option-seeds 0 --out round2/round2_reader_relations
```

Error: Stage D reader passed `--relations round2/relation_channel_kept.jsonl`, but
`run_round2_reader.relation_lookup` reads the path as-is (not under `OUT`). The file is
`outputs/detectiveqa/round2/relation_channel_kept.jsonl`. Relaunched with the full path.
No method change.

Stage B complete: 820 rows (41 questions x 10 conditions x 2 option seeds).
Stage D complete: 164 rows (4 conditions x 41, seed 0).

### Analysis

```bash
python scripts/detectiveqa/analyze_round2.py --retrieval round2/round2_retrieval.jsonl \
  --reader round2/round2_reader.jsonl --relation round2/relation_channel_kept.jsonl \
  --out round2/round2_analysis
python scripts/detectiveqa/round2_reader_tables.py --reader round2/round2_reader.jsonl \
  --relations-reader round2/round2_reader_relations.jsonl --out round2/round2_reader_tables
python scripts/detectiveqa/judge_round2_cases.py --reader round2/round2_reader.jsonl \
  --out round2/round2_case_judge
```

## Final results (2026-11-12)

Retrieval (q-only, n=41): q_dense R@10 .418 / All@20 .220 / MRR .544; q_graph_f5
.425/.244/.497; q_rrf .429/.220/.516; BM25 .287/.122/.455. q_dense->q_graph_f5
R@10 +0.007 (cluster CI [-0.035,+0.043]), All@20 +0.024 (CI [0.000,+0.081]),
MRR -0.048 (CI [-0.128,+0.032], McNemar p=.012, 4/16). Graph stop/go gate FAILS.

Reader (seed0/seed1): q_dense .658/.585; q_graph .683/.658; q_rrf .634/.634;
qo_dense .707/.707; recent .805/.805; nonrecent_q_dense .439/.390;
recent_plus_q_dense .805/.732; oracle .805/.707; options_only .293/.439;
options_only_memory .463/.439. recent->nonrecent -36.6pp (p=.0015) / -41.5pp
(p=.0005). Recency effect holds in early/middle/late buckets.

Read-time v3: 120 packets -> 5,114 joint facts -> 1,435 raw -> 1,122 validated ->
601 kept. Manual audit 60: ccc 26, cbfr 19, insufficient 10, entity 3, generic 2,
claim_as_fact 0. Kept ccc 20/30 (66.7%); ccc+cbfr 24/30 (80%). Graph-injection
gate FAILS on precision. Relation channel: 17/41 gold-touch; RRF_all R@10 .312
(vs dense .418). Relation QA: q_dense .658 -> relations_q_dense .658;
q_rrf .634 -> relations_q_rrf .658 (CI crosses 0). R2-R5 NOT built.

LLM case judge: 41 items, 0 errors; reveal_before_cutoff 30/41, gold_sufficient
false 16/41, gold_ambiguous_or_wrong 23, gold_incomplete 12, entity_alias 22.
Case table: 34 unique questions in `round2_case_table.md/.jsonl`.

Final report: `outputs/detectiveqa/round2/ROUND2_REPORT.md`.
Decision: do not scale; change benchmark split/question construction (anti-reveal,
re-audited gold); stop graph direction on the current split.

Token cost: readtime_v3 1,935,876; reader B 1,895,631; relation QA 411,134;
case judge 238,946; total 4,481,587.

All smoke and full retrieval/reader contexts were checked for future chunks:
Stage A 41/41 no future ranked chunks; reader 820+164 rows no future context
chunks; prefix_safety_round2 41/41 pass.

Note: HY4 logs recorded one transient error during the Stage B reader run; the client retried and
the final `round2_reader.jsonl` has 0 rows with `error`. No data was lost.
