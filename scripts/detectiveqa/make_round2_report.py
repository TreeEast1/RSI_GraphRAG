"""Generate outputs/detectiveqa/round2/ROUND2_REPORT.md from real artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "detectiveqa" / "round2"


def load(name, default=None):
    p = OUT / name
    if not p.exists():
        return default
    if name.endswith(".jsonl"):
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return json.loads(p.read_text())


def main():
    stage_a = load("stageA_cases.json")
    reader_tables = load("round2_reader_tables.json")
    analysis = load("round2_analysis.json")
    judge = load("round2_case_judge.summary.json")
    rt = load("readtime_v3.summary.json")
    audit = load("readtime_v3_manual_audit.jsonl")
    rel_kept = load("relation_channel_kept.summary.json")
    rel_val = load("relation_channel_validated.summary.json")
    rel_reader = load("round2_reader_relations.summary.json")
    case_rows = load("round2_case_table.jsonl")
    lines = []
    add = lines.append

    add("# DetectiveQA Round 2 — Diagnosis Report")
    add("")
    add("Date: 2026-11-12. All numbers come from files under `outputs/detectiveqa/round2/`; "
        "commands and errors are in `RUN_LOG.md`. Official HippoRAG `index()`, PPR, OpenIE prompts "
        "and raw graph directories were not modified.")
    add("")

    add("## 1. Executive verdict")
    add("")
    add("**Graph retrieval does not work on this benchmark split, and the reason is not the graph.**")
    add("")
    add("- **Graph fails the pre-registered gate.** q-only `q_graph_f5` R@10 .425 vs dense .418 "
        "(delta +0.007, novel-clustered CI [-0.035,+0.043]); All@20 .244 vs .220 (delta +0.024, "
        "CI [+0.000,+0.081]); MRR .497 vs .544 (delta -0.048, CI [-0.128,+0.032], McNemar p=.012, "
        "4 wins / 16 losses). RRF does not meet the gate either (R@10 .429, +1.1pp, CI crosses 0; "
        "All@20 .220, no gain). **No positive clustered-CI lower bound on R@10 or All@20.**")
    add("- **Recency dominates.** `recent` (last 4096 tokens before cutoff) scores **.805 on both "
        "option seeds**, equal to oracle seed 0 and above q_dense (.658/.585), q_graph (.683/.658) "
        "and q_rrf (.634/.634). Removing recent chunks (`nonrecent_q_dense`) collapses accuracy to "
        "**.439/.390**, near the memory-only baseline (.463/.439). Paired recent→nonrecent: "
        "-36.6pp (CI [-51.2,-17.1], p=.0015) at seed 0 and -41.5pp (CI [-54.5,-22.2], p=.0005) at seed 1.")
    add("- **This is not only a late-cutoff artifact.** `recent` is .833/.833 (early), .75/.75 "
        "(middle), .80/.80 (late) across seeds; `nonrecent_q_dense` is .50/.39, .375/.375, .40/.40. "
        "The benchmark often places decisive evidence near the cutoff in all buckets.")
    add("- **Gold clues are not a reliable upper bound.** Oracle is .805/.707; recent is ≥ oracle. "
        "The LLM judge calls 16/41 gold sets insufficient, 23/41 ambiguous/questionable, and 30/41 "
        "have explicit reveal before cutoff. 27:00, 27:06, 117:01, 118:06 are concrete cases.")
    add("- **Reader remains a bottleneck, but evidence/locality comes first.** Dense seed-0 errors: 1 all-gold, "
        "10 partial-gold, 3 no-gold. Oracle still errs on 6 all-gold questions. However, the dominant "
        "loss is evidence/locality and label ambiguity, not a missing graph.")
    add("- **Comprehension v3 is higher-yield but not precise enough.** 601 kept relations (gate "
        "quantity passes), but manual kept precision is **20/30 = 66.7%** genuine cross-chunk "
        "(24/30 = 80% if fact rewrites count as correct). Relation channel touches gold for 17/41 "
        "questions, yet equal-weight RRF lowers R@10 (.312 vs dense .418) and relation notes give no "
        "QA gain (q_dense .658→.658; q_rrf .634→.658, CI crosses 0). **Graph-injection gate fails; "
        "R2–R5 were not built.**")
    add("- **Decision: do not scale, and stop the graph direction on the current split.** Change the "
        "benchmark split/question construction (anti-reveal, re-audited gold) before any scale-up; "
        "see §9.")
    add("")

    add("## 2. Retrieval table (q-only primary, qo diagnostic)")
    add("")
    add("| condition | R@10 | R@20 | All@20 | All@50 | MRR |")
    add("|---|---:|---:|---:|---:|---:|")
    for cond, v in stage_a["condition_means"].items():
        add(f"| {cond} | {v['recall@10']:.3f} | {v['recall@20']:.3f} | {v['all@20']:.3f} | "
            f"{v['all@50']:.3f} | {v['mrr']:.3f} |")
    add("")
    add("MRR counts a missed gold chunk as 0; all metrics are over all 41 questions. "
        "`qo_*` is diagnostic only (options are not available at retrieval time in the paper setting).")
    add("")
    add("**Novel-clustered paired tests (bootstrap 5000):**")
    add("")
    add("| comparison | metric | delta | 95% CI | wins/losses | McNemar p |")
    add("|---|---|---:|---|---:|---:|")
    for key, v in stage_a["paired"].items():
        left, rest = key.split("->", 1)
        right, metric = rest.rsplit(":", 1)
        add(f"| {left}→{right} | {metric} | {v['delta']:+.4f} | "
            f"[{v['ci_low']:+.4f},{v['ci_high']:+.4f}] | {v['wins']}/{v['losses']} | "
            f"{v['mcnemar_exact_p']:.4g} |")
    add("")
    add("**By cutoff bucket (q-only):**")
    add("")
    add("| bucket | n | q_dense R@10 / All@20 / MRR | q_graph_f5 | q_graph_f20 | q_rrf | q_bm25 |")
    add("|---|---:|---|---|---|---|---|")
    for bucket, v in stage_a["by_bucket"].items():
        def f(name):
            x = v.get(name)
            return "—" if not x else f"{x['recall@10']:.3f} / {x['all@20']:.3f} / {x['mrr']:.3f}"
        add(f"| {bucket} | {v['n']} | {f('q_dense')} | {f('q_graph_f5')} | {f('q_graph_f20')} | "
            f"{f('q_rrf_dense_graph')} | {f('q_bm25')} |")
    add("")
    add("The graph is slightly better on early/middle R@10 but worse on late R@10; All@20 is 0 for "
        "every method in the late bucket. The only stable graph signal is a small early/middle "
        "coverage gain, not a deployable overall gain.")
    add("")
    add("**Per novel (q-only, R@10 / All@20 / MRR):**")
    add("")
    add("| novel | n | q_dense | q_graph_f5 | q_rrf |")
    add("|---|---:|---|---|---|")
    for novel, v in stage_a["by_novel"].items():
        def f(name):
            x = v.get(name)
            return "—" if not x else f"{x['recall@10']:.3f} / {x['all@20']:.3f} / {x['mrr']:.3f}"
        add(f"| {novel} | {v['n']} | {f('q_dense')} | {f('q_graph_f5')} | {f('q_rrf_dense_graph')} |")
    add("")
    add("**Fact-seed mechanism:** q_graph_f5 has at least one gold-touching fact for 27/41 questions "
        "(mean 1.32 of 5 facts); f10 30/41, f20 33/41. Expanding seeds raises coverage but lowers "
        "MRR (f5 .497, f10 .462, f20 .452), matching the round-1 'top20 is worse' result. Synonym "
        "edges add little (q_graph_f5_nosyn: R@10 .412, All@20 .220, MRR .501 vs .425/.244/.497).")
    add("")

    add("## 3. Reader table")
    add("")
    add("| condition | seed0 acc | seed0 gold-any | seed0 gold-all | seed1 acc | seed1 gold-any | seed1 gold-all |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for cond, v in reader_tables["conditions"].items():
        v0 = v.get("seed0", {})
        v1 = v.get("seed1", {})
        add(f"| {cond} | {v0.get('accuracy', float('nan')):.3f} | {v0.get('gold_any', float('nan')):.3f} | "
            f"{v0.get('gold_all', float('nan')):.3f} | {v1.get('accuracy', float('nan')):.3f} | "
            f"{v1.get('gold_any', float('nan')):.3f} | {v1.get('gold_all', float('nan')):.3f} |")
    add("")
    add("`nonrecent_q_dense` removes the most recent 4096-token suffix and then packs q_dense; "
        "`recent_plus_q_dense` spends half the budget on recent text and half on q_dense. "
        "`oracle` gets only annotated gold chunks. `options_only_memory` sees no text but may use prior knowledge.")
    add("")
    add("**By cutoff bucket (accuracy, seed0/seed1):**")
    add("")
    add("| condition | early_lt_70 | middle_70_90 | late_ge_90 |")
    add("|---|---:|---:|---:|")
    for cond in ("q_dense", "q_graph", "q_rrf", "recent", "nonrecent_q_dense",
                 "recent_plus_q_dense", "oracle", "qo_dense", "options_only_memory"):
        v = reader_tables["by_bucket"].get(cond)
        if not v:
            continue
        cells = []
        for bucket in ("early_lt_70", "middle_70_90", "late_ge_90"):
            s0 = v.get("seed0", {}).get(bucket, float("nan"))
            s1 = v.get("seed1", {}).get(bucket, float("nan"))
            cells.append(f"{s0:.2f}/{s1:.2f}")
        add(f"| {cond} | {cells[0]} | {cells[1]} | {cells[2]} |")
    add("")
    add("(Each cell is seed0/seed1.)")
    add("")
    add("**Error decomposition, seed 0 (wrong answers):**")
    add("")
    add("| condition | all gold present | partial gold | no gold |")
    add("|---|---:|---:|---:|")
    for cond in ("q_dense", "q_graph", "q_rrf", "qo_dense", "recent", "recent_plus_q_dense",
                 "oracle", "nonrecent_q_dense", "options_only_memory"):
        v = reader_tables["conditions"].get(cond, {}).get("seed0")
        if not v:
            continue
        add(f"| {cond} | {v['wrong_all_gold']} | {v['wrong_partial_gold']} | {v['wrong_no_gold']} |")
    add("")
    add("Dense's 14 errors are mostly partial-gold (10), not no-gold (3). Oracle errs on 6 all-gold "
        "questions. Removing recent text increases no-gold errors to 11. This is a reader/locality "
        "problem, not a missing graph.")
    add("")
    add("**Option-order stability (same original answer, seeds 0 vs 1):**")
    add("")
    add("| condition | agreement |")
    add("|---|---:|")
    for cond, v in reader_tables["option_order"].items():
        add(f"| {cond} | {v['same_answer']:.3f} |")
    add("")
    add("`recent` is the most stable condition (.976); `recent_plus_q_dense` .902; q_dense/graph/rrf "
        ".902; oracle .854; options-only memory .732; strict options-only .488. Retrieval and oracle "
        "answers are option-order sensitive, so single-seed deltas must be treated as exploratory.")
    add("")

    add("## 4. Paired tests (novel-clustered, McNemar)")
    add("")
    add("Seed 0 reader comparisons (delta = right − left; positive means right is better):")
    add("")
    add("| comparison | delta | 95% CI | wins | losses | McNemar p |")
    add("|---|---:|---|---:|---:|---:|")
    for key, v in reader_tables["paired_seed0"].items():
        add(f"| {key} | {v['delta']:+.4f} | [{v['ci_low']:+.4f},{v['ci_high']:+.4f}] | "
            f"{v['wins']} | {v['losses']} | {v['mcnemar_exact_p']:.4g} |")
    add("")
    add("Seed 1:")
    add("")
    add("| comparison | delta | 95% CI | wins | losses | McNemar p |")
    add("|---|---:|---|---:|---:|---:|")
    for key, v in reader_tables["paired_seed1"].items():
        add(f"| {key} | {v['delta']:+.4f} | [{v['ci_low']:+.4f},{v['ci_high']:+.4f}] | "
            f"{v['wins']} | {v['losses']} | {v['mcnemar_exact_p']:.4g} |")
    add("")
    add("Key readings: q_dense→recent +14.6pp (seed0, CI [+5.7,+21.4], p=.109) and +22.0pp "
        "(seed1, CI [+10.8,+32.0], p=.0117); q_dense→nonrecent −22.0pp (seed0, p=.0117) and "
        "−19.5pp (seed1, p=.0386); recent→nonrecent −36.6pp / −41.5pp (p=.0015 / .0005). "
        "q_dense→q_graph is +2.4pp seed0 (CI [0,+8.1], 2/1, p=1) and +7.3pp seed1 "
        "(CI [+2.3,+13.9], 4/1, p=.375) — directionally positive but not robust across seeds and "
        "not supported by retrieval R@10/All@20. All results are exploratory: 6 novels, 41 questions.")
    add("")

    add("## 5. Failure taxonomy and case table")
    add("")
    if judge:
        add("**LLM case judge (41 items, 0 errors).** Item labels:")
        add("")
        add("| label | count |")
        add("|---|---:|")
        for k, v in sorted(judge["item_labels"].items(), key=lambda x: -x[1]):
            add(f"| {k} | {v} |")
        add("")
        add(f"Explicit reveal before cutoff: **{judge['explicit_reveal_before_cutoff']}/41**; "
            f"gold set judged insufficient: **{judge['gold_sufficient_false']}/41**.")
        add("")
        add("Condition-level judge labels (correct_supported / correct_by_shortcut / retrieval_miss / "
            "reader_reasoning_error / partial_evidence):")
        add("")
        add("| condition | correct_supported | correct_by_shortcut | retrieval_miss | reader_reasoning_error | partial_evidence |")
        add("|---|---:|---:|---:|---:|---:|")
        for cond in ("q_dense", "q_graph", "q_rrf", "recent", "oracle"):
            cl = judge["condition_labels"]
            add(f"| {cond} | {cl.get(cond+':correct_supported',0)} | "
                f"{cl.get(cond+':correct_by_shortcut',0)} | {cl.get(cond+':retrieval_miss',0)} | "
                f"{cl.get(cond+':reader_reasoning_error',0)} | {cl.get(cond+':partial_evidence',0)} |")
        add("")
        add("The judge marks 40/41 memory-only conditions as `option_bias_or_memory`; memorization is "
            "not a marginal caveat. It also flags 22/41 item-level `entity_alias_error`.")
        add("")
    add("**Corrected case labels (34 unique questions across all case queues).** "
        "Full row-by-row table: `round2_case_table.md`; machine-readable: `round2_case_table.jsonl`.")
    add("")
    add("| qid | bucket | dense/graph/rrf/recent/nonrecent/recent+dense/oracle/mem | corrected label | implication |")
    add("|---|---|---|---|---|")
    key = ["human_anno:27:00", "human_anno:27:06", "human_anno:117:01", "human_anno:117:149:01",
           "human_anno:118:06", "human_anno:124:05", "human_anno:27:04", "human_anno:79:04",
           "human_anno:124:00", "human_anno:118:01", "human_anno:121:00", "human_anno:79:02"]
    by_qid = {r["qid"]: r for r in case_rows}
    for qid in key:
        r = by_qid.get(qid)
        if not r:
            continue
        c = r["correct"]
        corr = "/".join("Y" if c.get(k) else "N" for k in
                        ["q_dense", "q_graph", "q_rrf", "recent", "nonrecent_q_dense",
                         "recent_plus_q_dense", "oracle", "options_only_memory"])
        add(f"| {qid} | {r['bucket']} | {corr} | {r['my_label']} | {r['implication']} |")
    add("")
    add("The full 34-row table adds the remaining recent-only / memory-correct / oracle-wrong cases. "
        "The dominant pattern is `recent` or `recent_plus` correct while dense/graph miss; a second "
        "pattern is all methods correct including memory (non-discriminative); a third is gold "
        "ambiguous/wrong (124:05, 79:03, 27:01, 79:01). The only dense-vs-graph disagreement cases "
        "(118:01, 118:06, 121:00) all have `recent` correct too, so they are not graph-specific wins.")
    add("")

    add("## 6. Read-time v3, manual audit, relation channel and relation QA")
    add("")
    add(f"**Funnel:** {rt['n_packets']} packets → {rt['n_joint_facts']:,} joint facts → "
        f"{rt['n_raw']:,} comprehension candidates → {rt['n_validated']:,} rule-validated → "
        f"**{rt['n_kept']:,} verifier-kept**. Rule rejects: "
        f"too_long {rt['rule_reject_reasons'].get('too_long',0)}, "
        f"needs_two_chunks {rt['rule_reject_reasons'].get('needs_two_chunks',0)}, "
        f"quote_not_found {rt['rule_reject_reasons'].get('quote_not_found',0)}, "
        f"generic_predicate {rt['rule_reject_reasons'].get('generic_predicate',0)}, "
        f"duplicate_fact {rt['rule_reject_reasons'].get('duplicate_fact',0)}. Errors: 0.")
    add("")
    add("**Manual audit (60; 30 kept / 20 verifier-dropped / 10 rule-rejected; all 6 novels).**")
    add("")
    from collections import Counter
    lab = Counter(r["manual_label"] for r in audit)
    add("| label | count | share |")
    add("|---|---:|---:|")
    for k, v in lab.most_common():
        add(f"| {k} | {v} | {v/len(audit):.0%} |")
    add("")
    kept = [r for r in audit if r["stage"] == "kept"]
    dropped = [r for r in audit if r["stage"] == "verifier_dropped"]
    rejected = [r for r in audit if r["stage"] == "rule_rejected"]
    add(f"- Kept genuine cross-chunk: **{sum(r['manual_label']=='correct_cross_chunk' for r in kept)}/{len(kept)} "
        f"({sum(r['manual_label']=='correct_cross_chunk' for r in kept)/len(kept):.1%})**; "
        f"including fact rewrites as 'correct': "
        f"{sum(r['manual_label'] in ('correct_cross_chunk','correct_but_fact_rewrite') for r in kept)}/{len(kept)} "
        f"({sum(r['manual_label'] in ('correct_cross_chunk','correct_but_fact_rewrite') for r in kept)/len(kept):.1%}).")
    add(f"- Verifier false rejects: {sum(r['manual_label']=='correct_cross_chunk' for r in dropped)}/{len(dropped)} "
        "were genuine cross-chunk (a21-like alias/status cases are still an issue).")
    add(f"- Rule false rejects: {sum(r['manual_label']=='correct_cross_chunk' for r in rejected)}/{len(rejected)} "
        "genuine cross-chunk were rejected, mostly by the `too_long` rule (a54/a55/a59).")
    add("")
    add("**Audit precision by relation type:** alias_or_identity 11/17 (65%); alibi_or_opportunity 3/3; "
        "contradiction_or_deception 3/9; method_or_capability 1/4; motive_or_interest 2/6; "
        "state_change 0/3; timeline_or_causal 6/18. The types most useful for graph injection "
        "(timeline, motive, contradiction) are mostly fact rewrites or insufficient; alias relations "
        "are precise but mostly entity canonicalization, not new cross-chunk understanding.")
    add("")
    add("**Graph-injection gate:** quantity 601 ≥ 100 passes; manual kept precision 66.7% < 75% fails. "
        "Even the 80% ccc+cbfr count would pass only by counting fact rewrites, which defeats the "
        "comprehension objective. **R2–R5 variants were not built.**")
    add("")
    add("**Relation channel (kept relations):** mean visible relations/question "
        f"{rel_kept['mean_visible_relations']}; gold-touching top relation "
        f"{rel_kept['questions_with_gold_touching_top_relation']}/41 (41.5%, passes the 25% coverage gate). "
        "But equal-weight RRF displaces good chunks: relation_chunks alone R@10 .137; "
        f"RRF (dense+graph+relation) R@10 .312 / All@20 .195 / MRR .352 vs q_dense .418/.220/.544. "
        "Validated (include-dropped) channel is similar (R@10 .340, All@20 .195, MRR .387). ")
    add("")
    add("**Relation-note QA (Stage D, 4 conditions × 41):** q_dense .658, relations_q_dense .658; "
        "q_rrf .634, relations_q_rrf .658. Paired q_dense→relations_q_dense delta 0.0 "
        "(CI [-7.0,+7.3]pp, 2/2, p=1); q_rrf→relations_q_rrf +2.4pp (CI [-5.9,+10.5], 2/1, p=1). "
        "Relation notes add no measurable QA gain.")
    add("")

    add("## 7. Graph variants")
    add("")
    add("**Not run.** The pre-registered gate requires ≥100 kept relations (passes: 601) and manual "
        "kept precision ≥75% (fails: 20/30 = 66.7% genuine cross-chunk). The relation channel also "
        "lowers retrieval R@10 and relation notes do not improve QA. Building R2–R5 would test a "
        "representation that the audit shows is mostly alias variants and fact rewrites; it would "
        "not answer a live question. This is a gate failure, not a missing run.")
    add("")

    add("## 8. Cost table")
    add("")
    add("| stage | calls | prompt tok | completion tok | total tok | note |")
    add("|---|---:|---:|---:|---:|---|")
    add("| Read-time v3 (120 packets) | 350 | 1,344,190 | 591,686 | 1,935,876 | six parallel per-novel processes |")
    add("| Reader Stage B (10 cond × 2 seeds × 41) | 534 | 1,844,218 | 51,413 | 1,895,631 | 287 cache hits |")
    add("| Relation QA Stage D (4 cond × 41) | 96 | 401,409 | 9,725 | 411,134 | 68 cache hits |")
    add("| Case judge (41 questions) | 41 | 226,055 | 12,891 | 238,946 | post-hoc only |")
    add("| **Total** | **1,021** | **3,815,872** | **665,715** | **4,481,587** | |")
    add("")
    add("Stage A retrieval, prefix-safety checks, relation channel and graph inspection use no LLM "
        "calls (NV-Embed only). R2–R5 were not built, so no additional embedding/indexing cost.")
    add("")

    add("## 9. Next decision")
    add("")
    add("**Do not scale to 23 novels. Change the benchmark split/question construction, and stop the "
        "graph direction on the current split.**")
    add("")
    add("Evidence for the decision:")
    add("")
    add("1. The graph stop/go gate fails (R@10 +0.7pp, CI crosses 0; All@20 +2.4pp, CI lower bound 0; "
        "MRR −4.8pp, p=.012). RRF and fact-top-k/synonym variants do not rescue it.")
    add("2. Recency explains the benchmark: recent .805 on both seeds; nonrecent collapses to .439/.390; "
        "the effect holds in early, middle and late buckets. A graph that reads the story differently "
        "cannot beat a 4096-token recency window on this split.")
    add("3. Gold labels are unreliable: 16/41 insufficient, 23/41 ambiguous/questionable, 30/41 "
        "explicit reveal; oracle is below or equal to recent on both seeds.")
    add("4. Comprehension v3 is higher-yield but not precise enough (kept ccc 66.7%), and the relation "
        "channel/notes do not improve retrieval or QA.")
    add("5. Memory-only is .463/.439; 40/41 memory conditions are judged option-bias/memory. Any claim "
        "must be reported over this baseline.")
    add("")
    add("Recommended next steps, in order:")
    add("")
    add("1. **Build an anti-reveal evaluation subset:** keep only questions where the decisive answer "
        "evidence is not in the last 4096 tokens, and re-audit gold clues/answers (especially 124:05, "
        "79:03, 27:01, 79:01). Re-run Stage A/B there before any graph claim.")
    add("2. **Make `recent` and `options_only_memory` mandatory baselines** in every retrieval/reader "
        "table; report gains over both.")
    add("3. **Fix the relation layer before injecting:** filter alias-only/fact-rewrite relations, "
        "use `fixed_status`, relax `too_long`, and target timeline/motive/contradiction relations "
        "with explicit cross-chunk necessity. Re-audit 60 relations; require kept ccc ≥75%.")
    add("4. **Only then test R3 on the anti-reveal subset** with paired novel-clustered CIs; require "
        "R@10 or All@20 positive lower bound and reader gain outside late cutoffs.")
    add("5. Do not start the Question-Time loop until the anti-reveal split and relation gate pass.")
    add("")

    add("## Appendix A. Case table")
    add("")
    case_md = OUT / "round2_case_table.md"
    if case_md.exists():
        add(case_md.read_text())
    add("")
    add("## Appendix B. Stage A rank-disagreement cases")
    add("")
    stage_cases = OUT / "stageA_cases_table.md"
    if stage_cases.exists():
        add(stage_cases.read_text())
    else:
        add("Not generated.")
    add("")
    add("## Appendix C. Reproduction")
    add("")
    add("All commands and errors are in `outputs/detectiveqa/round2/RUN_LOG.md`. Key artifacts: "
        "`round2_retrieval.jsonl`, `round2_reader.jsonl`, `readtime_v3.jsonl`, "
        "`readtime_v3_manual_audit.jsonl`, `relation_channel_kept.jsonl`, "
        "`round2_reader_relations.jsonl`, `round2_case_judge.jsonl`, `round2_case_table.md`.")
    add("")

    (OUT / "ROUND2_REPORT.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "ROUND2_REPORT.md")


if __name__ == "__main__":
    main()
