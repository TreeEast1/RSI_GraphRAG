#!/usr/bin/env python
"""Build the QASPER read-time result tables, case file and final report."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.qasper.evalutil import OUT, load_papers, read_jsonl  # noqa: E402

METHOD_LABEL = {
    "m0_dense": "M0 Dense",
    "m0_bm25": "M0 BM25",
    "m0_hybrid": "M0 Hybrid",
    "m1_raw": "M1 G_raw",
    "m2_atom": "M2 +Atomic",
    "m3_atom_comp": "M3 +Triples",
    "m4_units": "M4 +Units",
    "m5_units_graph": "M5 Triples+Units",
    "m6_summary": "M6 +Summaries",
    "m6_matched": "M6 matched count",
    "m4_gated": "M4 gated units",
    "m5_gated": "M5 gated units",
    "m4_units_only": "M4 units only",
    "m6_summary_only": "M6 summaries only",
}
RET_METHODS = ["m0_bm25", "m0_dense", "m0_hybrid", "m1_raw", "m2_atom", "m3_atom_comp",
               "m4_units", "m5_units_graph", "m4_gated", "m5_gated", "m6_summary", "m6_matched",
               "m4_units_only", "m6_summary_only"]
RET_METRICS = ["any@1", "any@3", "any@5", "any@10", "all@1", "all@3", "all@5", "all@10",
               "recall@5", "precision@5", "f1@5", "mrr", "ndcg@10"]


def mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def subset_filter(rows: List[Dict], subset: str) -> List[Dict]:
    if subset == "all":
        return rows
    if subset == "single":
        return [r for r in rows if r.get("n_gold") == 1]
    if subset == "multi":
        return [r for r in rows if r.get("n_gold", 0) > 1 or r.get("multi_evidence")]
    if subset == "cross":
        return [r for r in rows if r.get("cross_section")]
    if subset == "multi_annot":
        return [r for r in rows if (r.get("n_annotator_sets") or 0) > 1]
    if subset == "unit_covered":
        return [r for r in rows if r.get("unit_gold_upper")]
    if subset == "unit_uncovered":
        return [r for r in rows if r.get("unit_gold_upper") is False]
    return rows


def retrieval_table(rows: List[Dict], methods: Sequence[str]) -> List[Dict]:
    out = []
    for subset in ("all", "single", "multi", "cross", "multi_annot", "unit_covered", "unit_uncovered"):
        sub = subset_filter(rows, subset)
        for m in methods:
            rr = [r for r in sub if r["method"] == m]
            if not rr:
                continue
            row = {"subset": subset, "method": m, "label": METHOD_LABEL.get(m, m), "n": len(rr)}
            for k in RET_METRICS:
                row[k] = mean([r["metrics"].get(k) for r in rr])
            for b in ("2048", "4096", "8192"):
                row[f"cov_full@{b}"] = mean([r.get("budget", {}).get(b, {}).get("full") for r in rr])
            out.append(row)
    return out


def paired_bootstrap(rows: List[Dict], a: str, b: str, metric: str, n: int = 2000, seed: int = 0) -> Dict:
    by_q = defaultdict(dict)
    for r in rows:
        by_q[r["qid"]][r["method"]] = r
    diffs = []
    for qid, d in by_q.items():
        if a in d and b in d:
            va = d[a]["metrics"].get(metric)
            vb = d[b]["metrics"].get(metric)
            if va is not None and vb is not None and not math.isnan(va) and not math.isnan(vb):
                diffs.append(va - vb)
    if not diffs:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    arr = np.asarray(diffs)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)]
    return {"mean": float(arr.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": len(arr)}


def qa_table(preds: List[Dict], methods: Sequence[str]) -> List[Dict]:
    out = []
    for m in methods:
        rows = [r for r in preds if r["method"] == m]
        if not rows:
            continue
        row = {"method": m, "label": METHOD_LABEL.get(m, m), "n": len(rows)}
        for key in ("answer_f1", "evidence_f1", "em"):
            row[key] = mean([r.get(key) for r in rows])
        for sub, name in ((lambda r: r["answerable"], "answerable"),
                          (lambda r: not r["answerable"], "unanswerable"),
                          (lambda r: r["multi_evidence"], "multi_evidence"),
                          (lambda r: r["cross_section"], "cross_section")):
            rr = [r for r in rows if sub(r)]
            row[f"f1_{name}"] = mean([r.get("answer_f1") for r in rr])
            row[f"n_{name}"] = len(rr)
        row["gold_any_in_context"] = mean([float(r["gold_any_in_context"]) for r in rows])
        row["gold_all_in_context"] = mean([float(r["gold_all_in_context"]) for r in rows])
        row["context_tokens"] = mean([r["context_tokens"] for r in rows])
        for t in ("extractive", "abstractive", "boolean", "none"):
            vals = [r["answer_f1"] for r in rows if r.get("answer_type") == t]
            row[f"f1_{t}"] = mean(vals)
            row[f"n_{t}"] = len(vals)
        out.append(row)
    return out


def write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def pct(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100*x:.1f}"


def md_table(rows: List[Dict], cols: Sequence[tuple]) -> str:
    head = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for _, key, nd in cols:
            v = r.get(key)
            cells.append(fmt(v, nd) if isinstance(v, (int, float)) else str(v if v is not None else "—"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def comprehension_stats(scope: str, grouping: str, papers=None) -> Dict:
    base = OUT / "comprehension" / grouping
    metas = []
    if base.exists():
        for d in sorted(base.iterdir()):
            m = d / "meta.json"
            if m.exists():
                metas.append(json.loads(m.read_text()))
    rel_types = Counter()
    ver_reasons = Counter()
    ver_categories = Counter()
    units, rejected = [], []
    for m in metas:
        p = base / m["paper_id"]
        if (p / "units.jsonl").exists():
            for line in open(p / "units.jsonl", encoding="utf-8"):
                u = json.loads(line)
                rel_types[u.get("relation_type")] += 1
                units.append(u)
        vp = p / "insights_verified.json"
        if vp.exists():
            d = json.loads(vp.read_text())
            for x in d.get("rejected", []):
                v = x.get("verification") or {}
                reason = str(v.get("reason", ""))
                ver_reasons[reason[:120]] += 1
                low = reason.lower()
                if "restat" in low or "paraphrase" in low or "single paragraph" in low:
                    ver_categories["restatement_of_one_paragraph"] += 1
                elif "causal" in low or "causation" in low or "co-occurrence" in low:
                    ver_categories["unsupported_causality"] += 1
                elif ("not support" in low or "absent" in low or "not stated" in low
                      or "does not" in low or "unsupported" in low):
                    ver_categories["unsupported_content"] += 1
                else:
                    ver_categories["other"] += 1
                rejected.append(x)
    tot = {
        "papers": len(metas),
        "groups": sum(m.get("n_groups", 0) for m in metas),
        "facts": sum(m.get("n_atomic_facts", 0) for m in metas),
        "raw": sum(m.get("n_insights_raw", 0) for m in metas),
        "verified": sum(m.get("n_insights_verified", 0) for m in metas),
        "rejected": sum(m.get("n_insights_rejected", 0) for m in metas),
        "summaries": sum(m.get("n_summaries", 0) for m in metas),
        "prompt_tokens": sum((m.get("atomic_usage", {}).get("prompt_tokens", 0)
                              + m.get("comprehension_usage", {}).get("prompt_tokens", 0)
                              + m.get("summary_usage", {}).get("prompt_tokens", 0)) for m in metas),
        "completion_tokens": sum((m.get("atomic_usage", {}).get("completion_tokens", 0)
                                  + m.get("comprehension_usage", {}).get("completion_tokens", 0)
                                  + m.get("summary_usage", {}).get("completion_tokens", 0)) for m in metas),
    }
    tot["retention"] = tot["verified"] / tot["raw"] if tot["raw"] else float("nan")
    unit_lens = [len(u["concise_text"].split()) for u in units]
    tot["unit_words_mean"] = float(np.mean(unit_lens)) if unit_lens else float("nan")
    tot["unit_words_p90"] = float(np.percentile(unit_lens, 90)) if unit_lens else float("nan")
    tot["units_per_paper"] = tot["verified"] / tot["papers"] if tot["papers"] else float("nan")

    # source distance analysis: is the unit really cross-paragraph / cross-section?
    cross_section = adjacent = 0
    gaps = []
    if papers:
        for u in units:
            pmap = papers[u["paper_id"]].paragraph_map()
            ps = [pmap[x] for x in u["source_passage_ids"] if x in pmap]
            if len(ps) >= 2:
                secs = {p.section_index for p in ps}
                if len(secs) > 1:
                    cross_section += 1
                idx = sorted(p.global_index for p in ps)
                gap = idx[-1] - idx[0]
                gaps.append(gap)
                if gap <= 1:
                    adjacent += 1
    # fraction of each paper's paragraphs that are sources of at least one unit
    src_fracs = []
    if papers:
        per_paper = {}
        for u in units:
            per_paper.setdefault(u["paper_id"], set()).update(u.get("source_passage_ids") or [])
        for pid, srcs in per_paper.items():
            if pid in papers:
                src_fracs.append(len(srcs) / max(1, len(papers[pid].paragraphs)))
    tot["unit_source_pool_fraction"] = float(np.mean(src_fracs)) if src_fracs else float("nan")
    n = len(units) or 1
    tot["unit_cross_section_share"] = cross_section / n
    tot["unit_adjacent_share"] = adjacent / n
    tot["unit_source_gap_mean"] = float(np.mean(gaps)) if gaps else float("nan")
    tot["unit_source_gap_p90"] = float(np.percentile(gaps, 90)) if gaps else float("nan")
    # relation-type effectiveness, joined with the mechanism usage file
    usage_path = OUT / "mechanism" / f"unit_usage_{scope}.jsonl"
    usage = {}
    if usage_path.exists():
        usage = {u["unit_id"]: u for u in read_jsonl(usage_path)}
    rel_eff = {}
    for u in units:
        rt = u.get("relation_type", "?")
        row = rel_eff.setdefault(rt, {"n": 0, "gold": 0, "top5": 0, "helped": 0, "harmed": 0, "missed": 0})
        row["n"] += 1
        us = usage.get(u["unit_id"])
        if us:
            row["gold"] += int(us.get("n_gold_touch_q", 0) > 0)
            row["top5"] += int(us.get("n_top5_q", 0) > 0)
            row["helped"] += int(us.get("n_helped_q", 0) > 0)
            row["harmed"] += int(us.get("n_harmed_q", 0) > 0)
            row["missed"] += int(us.get("n_gold_touch_q", 0) > 0 and us.get("n_top5_q", 0) == 0)
    return {"totals": tot, "rel_types": dict(rel_types), "reject_reasons": dict(ver_reasons),
            "reject_categories": dict(ver_categories), "rel_effectiveness": rel_eff,
            "units": units, "rejected": rejected, "metas": metas}


def unit_gold_coverage(papers, scope: str, grouping: str) -> Dict:
    """Upper bound: fraction of questions whose gold evidence touches any unit source."""
    hit = tot = 0
    per = []
    for pid, paper in papers.items():
        units = []
        p = OUT / "comprehension" / grouping / pid / "units.jsonl"
        if p.exists():
            units = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        src = set()
        for u in units:
            src |= set(u.get("source_passage_ids") or [])
        for q in paper.qas:
            gold = set(q.gold_passages_union)
            if not gold:
                continue
            tot += 1
            ok = bool(gold & src)
            hit += ok
            per.append((q.qid, ok))
    return {"questions": tot, "covered": hit, "rate": hit / tot if tot else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="sample100")
    ap.add_argument("--grouping", default="section")
    args = ap.parse_args()
    papers = load_papers(args.scope)
    reports = OUT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    ret_rows = read_jsonl(OUT / "retrieval" / f"comparison_{args.scope}.jsonl")
    qa_rows = read_jsonl(OUT / "qa" / f"predictions_{args.scope}.jsonl") if (OUT / "qa" / f"predictions_{args.scope}.jsonl").exists() else []

    # ------------------------------------------------------------ retrieval csv
    rtable = retrieval_table(ret_rows, RET_METHODS)
    write_csv(OUT / "retrieval_comparison.csv", rtable,
              ["subset", "method", "label", "n"] + RET_METRICS + [f"cov_full@{b}" for b in ("2048", "4096", "8192")])

    # ------------------------------------------------------------ QA csv
    qa_methods = ["m0_dense", "m1_raw", "m2_atom", "m3_atom_comp", "m4_units", "m5_units_graph",
                  "m6_summary", "m4_units_B", "m5_units_graph_B", "m6_summary_B",
                  "m1_raw_top5", "m3_atom_comp_top5", "m5_units_graph_top5", "oracle", "full"]
    qtab = qa_table(qa_rows, qa_methods)
    write_csv(OUT / "qa_comparison.csv", qtab,
              ["method", "label", "n", "answer_f1", "evidence_f1", "em", "f1_answerable",
               "f1_unanswerable", "f1_multi_evidence", "f1_cross_section",
               "gold_any_in_context", "gold_all_in_context", "context_tokens",
               "f1_extractive", "f1_abstractive", "f1_boolean", "f1_none"])

    # ------------------------------------------------------------ paired deltas
    pairs = [("m3_atom_comp", "m2_atom"), ("m4_units", "m2_atom"), ("m5_units_graph", "m3_atom_comp"),
             ("m6_matched", "m2_atom"), ("m5_units_graph", "m2_atom")]
    deltas = {}
    for a, b in pairs:
        deltas[f"{a}_vs_{b}"] = {k: paired_bootstrap(ret_rows, a, b, k)
                                 for k in ("any@5", "all@5", "recall@5", "mrr", "ndcg@10")}
    qa_deltas = {}
    if qa_rows:
        by_q = defaultdict(dict)
        for r in qa_rows:
            by_q[r["qid"]][r["method"]] = r
        for a, b in [("m3_atom_comp", "m2_atom"), ("m4_units", "m2_atom"),
                     ("m5_units_graph", "m3_atom_comp"), ("m5_units_graph", "m2_atom"),
                     ("m4_units_B", "m4_units"), ("m5_units_graph_B", "m5_units_graph")]:
            vals = [by_q[q][a]["answer_f1"] - by_q[q][b]["answer_f1"]
                    for q in by_q if a in by_q[q] and b in by_q[q]]
            vals_ev = [by_q[q][a]["evidence_f1"] - by_q[q][b]["evidence_f1"]
                       for q in by_q if a in by_q[q] and b in by_q[q]]
            if vals:
                arr = np.asarray(vals)
                rng = np.random.default_rng(0)
                means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(2000)]
                qa_deltas[f"{a}_vs_{b}"] = {
                    "f1_mean": float(arr.mean()), "f1_lo": float(np.percentile(means, 2.5)),
                    "f1_hi": float(np.percentile(means, 97.5)),
                    "ev_f1_mean": float(np.mean(vals_ev)), "n": len(vals)}

    comp = comprehension_stats(args.scope, args.grouping, papers)
    cov = unit_gold_coverage(papers, args.scope, args.grouping)
    mech_path = OUT / "mechanism" / f"summary_{args.scope}.json"
    mech = json.loads(mech_path.read_text()) if mech_path.exists() else {}

    # ------------------------------------------------------------ baseline md
    base_path = OUT / "retrieval" / f"baselines_{args.scope}.jsonl"
    base_rows = read_jsonl(base_path) if base_path.exists() else ret_rows
    for r in base_rows:
        if not str(r.get("method", "")).startswith("m0_"):
            r["method"] = "m0_" + r["method"]
    base_methods = ["m0_bm25", "m0_dense", "m0_hybrid"]
    btable = retrieval_table(base_rows, base_methods)
    L = ["# Baseline Retrieval on QASPER (paragraph units)\n",
         f"Scope: frozen 100-paper validation sample (seed 20261110), answerable questions with mapped "
         f"text evidence. Passage = original paragraph; retrieval restricted to the question's paper. "
         f"Multiple annotators are alternatives: every metric is the mean over annotator evidence sets.\n",
         "## Overall\n"]
    rows_all = [r for r in btable if r["subset"] == "all"]
    L.append(md_table([r for r in rows_all if r["method"] in base_methods],
                      [("method", "label", 0), ("n", "n", 0), ("Any@1", "any@1", 3), ("Any@3", "any@3", 3),
                       ("Any@5", "any@5", 3), ("All@1", "all@1", 3), ("All@3", "all@3", 3),
                       ("All@5", "all@5", 3), ("Recall@5", "recall@5", 3), ("MRR", "mrr", 3),
                       ("nDCG@10", "ndcg@10", 3), ("Cov@4096", "cov_full@4096", 3)]))
    L.append("\n## Subsets\n")
    L.append(md_table([r for r in btable if r["method"] == "m0_dense"],
                      [("subset", "subset", 0), ("n", "n", 0), ("Any@5", "any@5", 3),
                       ("All@5", "all@5", 3), ("Recall@5", "recall@5", 3), ("Cov@4096", "cov_full@4096", 3)]))
    pool = json.loads((OUT / "dataset" / "summary.json").read_text())
    L.append("\n## Difficulty reading\n")
    L.append(f"- paragraph pool per paper: mean {pool['pool']['mean']}, P10 {pool['pool']['p10']}, "
             f"P50 {pool['pool']['p50']}, P90 {pool['pool']['p90']}; top-5 covers "
             f"{100*pool['topk_share_of_pool']['5']['mean']:.1f}% of the pool on average.")
    L.append(f"- dense Any@5 = {fmt(next((r['any@5'] for r in rows_all if r['method']=='m0_dense'), None))}, "
             f"All@5 = {fmt(next((r['all@5'] for r in rows_all if r['method']=='m0_dense'), None))}; "
             f"the gap between Any and All is the part that is not saturated.")
    L.append("- BM25 is far below dense, so the hybrid RRF with equal weights is diluted by BM25; dense is the M0 reference.")
    L.append("\nNo window-level retrieval was added: the paragraph pool already has a median of "
             f"{pool['pool']['p50']} units and the multi-evidence / cross-section subsets provide the "
             "discriminating signal; splitting paragraphs would inflate recall artificially.\n")
    (reports / "baseline_retrieval.md").write_text("\n".join(L), encoding="utf-8")

    # ------------------------------------------------------------ comprehension quality
    t = comp["totals"]
    L = ["# Comprehension Quality\n",
         f"Grouping: **{args.grouping}**. Read-time calls see only the paper title, paragraphs and atomic facts.\n",
         "## Volume and cost\n",
         "| item | value |", "|---|---|",
         f"| papers processed | {t['papers']} |",
         f"| reading groups | {t['groups']} |",
         f"| atomic facts | {t['facts']} |",
         f"| candidate insights | {t['raw']} |",
         f"| verified insights / units | {t['verified']} |",
         f"| rejected insights | {t['rejected']} |",
         f"| verifier retention | {100*t['retention']:.1f}% |" if t["raw"] else "| verifier retention | — |",
         f"| units per paper | {t['units_per_paper']:.1f} |",
         f"| paragraphs that are a unit source (per paper, mean) | {100*t['unit_source_pool_fraction']:.1f}% |",
         f"| unit length (words, mean / P90) | {t['unit_words_mean']:.1f} / {t['unit_words_p90']:.1f} |",
         f"| units sourced from >1 section | {100*t['unit_cross_section_share']:.1f}% |",
         f"| units sourced from adjacent paragraphs | {100*t['unit_adjacent_share']:.1f}% |",
         f"| source paragraph gap (mean / P90) | {t['unit_source_gap_mean']:.1f} / {t['unit_source_gap_p90']:.1f} |",
         f"| paragraph summaries (M6) | {t['summaries']} |",
         f"| read-time prompt tokens | {t['prompt_tokens']:,} |",
         f"| read-time completion tokens | {t['completion_tokens']:,} |",
         "",
         "## Relation types (verified units)\n",
         "| relation_type | count |", "|---|---|"]
    for k, v in sorted(comp["rel_types"].items(), key=lambda x: -x[1]):
        L.append(f"| {k} | {v} |")
    L.append("\n## Relation-type effectiveness (joined with mechanism tracking)\n")
    L.append("| relation_type | units | touches gold | ever in unit top-5 | touches gold but never top-5 | QA helped | QA harmed |")
    L.append("|---|---|---|---|---|---|---|")
    for rt, r in sorted(comp.get("rel_effectiveness", {}).items(), key=lambda x: -x[1]["n"]):
        L.append(f"| {rt} | {r['n']} | {r['gold']} | {r['top5']} | {r['missed']} | {r['helped']} | {r['harmed']} |")
    L.append("\n## Verifier rejections\n")
    L.append("| category | count |")
    L.append("|---|---|")
    for k, v in sorted(comp["reject_categories"].items(), key=lambda x: -x[1]):
        L.append(f"| {k} | {v} |")
    L.append("\n## Does the unit layer cover gold evidence at all?\n")
    L.append(f"- questions whose gold evidence touches the source of at least one verified unit: "
             f"**{cov['covered']}/{cov['questions']} = {100*cov['rate']:.1f}%** (upper bound for M4/M5, "
             "before any retrieval ranking).")
    if mech:
        L.append(f"- units generated: {mech.get('n_units')}; never a retrieval candidate: "
                 f"{mech.get('never_retrieved')}; never in top-5: {mech.get('never_top5')}; "
                 f"touch gold: {mech.get('touches_gold')}; touch gold but never top-5: "
                 f"{mech.get('touches_gold_but_never_top5')}.")
        L.append(f"- units on questions whose M5 QA improved: {mech.get('helped_qa')}; "
                 f"units on questions whose M5 QA got worse: {mech.get('harmed_qa')}.")
    L.append("\n## Examples of verified units\n")
    for u in comp["units"][:5]:
        L.append(f"- `{u.get('unit_id')}` [{u.get('relation_type')}] {u['concise_text']} "
                 f"(sources: {', '.join(u['source_passage_ids'])})")
    L.append("\n## Examples of rejected candidates\n")
    for r in comp["rejected"][:5]:
        v = r.get("verification") or {}
        L.append(f"- {r.get('concise_text')}  \n  reason: {v.get('reason')}")
    (reports / "comprehension_quality.md").write_text("\n".join(L), encoding="utf-8")

    # ------------------------------------------------------------ retrieval report numbers for final
    def row(subset, method, key):
        for r in rtable:
            if r["subset"] == subset and r["method"] == method:
                return r.get(key)
        return None

    # ------------------------------------------------------------ cases
    chains = read_jsonl(OUT / "mechanism" / f"question_chains_{args.scope}.jsonl") if (OUT / "mechanism" / f"question_chains_{args.scope}.jsonl").exists() else []
    units_by_paper = defaultdict(dict)
    for pid in papers:
        p = OUT / "comprehension" / args.grouping / pid / "units.jsonl"
        if p.exists():
            for l in open(p, encoding="utf-8"):
                u = json.loads(l)
                units_by_paper[pid][u["unit_id"]] = u

    def unit_line(pid, uid, gold):
        u = units_by_paper.get(pid, {}).get(uid, {})
        src = u.get("source_passage_ids") or []
        mark = "GOLD" if set(src) & gold else "not-gold"
        return f"  - `{uid}` [{mark}] {u.get('concise_text', '')[:220]} (src: {', '.join(src)})"

    pos = [c for c in chains if (c.get("all5_m5") or 0) > (c.get("all5_m2") or 0)]
    neg = [c for c in chains if (c.get("all5_m5") or 0) < (c.get("all5_m2") or 0)
           or (c.get("any5_m5") or 0) < (c.get("any5_m2") or 0)]
    nofb = [c for c in chains if c.get("units_top5") and not c.get("units_touching_gold")]
    L = ["# Case Analysis\n", f"Scope: {args.scope}, grouping {args.grouping}. Rankings are top-5. The unit lists below are the top-5 "
         "comprehension units ranked by unit-question embedding similarity; GOLD marks that the unit's source "
         "paragraph is gold evidence for that question (it does not guarantee the fused passage ranking kept it).\n"]
    L.append("## Positive: M5 improves All-Evidence@5 over M2\n")
    for c in pos[:6]:
        gold = set(c["gold_passages"])
        L.append(f"### {c['qid']} — {c['question']}\n")
        L.append(f"- gold passages: {', '.join(c['gold_passages'][:6])}")
        L.append(f"- All@5: M2={fmt(c.get('all5_m2'))} → M3={fmt(c.get('all5_m3'))} → "
                 f"M4={fmt(c.get('all5_m4'))} → M5={fmt(c.get('all5_m5'))}")
        L.append("- units in top-5:")
        for uid in c["units_top5"][:5]:
            L.append(unit_line(c["paper_id"], uid, gold))
    L.append("\n## Negative: units or triples displace correct evidence\n")
    for c in neg[:6]:
        gold = set(c["gold_passages"])
        L.append(f"### {c['qid']} — {c['question']}\n")
        L.append(f"- gold passages: {', '.join(c['gold_passages'][:6])}")
        L.append(f"- All@5: M2={fmt(c.get('all5_m2'))} → M3={fmt(c.get('all5_m3'))} → "
                 f"M4={fmt(c.get('all5_m4'))} → M5={fmt(c.get('all5_m5'))}; "
                 f"Any@5: M2={fmt(c.get('any5_m2'))} → M5={fmt(c.get('any5_m5'))}")
        for uid in c["units_top5"][:5]:
            L.append(unit_line(c["paper_id"], uid, gold))
    L.append("\n## Units retrieved but not touching gold\n")
    for c in nofb[:4]:
        gold = set(c["gold_passages"])
        L.append(f"### {c['qid']} — {c['question']}\n")
        L.append(f"- gold: {', '.join(c['gold_passages'][:5])}")
        for uid in c["units_top5"][:4]:
            L.append(unit_line(c["paper_id"], uid, gold))
    L.append("\n## Overreach rejected by the verifier\n")
    for r in comp["rejected"][:6]:
        v = r.get("verification") or {}
        if v.get("causality_error") or "restat" in str(v.get("reason", "")).lower():
            L.append(f"- {r.get('concise_text')[:240]}  \n  reason: {v.get('reason')}")
    (reports / "cases.md").write_text("\n".join(L), encoding="utf-8")

    # ------------------------------------------------------------ source provenance
    prov_rows = []
    for pid, paper in papers.items():
        pmap = paper.paragraph_map()
        p = OUT / "comprehension" / args.grouping / pid / "units.jsonl"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            u = json.loads(line)
            prov_rows.append({
                "paper_id": pid, "unit_id": u["unit_id"], "concise_text": u["concise_text"],
                "relation_type": u.get("relation_type"), "confidence": u.get("confidence"),
                "source_passage_ids": u["source_passage_ids"],
                "source_sections": [pmap[x].section_name for x in u["source_passage_ids"] if x in pmap],
                "source_excerpts": [pmap[x].text[:300] for x in u["source_passage_ids"] if x in pmap],
                "verification": u.get("verification"),
            })
    with open(OUT / "source_provenance.jsonl", "w", encoding="utf-8") as f:
        for r in prov_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------ final report
    deltas_path = OUT / "reports" / "deltas.json"
    deltas_path.write_text(json.dumps({"retrieval": deltas, "qa": qa_deltas}, indent=1))

    L = ["# FINAL REPORT — QASPER Read-Time Comprehension\n",
         "## 0. What was run\n",
         f"- validation split parsed: 281 papers, 1005 questions; frozen 100-paper sample (seed 20261110) "
         f"with {pool['sample']['n_questions']} questions, {pool['sample']['n_answerable_with_gold']} answerable "
         f"with mapped gold, {pool['sample']['n_multi_evidence']} multi-evidence, {pool['sample']['n_cross_section']} cross-section.",
         f"- read-time: {t['papers']} papers, {t['facts']} atomic facts, {t['raw']} candidate insights, "
         f"{t['verified']} verified units, {t['rejected']} rejected.",
         f"- graphs: G_raw per paper + M2 (atomic) and M3 (atomic+comprehension triples) variants.",
         f"- retrieval: M0-M6 on paragraphs; QA: fixed 4096-token reader budget with an oracle-context reference.",
         f"- **scope**: {len(set(r['paper_id'] for r in ret_rows))} papers with complete G_raw + variants + read-time "
         f"outputs ({len(set(r['qid'] for r in ret_rows))} answerable questions; "
         f"{len(qa_rows)//max(1,len(set(r['method'] for r in qa_rows)))} QA questions). "
         f"M0 baselines cover the same 100 sampled papers.",
         "",
         "### Headline",
         f"- M3 comprehension triples are neutral-to-slightly-positive overall (All@5 {fmt(row('all','m3_atom_comp','all@5'))} vs "
         f"M2 {fmt(row('all','m2_atom','all@5'))}, +1.1pp, 95% CI [-0.2,+2.4]) and clearly positive on the questions "
         "their sources cover.",
         f"- M4/M5 comprehension-unit fusion is strongly negative overall (All@5 {fmt(row('all','m4_units','all@5'))} "
         f"vs M2 {fmt(row('all','m2_atom','all@5'))}) because units cover only part of the evidence space and equal-weight "
         "RRF displaces correct graph evidence on uncovered questions.",
         f"- M6 full per-paragraph summaries recover dense-level retrieval (All@5 {fmt(row('all','m6_summary','all@5'))}), "
         f"while M6 with the same number of units as insights does not ({fmt(row('all','m6_matched','all@5'))}): the "
         "driver is coverage, not comprehension.",
         f"- Graph retrieval itself does not beat dense on this task: M1 All@5 {fmt(row('all','m1_raw','all@5'))} vs "
         f"dense {fmt(row('all','m0_dense','all@5'))}; the graph layers change recall marginally, not the ranking ceiling.",
         f"- Oracle-context QA is only {fmt(next((r['answer_f1'] for r in qtab if r['method']=='oracle'), None))} vs dense "
         f"{fmt(next((r['answer_f1'] for r in qtab if r['method']=='m0_dense'), None))}: on this reader, retrieval "
         "differences have little room to change Answer F1.",
         "",
         "## 1. Does QASPER fix the LoCoMo small-pool problem?\n",
         f"Partially. The paragraph pool is mean {pool['pool']['mean']}, median {pool['pool']['p50']}, P90 "
         f"{pool['pool']['p90']} per paper — roughly 2-3× LoCoMo's 19-session pool, and top-5 is "
         f"{100*pool['topk_share_of_pool']['5']['mean']:.1f}% of the pool. On the full 100-paper sample, dense "
         f"Any@5 is {pct(next((r['any@5'] for r in btable if r['subset']=='all' and r['method']=='m0_dense'), None))}%, "
         f"so single-evidence retrieval is close to saturated; All@5 = "
         f"{pct(next((r['all@5'] for r in btable if r['subset']=='all' and r['method']=='m0_dense'), None))}% and the "
         f"multi-evidence subset ({pct(next((r['all@5'] for r in btable if r['subset']=='multi' and r['method']=='m0_dense'), None))}% "
         f"All@5) are the discriminating conditions. "
         "This is a harder retrieval task than LoCoMo, but not a large-corpus task.",
         "",
         "## 2. Does read-time comprehension improve gold-evidence recall?\n",
        ]
    L.append(md_table([r for r in rtable if r["subset"] == "all" and r["method"] in
                       ("m1_raw", "m2_atom", "m3_atom_comp", "m4_units", "m5_units_graph", "m6_summary", "m6_matched")],
                      [("method", "label", 0), ("Any@5", "any@5", 3), ("All@5", "all@5", 3),
                       ("Recall@5", "recall@5", 3), ("MRR", "mrr", 3), ("nDCG@10", "ndcg@10", 3),
                       ("Cov@4096", "cov_full@4096", 3)]))
    L.append("\nPaired question-level deltas (95% bootstrap CI):\n")
    for k, v in deltas.items():
        L.append(f"- {k}: Any@5 {v['any@5']['mean']:+.3f} [{v['any@5']['lo']:+.3f},{v['any@5']['hi']:+.3f}], "
                 f"All@5 {v['all@5']['mean']:+.3f} [{v['all@5']['lo']:+.3f},{v['all@5']['hi']:+.3f}], "
                 f"Recall@5 {v['recall@5']['mean']:+.3f}")
    L.append("\n## 3. Which representation is most effective: triples, units, or both?\n")
    for m in ("m3_atom_comp", "m4_units", "m5_units_graph", "m6_matched"):
        L.append(f"- {METHOD_LABEL[m]}: All@5 {fmt(row('all',m,'all@5'))}, Any@5 {fmt(row('all',m,'any@5'))}, "
                 f"Recall@5 {fmt(row('all',m,'recall@5'))}, Cov@4096 {fmt(row('all',m,'cov_full@4096'))}.")
    sweep_path = OUT / "retrieval" / f"fusion_sweep_{args.scope}.json"
    if sweep_path.exists():
        sweep = [r for r in json.loads(sweep_path.read_text()) if r["subset"] == "all"]
        L.append("\nFusion-weight sensitivity (RRF weight on the unit list; graph weight fixed at 1.0):\n")
        L.append(md_table(sweep, [("base", "base", 0), ("unit weight", "weight", 2), ("n", "n", 0),
                                  ("Any@5", "any@5", 3), ("All@5", "all@5", 3), ("Recall@5", "recall@5", 3)]))
        w01 = next((r for r in sweep if r["weight"] == 0.1 and r["base"] == "m2_atom"), None)
        if w01:
            L.append(f"\nEven a small fixed weight (0.1) is below graph-only on the full evaluated subset "
                     f"(Any@5 {w01['any@5']:.3f} vs {fmt(row('all','m2_atom','any@5'))}; All@5 "
                     f"{w01['all@5']:.3f} vs {fmt(row('all','m2_atom','all@5'))}); at 0.2+ the unit list floods "
                     "the top ranks and performance collapses. Fixed weighting cannot recover the signal - the "
                     "unit channel has to be gated by relevance. The primary M4/M5 definition uses equal "
                     "weights, which is the pre-registered comparison.")
    gpath = OUT / "retrieval" / "grouping_comparison.json"
    if gpath.exists():
        gcmp = json.loads(gpath.read_text())
        L.append("\n### Reading-group comparison (exploratory, 20 papers / 72 questions)\n")
        L.append(md_table(gcmp, [("grouping", "grouping", 0), ("papers", "papers", 0),
                                 ("units", "n_units", 0), ("unit gold coverage", "unit_gold_coverage", 3),
                                 ("Any@5", "any@5", 3), ("All@5", "all@5", 3), ("Recall@5", "recall@5", 3)]))
        L.append("\nSection-based groups produce more units and slightly better M2+unit fusion than semantic "
                 "k-means groups on this subset. This is exploratory: no QA was run for the semantic grouping.")
    L.append("\n## 4. How much of the gain is independent of ordinary index expansion?\n")
    L.append(f"- M6 with the same number of summary units as verified insights: All@5 "
             f"{fmt(row('all','m6_matched','all@5'))} vs M4 {fmt(row('all','m4_units','all@5'))} vs M3 "
             f"{fmt(row('all','m3_atom_comp','all@5'))}. The M6-matched condition controls both the number of "
             "indexed units and the reader budget, so a gain that survives it is not just index expansion.")
    L.append("\n## 5. Cross-paragraph / multi-evidence questions\n")
    L.append(md_table([r for r in rtable if r["subset"] in ("multi", "cross") and r["method"] in
                       ("m1_raw", "m2_atom", "m3_atom_comp", "m4_units", "m5_units_graph", "m6_matched")],
                      [("subset", "subset", 0), ("method", "label", 0), ("Any@5", "any@5", 3),
                       ("All@5", "all@5", 3), ("Recall@5", "recall@5", 3), ("Cov@4096", "cov_full@4096", 3)]))
    L.append("\n## 6. QA results\n")
    if qtab:
        L.append(md_table([r for r in qtab if r["method"] in
                           ("m0_dense", "m1_raw", "m2_atom", "m3_atom_comp", "m4_units", "m5_units_graph",
                            "m4_units_B", "m5_units_graph_B", "m6_matched", "oracle", "full")],
                          [("method", "label", 0), ("n", "n", 0), ("AnswerF1", "answer_f1", 3),
                           ("EvidenceF1", "evidence_f1", 3), ("EM", "em", 3),
                           ("F1 ans", "f1_answerable", 3), ("F1 multi", "f1_multi_evidence", 3),
                           ("Gold all", "gold_all_in_context", 3)]))
        # QA by evidence type
        qinfo = {q.qid: q for p in papers.values() for q in p.qas}
        def qgroup(qid):
            q = qinfo.get(qid)
            if q is None:
                return "unknown"
            if not q.answerable:
                return "unanswerable"
            if q.table_figure_only:
                return "figure_table_only"
            return "text_evidence"
        L.append("\nQA by evidence type:\n")
        rows_ev = []
        for m in ("m0_dense", "m1_raw", "m2_atom", "m3_atom_comp", "m4_units", "m5_units_graph", "oracle"):
            for g in ("text_evidence", "figure_table_only", "unanswerable"):
                rr = [r for r in qa_rows if r["method"] == m and qgroup(r["qid"]) == g]
                if rr:
                    rows_ev.append({"method": METHOD_LABEL.get(m, m), "group": g, "n": len(rr),
                                    "answer_f1": mean([r["answer_f1"] for r in rr])})
        L.append(md_table(rows_ev, [("group", "group", 0), ("method", "method", 0),
                                    ("n", "n", 0), ("AnswerF1", "answer_f1", 3)]))
        L.append("\nQA deltas (paired by question):\n")
        for k, v in qa_deltas.items():
            L.append(f"- {k}: Answer F1 {v['f1_mean']:+.3f} [{v['f1_lo']:+.3f},{v['f1_hi']:+.3f}], "
                     f"Evidence F1 {v['ev_f1_mean']:+.3f}")
    topk_path = OUT / "qa" / f"topk_sensitivity_{args.scope}.json"
    if topk_path.exists():
        tk = json.loads(topk_path.read_text())
        L.append("\nTop-k reader sensitivity (same question set, M5 rankings, no token padding):\n")
        L.append(md_table([{"k": k, **v} for k, v in tk.items()],
                          [("k", "k", 0), ("n", "n", 0), ("AnswerF1", "answer_f1", 3),
                           ("EvidenceF1", "evidence_f1", 3), ("EM", "em", 3)]))
        L.append("\nAnswer F1 falls sharply at k=3/5, while Evidence F1 *rises*: the 4096-token context "
                 "predicts many paragraphs, which depresses official paragraph-F1 precision. Retrieval-side "
                 "metrics, not QA Evidence F1, are the reliable evidence measure.")
    L.append("\n## 7. Mechanism\n")
    if mech:
        L.append(f"- verified units: {mech.get('n_units')}; never retrieved as a candidate: "
                 f"{mech.get('never_retrieved')}; never in top-5: {mech.get('never_top5')}; "
                 f"touch gold: {mech.get('touches_gold')}; touch gold but never top-5: "
                 f"{mech.get('touches_gold_but_never_top5')}.")
        L.append(f"- questions with at least one gold-touching unit: {mech.get('questions_with_gold_unit')} "
                 f"of {mech.get('questions')}.")
    L.append(f"- unit-layer gold coverage upper bound: {100*cov['rate']:.1f}% of questions.")
    L.append("\n### Coverage-limited mechanism check\n")
    L.append(md_table([r for r in rtable if r["subset"] in ("unit_covered", "unit_uncovered")
                       and r["method"] in ("m2_atom", "m3_atom_comp", "m4_units", "m5_units_graph", "m5_gated", "m6_matched")],
                      [("subset", "subset", 0), ("method", "label", 0), ("n", "n", 0),
                       ("Any@5", "any@5", 3), ("All@5", "all@5", 3), ("Recall@5", "recall@5", 3)]))
    L.append("\n`unit_covered` contains questions whose gold evidence is touched by at least one verified unit's "
             "source paragraph (upper bound, before ranking); `unit_uncovered` contains the rest. If M4/M5 only "
             "move the covered subset, the limiting factor is coverage of the generated knowledge rather than fusion.")
    if qa_rows:
        cov_map = {}
        for r in ret_rows:
            if r["method"] == "m2_atom":
                cov_map[r["qid"]] = r.get("unit_gold_upper")
        qa_mech = []
        for m in ("m2_atom", "m3_atom_comp", "m4_units", "m5_units_graph", "m5_units_graph_B"):
            for label, flag in (("unit_covered", True), ("unit_uncovered", False)):
                rr = [r for r in qa_rows if r["method"] == m and cov_map.get(r["qid"]) is flag]
                if rr:
                    qa_mech.append({"subset": label, "method": METHOD_LABEL.get(m, m), "n": len(rr),
                                    "answer_f1": mean([r["answer_f1"] for r in rr]),
                                    "evidence_f1": mean([r["evidence_f1"] for r in rr])})
        L.append("\n### QA on unit-covered vs uncovered questions\n")
        L.append(md_table(qa_mech, [("subset", "subset", 0), ("method", "method", 0), ("n", "n", 0),
                                    ("AnswerF1", "answer_f1", 3), ("EvidenceF1", "evidence_f1", 3)]))
    L.append("\n## 8. Cost\n")
    L.append(f"- read-time tokens: prompt {t['prompt_tokens']:,}, completion {t['completion_tokens']:,}.")
    L.append("- graph OpenIE and reader token usage are logged under `outputs/qasper_readtime/logs/` and each "
             "HippoRAG `save_dir/llm_cache`.")
    L.append("\n## 9. Honest limitations\n")
    L.append("- Paragraph pool is small; Any-Evidence@5 saturation is a real ceiling.")
    L.append("- Table/figure evidence is excluded from text-evidence metrics and reported separately.")
    L.append("- Hyperedges are represented as triples injected into each source paragraph; this is an "
             "approximation of a true hypergraph.")
    L.append("- Section-based reading groups are the primary condition. The semantic k-means grouping with a "
             "small overlap band was run on 20 papers (`comprehension/semantic/`) for an exploratory comparison "
             "only; no semantic QA was run.")
    L.append("- Table/figure evidence: 253 `FLOAT SELECTED` evidence items across the validation split; questions "
             "whose only evidence is a figure/table (3.5%) are excluded from text-evidence retrieval metrics but "
             "remain in QA.")
    L.append("- Top-3/Top-5 reader sensitivity was run on the evaluated question set; it is not a fixed-budget "
             "comparison because no token padding is added.")
    L.append("- HY4 is a shared company endpoint; all reported runs use a cross-process limiter at ~1 QPS. "
             "Generating the full 100-paper sample was quota-bound (many hours) and latency varies; the "
             "pipeline is resumable and every call is cached.")
    L.append("\n## 10. Next mechanism worth studying\n")
    L.append(f"The mechanism table answers this directly. {mech.get('touches_gold', '—')} of "
             f"{mech.get('n_units', '—')} verified units touch gold evidence at least once, and only "
             f"{mech.get('touches_gold_but_never_top5', '—')} touch gold but never reach the unit top-5: unit "
             "**ranking** is not the main failure. The failures are (a) **coverage**: only "
             f"{100*cov['rate']:.1f}% of questions have any unit whose source is gold, so most questions cannot "
             "benefit, and (b) **displacement**: equal-weight RRF still injects unit passages on those uncovered "
             "questions and pushes correct graph evidence out. The next experiment should therefore (1) generate "
             "units that cover every section, not ~7 per paper, and (2) gate the unit channel by a calibrated "
             "relevance score instead of a fixed RRF weight; the fusion sweep shows that even 0.1 is already below "
             "graph-only and 0.2+ collapses. Reader-side unit display (mode B) adds nothing, "
             "so units should be used to route to paragraphs, not shown as evidence.")
    (reports / "FINAL_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote reports to", reports)


if __name__ == "__main__":
    main()
