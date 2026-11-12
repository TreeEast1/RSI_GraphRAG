#!/usr/bin/env python
"""Stage 1 - fixed baseline Readers (P0, P7, P8) on the three subsets.

P0/P7 rows are reused from the frozen prompt-optimization joint run with the
same COMP-P4_top5 retrieval and 4096-token context.  P8 is run here because it
was not previously evaluated on this retrieval variant.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from . import common


def p8_one(qidx: int) -> dict:
    context, sel, ntok = common.record_budget_docs(qidx)
    rec = common.answer(qidx, pid="P8", stage="baseline", purpose="P8_reader",
                        context_override=(context, sel, ntok))
    rec["pid"] = "P8"
    rec["context"] = context
    rec["answer_support"] = common.answer_support(rec["pred_answer"], context)
    return rec


def enrich(row: dict, pid: str) -> dict:
    row = dict(row)
    row["pid"] = pid
    context, sel, ntok = common.record_budget_docs(int(row["qidx"]))
    row["context"] = context
    row["answer_support"] = common.answer_support(row.get("pred_answer", ""), context)
    return row


def main():
    qids = sorted(common.qa_map())
    rows = {"P0": [], "P7": [], "P8": []}
    for pid in ("P0", "P7"):
        for q in qids:
            r = common.answer_context_cached(q, pid=pid)
            if r:
                rows[pid].append(enrich(r, pid))
    todo = [q for q in qids if not any(r["qidx"] == q for r in rows["P8"])]
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows["P8"] = list(ex.map(p8_one, todo))

    out_rows = []
    summary = {}
    split = common.split()
    for pid, rs in rows.items():
        out_rows.extend(rs)
        s = common.summarize(rs)
        summary[pid] = {"all": s, "subsets": {}}
        for name in ("adaptation", "development", "evaluation"):
            qs = set(common.qidx_list(name))
            sub = [r for r in rs if int(r["qidx"]) in qs]
            summary[pid]["subsets"][name] = common.summarize(sub)
        summary[pid]["answer_support"] = common.mean(r["answer_support"] for r in rs)
    common.write_jsonl(common.OUT / "artifacts" / "baseline_predictions.jsonl", out_rows)
    common.write_json(common.OUT / "artifacts" / "baseline_summary.json",
                      {"summaries": summary, "base_variant": common.BASE_VARIANT,
                       "context_budget": 4096})

    lines = ["# Baseline Results (conv-30)", "",
             f"Retrieval: frozen **{common.BASE_VARIANT}** ranked sessions, 4096-token context. "
             "P0/P7 reused from `outputs/prompt_optimization_conv30/joint`; P8 run here.", "",
             "## All 105 questions", "",
             "| Reader | EM | F1 | official F1 | adversarial refusal | answer support |",
             "|---|---:|---:|---:|---:|---:|"]
    for pid in ("P0", "P7", "P8"):
        s = summary[pid]["all"]
        lines.append(f"| {pid} | {s['project_em']:.3f} | {s['project_f1']:.3f} | "
                     f"{s['official_f1']:.3f} | {s['adversarial_refusal']:.3f} | "
                     f"{summary[pid]['answer_support']:.3f} |")
    lines += ["", "## By subset (project EM / F1 / official F1 / adversarial refusal)", "",
              "| Reader | subset | n | EM | F1 | official F1 | adv refusal |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for pid in ("P0", "P7", "P8"):
        for name in ("adaptation", "development", "evaluation"):
            s = summary[pid]["subsets"][name]
            lines.append(f"| {pid} | {name} | {s['n']} | {s['project_em']:.3f} | "
                         f"{s['project_f1']:.3f} | {s['official_f1']:.3f} | "
                         f"{s['adversarial_refusal']:.3f} |")
    lines += ["", "## By category (all 105; project EM / F1)", "",
              "| Reader | multi-hop | temporal | single-hop |", "|---|---:|---:|---:|"]
    for pid in ("P0", "P7", "P8"):
        bc = summary[pid]["all"]["by_category"]
        lines.append(f"| {pid} | " + " | ".join(
            f"{bc[c]['em']:.3f} / {bc[c]['f1']:.3f}" for c in ("1", "2", "4")) + " |")
    lines += ["", "Notes:",
              "- Project EM/F1 use the SQuAD-style normalizer; official F1 is the LoCoMo scorer.",
              "- Adversarial refusal is the fraction of category-5 questions answered with a refusal.",
              "- The adaptation and development columns are the fixed selection signals; the "
              "evaluation column is untouched until the final snapshot is chosen.", ""]
    (common.OUT / "baseline_results.md").write_text("\n".join(lines), encoding="utf-8")
    for pid in ("P0", "P7", "P8"):
        print(pid, {k: round(v, 3) for k, v in summary[pid]["all"].items()
                    if isinstance(v, float)})


if __name__ == "__main__":
    main()
