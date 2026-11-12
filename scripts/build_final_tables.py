#!/usr/bin/env python
"""Build the final result tables from artefacts on disk.

Outputs `outputs/reports/FINAL_TABLES.md`.  Every value is computed from files
written by actual runs; missing experiments are shown as `N/A (not run)`.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PATHS  # noqa: E402
from src.evaluation.pipeline import summarize  # noqa: E402

SPLIT = {}


def fmt(v, nd=3):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def load_rows(prefix: str, split: str | None = None):
    rows = []
    for p in sorted((PATHS["qa"]).glob(f"*.jsonl")):
        if "smoke" in p.name or not p.name.startswith(prefix):
            continue
        try:
            recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        except json.JSONDecodeError:
            continue
        if not recs:
            continue
        if split and recs[0].get("conv_id") not in SPLIT.get(split, set()):
            continue
        rows.extend(recs)
    return rows


def metric_line(name, s, n=None):
    # full-context has no retrieval stage: those columns are genuinely N/A
    no_retrieval = "fullconv" in name or name == "fullcontext"
    r5 = None if no_retrieval else s.get("recall@5")
    r20 = None if no_retrieval else s.get("recall@20")
    ev = None if no_retrieval else s.get("evidence_full@4096")
    return (f"| {name} | {n if n is not None else s.get('n')} | {fmt(s.get('em_answerable'))} | "
            f"{fmt(s.get('f1_answerable'))} | {fmt(r5)} | {fmt(r20)} | "
            f"{fmt(ev)} | {fmt(s.get('refusal_rate'))} |")


def base_table(f, split):
    rows = load_rows("baselines_", split)
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    f.write(f"\n### Baselines — {split}\n\n")
    f.write("| method | n | EM (ans.) | F1 (ans.) | R@5 | R@20 | Evidence full@4096 | Refusal (cat5) |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    for m in ("dense", "hipporag", "fullcontext"):
        if m in by:
            f.write(metric_line(m, summarize(by[m], m)) + "\n")
        else:
            f.write(f"| {m} | N/A (not run) | | | | | | |\n")


def readtime_table(f, split):
    rows = load_rows("readtime_", split)
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    f.write(f"\n### Read-time — {split}\n\n")
    f.write("| graph variant | n | EM (ans.) | F1 (ans.) | R@5 | R@20 | Evidence full@4096 | Refusal |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    if not by:
        f.write("| N/A (not run) | | | | | | | |\n")
    for m, rs in sorted(by.items()):
        f.write(metric_line(m, summarize(rs, m)) + "\n")


def qtime_table(f, split):
    rows = load_rows("qtime_", split)
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    f.write(f"\n### Question-time — {split}\n\n")
    f.write("| strategy | n | EM (ans.) | F1 (ans.) | R@5 | R@20 | Evidence full@4096 |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    if not by:
        f.write("| N/A (not run) | | | | | | |\n")
    for m, rs in sorted(by.items()):
        s = summarize(rs, m)
        f.write(f"| {m} | {s.get('n')} | {fmt(s.get('em_answerable'))} | {fmt(s.get('f1_answerable'))} | "
                f"{fmt(s.get('recall@5'))} | {fmt(s.get('recall@20'))} | "
                f"{fmt(s.get('evidence_full@4096'))} |\n")


def grouping_table(f):
    recs = []
    for p in sorted((PATHS["groupings"]).glob("*/*.json")):
        try:
            recs.append(json.loads(p.read_text())["result"])
        except Exception:
            continue
    if not recs:
        f.write("\n### Grouping\n\n| N/A (not run) |\n")
        return
    agg = defaultdict(list)
    for r in recs:
        agg[(r["method"], r["budget"])].append(r)
    f.write("\n### Grouping (mean over dev+test conversations, all budgets)\n\n")
    f.write("| method | budget | convs | gold full | gold partial | cross-session co-location | "
            "avg group tokens | max group tokens |\n|---|---|---|---|---|---|---|---|\n")
    for (m, b), rs in sorted(agg.items()):
        import numpy as np
        f.write(f"| {m} | {b} | {len(rs)} | {fmt(float(np.mean([x['gold_full_coverage'] for x in rs])))} | "
                f"{fmt(float(np.mean([x['gold_partial_coverage'] for x in rs])))} | "
                f"{fmt(float(np.nanmean([x['cross_session_colocation'] for x in rs])))} | "
                f"{fmt(float(np.mean([x['avg_group_tokens'] for x in rs])),0)} | "
                f"{int(max(x['max_group_tokens'] for x in rs))} |\n")


def main():
    global SPLIT
    sp = json.loads((PATHS["splits"] / "split.json").read_text())
    SPLIT = {"dev": set(sp["dev"]), "test": set(sp["test"])}

    out = PATHS["reports"] / "FINAL_TABLES.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Final result tables (auto-generated from run artefacts)\n")
        f.write("\nMain QA columns exclude category-5 (adversarial) questions; those are scored as refusal.\n")
        for split in ("dev", "test"):
            f.write(f"\n## {split.upper()}\n")
            base_table(f, split)
            readtime_table(f, split)
            qtime_table(f, split)
        f.write("\n## Grouping\n")
        grouping_table(f)
        audit = PATHS["reports"] / "comprehension_audit.json"
        if audit.exists():
            data = json.loads(audit.read_text())
            f.write("\n### Comprehension audit\n\n| conv | scope | relations | A | B | C | D | E | %D |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in data:
                c = r["counts"]
                f.write(f"| {r['conv_id']} | {r['scope']} | {r['n_relations']} | {c.get('A',0)} | "
                        f"{c.get('B',0)} | {c.get('C',0)} | {c.get('D',0)} | {c.get('E',0)} | "
                        f"{fmt(r['pct_D'])} |\n")
    print("wrote", out)
    print(out.read_text()[:3000])


if __name__ == "__main__":
    main()
