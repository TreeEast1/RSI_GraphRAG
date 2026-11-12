#!/usr/bin/env python
"""Generate the experiment reports from measured artefacts only.

Every number written here comes from a file produced by an actual run:
  outputs/reports/metrics_summary.json     (scripts/aggregate.py)
  outputs/reports/grouping_runs.json       (scripts/run_grouping.py)
  outputs/reports/comprehension_audit.json (scripts/audit_comprehension.py)
  outputs/reports/qa_*_summary.json        (read-time / question-time runners)
Missing data is printed as "not run" — never fabricated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PATHS  # noqa: E402


def fmt(v, nd=3):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def load(name):
    p = PATHS["reports"] / name
    return json.loads(p.read_text()) if p.exists() else None


def metric_row(m, s):
    return (f"| {m} | {fmt(s.get('recall@5'))} | {fmt(s.get('recall@20'))} | "
            f"{fmt(s.get('evidence_full@4096'))} | {fmt(s.get('em'))} | {fmt(s.get('f1'))} |")


def write_grouping():
    data = load("grouping_runs.json")
    lines = ["# GROUPING_REPORT — experiment B", ""]
    if not data:
        lines.append("**Not run.** `outputs/reports/grouping_runs.json` is missing.")
        (PATHS["reports"] / "GROUPING_REPORT.md").write_text("\n".join(lines))
        return
    lines += [
        "All groupings are built from unsupervised signals only (temporal order, entity sets "
        "from OpenIE NER, NV-Embed-v2 document vectors). Gold evidence is used exclusively to score them.",
        "",
        "Columns: gold_full = fraction of questions whose *complete* gold session set falls inside a single "
        "reading group; gold_partial = best fraction of gold sessions co-located in one group; "
        "cross = same as gold_full but restricted to questions with >=2 gold sessions; "
        "cost = total reading tokens / corpus tokens.",
        "",
        "| conv | method | budget | groups | gold_full | gold_partial | cross | avg tokens | max tokens | cost |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(data, key=lambda x: (x["conv_id"], x["budget"], x["method"], x["seed"])):
        lines.append(
            f"| {r['conv_id']} | {r['method']}{'(s'+str(r['seed'])+')' if r['method']=='random' else ''} | "
            f"{r['budget']} | {r['n_groups']} | {fmt(r['gold_full_coverage'])} | "
            f"{fmt(r['gold_partial_coverage'])} | {fmt(r['cross_session_colocation'])} | "
            f"{fmt(r['avg_group_tokens'],0)} | {r['max_group_tokens']} | {fmt(r['reading_cost_ratio'],2)} |")
    lines += ["", "See `outputs/groupings/` for the exact session-id membership of every group."]
    (PATHS["reports"] / "GROUPING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_readtime():
    base = load("metrics_summary.json") or {}
    rt = load("readtime_summary.json") or {}
    lines = ["# READ_TIME_REPORT — experiment C", "",
             "Every method below uses the same reader, top-20 cut and 4096-token reader budget.", "",
             "| method | Recall@5 | Recall@20 | Evidence full@4096 | EM | F1 |",
             "|---|---|---|---|---|---|"]
    for key, s in base.items():
        m, split = key.split("|")
        if split == "all":
            lines.append(metric_row(m, s))
    if rt.get("summaries"):
        for m, s in rt["summaries"].items():
            lines.append(metric_row(f"{m} (dev)", s))
    lines += ["", "## Graph builds", ""]
    if rt.get("builds"):
        lines += ["| conv | variant | extra records | graph |", "|---|---|---|---|"]
        for b in rt["builds"]:
            lines.append(f"| {b.get('conv_id')} | {b.get('variant')} | "
                         f"{b.get('injection', {}).get('added', 'reused')} | "
                         f"{json.dumps(b.get('graph_info', {}), ensure_ascii=False)[:120]} |")
    else:
        lines.append("Read-time graph variants: not run.")
    (PATHS["reports"] / "READ_TIME_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_question():
    qt = load("qtime_summary.json")
    lines = ["# QUESTION_TIME_REPORT — experiment E", ""]
    if not qt:
        lines.append("**Not run.**")
    else:
        lines += ["| method | Recall@5 | Recall@20 | Evidence full@4096 | EM | F1 |",
                  "|---|---|---|---|---|---|"]
        for m, s in qt["summaries"].items():
            lines.append(metric_row(m, s))
        lines += ["", "## Behaviour statistics", "```json",
                  json.dumps(qt.get("stats", {}), indent=2), "```"]
    (PATHS["reports"] / "QUESTION_TIME_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_audit():
    data = load("comprehension_audit.json")
    lines = ["# Comprehension quality audit", ""]
    if not data:
        lines.append("**Not run.**")
    else:
        lines += ["Classes: A = duplicate of G_raw, B = paraphrase of G_raw, C = single-session, "
                  "D = cross-session supported, E = low-confidence inference.", "",
                  "| conv | scope | relations | A | B | C | D | E | %D |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in data:
            c = r["counts"]
            lines.append(f"| {r['conv_id']} | {r['scope']} | {r['n_relations']} | {c.get('A',0)} | "
                         f"{c.get('B',0)} | {c.get('C',0)} | {c.get('D',0)} | {c.get('E',0)} | "
                         f"{fmt(r['pct_D'])} |")
    (PATHS["reports"] / "COMPREHENSION_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    write_grouping()
    write_readtime()
    write_question()
    write_audit()
    print("reports written to", PATHS["reports"])


if __name__ == "__main__":
    main()
