"""Summarize DetectiveQA retrieval runs into report-ready tables."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detectiveqa.data import OUT  # noqa: E402

CONDS = ("q_dense", "q_graph", "qo_dense", "qo_graph")
METRICS = ("any@5", "any@10", "any@20", "recall@5", "recall@10", "recall@20",
           "all@5", "all@10", "all@20")


def m(rows: List[Dict], cond: str, metric: str) -> float:
    key = f"{cond}_{metric}"
    vals = [r[key] for r in rows if key in r]
    return round(mean(vals), 4) if vals else float("nan")


def mrr(rows: List[Dict], cond: str) -> float:
    """Standard MRR@k over all questions: a missed gold chunk contributes 0."""
    vals = [(1.0 / r[f"{cond}_gold_rank"]) if r.get(f"{cond}_gold_rank") else 0.0 for r in rows]
    return round(mean(vals), 4) if vals else 0.0


def hit(rows: List[Dict], cond: str, k: int) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for r in rows if 0 < r.get(f"{cond}_gold_rank", 0) <= k) / len(rows), 4)


def table(rows: List[Dict], title: str) -> List[str]:
    lines = [f"### {title} (n={len(rows)})", "",
             "| method | Any@5 | Any@10 | Any@20 | R@5 | R@10 | R@20 | All@5 | All@10 | All@20 | MRR | Hit@20 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cond in CONDS:
        if not any(f"{cond}_any@5" in r for r in rows):
            continue
        vals = [m(rows, cond, x) for x in METRICS]
        lines.append(f"| {cond} | " + " | ".join(f"{v:.3f}" for v in vals) +
                     f" | {mrr(rows, cond):.3f} | {hit(rows, cond, 20):.3f} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="retrieval_eval_human_raw_both.jsonl")
    ap.add_argument("--out", default="retrieval_summary.md")
    args = ap.parse_args()
    rows = [json.loads(line) for line in (OUT / args.input).read_text().splitlines() if line.strip()]
    dedup = [r for r in rows if ":149:" not in r["qid"]]
    lines = ["# DetectiveQA retrieval summary", "", f"Input: `{args.input}`",
             "", "All metrics are over all questions; MRR counts a missed gold chunk as 0.", ""]
    lines += table(rows, "All pilot human questions") + [""]
    lines += table(dedup, "Sensitivity: drop the five duplicate novel-117 annotations") + [""]
    # fallback
    lines += ["### Dense fallback (graph mode)", "",
              "| condition | fallback ratio |", "|---|---:|"]
    for cond in ("q_graph", "qo_graph"):
        vals = [r[f"{cond}_fallback"] for r in rows if f"{cond}_fallback" in r]
        if vals:
            lines.append(f"| {cond} | {mean(vals):.3f} |")
    lines.append("")
    # per novel
    by_novel = defaultdict(list)
    for r in rows:
        by_novel[r["novel_id"]].append(r)
    lines += ["### Per-novel Recall@10 (qo)", "", "| novel | n | dense R@10 | graph R@10 | dense MRR | graph MRR |", "|---|---:|---:|---:|---:|---:|"]
    for novel in sorted(by_novel, key=int):
        sub = by_novel[novel]
        lines.append(f"| {novel} | {len(sub)} | {m(sub,'qo_dense','recall@10'):.3f} | "
                     f"{m(sub,'qo_graph','recall@10'):.3f} | {mrr(sub,'qo_dense'):.3f} | {mrr(sub,'qo_graph'):.3f} |")
    lines.append("")
    # cases where graph beats / loses
    wins = [r for r in rows if r.get("qo_graph_gold_rank") and
            (not r.get("qo_dense_gold_rank") or r["qo_graph_gold_rank"] < r["qo_dense_gold_rank"])]
    losses = [r for r in rows if r.get("qo_dense_gold_rank") and
              (not r.get("qo_graph_gold_rank") or r["qo_dense_gold_rank"] < r["qo_graph_gold_rank"])]
    lines += [f"### Rank cases (qo): graph better {len(wins)}, dense better {len(losses)}", ""]
    for label, cases in (("graph_better", wins[:8]), ("dense_better", losses[:8])):
        lines.append(f"**{label}**")
        for r in cases:
            lines.append(f"- {r['qid']}: dense rank {r.get('qo_dense_gold_rank')}, "
                         f"graph rank {r.get('qo_graph_gold_rank')}, gold {r['n_gold_chunks']}")
        lines.append("")
    (OUT / args.out).write_text("\n".join(lines))
    summary = {"n": len(rows), "n_dedup": len(dedup)}
    for cond in CONDS:
        summary[cond] = {x: m(rows, cond, x) for x in METRICS}
        summary[cond]["mrr"] = mrr(rows, cond)
        summary[cond]["hit@20"] = hit(rows, cond, 20)
    (OUT / "retrieval_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
