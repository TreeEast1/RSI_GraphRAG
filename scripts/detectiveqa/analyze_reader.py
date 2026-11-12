"""Reader failure decomposition and duplicate-question agreement analysis."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detectiveqa.data import OUT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reader_pilot.jsonl")
    args = ap.parse_args()
    rows = [json.loads(line) for line in (OUT / args.input).read_text().splitlines() if line.strip()]
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    summary: Dict = {"n": len(rows), "conditions": {}}
    for cond in conditions:
        sub = [r for r in rows if r["condition"] == cond]
        reached = [r for r in sub if set(r["gold_chunk_ids"]) & set(r["context_chunk_ids"])]
        unreached = [r for r in sub if not (set(r["gold_chunk_ids"]) & set(r["context_chunk_ids"]))]
        wrong = [r for r in sub if not r["scores"]["correct"]]
        summary["conditions"][cond] = {
            "n": len(sub),
            "accuracy": round(mean(r["scores"]["correct"] for r in sub), 4),
            "n_wrong": len(wrong),
            "n_gold_reached": len(reached),
            "reached_but_wrong": sum(1 for r in reached if not r["scores"]["correct"]),
            "not_reached": len(unreached),
            "not_reached_and_wrong": sum(1 for r in unreached if not r["scores"]["correct"]),
            "n_insufficient": sum(r["scores"]["insufficient"] for r in sub),
            "accuracy_by_novel": {n: round(mean(r["scores"]["correct"] for r in sub if r["novel_id"] == n), 3)
                                  for n in sorted({r["novel_id"] for r in sub}, key=int)},
        }
    # duplicate 117 annotations: same underlying question, different wording
    agree = {}
    for cond in conditions:
        base_pred = {}
        dup_pred = {}
        for r in rows:
            if r["condition"] != cond:
                continue
            if r["qid"].startswith("human_anno:117:149:"):
                dup_pred[r["qid"].split(":")[-1]] = (r["prediction"]["answer"], r["scores"]["correct"])
            elif r["qid"].startswith("human_anno:117:"):
                base_pred[r["qid"].split(":")[-1]] = (r["prediction"]["answer"], r["scores"]["correct"])
        common = sorted(set(base_pred) & set(dup_pred))
        agree[cond] = {
            "n_pairs": len(common),
            "same_answer": sum(base_pred[i][0] == dup_pred[i][0] for i in common),
            "both_correct": sum(base_pred[i][1] and dup_pred[i][1] for i in common),
            "exactly_one_correct": sum(base_pred[i][1] != dup_pred[i][1] for i in common),
        }
    summary["duplicate_117_pairs"] = agree
    # samples
    summary["reached_but_wrong_samples"] = [
        {k: r[k] for k in ("qid", "condition", "question", "gold_answer", "prediction",
                           "gold_chunk_ids", "context_chunk_ids")}
        for r in rows if not r["scores"]["correct"] and
        set(r["gold_chunk_ids"]) & set(r["context_chunk_ids"])][:20]
    summary["not_reached_samples"] = [
        {k: r[k] for k in ("qid", "condition", "question", "gold_answer", "prediction",
                           "gold_chunk_ids", "context_chunk_ids")}
        for r in rows if not r["scores"]["correct"] and
        not (set(r["gold_chunk_ids"]) & set(r["context_chunk_ids"]))][:20]
    (OUT / "reader_analysis.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False)[:12000])


if __name__ == "__main__":
    main()
