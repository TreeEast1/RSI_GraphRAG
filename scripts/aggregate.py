#!/usr/bin/env python
"""Aggregate every per-question result file into metrics.csv + a JSON summary."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PATHS  # noqa: E402
from src.evaluation.pipeline import summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*.jsonl")
    ap.add_argument("--out", default="metrics.csv")
    args = ap.parse_args()

    split = json.loads((PATHS["splits"] / "split.json").read_text())
    dev, test = set(split["dev"]), set(split["test"])

    rows = defaultdict(list)  # (method, split) -> rows
    for p in sorted((PATHS["qa"]).glob(args.pattern)):
        if "smoke" in p.name:
            continue
        try:
            recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        except json.JSONDecodeError:
            continue
        if not recs or "method" not in recs[0]:
            continue
        method = recs[0]["method"]
        if method == "fullcontext":
            # full-context has no retrieval step; retrieval/budget metrics are N/A
            for r in recs:
                r["retrieval"] = {}
                r["budget"] = {}
        for split_name, ids in (("dev", dev), ("test", test), ("all", dev | test)):
            if split_name == "all" and {r.get("conv_id") for r in recs} != (dev | test):
                continue  # do not emit a mixed-coverage "all" row
            sub = [r for r in recs if r.get("conv_id") in ids]
            if sub:
                rows[(method, split_name)].extend(sub)

    fields = ["method", "split", "n", "em", "f1", "recall@1", "recall@3", "recall@5", "recall@10",
              "recall@20", "mrr", "ndcg@10", "em_answerable", "f1_answerable", "refusal_rate", "evidence_full@2048", "evidence_partial@2048",
              "evidence_full@4096", "evidence_partial@4096", "evidence_full@8192",
              "evidence_partial@8192", "session_full@4096"]
    out_rows = []
    summary = {}
    for (method, split_name), recs in sorted(rows.items()):
        s = summarize(recs, method)
        row = {"method": method, "split": split_name}
        for f in fields[2:]:
            row[f] = s.get(f)
        out_rows.append(row)
        summary[f"{method}|{split_name}"] = s

    csv_path = PATHS["reports"] / args.out
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    (PATHS["reports"] / "metrics_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"wrote {csv_path} ({len(out_rows)} rows)")

    for r in out_rows:
        if r["split"] == "all":
            fmt = lambda v: "  N/A" if v is None else f"{v:.3f}"
            print(f"{r['method']:38s} EM={fmt(r['em'])} F1={fmt(r['f1'])} "
                  f"R@5={fmt(r['recall@5'])} R@20={fmt(r['recall@20'])} "
                  f"ev@4096={fmt(r['evidence_full@4096'])}")


if __name__ == "__main__":
    main()
