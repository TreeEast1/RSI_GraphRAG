"""Stage A case diagnosis: dense vs graph ranks, facts and provenance.

Reads the round2 retrieval JSONL (no API).  Produces condition means, novel-clustered
paired tests and the dense_better / graph_better case lists required by the plan.
"""
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

from src.detectiveqa.data import OUT, read_jsonl  # noqa: E402
from src.detectiveqa.round2 import bootstrap_paired_by_novel  # noqa: E402

CONDS = ["q_dense", "q_bm25", "q_graph_f5", "q_graph_f10", "q_graph_f20",
         "q_graph_f5_nosyn", "q_rrf_dense_graph", "q_rrf_dense_bm25", "q_rrf_all",
         "qo_dense", "qo_graph_f5", "qo_rrf_dense_graph"]
METRICS = ("recall@10", "recall@20", "all@20", "all@50", "mrr")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", default="round2_retrieval.jsonl")
    ap.add_argument("--out", default="stageA_cases")
    args = ap.parse_args()
    rows = read_jsonl(OUT / args.retrieval)
    report: Dict = {"n": len(rows), "condition_means": {}, "paired": {}, "cases": {},
                    "by_bucket": {}, "by_novel": {}}
    for cond in CONDS:
        if cond not in rows[0]["conditions"]:
            continue
        report["condition_means"][cond] = {
            m: round(mean(r["conditions"][cond][m] for r in rows), 4) for m in METRICS}
    for right in ("q_graph_f5", "q_graph_f10", "q_graph_f20", "q_graph_f5_nosyn",
                  "q_bm25", "q_rrf_dense_graph", "q_rrf_dense_bm25", "q_rrf_all"):
        if "q_dense" not in rows[0]["conditions"] or right not in rows[0]["conditions"]:
            continue
        paired = [{"novel_id": r["novel_id"], "q_dense": r["conditions"]["q_dense"],
                   right: r["conditions"][right]} for r in rows]
        for metric in ("recall@10", "all@20", "mrr"):
            report["paired"][f"q_dense->{right}:{metric}"] = bootstrap_paired_by_novel(
                paired, "q_dense", right, metric=metric)
    # q-only vs qo (diagnostic)
    for left, right in (("q_dense", "qo_dense"), ("q_graph_f5", "qo_graph_f5")):
        if left in rows[0]["conditions"] and right in rows[0]["conditions"]:
            paired = [{"novel_id": r["novel_id"], left: r["conditions"][left],
                       right: r["conditions"][right]} for r in rows]
            report["paired"][f"{left}->{right}:recall@10"] = bootstrap_paired_by_novel(
                paired, left, right, metric="recall@10")

    def case_row(r: Dict) -> Dict:
        d = r["conditions"]["q_dense"]["first_gold_rank"]
        g = r["conditions"]["q_graph_f5"]["first_gold_rank"]
        facts = r["conditions"]["q_graph_f5"].get("facts", [])
        prov = r["conditions"]["q_graph_f5"].get("fact_provenance", [])
        return {"qid": r["qid"], "novel_id": r["novel_id"], "bucket": r["cutoff_bucket"],
                "cutoff_ratio": round(r["cutoff_ratio"], 3), "n_gold": len(r["gold_chunk_ids"]),
                "dense_rank": d, "graph_rank": g,
                "gold_chunk_ids": r["gold_chunk_ids"],
                "dense_gold_ranks": r["conditions"]["q_dense"]["gold_chunk_ranks"],
                "graph_gold_ranks": r["conditions"]["q_graph_f5"]["gold_chunk_ranks"],
                "dense_top10": r["conditions"]["q_dense"]["ranked"][:10],
                "graph_top10": r["conditions"]["q_graph_f5"]["ranked"][:10],
                "graph_facts": facts,
                "graph_fact_provenance": prov,
                "graph_mode": r["conditions"]["q_graph_f5"].get("mode")}
    dense_better, graph_better = [], []
    for r in rows:
        c = case_row(r)
        d, g = c["dense_rank"], c["graph_rank"]
        if d == g:
            continue
        if (g and not d) or (g and d and g < d):
            graph_better.append(c)
        else:
            dense_better.append(c)
    report["cases"]["dense_better"] = dense_better
    report["cases"]["graph_better"] = graph_better
    report["cases"]["counts"] = {"dense_better": len(dense_better), "graph_better": len(graph_better)}
    for name, cases in (("dense_better", dense_better), ("graph_better", graph_better)):
        report["cases"][f"{name}_by_bucket"] = dict(Counter(c["bucket"] for c in cases))
        report["cases"][f"{name}_by_novel"] = dict(Counter(c["novel_id"] for c in cases))
    for bucket in sorted({r["cutoff_bucket"] for r in rows}):
        sub = [r for r in rows if r["cutoff_bucket"] == bucket]
        report["by_bucket"][bucket] = {"n": len(sub)}
        for cond in ("q_dense", "q_bm25", "q_graph_f5", "q_graph_f20", "q_rrf_dense_graph"):
            if cond in sub[0]["conditions"]:
                report["by_bucket"][bucket][cond] = {
                    m: round(mean(r["conditions"][cond][m] for r in sub), 4)
                    for m in ("recall@10", "all@20", "mrr")}
    for novel in sorted({r["novel_id"] for r in rows}, key=int):
        sub = [r for r in rows if r["novel_id"] == novel]
        report["by_novel"][novel] = {"n": len(sub)}
        for cond in ("q_dense", "q_bm25", "q_graph_f5", "q_graph_f20", "q_rrf_dense_graph"):
            if cond in sub[0]["conditions"]:
                report["by_novel"][novel][cond] = {
                    m: round(mean(r["conditions"][cond][m] for r in sub), 4)
                    for m in ("recall@10", "all@20", "mrr")}
    (OUT / f"{args.out}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    lines = ["# Stage A retrieval diagnosis", "", f"n={len(rows)}", "",
             "## Condition means", "", "| condition | R@10 | R@20 | All@20 | All@50 | MRR |",
             "|---|---:|---:|---:|---:|---:|"]
    for cond, vals in report["condition_means"].items():
        lines.append(f"| {cond} | {vals['recall@10']:.3f} | {vals['recall@20']:.3f} | "
                     f"{vals['all@20']:.3f} | {vals['all@50']:.3f} | {vals['mrr']:.3f} |")
    lines += ["", "## Paired (novel-clustered bootstrap, delta, 95% CI, wins/losses, McNemar p)", ""]
    for key, val in report["paired"].items():
        lines.append(f"- {key}: delta {val['delta']:+.4f}, CI [{val['ci_low']:+.4f},{val['ci_high']:+.4f}], "
                     f"wins {val['wins']}, losses {val['losses']}, p={val['mcnemar_exact_p']:.4g}")
    lines += ["", "## Case counts", "",
              f"- dense_better {report['cases']['counts']['dense_better']} "
              f"({report['cases']['dense_better_by_bucket']})",
              f"- graph_better {report['cases']['counts']['graph_better']} "
              f"({report['cases']['graph_better_by_bucket']})"]
    (OUT / f"{args.out}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
