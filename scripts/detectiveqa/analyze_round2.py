"""Deterministic round-two analysis tables and case queues."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detectiveqa.data import OUT, read_jsonl  # noqa: E402
from src.detectiveqa.round2 import bootstrap_paired_by_novel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", default="round2_retrieval.jsonl")
    ap.add_argument("--reader", default="round2_reader.jsonl")
    ap.add_argument("--relation", default="relation_channel.jsonl")
    ap.add_argument("--out", default="round2_analysis")
    args = ap.parse_args()
    retrieval = read_jsonl(OUT / args.retrieval)
    reader = read_jsonl(OUT / args.reader) if (OUT / args.reader).exists() else []
    relation = read_jsonl(OUT / args.relation) if (OUT / args.relation).exists() else []
    report = {"retrieval": {}, "reader": {}, "cases": {}}
    for name in ("q_dense", "q_bm25", "q_graph_f5", "q_graph_f5_nosyn", "q_graph_f20", "q_rrf_dense_graph", "qo_dense", "qo_graph_f5"):
        if name not in retrieval[0]["conditions"]:
            continue
        rows = [r["conditions"][name] for r in retrieval]
        report["retrieval"][name] = {k: round(mean(x[k] for x in rows), 4)
                                     for k in ("recall@10", "recall@20", "all@20", "all@50", "mrr")}
    comparisons = (("q_dense", "q_graph_f5"), ("q_dense", "q_rrf_dense_graph"), ("q_graph_f5", "q_graph_f5_nosyn"))
    report["retrieval_paired"] = {}
    for left, right in comparisons:
        if left in retrieval[0]["conditions"] and right in retrieval[0]["conditions"]:
            paired = [{"novel_id": r["novel_id"], left: r["conditions"][left], right: r["conditions"][right]} for r in retrieval]
            for metric in ("recall@10", "all@20", "mrr"):
                report["retrieval_paired"][f"{left}->{right}:{metric}"] = bootstrap_paired_by_novel(
                    paired, left, right, metric=metric)
    by_q = defaultdict(dict)
    for row in reader:
        if row["option_seed"] == 0:
            by_q[row["qid"]][row["condition"]] = row
    conditions = sorted({r["condition"] for r in reader})
    for condition in conditions:
        rows = [r for r in reader if r["condition"] == condition and r["option_seed"] == 0]
        if not rows:
            continue
        wrong = [r for r in rows if not r["scores"]["correct"]]
        report["reader"][condition] = {
            "n": len(rows), "accuracy": round(mean(r["scores"]["correct"] for r in rows), 4),
            "gold_any": round(mean(r["scores"]["gold_any_in_context"] for r in rows), 4),
            "gold_all": round(mean(r["scores"]["gold_all_in_context"] for r in rows), 4),
            "wrong_with_all_gold": sum(r["scores"]["gold_all_in_context"] > 0 for r in wrong),
            "wrong_with_partial_gold": sum(0 < r["scores"]["gold_recall_in_context"] < 1 for r in wrong),
            "wrong_without_gold": sum(r["scores"]["gold_any_in_context"] == 0 for r in wrong),
            "by_bucket": {b: round(mean(r["scores"]["correct"] for r in rows if r["cutoff_bucket"] == b), 4)
                          for b in sorted({r["cutoff_bucket"] for r in rows})},
        }
    robust = defaultdict(list)
    for row in reader:
        robust[(row["qid"], row["condition"])].append(row["prediction"].get("answer_original"))
    report["option_order_agreement"] = {
        c: round(mean(len(set(v)) == 1 for (qid, cond), v in robust.items() if cond == c and len(v) > 1), 4)
        for c in conditions if any(cond == c and len(v) > 1 for (_, cond), v in robust.items())
    }
    queues = defaultdict(list)
    for qid, rows in by_q.items():
        if not {"q_dense", "q_graph", "recent"} <= set(rows):
            continue
        d, g, rec = rows["q_dense"], rows["q_graph"], rows["recent"]
        if d["scores"]["correct"] != g["scores"]["correct"]:
            queues["dense_graph_disagree"].append(qid)
        if rec["scores"]["correct"] and not d["scores"]["correct"]:
            queues["recent_only"].append(qid)
        if rec["scores"]["correct"] and rec["scores"]["gold_any_in_context"] == 0:
            queues["recent_correct_without_gold"].append(qid)
        if d["scores"]["gold_all_in_context"] and not d["scores"]["correct"]:
            queues["dense_wrong_all_gold"].append(qid)
        if rows.get("oracle") and not rows["oracle"]["scores"]["correct"]:
            queues["oracle_wrong"].append(qid)
        if rows.get("options_only_memory") and rows["options_only_memory"]["scores"]["correct"]:
            queues["memory_correct"].append(qid)
    report["case_queues"] = {k: sorted(v) for k, v in queues.items()}
    if relation:
        report["relation_channel"] = {
            "questions_with_visible_relations": sum(r["n_visible_relations"] > 0 for r in relation),
            "questions_with_gold_touching_top_relation": sum(r["n_top_relations_touching_gold"] > 0 for r in relation),
            "relation_types_in_top": dict(Counter(x["relation_type"] for r in relation for x in r["relations"])),
        }
    (OUT / f"{args.out}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    lines = ["# DetectiveQA Round 2 deterministic analysis", "", "## Retrieval", "",
             "| condition | R@10 | R@20 | All@20 | All@50 | MRR |", "|---|---:|---:|---:|---:|---:|"]
    for name, value in report["retrieval"].items():
        lines.append(f"| {name} | {value['recall@10']:.3f} | {value['recall@20']:.3f} | {value['all@20']:.3f} | "
                     f"{value['all@50']:.3f} | {value['mrr']:.3f} |")
    lines += ["", "## Reader seed 0", "", "| condition | Acc | gold any | gold all | wrong all-gold | wrong partial | wrong no-gold |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in report["reader"].items():
        lines.append(f"| {name} | {value['accuracy']:.3f} | {value['gold_any']:.3f} | {value['gold_all']:.3f} | "
                     f"{value['wrong_with_all_gold']} | {value['wrong_with_partial_gold']} | {value['wrong_without_gold']} |")
    lines += ["", "## Case queues", ""]
    for name, ids in report["case_queues"].items():
        lines.append(f"- {name} ({len(ids)}): {', '.join(ids)}")
    (OUT / f"{args.out}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
