"""Round-two reader tables: seeds, buckets, error decomposition, paired tests."""
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

from src.detectiveqa.data import OUT, read_jsonl  # noqa: E402
from src.detectiveqa.round2 import bootstrap_paired_by_novel  # noqa: E402


def load(name: str) -> List[Dict]:
    path = OUT / name
    return read_jsonl(path) if path.exists() else []


def paired(rows: List[Dict], left: str, right: str, metric: str = "correct") -> Dict:
    by_q = {}
    for r in rows:
        if r["option_seed"] != 0:
            continue
        by_q.setdefault(r["qid"], {"novel_id": r["novel_id"]})[r["condition"]] = r["scores"]
    pairs = [v for v in by_q.values() if left in v and right in v]
    return bootstrap_paired_by_novel(pairs, left, right, metric=metric)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="round2/round2_reader.jsonl")
    ap.add_argument("--relations-reader", default="round2/round2_reader_relations.jsonl")
    ap.add_argument("--out", default="round2/round2_reader_tables")
    args = ap.parse_args()
    rows = load(args.reader)
    rel_rows = load(args.relations_reader)
    conditions = sorted({r["condition"] for r in rows})
    seeds = sorted({r["option_seed"] for r in rows})
    report: Dict = {"n_rows": len(rows), "conditions": {}, "by_bucket": {},
                    "option_order": {}, "paired_seed0": {}, "paired_seed1": {},
                    "relation_qa": {}, "relation_paired": {}}
    for cond in conditions:
        report["conditions"][cond] = {}
        for seed in seeds:
            sub = [r for r in rows if r["condition"] == cond and r["option_seed"] == seed]
            if not sub:
                continue
            report["conditions"][cond][f"seed{seed}"] = {
                "n": len(sub), "accuracy": round(mean(r["scores"]["correct"] for r in sub), 4),
                "gold_any": round(mean(r["scores"]["gold_any_in_context"] for r in sub), 4),
                "gold_all": round(mean(r["scores"]["gold_all_in_context"] for r in sub), 4),
                "invalid": round(mean(r["scores"]["invalid"] for r in sub), 4),
                "wrong_all_gold": sum(not r["scores"]["correct"] and r["scores"]["gold_all_in_context"] > 0 for r in sub),
                "wrong_partial_gold": sum(not r["scores"]["correct"] and 0 < r["scores"]["gold_recall_in_context"] < 1 for r in sub),
                "wrong_no_gold": sum(not r["scores"]["correct"] and r["scores"]["gold_any_in_context"] == 0 for r in sub),
            }
            report["by_bucket"].setdefault(cond, {})[f"seed{seed}"] = {
                b: round(mean(r["scores"]["correct"] for r in sub if r["cutoff_bucket"] == b), 4)
                for b in sorted({r["cutoff_bucket"] for r in sub})}
    # option-order agreement per condition
    by_qc = defaultdict(list)
    for r in rows:
        by_qc[(r["qid"], r["condition"])].append(r["prediction"].get("answer_original"))
    for cond in conditions:
        pairs = [v for (qid, c), v in by_qc.items() if c == cond and len(v) > 1]
        if pairs:
            report["option_order"][cond] = {
                "n_pairs": len(pairs),
                "same_answer": round(mean(len(set(v)) == 1 for v in pairs), 4),
            }
    # paired tests on seeds 0 and 1
    comparisons = [("q_dense", "q_graph"), ("q_dense", "q_rrf"), ("q_dense", "recent"),
                   ("q_dense", "nonrecent_q_dense"), ("q_dense", "recent_plus_q_dense"),
                   ("q_dense", "qo_dense"), ("q_dense", "oracle"),
                   ("q_dense", "options_only_memory"), ("recent", "nonrecent_q_dense"),
                   ("recent", "recent_plus_q_dense")]
    for seed in seeds:
        sub = [r for r in rows if r["option_seed"] == seed]
        by_q = {}
        for r in sub:
            by_q.setdefault(r["qid"], {"novel_id": r["novel_id"]})[r["condition"]] = r["scores"]
        pairs = list(by_q.values())
        for left, right in comparisons:
            if left in conditions and right in conditions and all(left in p and right in p for p in pairs):
                report[f"paired_seed{seed}"][f"{left}->{right}"] = bootstrap_paired_by_novel(pairs, left, right)
    # seed 0 vs seed 1 agreement and accuracy delta per condition
    seed_pairs = defaultdict(dict)
    for r in rows:
        seed_pairs[(r["qid"], r["condition"])][r["option_seed"]] = r["prediction"].get("answer_original")
    report["seed_agreement"] = {
        cond: round(mean(len(set(v.values())) == 1 for (qid, c), v in seed_pairs.items()
                         if c == cond and 0 in v and 1 in v), 4)
        for cond in conditions if any(c == cond and 0 in v and 1 in v for (_, c), v in seed_pairs.items())}
    # relation QA
    if rel_rows:
        rel_conditions = sorted({r["condition"] for r in rel_rows})
        for cond in rel_conditions:
            sub = [r for r in rel_rows if r["condition"] == cond]
            report["relation_qa"][cond] = {
                "n": len(sub), "accuracy": round(mean(r["scores"]["correct"] for r in sub), 4),
                "gold_any": round(mean(r["scores"]["gold_any_in_context"] for r in sub), 4),
            }
        by_q = {}
        for r in rel_rows:
            by_q.setdefault(r["qid"], {"novel_id": r["novel_id"]})[r["condition"]] = r["scores"]
        pairs = list(by_q.values())
        for left, right in (("q_dense", "relations_q_dense"), ("q_rrf", "relations_q_rrf"),
                            ("q_dense", "q_rrf"), ("relations_q_dense", "relations_q_rrf")):
            if left in rel_conditions and right in rel_conditions and all(left in p and right in p for p in pairs):
                report["relation_paired"][f"{left}->{right}"] = bootstrap_paired_by_novel(pairs, left, right)
    (OUT / f"{args.out}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    lines = ["# Round 2 reader tables", "", f"n_rows={len(rows)}", "", "## Accuracy by condition and seed",
             "", "| condition | seed0 acc | seed0 gold-any | seed0 gold-all | seed1 acc | seed1 gold-any | seed1 gold-all |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for cond in conditions:
        v0 = report["conditions"].get(cond, {}).get("seed0", {})
        v1 = report["conditions"].get(cond, {}).get("seed1", {})
        lines.append(f"| {cond} | {v0.get('accuracy', float('nan')):.3f} | {v0.get('gold_any', float('nan')):.3f} | "
                     f"{v0.get('gold_all', float('nan')):.3f} | {v1.get('accuracy', float('nan')):.3f} | "
                     f"{v1.get('gold_any', float('nan')):.3f} | {v1.get('gold_all', float('nan')):.3f} |")
    lines += ["", "## Paired tests (novel-clustered bootstrap; seed 0)", "",
              "| comparison | delta | 95% CI | wins | losses | McNemar p |", "|---|---:|---|---:|---:|---:|"]
    for key, val in report["paired_seed0"].items():
        lines.append(f"| {key} | {val['delta']:+.4f} | [{val['ci_low']:+.4f},{val['ci_high']:+.4f}] | "
                     f"{val['wins']} | {val['losses']} | {val['mcnemar_exact_p']:.4g} |")
    lines += ["", "## Option-order agreement", ""]
    for cond, val in report["option_order"].items():
        lines.append(f"- {cond}: {val['same_answer']:.3f} ({val['n_pairs']} pairs)")
    if report["relation_qa"]:
        lines += ["", "## Relation QA", "", "| condition | n | accuracy | gold any |", "|---|---:|---:|---:|"]
        for cond, val in report["relation_qa"].items():
            lines.append(f"| {cond} | {val['n']} | {val['accuracy']:.3f} | {val['gold_any']:.3f} |")
        lines += ["", "| comparison | delta | 95% CI | wins | losses | McNemar p |", "|---|---:|---|---:|---:|---:|"]
        for key, val in report["relation_paired"].items():
            lines.append(f"| {key} | {val['delta']:+.4f} | [{val['ci_low']:+.4f},{val['ci_high']:+.4f}] | "
                         f"{val['wins']} | {val['losses']} | {val['mcnemar_exact_p']:.4g} |")
    (OUT / f"{args.out}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:80]))


if __name__ == "__main__":
    main()
