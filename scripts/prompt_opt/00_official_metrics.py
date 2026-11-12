#!/usr/bin/env python
"""Phase 0.2 — recompute existing conv-30 predictions under the official LoCoMo
scorer WITHOUT calling any LLM, and diff it against the project's scorer.

Outputs:
  outputs/prompt_optimization_conv30/audit/official_metric_comparison.{json,md}
  outputs/prompt_optimization_conv30/audit/official_metric_diff_cases.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.prompt_optimization.official_metrics import official_qa_score  # noqa: E402

OUT = ROOT / "outputs" / "prompt_optimization_conv30" / "audit"
QA = ROOT / "outputs" / "qa"

METHODS = {
    "G_raw": "baselines_hipporag_conv-30.jsonl",
    "C2": "readtime_cluster_hybrid_0.5_b8192_triple_conv-30.jsonl",
    "C3": "readtime_cluster_hybrid_0.5_b8192_comp_conv-30.jsonl",
    "C5": "readtime_full_conv_triple_conv-30.jsonl",
    "C6": "readtime_full_conv_comp_conv-30.jsonl",
    "C3_composed": "readtime_ablation_cluster_hybrid_0.5_b8192_comp_composed_conv-30.jsonl",
    "C3_inferred": "readtime_ablation_cluster_hybrid_0.5_b8192_comp_inferred_conv-30.jsonl",
}


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def main():
    rows_by = {}
    for name, fn in METHODS.items():
        path = QA / fn
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        for r in rows:
            off = official_qa_score(r.get("pred_answer"), r.get("gold_answer"), int(r["category"]))
            r["off"] = off
        rows_by[name] = rows

    table = {}
    for name, rows in rows_by.items():
        ans = [r for r in rows if r["category"] != 5]
        adv = [r for r in rows if r["category"] == 5]
        table[name] = {
            "n_all": len(rows), "n_answerable": len(ans), "n_adversarial": len(adv),
            # project scorer (originals, untouched)
            "project_em": mean([r["em"] for r in ans]),
            "project_f1": mean([r["f1"] for r in ans]),
            # official scorer, answerable questions only
            "official_f1": mean([r["off"]["official_f1"] for r in ans]),
            "official_em": mean([r["off"]["official_em"] for r in ans]),
            "official_f1_all105": mean([r["off"]["official_f1"] for r in rows]),
            # per-category official F1
            "official_f1_by_cat": {
                int(c): mean([r["off"]["official_f1"] for r in ans if int(r["category"]) == c])
                for c in sorted({int(r["category"]) for r in ans})
            },
            "project_f1_by_cat": {
                int(c): mean([r["f1"] for r in ans if int(r["category"]) == c])
                for c in sorted({int(r["category"]) for r in ans})
            },
            # adversarial: project rule vs official keyword rule
            "project_refusal": mean([r["em"] for r in adv]),
            "official_refusal_strict": mean([r["off"]["official_refused_strict"] for r in adv]),
            "official_refusal_adapted": mean([r["off"]["official_refused_adapted"] for r in adv]),
        }

    # per-question diffs (official F1 - project F1) on answerable questions
    diff = []
    for name in ("G_raw", "C3"):
        for r in rows_by[name]:
            if r["category"] == 5:
                continue
            diff.append({
                "method": name, "qid": r["qid"], "qidx": int(r["qid"].split("q")[-1]),
                "category": r["category"], "category_name": r["category_name"],
                "question": r["question"], "gold_answer": r["gold_answer"],
                "pred_answer": r.get("pred_answer"),
                "project_em": r["em"], "project_f1": r["f1"],
                "official_f1": r["off"]["official_f1"], "official_em": r["off"]["official_em"],
                "delta_official_minus_project": r["off"]["official_f1"] - r["f1"],
            })
    diff.sort(key=lambda d: -abs(d["delta_official_minus_project"]))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "official_metric_comparison.json").write_text(
        json.dumps({"table": table, "diffs": diff}, indent=2, ensure_ascii=False), encoding="utf-8")

    import csv
    with open(OUT / "official_metric_diff_cases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(diff[0].keys()))
        w.writeheader()
        w.writerows(diff)

    # markdown
    lines = ["# 官方 LoCoMo 评分口径 vs 项目口径（conv-30，无 LLM 重跑）", "",
             "官方实现：`snap-research/locomo task_eval/evaluation.py`（2026-11-09 拉取）。", "",
             "| 方法 | n(答) | 项目EM | 项目F1 | 官方F1 | 官方EM | 官方F1(含cat5) | 拒答率(项目) | 拒答率(官方严格) | 拒答率(官方+No answer) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, t in table.items():
        lines.append(
            f"| {name} | {t['n_answerable']} | {t['project_em']:.4f} | {t['project_f1']:.4f} | "
            f"{t['official_f1']:.4f} | {t['official_em']:.4f} | {t['official_f1_all105']:.4f} | "
            f"{t['project_refusal']:.4f} | {t['official_refusal_strict']:.4f} | {t['official_refusal_adapted']:.4f} |")
    lines += ["", "## 分题型（官方F1 / 项目F1）", "",
              "| 方法 | cat1 multi-hop | cat2 temporal | cat3 open | cat4 single |", "|---|---:|---:|---:|---:|"]
    for name, t in table.items():
        cells = []
        for c in (1, 2, 3, 4):
            o = t["official_f1_by_cat"].get(c)
            p = t["project_f1_by_cat"].get(c)
            cells.append(f"{o:.3f} / {p:.3f}" if o is not None else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## 差异最大的 15 题（官方F1 − 项目F1；G_raw + C3）", "",
              "| 方法 | qid | cat | question | gold | pred | 项目F1 | 官方F1 | Δ |",
              "|---|---|---|---|---|---|---:|---:|---:|"]
    for d in diff[:15]:
        q = d["question"].replace("|", "\\|")
        g = str(d["gold_answer"]).replace("|", "\\|")[:60]
        p = str(d["pred_answer"]).replace("|", "\\|")[:60]
        lines.append(f"| {d['method']} | q{d['qidx']} | {d['category']} | {q} | {g} | {p} | "
                     f"{d['project_f1']:.3f} | {d['official_f1']:.3f} | {d['delta_official_minus_project']:+.3f} |")
    (OUT / "official_metric_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[:20]))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
