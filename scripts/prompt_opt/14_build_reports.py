#!/usr/bin/env python
"""Build every table in FINAL_REPORT.md from the saved artifacts (no LLM calls)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import PATHS  # noqa: E402
from src.prompt_optimization.runner import evaluate, mean  # noqa: E402

OUT = ROOT / "outputs" / "prompt_optimization_conv30"
BASE = [json.loads(l) for l in open(PATHS["qa"] / "baselines_hipporag_conv-30.jsonl", encoding="utf-8")]
SPLIT = json.loads((OUT / "splits" / "split_dev40_val65.json").read_text())
DEV = set(SPLIT["dev_qids"])
VAL = set(SPLIT["val_qids"])


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def eval_rows(rows, qfilter=None):
    out = []
    for r in rows:
        q = r["qidx"]
        if qfilter is not None and q not in qfilter:
            continue
        b = BASE[q]
        r = dict(r)
        r["category"] = int(b["category"])
        r.update(evaluate(r.get("pred_answer"), b["gold_answer"], int(b["category"])))
        out.append(r)
    return out


def reader_metrics(rows):
    ans = [r for r in rows if r["category"] != 5]
    adv = [r for r in rows if r["category"] == 5]
    m = {
        "n": len(rows), "n_ans": len(ans), "n_adv": len(adv),
        "em": mean(r["em"] for r in ans), "f1": mean(r["f1"] for r in ans),
        "off_f1": mean(r["official_f1"] for r in ans),
        "off_em": mean(r["official_em"] for r in ans),
        "refusal": mean(float(r["refused"]) for r in ans),
        "adv_refusal": mean(r["em"] for r in adv) if adv else float("nan"),
        "words": mean(len(str(r.get("pred_answer") or "").split()) for r in rows),
        "by_cat": {},
    }
    for c in (1, 2, 3, 4):
        sub = [r for r in ans if r["category"] == c]
        if not sub:
            continue
        m["by_cat"][c] = {
            "n": len(sub), "em": mean(r["em"] for r in sub), "f1": mean(r["f1"] for r in sub),
            "off_f1": mean(r["official_f1"] for r in sub),
            "refusal": mean(float(r["refused"]) for r in sub),
        }
    return m


def collect_reader():
    out = {}
    p0 = eval_rows([{"qidx": i, "pred_answer": r.get("pred_answer"),
                     "usage": r.get("reader_usage"), "finish_reason": None, "cached": True}
                    for i, r in enumerate(BASE)])
    out["P0"] = {"dev": reader_metrics([r for r in p0 if r["qidx"] in DEV]),
                 "val": reader_metrics([r for r in p0 if r["qidx"] in VAL]),
                 "all": reader_metrics(p0)}
    for pid in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
        pilot = OUT / "reader" / "pilot" / f"results_{pid}.jsonl"
        final = OUT / "reader" / "final" / f"results_{pid}.jsonl"
        rows = []
        if pilot.exists():
            rows += eval_rows(load_rows(pilot), DEV)
        if final.exists():
            rows += eval_rows(load_rows(final), VAL)
        if not rows:
            continue
        out[pid] = {"all": reader_metrics(rows)}
        out[pid]["dev"] = reader_metrics([r for r in rows if r["qidx"] in DEV])
        out[pid]["val"] = reader_metrics([r for r in rows if r["qidx"] in VAL])
    return out


def collect_comp():
    s = json.loads((OUT / "comprehension" / "retrieval" / "summary.json").read_text())["summaries"]
    quality = json.loads((OUT / "comprehension" / "quality_report.json").read_text())["summaries"]
    qmap = {x["name"].split(" —")[0]: x for x in quality}
    # per-question negative transfer vs C2
    c2_path = OUT / "comprehension" / "retrieval" / "C2" / "records.jsonl"
    c2 = {r["qid"]: r for r in load_rows(c2_path)} if c2_path.exists() else {}
    out = {}
    for name, summ in s.items():
        rec = {}
        if name != "C2" and c2 and (OUT / "comprehension" / "retrieval" / name / "records.jsonl").exists():
            var = {r["qid"]: r for r in load_rows(OUT / "comprehension" / "retrieval" / name / "records.jsonl")}
            lost_ev = gained_ev = lost_r5 = gained_r5 = 0
            for qid, v in var.items():
                c = c2.get(qid)
                if not c:
                    continue
                if c["budget"]["4096"]["evidence_full"] == 1 and v["budget"]["4096"]["evidence_full"] == 0:
                    lost_ev += 1
                if c["budget"]["4096"]["evidence_full"] == 0 and v["budget"]["4096"]["evidence_full"] == 1:
                    gained_ev += 1
                if c["retrieval"]["recall@5"] > v["retrieval"]["recall@5"]:
                    lost_r5 += 1
                if c["retrieval"]["recall@5"] < v["retrieval"]["recall@5"]:
                    gained_r5 += 1
            rec = {"lost_evidence": lost_ev, "gained_evidence": gained_ev,
                   "lost_r5": lost_r5, "gained_r5": gained_r5}
        key = name if name in ("G_raw", "C2") else None
        out[name] = {**summ, **rec}
    return out, qmap


def collect_joint():
    p = OUT / "joint" / "summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())["summaries"]


def main():
    reader = collect_reader()
    comp, qmap = collect_comp()
    joint = collect_joint()
    oracle = json.loads((OUT / "reader" / "oracle" / "summary.json").read_text())["summaries"] \
        if (OUT / "reader" / "oracle" / "summary.json").exists() else {}
    repeats = json.loads((OUT / "reader" / "repeats" / "summary.json").read_text())["summaries"] \
        if (OUT / "reader" / "repeats" / "summary.json").exists() else {}
    official_adv = json.loads((OUT / "reader" / "official_adversarial" / "summary.json").read_text())["summaries"] \
        if (OUT / "reader" / "official_adversarial" / "summary.json").exists() else {}

    payload = {"reader": reader, "comprehension": comp, "quality": qmap, "joint": joint,
               "oracle": oracle, "repeats": repeats, "official_adversarial": official_adv}
    (OUT / "reports" / "all_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")

    L = []
    L += ["# 自动汇总表（由 14_build_reports.py 生成）", ""]
    L += ["## Reader 实验结果", "",
          "| Prompt | n | EM | F1 | 官方F1 | 官方EM | 过拒率(可答) | 对抗拒答率 | 平均词数 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    names = {"P0": "P0 Baseline", "P1": "P1 concise-complete", "P2": "P2 integrate",
             "P3": "P3 temporal", "P4": "P4 refusal-calibration", "P5": "P5 question-type",
             "P6": "P6 official-style", "P7": "P7 synthesis", "P8": "P8 synthesis+presupposition"}
    for split in ("dev", "val", "all"):
        L += [f"", f"### {split}"]
        for pid, m in reader.items():
            if split not in m:
                continue
            x = m[split]
            L.append(f"| {names.get(pid,pid)} | {x['n']} | {x['em']:.3f} | {x['f1']:.3f} | "
                     f"{x['off_f1']:.3f} | {x['off_em']:.3f} | {x['refusal']:.3f} | {x['adv_refusal']:.3f} | {x['words']:.1f} |")

    L += ["", "## Comprehension 检索结果（105 题）", "",
          "| 图 | 新增关系 | R@3 | R@5 | MRR | nDCG@10 | Ev@4096 | 丢失证据题 | 新增证据题 | 丢失R@5 | 新增R@5 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in comp.items():
        nrel = 0
        if name.startswith("cmpo_"):
            pid = {"cmpo_compp1": "COMP-P1", "cmpo_compp2": "COMP-P2", "cmpo_compp3": "COMP-P3",
                   "cmpo_compp4": "COMP-P4", "cmpo_compp5": "COMP-P5",
                   "cmpo_compp4c10": "COMP-P4-c10"}.get(name)
            p = OUT / "comprehension" / "extractions" / f"{pid}.json"
            if p.exists():
                nrel = json.loads(p.read_text())["n_relations"]
        elif name == "C2":
            nrel = 0
        elif name == "G_raw":
            nrel = 0
        L.append(f"| {name} | {nrel} | {s.get('recall@3',float('nan')):.4f} | {s.get('recall@5',float('nan')):.4f} | "
                 f"{s.get('mrr',float('nan')):.4f} | {s.get('ndcg@10',float('nan')):.4f} | "
                 f"{s.get('evidence_full@4096',float('nan')):.4f} | {s.get('lost_evidence','-')} | "
                 f"{s.get('gained_evidence','-')} | {s.get('lost_r5','-')} | {s.get('gained_r5','-')} |")

    L += ["", "## Oracle / Full-context 诊断", "",
          "| 条件 | Prompt | n | EM | F1 | 官方F1 | 过拒率 | 平均 context tokens |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for k, s in oracle.items():
        L.append(f"| {k} | {k.split('_')[-1]} | {s['n']} | {s['project_em']:.3f} | {s['project_f1']:.3f} | "
                 f"{s['official_f1']:.3f} | {s['refusal_rate_answerable']:.3f} | {s.get('mean_context_tokens',float('nan')):.0f} |")

    L += ["", "## 联合实验", "",
          "| 图 | Prompt | EM | F1 | 官方F1 | 过拒率 | 对抗拒答率 |", "|---|---|---:|---:|---:|---:|---:|"]
    for k, s in joint.items():
        L.append(f"| {k.split('|')[0]} | {k.split('|')[1]} | {s['project_em']:.3f} | {s['project_f1']:.3f} | "
                 f"{s['official_f1']:.3f} | {s['refusal_rate_answerable']:.3f} | {s['adversarial_refusal_project']:.3f} |")

    L += ["", "## 重复运行稳定性", "",
          "| Prompt | 词面不稳定率 | 得分不稳定率 | 平均 Jaccard | 每次运行 EM | 多数票 EM |",
          "|---|---:|---:|---:|---|---:|"]
    for pid, s in repeats.items():
        L.append(f"| {pid} | {s['wording_unstable_rate']:.3f} | {s['scored_unstable_rate']:.3f} | "
                 f"{s['mean_pair_jaccard']:.3f} | {[round(x,3) for x in s['em_per_run']]} | {s['majority_vote_em']:.3f} |")

    L += ["", "## 官方 cat-5 多选协议", "", "| Prompt | 准确率 | n | 解析分布 |", "|---|---:|---:|---|"]
    for pid, s in official_adv.items():
        L.append(f"| {pid} | {s['accuracy']:.3f} | {s['n']} | {s['parse_breakdown']} |")

    (OUT / "reports" / "auto_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
