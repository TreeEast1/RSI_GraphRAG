#!/usr/bin/env python
"""Recompute pilot/final summaries from saved result jsonl files (no LLM calls)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import PATHS  # noqa: E402
from src.prompt_optimization import reader_prompts as RP  # noqa: E402
from src.prompt_optimization.runner import (OUT_ROOT, evaluate, summarize_prompt)  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default="pilot")
    ap.add_argument("--pids", default="P1,P2,P3,P4,P5,P6,P7")
    args = ap.parse_args()
    base = [json.loads(l) for l in open(PATHS["qa"] / "baselines_hipporag_conv-30.jsonl", encoding="utf-8")]
    outdir = OUT_ROOT / "reader" / args.subdir
    split = json.loads((OUT_ROOT / "splits" / "split_dev40_val65.json").read_text())
    meta0 = json.loads((outdir / "results_P1.jsonl.meta.json").read_text()) if (outdir / "results_P1.jsonl.meta.json").exists() else {}
    qids = meta0.get("qids")
    if qids is None:
        raise SystemExit("cannot infer qids")

    summaries = {}
    # P0 from saved baseline
    p0 = []
    for q in qids:
        r = base[q]
        row = {"pid": "P0", "qid": r["qid"], "qidx": q, "category": int(r["category"]),
               "pred_answer": r.get("pred_answer"), "usage": r.get("reader_usage"),
               "finish_reason": None, "cached": True}
        row.update(evaluate(row["pred_answer"], r["gold_answer"], int(r["category"])))
        p0.append(row)
    summaries["P0"] = summarize_prompt(p0)
    summaries["P0"]["name"] = RP.get("P0").name

    for pid in args.pids.split(","):
        p = outdir / f"results_{pid}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        for r in rows:
            b = base[r["qidx"]]
            r["category"] = int(b["category"])
            r.update(evaluate(r["pred_answer"], b["gold_answer"], int(b["category"])))
        summaries[pid] = summarize_prompt(rows)
        summaries[pid]["name"] = RP.get(pid).name

    out = {"subdir": args.subdir, "n": len(qids), "summaries": summaries}
    (outdir / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'pid':4s} {'EM':>6s} {'F1':>6s} {'offF1':>6s} {'refus':>6s} {'adv':>5s} {'words':>6s} {'trunc':>5s}")
    for pid, s in summaries.items():
        print(f"{pid:4s} {s['project_em']:6.3f} {s['project_f1']:6.3f} {s['official_f1']:6.3f} "
              f"{s['refusal_rate_answerable']:6.3f} {s['adversarial_refusal_project']:5.2f} "
              f"{s['mean_answer_words']:6.1f} {s['truncated']:5d}")


if __name__ == "__main__":
    main()
