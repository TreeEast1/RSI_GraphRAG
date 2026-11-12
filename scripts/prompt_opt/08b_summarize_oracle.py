#!/usr/bin/env python
"""Rebuild the oracle summary from saved jsonl files (no LLM calls)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import PATHS  # noqa: E402
from src.prompt_optimization.runner import OUT_ROOT, evaluate, mean, summarize_prompt  # noqa: E402


def main():
    base = [json.loads(l) for l in open(PATHS["qa"] / "baselines_hipporag_conv-30.jsonl", encoding="utf-8")]
    outdir = OUT_ROOT / "reader" / "oracle"
    summary = {}
    for f in sorted(outdir.glob("R*.jsonl")):
        key = f.stem
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        for r in rows:
            b = base[r["qidx"]]
            r["category"] = int(b["category"])
            r.update(evaluate(r["pred_answer"], b["gold_answer"], int(b["category"])))
        summary[key] = summarize_prompt(rows)
        summary[key]["mean_context_tokens"] = mean(r["context_tokens"] for r in rows)
        summary[key]["n_context_docs"] = mean(len(r["context_docs"]) for r in rows)

    # R3/P0 is the existing full-context baseline (same prompt, same full text)
    full = [json.loads(l) for l in open(PATHS["qa"] / "baselines_fullcontext_conv-30.jsonl", encoding="utf-8")]
    rows = []
    for i, r in enumerate(full):
        rows.append({"qidx": i, "category": int(r["category"]), "pred_answer": r.get("pred_answer"),
                     "usage": r.get("reader_usage"), "finish_reason": None, "cached": True,
                     "context_tokens": (r.get("extra") or {}).get("input_tokens"),
                     "context_docs": []})
        rows[-1].update(evaluate(rows[-1]["pred_answer"], r["gold_answer"], int(r["category"])))
    summary["R3_P0"] = summarize_prompt(rows)
    summary["R3_P0"]["mean_context_tokens"] = mean(r["context_tokens"] for r in rows)
    summary["R3_P0"]["n_context_docs"] = 19

    (outdir / "summary.json").write_text(json.dumps({"summaries": summary}, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"{'cond':8s} {'n':>4s} {'EM':>6s} {'F1':>6s} {'offF1':>6s} {'ref':>6s} {'adv':>5s} {'ctx':>7s} {'words':>6s}")
    for k, s in summary.items():
        print(f"{k:8s} {s['n']:4d} {s['project_em']:6.3f} {s['project_f1']:6.3f} {s['official_f1']:6.3f} "
              f"{s['refusal_rate_answerable']:6.3f} {s['adversarial_refusal_project']:5.2f} "
              f"{s.get('mean_context_tokens',0):7.0f} {s['mean_answer_words']:6.1f}")


if __name__ == "__main__":
    main()
