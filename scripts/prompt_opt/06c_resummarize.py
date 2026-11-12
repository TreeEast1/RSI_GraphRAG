#!/usr/bin/env python
"""Recompute comprehension/retrieval/summary.json from every records.jsonl (no LLM)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.pipeline import summarize  # noqa: E402

OUT = ROOT / "outputs" / "prompt_optimization_conv30" / "comprehension" / "retrieval"


def main():
    summaries = {}
    for d in sorted(OUT.iterdir()):
        p = d / "records.jsonl"
        if not d.is_dir() or not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        s = summarize(rows, d.name)
        s["fallback_dense_count"] = sum(1 for r in rows if r.get("diag", {}).get("fallback_dense"))
        summaries[d.name] = s
    (OUT / "summary.json").write_text(json.dumps({"summaries": summaries}, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"wrote {len(summaries)} variants")
    for n, s in summaries.items():
        print(f"{n:16s} R@3={s['recall@3']:.4f} R@5={s['recall@5']:.4f} MRR={s['mrr']:.4f} "
              f"Ev={s['evidence_full@4096']:.4f} fallback={s['fallback_dense_count']}")


if __name__ == "__main__":
    main()
