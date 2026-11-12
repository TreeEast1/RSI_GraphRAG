#!/usr/bin/env python
"""Phase 1.0 — fixed 40-question development split + 65-question internal validation split.

The 12 questions named in the experiment brief are always in dev; the rest are
sampled by category with a fixed seed.  Validation qids are written before any
new prompt is run and must not be inspected per-question until prompts are frozen.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import PATHS  # noqa: E402

OUT = ROOT / "outputs" / "prompt_optimization_conv30" / "splits"

KNOWN = [4, 16, 17, 24, 29, 31, 33, 39, 47, 54, 74, 85]
QUOTAS = {1: 11, 2: 12, 4: 11, 5: 6}
SEED = 42


def main():
    rows = [json.loads(line) for line in open(PATHS["qa"] / "baselines_hipporag_conv-30.jsonl", encoding="utf-8")]
    qidx = [int(r["qid"].split("q")[-1]) for r in rows]
    by_cat = {c: [i for i in qidx if rows[i]["category"] == c] for c in (1, 2, 4, 5)}

    dev = []
    picked = {}
    for cat, quota in QUOTAS.items():
        pool = by_cat[cat]
        seed_known = [i for i in KNOWN if i in pool]
        rest = [i for i in pool if i not in seed_known]
        random.Random(SEED + cat).shuffle(rest)
        chosen = seed_known + rest[: max(0, quota - len(seed_known))]
        chosen = sorted(chosen)
        picked[cat] = chosen
        dev.extend(chosen)

    dev = sorted(set(dev))
    val = sorted(set(qidx) - set(dev))
    split = {
        "seed": SEED, "known_case_qids": KNOWN, "quotas": QUOTAS,
        "dev_qids": dev, "val_qids": val,
        "dev_by_category": {str(c): picked[c] for c in picked},
        "n_dev": len(dev), "n_val": len(val),
        "note": ("Validation qids are fixed here, before any candidate prompt is run. "
                 "Only dev results may be used to choose the final prompt."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split_dev40_val65.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(json.dumps({k: split[k] for k in ("n_dev", "n_val", "dev_by_category")}, indent=2))
    print("dev:", dev)
    print("val n:", len(val))


if __name__ == "__main__":
    main()
