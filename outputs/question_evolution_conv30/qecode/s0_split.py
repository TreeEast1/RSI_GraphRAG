#!/usr/bin/env python
"""Stage 0 - fixed stratified Adaptation / Development / Evaluation split.

The split is created *before* any skill is learned and is written to
DATA_SPLIT.json.  The seed is fixed.  conv-30 was already used by earlier
prompt-optimization rounds, so the evaluation set is an internal, exploratory
test rather than a fresh unseen benchmark.
"""
from __future__ import annotations

import json
import random

from . import common  # noqa: F401  (run as module: python -m code.s0_split)


def main():
    seed = 20261110
    # category -> (adaptation, development, evaluation)
    alloc = {1: (4, 2, 5), 2: (11, 5, 10), 4: (18, 8, 18), 5: (10, 4, 10)}
    rng = random.Random(seed)
    by_cat = {c: sorted(q for q in common.qa_map() if common.category_of(q) == c) for c in alloc}
    sets = {"adaptation": [], "development": [], "evaluation": []}
    for c, (na, nd, ne) in alloc.items():
        ids = by_cat[c][:]
        rng.shuffle(ids)
        assert len(ids) == na + nd + ne, (c, len(ids))
        sets["adaptation"] += ids[:na]
        sets["development"] += ids[na:na + nd]
        sets["evaluation"] += ids[na + nd:]
    # Fix the order in which adaptation questions are experienced.
    rng.shuffle(sets["adaptation"])
    order = {"seed": seed, "allocation": {str(k): v for k, v in alloc.items()},
             "adaptation": sorted(sets["adaptation"]),
             "development": sorted(sets["development"]),
             "evaluation": sorted(sets["evaluation"]),
             "adaptation_order": sets["adaptation"],
             "by_category": {
                 name: {str(c): [q for q in sets[name] if common.category_of(q) == c] for c in alloc}
                 for name in sets},
             "note": ("conv-30 has been studied in earlier rounds. This split is fixed before "
                      "skill learning, but the evaluation set remains an internal exploratory "
                      "test, not a fully independent final benchmark. Only development questions "
                      "select the final snapshot; evaluation is untouched until the end.")}
    common.write_json(common.SPLIT_PATH, order)

    lines = ["# DATA_SPLIT - conv-30 Question-Time Self-Evolution", "",
             f"Seed: `{seed}`. Split fixed before any skill was learned.", "",
             "| subset | n | multi-hop (1) | temporal (2) | single-hop (4) | adversarial (5) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in ("adaptation", "development", "evaluation"):
        row = order["by_category"][name]
        lines.append(f"| {name} | {len(sets[name])} | " + " | ".join(
            str(len(row[str(c)])) for c in (1, 2, 4, 5)) + " |")
    lines += ["", "## Question IDs", ""]
    for name in ("adaptation", "development", "evaluation"):
        lines.append(f"### {name.capitalize()} ({len(sets[name])})")
        lines.append("")
        for c in (1, 2, 4, 5):
            ids = order["by_category"][name][str(c)]
            lines.append(f"- **{common.CATEGORY_NAMES[c]}**: " + ", ".join(f"q{q}" for q in ids))
        lines.append("")
    lines += ["## Adaptation order", "",
              ", ".join(f"q{q}" for q in sets["adaptation"]), "",
              "## Caveat", "",
              order["note"], ""]
    (common.OUT / "DATA_SPLIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: len(v) for k, v in sets.items()}, indent=2))
    for name in ("adaptation", "development", "evaluation"):
        print(name, order["by_category"][name])


if __name__ == "__main__":
    main()
