#!/usr/bin/env python
"""Create a label-free, count-controlled subset of COMP-P4 (top-10 by confidence).

The rule uses only the model's self-reported confidence and a deterministic
tie-break — no QA labels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

COMP = ROOT / "outputs" / "prompt_optimization_conv30" / "comprehension"
N_KEEP = 10


def main():
    d = json.loads((COMP / "extractions" / "COMP-P4.json").read_text())
    rels = d["relations"]
    ranked = sorted(enumerate(rels), key=lambda x: (-x[1].get("confidence", 0), x[0]))
    keep = [r for _, r in ranked[:N_KEEP]]
    out = dict(d)
    out["pid"] = "COMP-P4-c10"
    out["name"] = "COMP-P4 top-10 by confidence (count control)"
    out["relations"] = keep
    out["n_relations"] = len(keep)
    out["count_control"] = {
        "source": "COMP-P4", "rule": "confidence desc, original order tie-break",
        "n_source": len(rels), "n_kept": len(keep),
        "kept": [f"{r['subject']} | {r['predicate']} | {r['object']}" for r in keep],
    }
    (COMP / "extractions" / "COMP-P4-c10.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["count_control"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
