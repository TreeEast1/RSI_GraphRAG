#!/usr/bin/env python
"""Merge sharded retrieval/QA JSONL outputs into the canonical files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.qasper.evalutil import OUT  # noqa: E402


def merge(paths, key_fn, out_path: Path):
    seen = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            seen[key_fn(r)] = r
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in seen.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(seen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="sample100")
    ap.add_argument("--kind", default="all", choices=["retrieval", "qa", "all"])
    ap.add_argument("--include-base", action="store_true")
    args = ap.parse_args()
    if args.kind in ("retrieval", "all"):
        paths = sorted((OUT / "retrieval").glob(f"comparison_{args.scope}_s*.jsonl"))
        if args.include_base and (OUT / "retrieval" / f"comparison_{args.scope}.jsonl").exists():
            paths.append(OUT / "retrieval" / f"comparison_{args.scope}.jsonl")
        if paths:
            n = merge(paths, lambda r: (r["method"], r["qid"]),
                      OUT / "retrieval" / f"comparison_{args.scope}.jsonl")
            print("retrieval:", n, "rows from", len(paths), "shards")
    if args.kind in ("qa", "all"):
        paths = sorted((OUT / "qa").glob(f"predictions_{args.scope}_s*.jsonl"))
        if args.include_base and (OUT / "qa" / f"predictions_{args.scope}.jsonl").exists():
            paths.append(OUT / "qa" / f"predictions_{args.scope}.jsonl")
        if paths:
            n = merge(paths, lambda r: (r["method"], r["qid"]),
                      OUT / "qa" / f"predictions_{args.scope}.jsonl")
            print("qa:", n, "rows from", len(paths), "shards")


if __name__ == "__main__":
    main()
