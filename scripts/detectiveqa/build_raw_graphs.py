"""Build official HippoRAG2 graphs for selected DetectiveQA novels."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HY4_RATE_FILE", "detectiveqa")

from src.detectiveqa.data import OUT, load_prepared  # noqa: E402
from src.detectiveqa.graph import build_raw_graph  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="pilot", choices=["pilot", "human", "all"])
    ap.add_argument("--novels", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="raw_graph_builds.json")
    args = ap.parse_args()
    _, chunks, splits = load_prepared()
    if args.novels:
        novels = args.novels.split(",")
    elif args.split == "pilot":
        novels = splits["human_novels"][:6]
    elif args.split == "human":
        novels = splits["human_novels"]
    else:
        novels = sorted(chunks, key=int)
    rows = []
    for novel_id in novels:
        info = build_raw_graph(novel_id, chunks[novel_id], force=args.force)
        rows.append(info)
        print(json.dumps({k: info.get(k) for k in ("novel_id", "n_chunks", "num_total_nodes",
                                                    "num_total_triples", "build_seconds")}), flush=True)
    (OUT / args.out).write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
