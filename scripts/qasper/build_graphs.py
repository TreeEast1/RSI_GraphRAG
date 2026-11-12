#!/usr/bin/env python
"""Build G_raw (and later knowledge variants) for QASPER papers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.qasper.evalutil import OUT, load_papers  # noqa: E402
from src.qasper.graph import build_raw_graph, build_variant_graph  # noqa: E402


def load_triples(path: Path, key: str = "triples") -> dict:
    """Load per-passage triples from a knowledge JSON: {passage_id: [[s,p,o],...]}."""
    data = json.loads(path.read_text())
    out = {}
    for pid, items in data.get(key, {}).items():
        out[pid] = [tuple(x) for x in items]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="sample100")
    ap.add_argument("--stage", default="raw", choices=["raw", "variant"])
    ap.add_argument("--variant", default=None)
    ap.add_argument("--triples", default=None, help="JSON with {passage_id: [[s,p,o]]}")
    ap.add_argument("--base-variant", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    papers = load_papers(args.scope)
    if args.num_shards > 1:
        papers = {pid: p for i, (pid, p) in enumerate(papers.items())
                  if i % args.num_shards == args.shard}
    infos = []
    for i, (pid, paper) in enumerate(papers.items(), 1):
        print(f"=== [{i}/{len(papers)}] {pid} ({len(paper.paragraphs)} paragraphs)", flush=True)
        try:
            if args.stage == "raw":
                info = build_raw_graph(paper, force=args.force)
            else:
                if not args.variant or not args.triples:
                    raise SystemExit("--variant and --triples required for stage=variant")
                info = build_variant_graph(paper, args.variant, load_triples(Path(args.triples)),
                                           force=args.force, base_variant=args.base_variant)
            infos.append(info)
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            print(f"!! {pid} failed: {type(exc).__name__}: {exc}", flush=True)
            infos.append({"paper_id": pid, "error": f"{type(exc).__name__}: {exc}"})
    tag = args.variant if args.stage == "variant" else "raw"
    suffix = f"_{args.shard}of{args.num_shards}" if args.num_shards > 1 else ""
    path = OUT / "graphs" / f"build_log_{tag}_{args.scope}{suffix}.json"
    path.write_text(json.dumps(infos, indent=1, default=str))
    n_err = sum(1 for x in infos if x.get("error"))
    print(f"done: {len(infos)-n_err} ok, {n_err} errors -> {path}")


if __name__ == "__main__":
    main()
