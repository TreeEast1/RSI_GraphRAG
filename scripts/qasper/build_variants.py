#!/usr/bin/env python
"""Build M2 (atom) and M3 (atom + comprehension triples) graph variants."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.qasper.evalutil import OUT, load_papers  # noqa: E402
from src.qasper.graph import build_variant_graph  # noqa: E402


def load_triples(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {pid: [tuple(x) for x in items] for pid, items in data.items()}


def merge(a: dict, b: dict) -> dict:
    out = {k: list(v) for k, v in a.items()}
    for k, v in b.items():
        out.setdefault(k, []).extend(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="sample100")
    ap.add_argument("--grouping", default="section")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    papers = load_papers(args.scope)
    infos = []
    for i, (pid, paper) in enumerate(papers.items(), 1):
        if args.limit and i > args.limit:
            break
        base = OUT / "comprehension" / args.grouping / pid
        if not (base / "meta.json").exists():
            print(f"=== [{i}/{len(papers)}] {pid}: read-time not ready, skipped", flush=True)
            continue
        if not (OUT / "graphs" / "raw" / pid / "qasper_graph_info.json").exists():
            print(f"=== [{i}/{len(papers)}] {pid}: G_raw not ready, skipped", flush=True)
            continue
        atom = load_triples(base / "triples_atomic_by_passage.json")
        comp = load_triples(base / "triples_comp_by_passage.json")
        print(f"=== [{i}/{len(papers)}] {pid}: atom={sum(len(v) for v in atom.values())} "
              f"comp={sum(len(v) for v in comp.values())}", flush=True)
        try:
            info2 = build_variant_graph(paper, "atom", atom, force=args.force)
            info3 = build_variant_graph(paper, "atom_comp", comp, base_variant="atom", force=args.force)
            infos.extend([info2, info3])
        except Exception as exc:  # noqa: BLE001
            print(f"!! {pid}: {type(exc).__name__}: {exc}", flush=True)
            infos.append({"paper_id": pid, "error": f"{type(exc).__name__}: {exc}"})
    path = OUT / "graphs" / f"build_log_variants_{args.scope}.json"
    path.write_text(json.dumps(infos, indent=1, default=str))
    print(f"done: {sum(1 for x in infos if not x.get('error'))} ok, "
          f"{sum(1 for x in infos if x.get('error'))} errors -> {path}")


if __name__ == "__main__":
    main()
