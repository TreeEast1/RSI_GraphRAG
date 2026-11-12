#!/usr/bin/env python
"""Sensitivity of unit/graph fusion weight (exploratory, no LLM calls).

Uses the ranked lists already stored by run_retrieval and recomputes unit
rankings; sweeps the RRF weight of the unit list against the M2/M3 graph list.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.qasper.evalutil import OUT, load_papers, read_jsonl  # noqa: E402
from src.qasper.retrieval_metrics import question_metrics  # noqa: E402
from src.qasper.units import KnowledgeIndex, load_units, rrf_merge  # noqa: E402


def load_embedder():
    from src.embedding.nvembed import NVEmbedder
    from src.graph.hipporag_adapter import make_base_config
    from src.graph.runtime import EmbeddingProvider

    prov = EmbeddingProvider.get(make_base_config(OUT / "cache" / "embed_cfg"))
    return NVEmbedder(prov.model, cache_path=OUT / "cache" / "para_embeddings.pkl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="sample100")
    ap.add_argument("--grouping", default="section")
    args = ap.parse_args()
    rows = read_jsonl(OUT / "retrieval" / f"comparison_{args.scope}.jsonl")
    by_q = defaultdict(dict)
    meta = {}
    for r in rows:
        by_q[r["qid"]][r["method"]] = r
        meta[r["qid"]] = r
    papers = load_papers(args.scope)
    embedder = load_embedder()
    out = []
    for w in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        per_base = defaultdict(list)
        for pid, paper in papers.items():
            units = load_units(pid, args.grouping)
            if not units:
                continue
            idx = KnowledgeIndex(units, embedder, "unit")
            for q in paper.qas:
                if q.qid not in by_q:
                    continue
                urank = idx.rank_passages(q.question, 20)
                gold = [s for s in q.annotator_sets if s]
                if not gold:
                    continue
                for base in ("m2_atom", "m3_atom_comp"):
                    g = [{"passage_id": x} for x in by_q[q.qid][base]["ranked"]]
                    fused = rrf_merge([g, urank], k=60, top_k=20, weights=[1.0, w])
                    m = question_metrics([x["passage_id"] for x in fused], gold)
                    per_base[base].append({
                        "w": w, "base": base,
                        "unit_gold_upper": by_q[q.qid][base].get("unit_gold_upper"),
                        **m,
                    })
        for base, recs in per_base.items():
            for flag in (None, True, False):
                sub = [r for r in recs if flag is None or r["unit_gold_upper"] == flag]
                if not sub:
                    continue
                out.append({
                    "weight": w, "base": base,
                    "subset": "all" if flag is None else ("covered" if flag else "uncovered"),
                    "n": len(sub),
                    "any@5": sum(r["any@5"] for r in sub) / len(sub),
                    "all@5": sum(r["all@5"] for r in sub) / len(sub),
                    "recall@5": sum(r["recall@5"] for r in sub) / len(sub),
                })
    path = OUT / "retrieval" / f"fusion_sweep_{args.scope}.json"
    path.write_text(json.dumps(out, indent=1))
    for r in out:
        if r["subset"] == "all":
            print(r)


if __name__ == "__main__":
    main()
