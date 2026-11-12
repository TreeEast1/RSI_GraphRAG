#!/usr/bin/env python
"""Exploratory comparison of section vs semantic reading groups.

Uses the same frozen M2 graph ranking for both conditions; only the unit index
differs. Reports unit gold coverage and M2+unit fusion on a paper subset.
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
    ap.add_argument("--papers-file", required=True)
    ap.add_argument("--groupings", default="section,semantic")
    args = ap.parse_args()
    keep = {l.strip() for l in open(args.papers_file) if l.strip()}
    papers = {p: v for p, v in load_papers(args.scope).items() if p in keep}
    rows = read_jsonl(OUT / "retrieval" / f"comparison_{args.scope}.jsonl")
    m2 = defaultdict(dict)
    for r in rows:
        if r["method"] == "m2_atom":
            m2[r["qid"]] = r
    emb = load_embedder()
    out = []
    for grouping in args.groupings.split(","):
        stat = {k: [] for k in ("any@5", "all@5", "recall@5")}
        cov = tot = 0
        n_units = 0
        for pid, paper in papers.items():
            units = load_units(pid, grouping)
            n_units += len(units)
            if not units:
                continue
            idx = KnowledgeIndex(units, emb, "unit")
            src = set()
            for u in units:
                src |= set(u.get("source_passage_ids") or [])
            for q in paper.qas:
                if q.qid not in m2:
                    continue
                sets = [s for s in q.annotator_sets if s]
                if not sets:
                    continue
                gold = set().union(*[set(s) for s in sets])
                tot += 1
                cov += bool(gold & src)
                urank = idx.rank_passages(q.question, 20)
                fused = rrf_merge([[{"passage_id": x} for x in m2[q.qid]["ranked"]], urank], top_k=20)
                met = question_metrics([x["passage_id"] for x in fused], sets)
                for k in stat:
                    stat[k].append(met[k])
        out.append({"grouping": grouping, "papers": len(papers), "n_units": n_units,
                    "questions": tot, "unit_gold_coverage": cov / tot if tot else None,
                    **{k: sum(v) / len(v) if v else None for k, v in stat.items()}})
    (OUT / "retrieval" / "grouping_comparison.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
