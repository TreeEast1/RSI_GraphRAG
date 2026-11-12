#!/usr/bin/env python
"""QASPER paragraph retrieval baselines: BM25, dense (NV-Embed-v2), hybrid RRF."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.qasper.evalutil import OUT, evaluate_paper_questions, load_papers, summarize, write_jsonl  # noqa: E402
from src.qasper.retrieval import HybridRetriever  # noqa: E402


def load_embedder():
    from src.embedding.nvembed import NVEmbedder
    from src.graph.hipporag_adapter import make_base_config
    from src.graph.runtime import EmbeddingProvider

    cfg = make_base_config(OUT / "cache" / "embed_cfg")
    provider = EmbeddingProvider.get(cfg)
    return NVEmbedder(provider.model, cache_path=OUT / "cache" / "para_embeddings.pkl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="smoke20", choices=["smoke20", "sample100", "all"])
    ap.add_argument("--methods", default="bm25,dense,hybrid")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    papers = load_papers(args.scope)
    embedder = load_embedder() if any(m in ("dense", "hybrid") for m in methods) else None

    out_dir = OUT / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.out or f"baselines_{args.scope}"
    rows = []
    t0 = time.time()
    for i, (pid, paper) in enumerate(papers.items(), 1):
        hybrid = HybridRetriever([p.to_dict() for p in paper.paragraphs], embedder=embedder)
        rows.extend(evaluate_paper_questions(
            paper, lambda q, m, k: hybrid.retrieve(q, method=m, k=k), methods, top_k=args.top_k))
        print(f"[{i}/{len(papers)}] {pid}: {len(paper.paragraphs)} paras, {len(paper.qas)} qas, "
              f"{time.time()-t0:.0f}s", flush=True)
    write_jsonl(out_dir / f"{tag}.jsonl", rows)

    summary = {}
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        summary[m] = {"overall": summarize(sub)}
        for name in ("multi_evidence", "cross_section"):
            summary[m][name] = summarize([r for r in sub if r.get(name)])
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({m: summary[m]["overall"] for m in methods}, indent=1))


if __name__ == "__main__":
    main()
