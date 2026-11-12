#!/usr/bin/env python
"""M0-M6 retrieval comparison on QASPER paragraphs.

M0 dense/BM25/hybrid, M1 G_raw, M2 G_raw+atomic facts, M3 M2+comprehension
triples, M4 M2+unit index, M5 M3+unit index, M6 M2+paragraph-summary index.
Every unit/summary method is also reported unit-only for mechanism tracking.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.qasper.evalutil import OUT, gold_sets, load_papers, write_jsonl  # noqa: E402
from src.qasper.graph import open_graph, retrieve  # noqa: E402
from src.qasper.retrieval import HybridRetriever  # noqa: E402
from src.qasper.retrieval_metrics import coverage_at_budget, question_metrics  # noqa: E402
from src.qasper.units import KnowledgeIndex, load_summaries, load_units, rrf_merge  # noqa: E402

METHODS = ["m0_dense", "m0_bm25", "m0_hybrid", "m1_raw", "m2_atom", "m3_atom_comp",
           "m4_units", "m5_units_graph", "m4_gated", "m5_gated", "m6_summary", "m6_matched",
           "m4_units_only", "m6_summary_only"]


def matched_summaries(summaries, n_units: int, seed: str):
    """Sample summaries to the same count as the verified comprehension units."""
    if n_units <= 0 or not summaries:
        return []
    if len(summaries) <= n_units:
        return list(summaries)
    import hashlib

    order = sorted(summaries, key=lambda s: hashlib.sha1(f"{seed}:{s['unit_id']}".encode()).hexdigest())
    return order[:n_units]


def load_embedder():
    from src.embedding.nvembed import NVEmbedder
    from src.graph.hipporag_adapter import make_base_config
    from src.graph.runtime import EmbeddingProvider

    cfg = make_base_config(OUT / "cache" / "embed_cfg")
    prov = EmbeddingProvider.get(cfg)
    return NVEmbedder(prov.model, cache_path=OUT / "cache" / "para_embeddings.pkl")


def rank_of(r):
    return [x["passage_id"] for x in r]


def unit_diag(recs, gold: set) -> dict:
    touched = [r for r in recs if set(r.get("source_passage_ids") or []) & gold]
    return {"n_candidates": len(recs),
            "n_touching_gold": len(touched),
            "gold_units": [r["unit_id"] for r in touched][:10]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="sample100")
    ap.add_argument("--grouping", default="section")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--budgets", default="2048,4096,8192")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--only-ready", action="store_true",
                    help="skip papers whose M2/M3 variants are not built yet")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--papers-file", default=None, help="restrict to paper ids listed in this file")
    ap.add_argument("--out-suffix", default=None, help="suffix for the comparison output file")
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",")]
    papers = load_papers(args.scope)
    if args.papers_file:
        keep = {l.strip() for l in open(args.papers_file) if l.strip()}
        papers = {pid: p for pid, p in papers.items() if pid in keep}
    if args.num_shards > 1:
        papers = {pid: p for i, (pid, p) in enumerate(papers.items())
                  if i % args.num_shards == args.shard}
    missing = []
    for pid in papers:
        for v in ("atom", "atom_comp"):
            if not (OUT / "graphs" / "raw" / pid / "variants" / v / "qasper_graph_info.json").exists():
                missing.append(f"{pid}:{v}")
    if missing and not args.allow_missing and not args.only_ready:
        raise SystemExit(f"{len(missing)} graph variants missing (first: {missing[:5]}). "
                         f"Run scripts/qasper/build_variants.py first or pass --allow-missing.")
    if args.only_ready:
        skip = {m.split(":")[0] for m in missing}
        papers = {pid: p for pid, p in papers.items() if pid not in skip}
        print(f"only-ready: {len(papers)} papers with both variants")
    embedder = load_embedder()
    rows = []
    t0 = time.time()
    for i, (pid, paper) in enumerate(papers.items(), 1):
        if args.limit and i > args.limit:
            break
        paras = {p.passage_id: p.to_dict() for p in paper.paragraphs}
        tok = {p.passage_id: p.n_tokens for p in paper.paragraphs}
        hyb = HybridRetriever(list(paras.values()), embedder=embedder)
        units = load_units(pid, args.grouping)
        summaries = load_summaries(pid, args.grouping)
        uidx = KnowledgeIndex(units, embedder, "unit")
        sidx = KnowledgeIndex(summaries, embedder, "summary")
        smatched = matched_summaries(summaries, len(units), pid)
        sidx_matched = KnowledgeIndex(smatched, embedder, "summary")
        unit_src_union = set()
        for u in units:
            unit_src_union |= set(u.get("source_passage_ids") or [])

        qs = [q for q in paper.qas if q.answerable and gold_sets(q)]
        g1 = {}
        with open_graph(pid, None, provider=None) as h:
            for q in qs:
                g1[q.qid] = retrieve(h, q.question, args.top_k)
        g2 = {}
        atom_dir = OUT / "graphs" / "raw" / pid / "variants" / "atom"
        if (atom_dir / "qasper_graph_info.json").exists():
            with open_graph(pid, "atom") as h:
                for q in qs:
                    g2[q.qid] = retrieve(h, q.question, args.top_k)
        g3 = {}
        comp_dir = OUT / "graphs" / "raw" / pid / "variants" / "atom_comp"
        if (comp_dir / "qasper_graph_info.json").exists():
            with open_graph(pid, "atom_comp") as h:
                for q in qs:
                    g3[q.qid] = retrieve(h, q.question, args.top_k)

        for q in qs:
            sets = gold_sets(q)
            gold = set().union(*[set(s) for s in sets]) if sets else set()
            urank = uidx.rank_passages(q.question, args.top_k) if len(uidx) else []
            srank = sidx.rank_passages(q.question, args.top_k) if len(sidx) else []
            srank_m = sidx_matched.rank_passages(q.question, args.top_k) if len(sidx_matched) else []
            ugated = uidx.rank_passages(q.question, args.top_k, max_units=3) if len(uidx) else []
            rankings = {
                "m0_dense": hyb.retrieve(q.question, "dense", args.top_k),
                "m0_bm25": hyb.retrieve(q.question, "bm25", args.top_k),
                "m0_hybrid": hyb.retrieve(q.question, "hybrid", args.top_k),
                "m1_raw": g1.get(q.qid, []),
                "m2_atom": g2.get(q.qid, g1.get(q.qid, [])),
                "m3_atom_comp": g3.get(q.qid, g2.get(q.qid, g1.get(q.qid, []))),
                "m4_units": rrf_merge([g2.get(q.qid, []), urank], top_k=args.top_k) if urank else g2.get(q.qid, []),
                "m5_units_graph": rrf_merge([g3.get(q.qid, []), urank], top_k=args.top_k) if urank else g3.get(q.qid, []),
                "m4_gated": rrf_merge([g2.get(q.qid, []), ugated], k=60, top_k=args.top_k,
                                      weights=[1.0, 0.5]) if ugated else g2.get(q.qid, []),
                "m5_gated": rrf_merge([g3.get(q.qid, []), ugated], k=60, top_k=args.top_k,
                                      weights=[1.0, 0.5]) if ugated else g3.get(q.qid, []),
                "m6_summary": rrf_merge([g2.get(q.qid, []), srank], top_k=args.top_k) if srank else g2.get(q.qid, []),
                "m6_matched": rrf_merge([g2.get(q.qid, []), srank_m], top_k=args.top_k) if srank_m else g2.get(q.qid, []),
                "m4_units_only": urank,
                "m6_summary_only": srank,
            }
            for method, ranked in rankings.items():
                ids = rank_of(ranked)
                rec = {
                    "method": method, "paper_id": pid, "qid": q.qid, "question": q.question,
                    "n_gold": max(len(s) for s in sets), "n_annotator_sets": len(sets),
                    "cross_section": q.cross_section,
                    "multi_evidence": max(len(s) for s in sets) > 1 or len(gold) > 1,
                    "unit_gold_upper": bool(gold & unit_src_union),
                    "unit_src_gold_fraction": (len(gold & unit_src_union) / len(gold)) if gold else None,
                    "n_units": len(units),
                    "ranked": ids,
                    "metrics": question_metrics(ids, sets),
                    "budget": {str(b): coverage_at_budget(ids, sets, tok, b) for b in budgets},
                }
                if method in ("m4_units", "m5_units_graph", "m4_units_only", "m4_gated", "m5_gated"):
                    rec["unit_diag"] = unit_diag(uidx.rank_records(q.question, 50) if len(uidx) else [], gold)
                if method in ("m6_summary", "m6_summary_only", "m6_matched"):
                    idx = sidx_matched if method == "m6_matched" else sidx
                    rec["unit_diag"] = unit_diag(idx.rank_records(q.question, 50) if len(idx) else [], gold)
                if method in ("m1_raw", "m2_atom", "m3_atom_comp"):
                    rec["graph_seeds"] = (ranked[0].get("graph_seeds") if ranked else []) or []
                rows.append(rec)
        print(f"[{i}/{len(papers)}] {pid}: {len(qs)} q, units={len(uidx)}, summaries={len(sidx)}, "
              f"{time.time()-t0:.0f}s", flush=True)
    suffix = args.out_suffix or (f"_s{args.shard}" if args.num_shards > 1 else "")
    out = OUT / "retrieval" / f"comparison_{args.scope}{suffix}.jsonl"
    write_jsonl(out, rows)
    print("wrote", out, len(rows))


if __name__ == "__main__":
    main()
