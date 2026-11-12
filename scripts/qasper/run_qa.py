#!/usr/bin/env python
"""QA reader evaluation for M0-M6 with a fixed context budget and identical prompt.

Modes:
  *_4096   paragraphs only, greedy under 4096 tokens (primary)
  *_B      comprehension units (or summaries) + source paragraphs, same 4096 total
  *_top5   top-5 paragraphs only (reader precision sensitivity)
  oracle   gold evidence paragraphs (upper bound reference)
  full     whole paper truncated to 12000 tokens (not a fixed-budget baseline)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.adapters.hy4_client import HY4Client, HY4Config, load_env  # noqa: E402
from src.qasper.evalutil import OUT, load_papers, read_jsonl, write_jsonl  # noqa: E402
from src.qasper.qa import gold_refs, score_one, summarize_predictions  # noqa: E402
from src.qasper.reader import QasperReader, batch_answer, build_context  # noqa: E402
from src.qasper.units import KnowledgeIndex, load_summaries, load_units  # noqa: E402

load_env()

BASE_METHODS = ["m0_dense", "m1_raw", "m2_atom", "m3_atom_comp", "m4_units",
                "m5_units_graph"]
B_METHODS = ["m5_units_graph"]
TOP5_METHODS = []


def load_embedder():
    from src.embedding.nvembed import NVEmbedder
    from src.graph.hipporag_adapter import make_base_config
    from src.graph.runtime import EmbeddingProvider

    cfg = make_base_config(OUT / "cache" / "embed_cfg")
    prov = EmbeddingProvider.get(cfg)
    return NVEmbedder(prov.model, cache_path=OUT / "cache" / "para_embeddings.pkl")


def pick_context(ranked, paras, tok, budget=4096):
    return build_context(paras, ranked, budget=budget)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="sample100")
    ap.add_argument("--grouping", default="section")
    ap.add_argument("--comparison", default=None)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--unit-budget", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    papers = load_papers(args.scope)
    if args.num_shards > 1:
        papers = {pid: p for i, (pid, p) in enumerate(papers.items())
                  if i % args.num_shards == args.shard}
    suffix = f"_s{args.shard}" if args.num_shards > 1 else ""
    comp_path = (Path(args.comparison) if args.comparison
                 else OUT / "retrieval" / f"comparison_{args.scope}{suffix}.jsonl")
    rows = read_jsonl(comp_path)
    by_q = {}
    for r in rows:
        by_q.setdefault(r["qid"], {})[r["method"]] = r["ranked"]
    have = set(by_q)
    papers = {pid: p for pid, p in papers.items() if any(q.qid in have for q in p.qas)}
    print(f"QA scope: {len(papers)} papers, {len(have)} questions")

    embedder = load_embedder()
    import os
    tag = os.environ.get("QASPER_QA_CACHE_TAG", "")
    cfg = HY4Config.from_env(max_concurrency=int(os.environ.get("HY4_MAX_CONCURRENCY", "6")))
    client = HY4Client(cfg, cache_path=OUT / "cache" / f"hy4_qa{tag}.sqlite",
                       log_path=OUT / "logs" / f"hy4_calls_qa{tag}.jsonl", tag="qasper_qa")
    reader = QasperReader(client)

    out_rows = []
    t0 = time.time()
    for i, (pid, paper) in enumerate(papers.items(), 1):
        if args.limit and i > args.limit:
            break
        paras = {p.passage_id: p.to_dict() for p in paper.paragraphs}
        tok = {p.passage_id: p.n_tokens for p in paper.paragraphs}
        ordered_all = [p.passage_id for p in paper.paragraphs]
        units = load_units(pid, args.grouping)
        summaries = load_summaries(pid, args.grouping)
        uidx = KnowledgeIndex(units, embedder, "unit")
        sidx = KnowledgeIndex(summaries, embedder, "summary")

        items = []
        for q in paper.qas:
            qmap = by_q.get(q.qid, {})
            base = qmap.get("m2_atom") or qmap.get("m1_raw") or ordered_all
            gold_ids = q.gold_passages_union
            variants = {}
            for m in BASE_METHODS:
                variants[m] = (qmap.get(m, base), None, 0)
            for m in B_METHODS:
                idx = uidx if m != "m6_summary" else sidx
                recs = idx.rank_records(q.question, 30) if len(idx) else []
                variants[f"{m}_B"] = (qmap.get(m, base), recs, args.unit_budget)
            for m in TOP5_METHODS:
                variants[f"{m}_top5"] = (qmap.get(m, base)[:5], None, 0)
            # oracle and full paper
            variants["oracle"] = (gold_ids, None, 0)
            variants["full"] = (ordered_all, None, 0)
            for name, (ranked, urecs, ubudget) in variants.items():
                budget = {"oracle": 8192, "full": 12000}.get(name, args.budget)
                ctx = build_context(paras, ranked, budget=budget, units=urecs, unit_budget=ubudget,
                                    unit_label="Insight" if not name.startswith("m6_summary") else "Summary")
                items.append((q, name, ctx))

        answers = batch_answer(reader, [(q.question, ctx["context"]) for q, _, ctx in items],
                               max_workers=args.workers)
        for (q, name, ctx), ans in zip(items, answers):
            refs = gold_refs(q)
            pred = {"answer": ans.get("answer", ""),
                    "evidence": [paras[p]["text"] for p in ctx["passages"] if p in paras]}
            sc = score_one(pred, refs)
            gold = set(q.gold_passages_union)
            sel = set(ctx["passages"])
            out_rows.append({
                "method": name, "paper_id": pid, "qid": q.qid, "question": q.question,
                "answerable": q.answerable, "cross_section": q.cross_section,
                "multi_evidence": len(gold) > 1,
                "gold_passages": sorted(gold),
                **sc,
                "pred_answer": pred["answer"], "raw": ans.get("raw", ""),
                "n_context": len(ctx["passages"]), "context_tokens": ctx["tokens"],
                "context_passages": ctx["passages"], "context_units": ctx["units"],
                "gold_any_in_context": bool(gold & sel),
                "gold_all_in_context": bool(gold and gold <= sel),
                "usage": ans.get("usage", {}), "error": ans.get("error"),
            })
        print(f"[{i}/{len(papers)}] {pid}: {len(paper.qas)} q x {len(set(x[1] for x in items))} methods, "
              f"{time.time()-t0:.0f}s", flush=True)

    out = OUT / "qa" / f"predictions_{args.scope}{suffix}.jsonl"
    write_jsonl(out, out_rows)

    summary = {}
    annots = {}
    for r in out_rows:
        annots.setdefault(r["method"], []).append(r)
    subsets = json.loads((OUT / "dataset" / "subsets.json").read_text())
    for m, recs in annots.items():
        summary[m] = {"overall": summarize_predictions(recs)}
        summary[m]["answerable"] = summarize_predictions([r for r in recs if r["answerable"]])
        summary[m]["unanswerable"] = summarize_predictions([r for r in recs if not r["answerable"]])
        summary[m]["multi_evidence"] = summarize_predictions(
            [r for r in recs if r["multi_evidence"]])
        summary[m]["cross_section"] = summarize_predictions(
            [r for r in recs if r["cross_section"]])
        summary[m]["gold_in_context"] = {
            "any": sum(r["gold_any_in_context"] for r in recs) / max(1, len(recs)),
            "all": sum(r["gold_all_in_context"] for r in recs) / max(1, len(recs)),
        }
    (OUT / "qa" / f"summary_{args.scope}{suffix}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({m: summary[m]["overall"] for m in summary}, indent=1))


if __name__ == "__main__":
    main()
