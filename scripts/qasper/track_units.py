#!/usr/bin/env python
"""Track every verified comprehension unit through retrieval and QA.

For each unit: how often it is a retrieval candidate, how often it reaches the
final top-5, whether it touches gold evidence, and whether the questions that
retrieve it improve under M5 relative to M3. Also writes per-question chains.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.qasper.evalutil import OUT, load_papers, read_jsonl, write_jsonl  # noqa: E402
from src.qasper.units import KnowledgeIndex, load_units  # noqa: E402


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
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()
    papers = load_papers(args.scope)
    embedder = load_embedder()
    comp_rows = read_jsonl(OUT / "retrieval" / f"comparison_{args.scope}.jsonl")
    by_q = defaultdict(dict)
    for r in comp_rows:
        by_q[r["qid"]][r["method"]] = r
    qa_path = OUT / "qa" / f"predictions_{args.scope}.jsonl"
    qa = {}
    if qa_path.exists():
        for r in read_jsonl(qa_path):
            qa.setdefault(r["qid"], {})[r["method"]] = r

    unit_rows, chain_rows = [], []
    for i, (pid, paper) in enumerate(papers.items(), 1):
        if not any(q.qid in by_q for q in paper.qas):
            continue
        units = load_units(pid, args.grouping)
        if not units:
            continue
        idx = KnowledgeIndex(units, embedder, "unit")
        stats = {u["unit_id"]: {"unit_id": u["unit_id"], "paper_id": pid,
                                "text": u["concise_text"],
                                "source_passage_ids": u["source_passage_ids"],
                                "n_candidate_q": 0, "n_top5_q": 0, "n_gold_touch_q": 0,
                                "n_helped_q": 0, "n_harmed_q": 0}
                 for u in units}
        for q in paper.qas:
            recs = idx.rank_records(q.question, 50)
            gold = set(q.gold_passages_union)
            top50 = [r["unit_id"] for r in recs]
            top5 = [r["unit_id"] for r in recs[:5]]
            touch = [r["unit_id"] for r in recs if set(r["source_passage_ids"]) & gold]
            for uid in top50:
                stats[uid]["n_candidate_q"] += 1
            for uid in top5:
                stats[uid]["n_top5_q"] += 1
            for uid in touch:
                stats[uid]["n_gold_touch_q"] += 1
            qr = by_q.get(q.qid, {})
            m3 = qr.get("m3_atom_comp", {})
            m5 = qr.get("m5_units_graph", {})
            m2 = qr.get("m2_atom", {})
            m4 = qr.get("m4_units", {})
            qa3 = qa.get(q.qid, {}).get("m3_atom_comp", {})
            qa5 = qa.get(q.qid, {}).get("m5_units_graph", {})
            delta_f1 = (qa5.get("answer_f1", 0) - qa3.get("answer_f1", 0)) if qa3 and qa5 else None
            if delta_f1 is not None:
                for uid in top5:
                    if delta_f1 > 1e-9:
                        stats[uid]["n_helped_q"] += 1
                    elif delta_f1 < -1e-9:
                        stats[uid]["n_harmed_q"] += 1
            chain_rows.append({
                "paper_id": pid, "qid": q.qid, "question": q.question,
                "gold_passages": sorted(gold),
                "units_top50": top50, "units_top5": top5, "units_touching_gold": touch,
                "all5_m2": m2.get("metrics", {}).get("all@5"),
                "all5_m3": m3.get("metrics", {}).get("all@5"),
                "all5_m4": m4.get("metrics", {}).get("all@5"),
                "all5_m5": m5.get("metrics", {}).get("all@5"),
                "any5_m2": m2.get("metrics", {}).get("any@5"),
                "any5_m5": m5.get("metrics", {}).get("any@5"),
                "m3_f1": qa3.get("answer_f1"), "m5_f1": qa5.get("answer_f1"),
                "delta_f1": delta_f1,
            })
        unit_rows.extend(stats.values())
        print(f"[{i}/{len(papers)}] {pid}: {len(units)} units", flush=True)

    out_dir = OUT / "mechanism"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / f"unit_usage_{args.scope}.jsonl", unit_rows)
    write_jsonl(out_dir / f"question_chains_{args.scope}.jsonl", chain_rows)
    n = len(unit_rows)
    summary = {
        "n_units": n,
        "never_retrieved": sum(1 for u in unit_rows if u["n_candidate_q"] == 0),
        "never_top5": sum(1 for u in unit_rows if u["n_top5_q"] == 0),
        "touches_gold": sum(1 for u in unit_rows if u["n_gold_touch_q"] > 0),
        "touches_gold_but_never_top5": sum(1 for u in unit_rows
                                           if u["n_gold_touch_q"] > 0 and u["n_top5_q"] == 0),
        "helped_qa": sum(1 for u in unit_rows if u["n_helped_q"] > 0),
        "harmed_qa": sum(1 for u in unit_rows if u["n_harmed_q"] > 0),
        "questions": len(chain_rows),
        "questions_with_gold_unit": sum(1 for c in chain_rows if c["units_touching_gold"]),
    }
    (out_dir / f"summary_{args.scope}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
