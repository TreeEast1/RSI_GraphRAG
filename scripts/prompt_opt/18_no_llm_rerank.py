#!/usr/bin/env python
"""Ablation requested in review: disable HippoRAG2's per-query LLM fact reranker.

The official pipeline is:
    embedding top-5 fact candidates -> LLM filter -> PPR (empty filter -> dense)
This script replaces the LLM filter with a pure embedding top-k selection:
    embedding top-k facts -> PPR (no LLM call, no filter-induced dense fallback)

We run k=5 (same final fact count as the normal pipeline) and k=20 (wider seed set)
on G_raw, C2 and COMP-P4.  No LLM calls are made in this script.

Outputs:
  comprehension/retrieval_norerank/{graph}_top{k}/records.jsonl
  comprehension/retrieval_norerank/{graph}_top{k}/diagnostics.jsonl
  comprehension/retrieval_norerank/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import PATHS  # noqa: E402
from src.dataset.locomo import load_locomo  # noqa: E402
from src.evaluation.pipeline import question_record, summarize  # noqa: E402
from src.prompt_optimization.retrieval_diag import retrieve_with_diag  # noqa: E402
from src.graph.variants import raw_save_dir  # noqa: E402

OUT = ROOT / "outputs" / "prompt_optimization_conv30" / "comprehension" / "retrieval_norerank"
GRAPH_DIRS = {
    "G_raw": lambda: raw_save_dir("conv-30"),
    "C2": lambda: raw_save_dir("conv-30") / "variants" / "cluster_hybrid_0.5_b8192_triple",
    "COMP-P4": lambda: raw_save_dir("conv-30") / "variants" / "cmpo_compp4",
}


def ranked_from_diag(d, docs_by_id):
    out = []
    for meta, score in zip(d["doc_metadata"], d["doc_scores"]):
        did = meta.get("document_id")
        base = docs_by_id.get(did)
        if base is not None:
            out.append({**base, "score": float(score)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="G_raw,C2,COMP-P4")
    ap.add_argument("--fact-ks", default="5,20")
    ap.add_argument("--num-to-retrieve", type=int, default=20)
    args = ap.parse_args()

    from src.graph.runtime import EmbeddingProvider, make_base_config, open_hipporag
    from src.graph.hipporag_adapter import HY4CacheOpenAI

    conv = {c.sample_id: c for c in load_locomo(ROOT / "data" / "raw" / "locomo" / "locomo10.json")}["conv-30"]
    questions = conv.qa
    docs_by_id = {s.document_id: {"document_id": s.document_id, "session_id": s.session_id,
                                  "header": f"Session {s.session_id} | {s.date_time_raw}",
                                  "content": s.text} for s in conv.sessions}
    cfg0 = make_base_config(PATHS["raw_graphs"] / "_warmup", conv_id="warmup")
    provider = EmbeddingProvider.get(cfg0)
    llm = HY4CacheOpenAI.from_experiment_config(cfg0)

    OUT.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for graph in args.graphs.split(","):
        save_dir = GRAPH_DIRS[graph]()
        for k in [int(x) for x in args.fact_ks.split(",")]:
            name = f"{graph}_top{k}"
            print(f"\n=== {name} (no LLM rerank, link_top_k={k}) ===")
            with open_hipporag(str(save_dir), conv_id="conv-30", embedding_provider=provider, llm=llm) as hippo:
                t = time.time()
                diag = retrieve_with_diag(hippo, [q.question for q in questions],
                                          num_to_retrieve=args.num_to_retrieve,
                                          use_llm_rerank=False, fact_top_k=k, link_top_k=k)
                print(f"  retrieval {time.time()-t:.0f}s | {hippo.get_graph_info()}")
            rows, drows = [], []
            for qa, d in zip(questions, diag):
                ranked = ranked_from_diag(d, docs_by_id)
                rec = question_record(name, conv, qa, ranked, extra={
                    "fallback_dense": d["fallback_dense"],
                    "n_fact_candidates_before_rerank": d["n_fact_candidates_before_rerank"],
                    "n_facts_after_rerank": d["n_facts_after_rerank"],
                    "no_llm_rerank": True, "fact_top_k": k, "link_top_k": k,
                })
                rec["diag"] = {"fallback_dense": d["fallback_dense"],
                               "n_facts_before": d["n_fact_candidates_before_rerank"],
                               "n_facts_after": d["n_facts_after_rerank"],
                               "top_facts": d["top_facts"][:10]}
                rows.append(rec)
                drows.append({"qid": qa.qid, "question": qa.question, "category": qa.category,
                              "fallback_dense": d["fallback_dense"],
                              "top_facts": d["top_facts"][:10],
                              "ranked_docs": rec["ranked_docs"][:20]})
            d = OUT / name
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "records.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(d / "diagnostics.jsonl", "w", encoding="utf-8") as f:
                for r in drows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            s = summarize(rows, name)
            s["fallback_dense_count"] = sum(r["diag"]["fallback_dense"] for r in rows)
            s["no_llm_rerank"] = True
            s["fact_top_k"] = k
            summaries[name] = s
            print(f"  R@3={s['recall@3']:.4f} R@5={s['recall@5']:.4f} MRR={s['mrr']:.4f} "
                  f"nDCG@10={s['ndcg@10']:.4f} Ev@4096={s['evidence_full@4096']:.4f} "
                  f"fallback={s['fallback_dense_count']}")
    (OUT / "summary.json").write_text(json.dumps({"summaries": summaries}, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
