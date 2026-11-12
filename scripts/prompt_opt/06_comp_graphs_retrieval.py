#!/usr/bin/env python
"""Phase 3.2 — build one graph per comprehension prompt and run retrieval with
intermediate-state diagnostics on all 105 conv-30 questions (no Reader calls).

Graph = G_raw OpenIE state + the fixed C2 triples + this prompt's relations.
The C2 triples are held identical across variants, so any ranking change is
attributable to the comprehension relations.

Outputs:
  outputs/raw_graphs/conv-30/variants/{variant}/
  comprehension/retrieval/{variant}.jsonl
  comprehension/retrieval/summary.json
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
from src.dataset.locomo import load_locomo, num_tokens  # noqa: E402
from src.evaluation.pipeline import question_record, summarize  # noqa: E402
from src.prompt_optimization.retrieval_diag import retrieve_with_diag  # noqa: E402

OUT = ROOT / "outputs" / "prompt_optimization_conv30" / "comprehension" / "retrieval"


def load_c2():
    return json.loads((PATHS["comprehension"] / "conv-30" / "cluster_hybrid_0.5_b8192_triple.json").read_text())


def load_scope(scope: str):
    return json.loads((PATHS["comprehension"] / "conv-30" / f"{scope}.json").read_text())


def records_to_extra(records, conv):
    sid_to_doc = {s.session_id: s.document_id for s in conv.sessions}
    extra = {}
    for item in records:
        t = (item["subject"], item["predicate"], item["object"])
        for sid in item.get("source_sessions", []):
            doc = sid_to_doc.get(sid)
            if doc:
                extra.setdefault(doc, []).append(t)
    return extra


def fact_trace(records, origin: str):
    from src.prompt_optimization.retrieval_diag import norm_fact
    out = {}
    for r in records:
        note = (f"{origin}|{r.get('relation_type','')}|sessions={r.get('source_sessions')}|"
                f"conf={r.get('confidence')}|deriv={r.get('derivation')}")
        raw = tuple(str(x) for x in (r["subject"], r["predicate"], r["object"]))
        out[raw] = note
        out[norm_fact(raw)] = note
    return out


def ranked_from_diag(d, docs_by_id):
    out = []
    for doc_text, meta, score in zip(d["docs"], d["doc_metadata"], d["doc_scores"]):
        did = meta.get("document_id")
        base = docs_by_id.get(did)
        if base is None:
            continue
        out.append({**base, "score": float(score)})
    return out


def run_retrieval(hippo, conv, questions, docs_by_id, trace, method):
    diag = retrieve_with_diag(hippo, [q.question for q in questions], num_to_retrieve=20,
                              fact_trace=trace)
    rows, diag_rows = [], []
    for qa, d in zip(questions, diag):
        ranked = ranked_from_diag(d, docs_by_id)
        rec = question_record(method, conv, qa, ranked, extra={
            "fallback_dense": d["fallback_dense"],
            "n_fact_candidates_before_rerank": d["n_fact_candidates_before_rerank"],
            "n_facts_after_rerank": d["n_facts_after_rerank"],
            "top_facts_traced": d["top_facts_traced"][:10],
        })
        rec["diag"] = {
            "fallback_dense": d["fallback_dense"],
            "n_facts_before": d["n_fact_candidates_before_rerank"],
            "n_facts_after": d["n_facts_after_rerank"],
            "top_facts": d["top_facts"][:10],
            "top_facts_traced": d["top_facts_traced"][:10],
        }
        rows.append(rec)
        diag_rows.append({
            "qid": qa.qid, "question": qa.question, "category": qa.category,
            "fallback_dense": d["fallback_dense"],
            "n_facts_before": d["n_fact_candidates_before_rerank"],
            "n_facts_after": d["n_facts_after_rerank"],
            "top_facts": d["top_facts"][:10], "top_facts_traced": d["top_facts_traced"][:10],
            "ranked_docs": rec["ranked_docs"][:20],
            "ranked_scores": rec["ranked_scores"][:20],
        })
    return rows, diag_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", default="COMP-P1,COMP-P2,COMP-P3,COMP-P4,COMP-P5")
    ap.add_argument("--force-build", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    from src.graph.runtime import EmbeddingProvider, make_base_config, open_hipporag
    from src.graph.variants import build_variant_graph, raw_save_dir
    from src.graph.hipporag_adapter import HY4CacheOpenAI

    conv = {c.sample_id: c for c in load_locomo(ROOT / "data" / "raw" / "locomo" / "locomo10.json")}["conv-30"]
    questions = conv.qa
    docs_by_id = {s.document_id: {"document_id": s.document_id, "session_id": s.session_id,
                                  "header": f"Session {s.session_id} | {s.date_time_raw}",
                                  "content": s.text} for s in conv.sessions}
    c2 = load_c2()
    base_triples = c2["triples"]

    cfg0 = make_base_config(PATHS["raw_graphs"] / "_warmup", conv_id="warmup")
    provider = EmbeddingProvider.get(cfg0)
    llm = HY4CacheOpenAI.from_experiment_config(cfg0)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    manifest = []

    def evaluate_graph(name, save_dir, records_for_trace, origin):
        trace = fact_trace(records_for_trace, origin)
        with open_hipporag(str(save_dir), conv_id="conv-30", embedding_provider=provider, llm=llm) as hippo:
            t = time.time()
            rows, diag_rows = run_retrieval(hippo, conv, questions, docs_by_id, trace, name)
            info = hippo.get_graph_info()
            print(f"  {name}: retrieval {time.time()-t:.0f}s graph={info}")
        OUT.joinpath(name).mkdir(parents=True, exist_ok=True)
        with open(OUT / name / "records.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(OUT / name / "diagnostics.jsonl", "w", encoding="utf-8") as f:
            for r in diag_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = summarize(rows, name)
        s["graph_info"] = info
        s["fallback_dense_count"] = sum(r["diag"]["fallback_dense"] for r in rows)
        s["records"] = f"comprehension/retrieval/{name}/records.jsonl"
        summary[name] = s
        manifest.append({"name": name, "save_dir": str(save_dir), "graph_info": info})
        return rows

    if not args.skip_baselines:
        print("=== G_raw baseline ===")
        evaluate_graph("G_raw", raw_save_dir("conv-30"), [], "explicit")
        print("=== C2 baseline ===")
        evaluate_graph("C2", raw_save_dir("conv-30") / "variants" / "cluster_hybrid_0.5_b8192_triple",
                       base_triples, "C2_explicit")

    for pid in args.pids.split(","):
        comp = json.loads((ROOT / "outputs" / "prompt_optimization_conv30" / "comprehension" /
                           "extractions" / f"{pid}.json").read_text())
        rels = comp["relations"]
        variant = f"cmpo_{pid.replace('-', '').lower()}"
        print(f"\n=== {pid}: {len(rels)} relations -> variant {variant} ===")
        records = list(base_triples) + list(rels)
        extra = records_to_extra(records, conv)
        info = build_variant_graph(conv, variant, extra, provider=provider, llm=llm,
                                   force=args.force_build)
        print(f"  build: {json.dumps(info)[:240]}")
        evaluate_graph(variant, raw_save_dir("conv-30") / "variants" / variant,
                       records, pid)

    (OUT / "summary.json").write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(args),
        "summaries": summary, "manifest": manifest,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== retrieval summary ===")
    for name, s in summary.items():
        print(f"{name:14s} R@3={s['recall@3']:.4f} R@5={s['recall@5']:.4f} MRR={s['mrr']:.4f} "
              f"nDCG@10={s['ndcg@10']:.4f} Ev@4096={s['evidence_full@4096']:.4f} "
              f"fallback={s['fallback_dense_count']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
