#!/usr/bin/env python
"""P0 baselines: Dense RAG, HippoRAG2 (G_raw), Full-Context QA.

Usage
-----
python scripts/run_baselines.py --convs conv-30 --max-questions 8 --methods dense,hipporag
python scripts/run_baselines.py --split dev
python scripts/run_baselines.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PATHS  # noqa: E402
from src.dataset.locomo import load_locomo, num_tokens  # noqa: E402
from src.evaluation.metrics import exact_match, f1_score, select_by_budget  # noqa: E402
from src.evaluation.pipeline import question_record, summarize, write_jsonl  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.reader import Reader  # noqa: E402

sys.path.insert(0, str(PATHS["third_party"] / "src"))
from hipporag.utils.misc_utils import Chunk  # noqa: E402


def load_split() -> dict:
    p = PATHS["splits"] / "split.json"
    return json.loads(p.read_text()) if p.exists() else {"dev": [], "test": []}


def chunks_for(conv):
    return [
        Chunk(content=s.text, source_id=s.document_id,
              metadata={"document_id": s.document_id, "session_id": s.session_id,
                        "date_time": s.date_time_raw, "conversation_id": conv.conversation_id})
        for s in conv.sessions
    ]


def docs_for(conv):
    return [
        {"document_id": s.document_id, "session_id": s.session_id,
         "header": f"Session {s.session_id} | {s.date_time_raw}", "content": s.text}
        for s in conv.sessions
    ]


def hippo_ranked(res, docs) -> list:
    by_content = {d["content"]: d for d in docs}
    out = []
    for d, m, sc in zip(res.docs, res.doc_metadata, res.doc_scores):
        base = None
        if isinstance(m, dict) and m.get("document_id"):
            base = {**by_content.get(d, {}), "document_id": m["document_id"],
                    "session_id": m.get("session_id"),
                    "header": f"Session {m.get('session_id')} | {m.get('date_time','')}",
                    "content": d}
        else:
            base = by_content.get(d)
        if base:
            out.append({**base, "score": float(sc)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, choices=[None, "dev", "test"])
    ap.add_argument("--convs", default=None)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--methods", default="dense,hipporag,fullcontext")
    ap.add_argument("--force-index", action="store_true")
    ap.add_argument("--out-prefix", default="baselines")
    ap.add_argument("--max-workers", type=int, default=16)
    args = ap.parse_args()

    from src.adapters.hy4_client import HY4Client
    from src.embedding.nvembed import NVEmbedder
    from src.graph.runtime import EmbeddingProvider, open_hipporag, make_base_config

    convs = {c.sample_id: c for c in load_locomo(ROOT / "data" / "raw" / "locomo" / "locomo10.json")}
    split = load_split()
    if args.convs:
        conv_ids = args.convs.split(",")
    elif args.split:
        conv_ids = split[args.split]
    else:
        conv_ids = sorted(convs)

    client = HY4Client(tag="reader")
    reader = Reader(client, tag="reader")
    methods = set(args.methods.split(","))

    cfg0 = make_base_config(PATHS["raw_graphs"] / "_warmup", conv_id="warmup")
    provider = EmbeddingProvider.get(cfg0)
    import os as _os
    _shard = _os.environ.get("HY4_CACHE_SHARD", "")
    embedder = NVEmbedder(provider.model, PATHS["cache"] / f"doc_embeddings{_shard}.pkl")

    summaries = {}
    all_rows = []
    for conv_id in conv_ids:
        conv = convs[conv_id]
        docs = docs_for(conv)
        qs = conv.qa[: args.max_questions] if args.max_questions else conv.qa
        full_text = "\n\n".join(s.text for s in conv.sessions)
        full_tokens = num_tokens(full_text)
        print(f"\n=== {conv_id}: {len(conv.sessions)} sessions, {len(qs)} QA, full={full_tokens} tok ===")

        save_dir = PATHS["raw_graphs"] / conv_id
        from src.graph.runtime import working_dir as _wd
        _wd_path = _wd(save_dir, "hy4", __import__("src.config", fromlist=["NVEMBED_PATH"]).NVEMBED_PATH)
        _need_rebuild = args.force_index or (not (_wd_path / "graph.pickle").exists())
        t = time.time()
        with open_hipporag(str(save_dir), conv_id=conv_id, embedding_provider=provider,
                           force_index_from_scratch=_need_rebuild) as hippo:
            hippo.index(chunks_for(conv))
            print(f"  G_raw ready {time.time()-t:.1f}s | {hippo.get_graph_info()}")

            rows_by_method = {}

            if "hipporag" in methods:
                t = time.time()
                res = hippo.retrieve([q.question for q in qs], num_to_retrieve=args.top_k)
                print(f"  hipporag retrieval {time.time()-t:.1f}s")
                ranked_list = [hippo_ranked(r, docs) for r in res]
                ctxs = [select_by_budget(r, args.budget, num_tokens) for r in ranked_list]
                t = time.time()
                ans = reader.answer_batch([(q.question, c) for q, c in zip(qs, ctxs)],
                                          max_workers=args.max_workers)
                print(f"  hipporag QA {time.time()-t:.1f}s")
                rows_by_method["hipporag"] = [
                    question_record("hipporag", conv, q, r, reader_answer=a["answer"],
                                    reader_usage=a["usage"],
                                    extra={"context_tokens": sum(num_tokens(d["content"]) for d in c),
                                           "n_context": len(c), "budget": args.budget})
                    for q, r, c, a in zip(qs, ranked_list, ctxs, ans)
                ]

            if "dense" in methods:
                t = time.time()
                dense = DenseRetriever(embedder, docs)
                ranked_list = [dense.retrieve(q.question, k=args.top_k) for q in qs]
                ctxs = [select_by_budget(r, args.budget, num_tokens) for r in ranked_list]
                ans = reader.answer_batch([(q.question, c) for q, c in zip(qs, ctxs)],
                                          max_workers=args.max_workers)
                print(f"  dense {time.time()-t:.1f}s")
                rows_by_method["dense"] = [
                    question_record("dense", conv, q, r, reader_answer=a["answer"],
                                    reader_usage=a["usage"],
                                    extra={"context_tokens": sum(num_tokens(d["content"]) for d in c),
                                           "n_context": len(c), "budget": args.budget})
                    for q, r, c, a in zip(qs, ranked_list, ctxs, ans)
                ]

            if "fullcontext" in methods and full_tokens < 300000:
                t = time.time()
                ans = reader.answer_full_context_batch([(q.question, full_text) for q in qs],
                                                       max_workers=args.max_workers)
                print(f"  fullcontext QA {time.time()-t:.1f}s")
                rows_by_method["fullcontext"] = [
                    question_record("fullcontext", conv, q, docs, reader_answer=a["answer"],
                                    reader_usage=a["usage"],
                                    extra={"input_tokens": full_tokens, "budget": None})
                    for q, a in zip(qs, ans)
                ]

            for method, rows in rows_by_method.items():
                write_jsonl(PATHS["qa"] / f"{args.out_prefix}_{method}_{conv_id}.jsonl", rows)
                all_rows.extend(rows)

    for method in sorted({r["method"] for r in all_rows}):
        summaries[method] = summarize([r for r in all_rows if r["method"] == method], method)

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(args),
           "summaries": summaries, "hy4": client.summary()}
    p = PATHS["reports"] / f"{args.out_prefix}_summary.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n=== summary ===")
    for m, s in summaries.items():
        print(f"{m:12s} EM={s.get('em')} F1={s.get('f1')} R@5={s.get('recall@5')} "
              f"R@20={s.get('recall@20')} ev@4096={s.get('evidence_full@4096')}")
    print("written", p)


if __name__ == "__main__":
    main()
