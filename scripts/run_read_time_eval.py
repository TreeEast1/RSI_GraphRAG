#!/usr/bin/env python
"""Read-time evaluation: build graph variants from extracted knowledge and measure
retrieval + QA with the same reader and token budget as the baselines.
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
from src.evaluation.metrics import select_by_budget  # noqa: E402
from src.evaluation.pipeline import question_record, summarize, write_jsonl  # noqa: E402
from src.retrieval.reader import Reader  # noqa: E402

sys.path.insert(0, str(PATHS["third_party"] / "src"))


def doc_map(conv):
    return {s.session_id: s.document_id for s in conv.sessions}


def triples_to_extra(records, conv) -> dict:
    """Attach each triple/relation to every source session it cites."""
    sid_to_doc = doc_map(conv)
    extra = {}
    for item in records:
        t = (item["subject"], item["predicate"], item["object"])
        for sid in item.get("source_sessions", []):
            doc = sid_to_doc.get(sid)
            if doc:
                extra.setdefault(doc, []).append(t)
    return extra


def load_scope(conv_id: str, scope: str) -> dict:
    p = PATHS["comprehension"] / conv_id / f"{scope}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def ranked_from_res(res, docs):
    by_content = {d["content"]: d for d in docs}
    out = []
    for d, m, sc in zip(res.docs, res.doc_metadata, res.doc_scores):
        base = by_content.get(d)
        if isinstance(m, dict) and m.get("document_id") and base:
            base = {**base, "document_id": m["document_id"], "session_id": m.get("session_id"),
                    "header": f"Session {m.get('session_id')} | {m.get('date_time','')}"}
        if base:
            out.append({**base, "score": float(sc)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--variants", default=None,
                    help="comma separated '{scope}:{variant_suffix}' e.g. cluster_hybrid_0.5_b8192_comp:comp")
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--out-prefix", default="readtime")
    ap.add_argument("--force-build", action="store_true")
    ap.add_argument("--include-relations", default="true")
    ap.add_argument("--no-qa", action="store_true", help="retrieval metrics only (no reader calls)")
    ap.add_argument("--variant-suffix", default="", help="suffix for the variant directory/method name")
    ap.add_argument("--relation-filter", default="", choices=["", "Composed", "Inferred"], help="keep only relations with this derivation")
    args = ap.parse_args()

    from src.adapters.hy4_client import HY4Client
    from src.graph.runtime import EmbeddingProvider, open_hipporag, make_base_config
    from src.graph.variants import build_variant_graph, raw_save_dir

    convs = {c.sample_id: c for c in load_locomo(ROOT / "data" / "raw" / "locomo" / "locomo10.json")}
    split = json.loads((PATHS["splits"] / "split.json").read_text())
    conv_ids = args.convs.split(",") if args.convs else (
        sorted(convs) if args.split == "all" else split[args.split])

    if not args.variants:
        raise SystemExit("--variants required, e.g. cluster_hybrid_0.5_b8192_comp:comp")

    client = HY4Client(tag="reader")
    reader = Reader(client, tag="reader")
    cfg0 = make_base_config(PATHS["raw_graphs"] / "_warmup")
    provider = EmbeddingProvider.get(cfg0)

    from src.graph.hipporag_adapter import HY4CacheOpenAI
    llm = HY4CacheOpenAI.from_experiment_config(cfg0)

    all_rows = []
    build_report = []
    for conv_id in conv_ids:
        conv = convs[conv_id]
        docs = [{"document_id": s.document_id, "session_id": s.session_id,
                 "header": f"Session {s.session_id} | {s.date_time_raw}", "content": s.text}
                for s in conv.sessions]
        qs = conv.qa[: args.max_questions] if args.max_questions else conv.qa

        for spec in args.variants.split(","):
            scope, _, suffix = spec.partition(":")
            suffix = suffix or scope.rsplit("_", 1)[-1]
            data = load_scope(conv_id, scope)
            records = list(data.get("triples", []))
            if args.include_relations == "true":
                rels = data.get("relations", [])
                if args.relation_filter:
                    rels = [r for r in rels if str(r.get("derivation", "")).lower() == args.relation_filter.lower()]
                records += rels
            extra = triples_to_extra(records, conv)
            variant_name = scope + args.variant_suffix
            info = build_variant_graph(conv, variant_name, extra, provider=provider, llm=llm,
                                       force=args.force_build)
            build_report.append({"conv_id": conv_id, **info})
            print(f"\n=== {conv_id} / {variant_name} ===")
            print(f"  build: {json.dumps(info)[:300]}")

            save_dir = raw_save_dir(conv_id) / "variants" / variant_name
            with open_hipporag(str(save_dir), conv_id=conv_id, embedding_provider=provider,
                               llm=llm) as hippo:
                t = time.time()
                res = hippo.retrieve([q.question for q in qs], num_to_retrieve=args.top_k)
                print(f"  retrieval {time.time()-t:.1f}s | graph {hippo.get_graph_info()}")
                ranked_list = [ranked_from_res(r, docs) for r in res]
                ctxs = [select_by_budget(r, args.budget, num_tokens) for r in ranked_list]
                if args.no_qa:
                    ans = [None] * len(qs)
                    print("  QA skipped (--no-qa)")
                else:
                    t = time.time()
                    ans = reader.answer_batch([(q.question, c) for q, c in zip(qs, ctxs)],
                                              max_workers=args.max_workers)
                    print(f"  QA {time.time()-t:.1f}s")
                method = f"{variant_name}"
                rows = [question_record(method, conv, q, r,
                                        reader_answer=(a["answer"] if a else None),
                                        reader_usage=(a["usage"] if a else None),
                                        extra={"context_tokens": sum(num_tokens(d["content"]) for d in c),
                                               "n_context": len(c), "budget": args.budget,
                                               "n_extra_records": len(records),
                                               "variant": variant_name})
                        for q, r, c, a in zip(qs, ranked_list, ctxs, ans)]
                write_jsonl(PATHS["qa"] / f"{args.out_prefix}_{variant_name}_{conv_id}.jsonl", rows)
                all_rows.extend(rows)

    summaries = {}
    for method in sorted({r["method"] for r in all_rows}):
        summaries[method] = summarize([r for r in all_rows if r["method"] == method], method)
    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(args),
           "summaries": summaries, "builds": build_report, "hy4": client.summary()}
    p = PATHS["reports"] / f"{args.out_prefix}_summary.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for m, s in summaries.items():
        print(f"{m:40s} EM={s.get('em')} F1={s.get('f1')} R@5={s.get('recall@5')} R@20={s.get('recall@20')}")
    print("written", p)


if __name__ == "__main__":
    main()
