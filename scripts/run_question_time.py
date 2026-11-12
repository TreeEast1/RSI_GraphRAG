#!/usr/bin/env python
"""Question-time evolution (experiment E): Q0 raw, Q1 QDM, Q2 +retrieval skill,
Q3 +answer skill. Works on top of any graph (G_raw by default)."""
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
from src.question_skill.pipeline import (answer_skill, decompose, judge_missing,
                                          rrf_merge)  # noqa: E402
from src.retrieval.reader import Reader  # noqa: E402

sys.path.insert(0, str(PATHS["third_party"] / "src"))


def docs_of(conv):
    return [{"document_id": s.document_id, "session_id": s.session_id,
             "header": f"Session {s.session_id} | {s.date_time_raw}", "content": s.text}
            for s in conv.sessions]


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
    ap.add_argument("--graph", default="raw", help="raw | variants/<name>")
    ap.add_argument("--strategies", default="q0,q1,q2,q3")
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--out-prefix", default="qtime")
    ap.add_argument("--skill-max-new", type=int, default=2)
    args = ap.parse_args()

    from src.adapters.hy4_client import HY4Client
    from src.graph.hipporag_adapter import HY4CacheOpenAI
    from src.graph.runtime import EmbeddingProvider, open_hipporag, make_base_config
    from src.graph.variants import raw_save_dir

    convs = {c.sample_id: c for c in load_locomo(ROOT / "data" / "raw" / "locomo" / "locomo10.json")}
    split = json.loads((PATHS["splits"] / "split.json").read_text())
    conv_ids = args.convs.split(",") if args.convs else (
        sorted(convs) if args.split == "all" else split[args.split])

    client = HY4Client(tag="question_time")
    reader = Reader(client, tag="reader")
    cfg0 = make_base_config(PATHS["raw_graphs"] / "_warmup")
    provider = EmbeddingProvider.get(cfg0)
    llm = HY4CacheOpenAI.from_experiment_config(cfg0)

    strategies = args.strategies.split(",")
    all_rows = []
    stats = {"qdm_decomposed": 0, "skill_calls": 0, "skill_added_queries": 0,
             "answer_revised": 0, "n_questions": 0}

    for conv_id in conv_ids:
        conv = convs[conv_id]
        docs = docs_of(conv)
        qs = conv.qa[: args.max_questions] if args.max_questions else conv.qa
        save_dir = raw_save_dir(conv_id) if args.graph == "raw" else raw_save_dir(conv_id) / "variants" / args.graph
        print(f"\n=== {conv_id} | graph={args.graph} | {len(qs)} QA ===")

        with open_hipporag(str(save_dir), conv_id=conv_id, embedding_provider=provider, llm=llm) as hippo:
            def retrieve(q: str):
                r = hippo.retrieve([q], num_to_retrieve=args.top_k)[0]
                return ranked_from_res(r, docs)

            for strategy in strategies:
                t0 = time.time()
                rows = []
                for i, qa in enumerate(qs):
                    stats["n_questions"] += 1
                    extra = {}
                    # ---- Q0 / Q1
                    if strategy == "q0":
                        ranked = retrieve(qa.question)
                        extra["queries"] = [qa.question]
                    else:
                        dec = decompose(client, qa.question, tag=f"{conv_id}_qdm")
                        extra["qdm"] = dec
                        if dec["needs_decomposition"]:
                            stats["qdm_decomposed"] += 1
                        subqs = [qa.question] + dec["sub_questions"]
                        lists = [retrieve(q) for q in subqs]
                        ranked = rrf_merge(lists, top=args.top_k)
                        extra["queries"] = subqs

                        # ---- Q2 / Q3
                        if strategy in ("q2", "q3"):
                            stats["skill_calls"] += 1
                            miss = judge_missing(client, qa.question, dec["sub_questions"],
                                                 ranked, max_new=args.skill_max_new,
                                                 tag=f"{conv_id}_ret_skill")
                            extra["retrieval_skill"] = miss
                            if miss["missing"]:
                                stats["skill_added_queries"] += len(miss["missing"])
                                lists = lists + [retrieve(m["query"]) for m in miss["missing"]]
                                ranked = rrf_merge(lists, top=args.top_k)

                    ctx = select_by_budget(ranked, args.budget, num_tokens)
                    ans = reader.answer(qa.question, ctx)
                    pred = ans["answer"]
                    if strategy == "q3":
                        sk = answer_skill(client, qa.question, ctx, pred,
                                          tag=f"{conv_id}_ans_skill")
                        extra["answer_skill"] = sk
                        if sk["revised"]:
                            stats["answer_revised"] += 1
                        pred = sk["answer"]
                    rows.append(question_record(f"{args.graph}_{strategy}", conv, qa, ranked,
                                                reader_answer=pred, reader_usage=ans["usage"],
                                                extra={**extra,
                                                       "context_tokens": sum(num_tokens(d["content"]) for d in ctx),
                                                       "n_context": len(ctx), "budget": args.budget}))
                    if (i + 1) % 25 == 0:
                        print(f"  {strategy} {i+1}/{len(qs)} {time.time()-t0:.0f}s")
                write_jsonl(PATHS["qa"] / f"{args.out_prefix}_{args.graph}_{strategy}_{conv_id}.jsonl", rows)
                all_rows.extend(rows)
                print(f"  {strategy} done {time.time()-t0:.0f}s")

    summaries = {}
    for m in sorted({r["method"] for r in all_rows}):
        summaries[m] = summarize([r for r in all_rows if r["method"] == m], m)
    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(args),
           "summaries": summaries, "stats": stats, "hy4": client.summary()}
    p = PATHS["reports"] / f"{args.out_prefix}_summary.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for m, s in summaries.items():
        print(f"{m:28s} EM={s.get('em')} F1={s.get('f1')} R@5={s.get('recall@5')} R@20={s.get('recall@20')}")
    print("stats", stats)
    print("written", p)


if __name__ == "__main__":
    main()
