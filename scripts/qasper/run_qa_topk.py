#!/usr/bin/env python
"""Top-3 / Top-5 reader sensitivity on the already-evaluated questions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.adapters.hy4_client import HY4Client, HY4Config, load_env  # noqa: E402
from src.qasper.evalutil import OUT, load_papers, read_jsonl  # noqa: E402
from src.qasper.qa import gold_refs, score_one, summarize_predictions  # noqa: E402
from src.qasper.reader import QasperReader, batch_answer, build_context  # noqa: E402

load_env()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="sample100")
    ap.add_argument("--method", default="m5_units_graph")
    ap.add_argument("--ks", default="3,5")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    papers = load_papers(args.scope)
    rows = read_jsonl(OUT / "retrieval" / f"comparison_{args.scope}.jsonl")
    by_q = {}
    for r in rows:
        if r["method"] == args.method:
            by_q[r["qid"]] = r["ranked"]
    papers = {pid: p for pid, p in papers.items() if any(q.qid in by_q for q in p.qas)}
    client = HY4Client(HY4Config.from_env(max_concurrency=8),
                       cache_path=OUT / "cache" / "hy4_qa_topk.sqlite",
                       log_path=OUT / "logs" / "hy4_calls_qa_topk.jsonl", tag="qa_topk")
    reader = QasperReader(client, tag="qasper_reader_topk")
    out = []
    for pid, paper in papers.items():
        paras = {p.passage_id: p.to_dict() for p in paper.paragraphs}
        items = []
        for q in paper.qas:
            if q.qid not in by_q:
                continue
            for k in ks:
                ctx = build_context(paras, by_q[q.qid][:k], budget=100000)
                items.append((q, k, ctx))
        answers = batch_answer(reader, [(q.question, c["context"]) for q, _, c in items],
                               max_workers=args.workers)
        for (q, k, ctx), ans in zip(items, answers):
            pred = {"answer": ans.get("answer", ""),
                    "evidence": [paras[x]["text"] for x in ctx["passages"] if x in paras]}
            sc = score_one(pred, gold_refs(q))
            out.append({"qid": q.qid, "paper_id": pid, "k": k, **sc,
                        "answer": pred["answer"], "context_passages": ctx["passages"]})
    path = OUT / "qa" / f"topk_sensitivity_{args.scope}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {}
    for k in ks:
        summary[str(k)] = summarize_predictions([r for r in out if r["k"] == k])
    (OUT / "qa" / f"topk_sensitivity_{args.scope}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
