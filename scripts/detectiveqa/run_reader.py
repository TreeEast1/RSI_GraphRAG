"""Run the unified DetectiveQA multi-choice Reader across context conditions."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.adapters.hy4_client import HY4Client  # noqa: E402
from src.detectiveqa.data import OUT, gold_chunk_ids, load_prepared, write_jsonl  # noqa: E402
from src.detectiveqa.graph import PrefixGraphRetriever, open_graph  # noqa: E402
from src.detectiveqa.reader import (build_context, make_prompt, parse_answer,  # noqa: E402
                                    score_reader, system_prompt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval_human")
    ap.add_argument("--novels", default="27,79,117,118,121,124")
    ap.add_argument("--conditions", default="options_only,dense,graph,oracle,long_prefix")
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--tag", default="reader_pilot")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    conditions = [x for x in args.conditions.split(",") if x]
    questions, chunks, splits = load_prepared()
    novels = set(args.novels.split(",")) if args.novels else None
    selected = [questions[qid] for qid in splits[args.split]
                if novels is None or questions[qid].novel_id in novels]
    client = HY4Client(tag=args.tag)
    name = args.out or f"reader_{args.split}"
    partial_path = OUT / f"{name}.partial.jsonl"
    partial = open(partial_path, "w", encoding="utf-8")
    rows: List[Dict] = []
    for novel_id in sorted({q.novel_id for q in selected}, key=int):
        with open_graph(novel_id) as hippo:
            retriever = PrefixGraphRetriever(hippo, chunks[novel_id])
            for q in [x for x in selected if x.novel_id == novel_id]:
                query = f"{q.question}\n" + "\n".join(f"{k}. {v}" for k, v in q.options.items())
                dense_ids, _ = retriever._dense(query, q.cutoff)
                dense_ranked = [retriever.passage_chunk[int(i)].chunk_id for i in dense_ids[:args.top_k]]
                graph = retriever.retrieve(query, q.cutoff, args.top_k)
                graph_ranked = [x["chunk_id"] for x in graph["ranked"]]
                gold = gold_chunk_ids(q, chunks[novel_id])
                for condition in conditions:
                    built = build_context(condition, q, chunks[novel_id],
                                          dense_ranked if condition == "dense"
                                          else graph_ranked if condition == "graph" else [],
                                          args.token_budget)
                    prompt = make_prompt(q, built["context"])
                    t0 = time.time()
                    error = ""
                    parsed: Dict = {}
                    usage = {}
                    try:
                        out = client.chat([{"role": "system", "content": system_prompt(condition)},
                                           {"role": "user", "content": prompt}],
                                          json_mode=True, max_tokens=1024, tag=args.tag)
                        usage = out.get("usage") or {}
                        from src.adapters.hy4_client import parse_json_loose
                        parsed = parse_json_loose(out["content"]) or {}
                        if not parsed:
                            error = "unparseable_json"
                    except Exception as exc:  # noqa: BLE001
                        error = f"{type(exc).__name__}: {exc}"[:300]
                    pred = parse_answer(parsed, q)
                    scores = score_reader(pred, q, gold, built["chunk_ids"])
                    rows.append({
                        "qid": q.qid, "novel_id": novel_id, "condition": condition,
                        "cutoff": q.cutoff, "question": q.question, "options": q.options,
                        "gold_answer": q.answer, "gold_chunk_ids": gold,
                        "gold_rank_dense": next((i for i, c in enumerate(dense_ranked, 1) if c in set(gold)), 0),
                        "gold_rank_graph": next((i for i, c in enumerate(graph_ranked, 1) if c in set(gold)), 0),
                        "context_chunk_ids": built["chunk_ids"], "context_meta": built["meta"],
                        "context_tokens_est": built["meta"].get("context_tokens_est", 0),
                        "prediction": pred, "scores": scores,
                        "usage": usage, "seconds": round(time.time() - t0, 3),
                        "error": error,
                    })
                    partial.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
                    partial.flush()
                    print(q.qid, condition, pred["answer"], scores["correct"], error, flush=True)
    partial.close()
    write_jsonl(OUT / f"{name}.jsonl", rows)
    summary: Dict = {"n": len(rows), "conditions": conditions, "token_budget": args.token_budget}
    for condition in conditions:
        sub = [r for r in rows if r["condition"] == condition]
        if not sub:
            continue
        summary[condition] = {
            "n": len(sub),
            "accuracy": round(mean(r["scores"]["correct"] for r in sub), 4),
            "invalid_rate": round(mean(r["scores"]["invalid"] for r in sub), 4),
            "insufficient_rate": round(mean(r["scores"]["insufficient"] for r in sub), 4),
            "evidence_any": round(mean(r["scores"]["evidence_any"] for r in sub), 4),
            "evidence_recall": round(mean(r["scores"]["evidence_recall"] for r in sub), 4),
            "citation_valid_ratio": round(mean(r["scores"]["citation_valid_ratio"] for r in sub), 4),
            "mean_prompt_tokens": round(mean(r["usage"].get("prompt_tokens", 0) for r in sub), 1),
            "mean_completion_tokens": round(mean(r["usage"].get("completion_tokens", 0) for r in sub), 1),
            "total_tokens": sum(r["usage"].get("total_tokens", 0) for r in sub),
            "mean_seconds": round(mean(r["seconds"] for r in sub), 2),
            "n_errors": sum(bool(r["error"]) for r in sub),
            "mean_context_tokens_est": round(mean(r["context_tokens_est"] for r in sub), 1),
        }
    summary["client_stats"] = client.summary()
    (OUT / f"{name}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
