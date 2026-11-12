"""Prefix-safe dense and HippoRAG retrieval on DetectiveQA.

Reports the four primary retrieval conditions in one run:

* dense, question only
* dense, question + options
* HippoRAG2 prefix-safe, question only
* HippoRAG2 prefix-safe, question + options

and records Any/Recall/All@5/10/20, dense-fallback ratio, top-5 facts, the rank
of the first gold chunk, and per-condition latency.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detectiveqa.data import OUT, gold_chunk_ids, load_prepared, write_jsonl  # noqa: E402
from src.detectiveqa.graph import PrefixGraphRetriever, open_graph  # noqa: E402

KS = (5, 10, 20)


def metrics(ranked: Sequence[str], gold: set, ks=KS) -> Dict[str, float]:
    out = {}
    for k in ks:
        top = set(ranked[:k])
        out[f"any@{k}"] = float(bool(top & gold))
        out[f"recall@{k}"] = len(top & gold) / len(gold) if gold else 0.0
        out[f"all@{k}"] = float(bool(gold) and gold <= top)
    return out


def first_gold_rank(ranked: Sequence[str], gold: set) -> int:
    for i, chunk_id in enumerate(ranked, 1):
        if chunk_id in gold:
            return i
    return 0


def gold_chunk_ranks(ranked: Sequence[str], gold: Sequence[str]) -> Dict[str, int]:
    position = {chunk_id: i for i, chunk_id in enumerate(ranked, 1)}
    return {chunk_id: position.get(chunk_id, 0) for chunk_id in gold}


def queries(question) -> Dict[str, str]:
    options = "\n".join(f"{k}. {v}" for k, v in question.options.items())
    return {"q": question.question,
            "qo": f"{question.question}\n{options}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval_human")
    ap.add_argument("--novels", default="")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--query-mode", default="qo", choices=["q", "qo", "both"],
                    help="q=question only, qo=question+options, both=run both")
    ap.add_argument("--llm-rerank", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    questions, chunks, splits = load_prepared()
    novels = set(args.novels.split(",")) if args.novels else None
    selected = [questions[qid] for qid in splits[args.split]
                if novels is None or questions[qid].novel_id in novels]
    modes = ["q", "qo"] if args.query_mode == "both" else [args.query_mode]
    rows = []
    for novel_id in sorted({q.novel_id for q in selected}, key=int):
        with open_graph(novel_id, args.variant) as hippo:
            retriever = PrefixGraphRetriever(hippo, chunks[novel_id])
            for q in [x for x in selected if x.novel_id == novel_id]:
                gold = set(gold_chunk_ids(q, chunks[novel_id]))
                base = {"qid": q.qid, "novel_id": novel_id, "n_gold_chunks": len(gold),
                        "cutoff": q.cutoff}
                for mode in modes:
                    query = queries(q)[mode]
                    t0 = time.time()
                    dense_ids, _ = retriever._dense(query, q.cutoff)
                    dense = [retriever.passage_chunk[int(i)].chunk_id
                             for i in dense_ids[:args.top_k]]
                    dense_s = time.time() - t0
                    t0 = time.time()
                    graph = retriever.retrieve(query, q.cutoff, args.top_k,
                                               use_llm_rerank=args.llm_rerank)
                    graph_s = time.time() - t0
                    graph_ids = [x["chunk_id"] for x in graph["ranked"]]
                    for name, ranked, elapsed, extra in (
                        ("dense", dense, dense_s, {}),
                        ("graph", graph_ids, graph_s,
                         {"mode": graph["mode"], "facts": graph["facts"]}),
                    ):
                        prefix = f"{mode}_{name}"
                        base[f"{prefix}_ranked"] = ranked
                        base[f"{prefix}_gold_rank"] = first_gold_rank(ranked, gold)
                        base[f"{prefix}_gold_chunk_ranks"] = gold_chunk_ranks(
                            ranked, gold_chunk_ids(q, chunks[novel_id]))
                        base[f"{prefix}_seconds"] = round(elapsed, 4)
                        base.update({f"{prefix}_{k}": v for k, v in metrics(ranked, gold).items()})
                        if extra:
                            base[f"{prefix}_fallback"] = float(extra["mode"] == "dense_fallback")
                            base[f"{prefix}_facts"] = extra["facts"]
                rows.append(base)
                print(q.qid, " ".join(f"{m}:{base[f'{m}_graph_gold_rank']}" for m in modes), flush=True)
    name = args.out or f"retrieval_{args.split}_{args.variant or 'raw'}" \
                       f"{'_rerank' if args.llm_rerank else ''}_{args.query_mode}"
    write_jsonl(OUT / f"{name}.jsonl", rows)
    summary = {"n": len(rows), "modes": modes, "top_k": args.top_k}
    for mode in modes:
        for method in ("dense", "graph"):
            prefix = f"{mode}_{method}"
            keys = [k for k in rows[0] if k.startswith(prefix + "_") and
                    isinstance(rows[0][k], float)] if rows else []
            for k in keys:
                summary[k] = round(mean(r[k] for r in rows), 4)
            ranks = [r[f"{prefix}_gold_rank"] for r in rows]
            summary[f"{prefix}_gold_rank_mrr"] = round(
                mean((1.0 / x) if x else 0.0 for x in ranks), 4) if ranks else 0.0
            summary[f"{prefix}_gold_rank_at_{args.top_k}"] = round(
                sum(1 for x in ranks if x) / len(ranks), 4) if rows else 0.0
            secs = [r[f"{prefix}_seconds"] for r in rows]
            summary[f"{prefix}_seconds_mean"] = round(mean(secs), 4) if secs else 0.0
            summary[f"{prefix}_seconds_median"] = round(median(secs), 4) if secs else 0.0
        if rows and f"{mode}_graph_fallback" in rows[0]:
            summary[f"{mode}_graph_fallback_ratio"] = round(
                mean(r[f"{mode}_graph_fallback"] for r in rows), 4)
    (OUT / f"{name}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
