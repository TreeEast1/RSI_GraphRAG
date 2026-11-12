"""Retrieve prefix-visible comprehension relations as a separate channel.

This tests whether relations can surface useful evidence before injecting them
into HippoRAG.  Relations come only from question-agnostic read-time packets.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detectiveqa.data import OUT, gold_chunk_ids, load_prepared, read_jsonl, write_jsonl  # noqa: E402
from src.detectiveqa.graph import PrefixGraphRetriever, open_graph  # noqa: E402
from src.detectiveqa.round2 import relation_text, retrieval_metrics, rrf  # noqa: E402


def encode(hippo, texts: List[str], instruction: str = "") -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    kwargs = {"norm": True}
    if instruction:
        kwargs["instruction"] = instruction
    return np.asarray(hippo.embedding_model.batch_encode(texts, **kwargs), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readtime", default="readtime_v3.jsonl")
    ap.add_argument("--retrieval", default="round2_retrieval.jsonl")
    ap.add_argument("--top-relations", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--include-dropped", action="store_true")
    ap.add_argument("--out", default="relation_channel")
    args = ap.parse_args()
    questions, chunks, _ = load_prepared()
    relation_rows = read_jsonl(OUT / args.readtime)
    retrieval_rows = {r["qid"]: r for r in read_jsonl(OUT / args.retrieval)}
    by_novel: Dict[str, List[Dict]] = defaultdict(list)
    for row in relation_rows:
        source = row["validated"] if args.include_dropped else row["kept"]
        for relation in source:
            by_novel[row["novel_id"]].append(relation)
    out_rows = []
    for novel_id in sorted({questions[qid].novel_id for qid in retrieval_rows}, key=int):
        relations = by_novel.get(novel_id, [])
        with open_graph(novel_id) as hippo:
            retriever = PrefixGraphRetriever(hippo, chunks[novel_id])
            instruction = "Given a question, retrieve relevant relation facts that help answer the question."
            rel_texts = [relation_text(r) + " Cues: " + "; ".join(r.get("retrieval_cues") or []) for r in relations]
            rel_vecs = encode(hippo, rel_texts)
            for qid, base in retrieval_rows.items():
                q = questions[qid]
                if q.novel_id != novel_id:
                    continue
                gold = set(gold_chunk_ids(q, chunks[novel_id]))
                visible = [i for i, r in enumerate(relations) if int(r["available_at"]) <= q.cutoff]
                qv = encode(hippo, [q.question], instruction)
                scores = np.dot(rel_vecs[visible], qv[0]) if visible else np.asarray([])
                order = [visible[i] for i in np.argsort(scores)[::-1][:args.top_relations]] if visible else []
                top = []
                for rank, idx in enumerate(order, 1):
                    rel = dict(relations[idx])
                    rel["relation_rank"] = rank
                    rel["relation_score"] = float(scores[visible.index(idx)])
                    rel["gold_overlap"] = sorted(set(rel["source_chunk_ids"]) & gold)
                    top.append(rel)
                rel_chunks = list(dict.fromkeys(cid for rel in top for cid in rel["source_chunk_ids"]))
                conditions = {
                    "relation_chunks": rel_chunks[:args.top_k],
                    "rrf_q_dense_relation": rrf([base["conditions"]["q_dense"]["ranked"], rel_chunks])[:args.top_k],
                    "rrf_q_graph_relation": rrf([base["conditions"]["q_graph_f5"]["ranked"], rel_chunks])[:args.top_k],
                    "rrf_q_all_relation": rrf([base["conditions"]["q_dense"]["ranked"],
                                              base["conditions"]["q_graph_f5"]["ranked"], rel_chunks])[:args.top_k],
                }
                row = {"qid": qid, "novel_id": novel_id, "cutoff_bucket": base["cutoff_bucket"],
                       "n_visible_relations": len(visible), "relations": top,
                       "n_top_relations_touching_gold": sum(bool(x["gold_overlap"]) for x in top),
                       "conditions": {k: {"ranked": v, **retrieval_metrics(v, gold)} for k, v in conditions.items()}}
                out_rows.append(row)
                print(qid, "visible", len(visible), "gold-touching", row["n_top_relations_touching_gold"], flush=True)
    write_jsonl(OUT / f"{args.out}.jsonl", out_rows)
    summary = {"n": len(out_rows), "readtime": args.readtime, "include_dropped": args.include_dropped,
               "mean_visible_relations": round(mean(r["n_visible_relations"] for r in out_rows), 2) if out_rows else 0,
               "questions_with_gold_touching_top_relation": sum(r["n_top_relations_touching_gold"] > 0 for r in out_rows),
               "conditions": {}}
    for name in sorted({k for row in out_rows for k in row["conditions"]}):
        vals = [row["conditions"][name] for row in out_rows]
        summary["conditions"][name] = {k: round(mean(v[k] for v in vals), 4)
                                       for k in vals[0] if isinstance(vals[0][k], float)}
    (OUT / f"{args.out}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
