"""Round-two retrieval diagnostics with question-only queries by default.

Primary deployable query is the question only.  ``qo`` remains a separate
diagnostic because previous papers expose options only at answer time.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detectiveqa.data import OUT, gold_chunk_ids, load_prepared, write_jsonl  # noqa: E402
from src.detectiveqa.graph import PrefixGraphRetriever, open_graph  # noqa: E402
from src.detectiveqa.round2 import cutoff_bucket, cutoff_ratio, retrieval_metrics, rrf  # noqa: E402

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("the a an of to in and or for with on at by from is was were be are who what when where why how which did does do this that his her their he she it as".split())


def toks(text: str) -> List[str]:
    return [x for x in TOKEN.findall(text.lower()) if x not in STOP and len(x) > 1]


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.2, b: float = 0.75):
        self.docs = [Counter(toks(x)) for x in docs]
        self.lens = [sum(x.values()) for x in self.docs]
        self.avg = mean(self.lens) if self.lens else 1.0
        df = Counter(term for doc in self.docs for term in doc)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def rank(self, query: str, allowed: Sequence[int], k: int) -> List[int]:
        terms = toks(query)
        scores = []
        for idx in allowed:
            doc, length = self.docs[idx], self.lens[idx]
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    score += self.idf.get(term, 0) * tf * (self.k1 + 1) / (
                        tf + self.k1 * (1 - self.b + self.b * length / self.avg))
            scores.append((score, idx))
        return [idx for _, idx in sorted(scores, reverse=True)[:k]]


def query_text(question, mode: str) -> str:
    if mode == "q":
        return question.question
    return question.question + "\n" + "\n".join(f"{k}. {v}" for k, v in question.options.items())


def fact_provenance(retriever: PrefixGraphRetriever, facts: Sequence[Sequence[str]], cutoff: int) -> List[Dict]:
    out = []
    passage_index = {k: i for i, k in enumerate(retriever.hippo.passage_node_keys)}
    for fact in facts:
        docs = retriever.hippo.proc_triples_to_docs.get(str(tuple(fact)), set())
        chunk_ids = []
        for doc in docs:
            idx = passage_index.get(doc)
            if idx is None:
                continue
            chunk = retriever.passage_chunk[idx]
            if chunk is not None and chunk.available_at <= cutoff:
                chunk_ids.append(chunk.chunk_id)
        out.append({"fact": list(fact), "visible_source_chunk_ids": sorted(chunk_ids)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval_human")
    ap.add_argument("--novels", default="27,79,117,118,121,124")
    ap.add_argument("--modes", default="q,qo")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--fact-top-k", default="5,10,20")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test only: first N questions")
    ap.add_argument("--out", default="round2_retrieval")
    args = ap.parse_args()
    questions, chunks, splits = load_prepared()
    novels = set(args.novels.split(",")) if args.novels else None
    selected = [questions[qid] for qid in splits[args.split] if novels is None or questions[qid].novel_id in novels]
    if args.limit:
        selected = selected[:args.limit]
    rows = []
    fact_ks = [int(x) for x in args.fact_top_k.split(",") if x]
    for novel_id in sorted({q.novel_id for q in selected}, key=int):
        with open_graph(novel_id, args.variant) as hippo:
            retriever = PrefixGraphRetriever(hippo, chunks[novel_id])
            for q in [x for x in selected if x.novel_id == novel_id]:
                gold = gold_chunk_ids(q, chunks[novel_id])
                ratio = cutoff_ratio(q, chunks[novel_id])
                row = {"qid": q.qid, "novel_id": novel_id, "question": q.question, "options": q.options,
                       "gold_answer": q.answer, "cutoff": q.cutoff, "cutoff_ratio": ratio,
                       "cutoff_bucket": cutoff_bucket(ratio), "gold_chunk_ids": gold,
                       "n_visible_chunks": int(sum(retriever.passage_available <= q.cutoff)),
                       "conditions": {}}
                visible = [int(x) for x in (retriever.passage_available <= q.cutoff).nonzero()[0]]
                for mode in args.modes.split(","):
                    query = query_text(q, mode)
                    t0 = time.time()
                    dense_idx, _ = retriever._dense(query, q.cutoff)
                    dense = [retriever.passage_chunk[int(i)].chunk_id for i in dense_idx[:args.top_k]]
                    dense_s = time.time() - t0
                    # Prefix-safe BM25: corpus statistics and scoring use only visible chunks.
                    visible_docs = [retriever.passage_chunk[i].text if retriever.passage_chunk[i] else ""
                                    for i in visible]
                    bm25_visible = BM25(visible_docs)
                    bm25_rel = bm25_visible.rank(query, range(len(visible_docs)), args.top_k)
                    bm25_ids = [retriever.passage_chunk[visible[j]].chunk_id for j in bm25_rel]
                    conditions = {f"{mode}_dense": {"ranked": dense, "seconds": dense_s},
                                  f"{mode}_bm25": {"ranked": bm25_ids, "seconds": 0.0}}
                    for fact_k in fact_ks:
                        for use_synonym in (True, False):
                            name = f"{mode}_graph_f{fact_k}" + ("" if use_synonym else "_nosyn")
                            t0 = time.time()
                            result = retriever.retrieve(query, q.cutoff, args.top_k, fact_top_k=fact_k,
                                                        use_synonym=use_synonym)
                            conditions[name] = {
                                "ranked": [x["chunk_id"] for x in result["ranked"]],
                                "seconds": time.time() - t0, "mode": result["mode"],
                                "facts": result["facts"],
                                "fact_provenance": fact_provenance(retriever, result["facts"], q.cutoff),
                            }
                    graph = conditions[f"{mode}_graph_f5"]["ranked"]
                    conditions[f"{mode}_rrf_dense_graph"] = {"ranked": rrf([dense, graph])[:args.top_k]}
                    conditions[f"{mode}_rrf_dense_bm25"] = {"ranked": rrf([dense, bm25_ids])[:args.top_k]}
                    conditions[f"{mode}_rrf_all"] = {"ranked": rrf([dense, graph, bm25_ids])[:args.top_k]}
                    for name, value in conditions.items():
                        value.update(retrieval_metrics(value["ranked"], gold))
                        value["gold_chunk_ranks"] = {g: value["ranked"].index(g) + 1 if g in value["ranked"] else 0
                                                     for g in gold}
                        row["conditions"][name] = value
                rows.append(row)
                print(q.qid, row["cutoff_bucket"], "q dense/graph",
                      row["conditions"].get("q_dense", {}).get("first_gold_rank"),
                      row["conditions"].get("q_graph_f5", {}).get("first_gold_rank"), flush=True)
    write_jsonl(OUT / f"{args.out}.jsonl", rows)
    summary = {"n": len(rows), "modes": args.modes.split(","), "conditions": {}}
    names = sorted({name for row in rows for name in row["conditions"]})
    for name in names:
        vals = [row["conditions"][name] for row in rows]
        summary["conditions"][name] = {k: round(mean(v[k] for v in vals), 4)
                                       for k in vals[0] if isinstance(vals[0][k], float)}
    by_bucket = defaultdict(list)
    for row in rows:
        by_bucket[row["cutoff_bucket"]].append(row)
    summary["by_bucket"] = {}
    for bucket, sub in by_bucket.items():
        summary["by_bucket"][bucket] = {"n": len(sub)}
        for name in ("q_dense", "q_graph_f5", "q_rrf_dense_graph", "qo_dense", "qo_graph_f5"):
            if name in sub[0]["conditions"]:
                summary["by_bucket"][bucket][name] = {
                    k: round(mean(r["conditions"][name][k] for r in sub), 4)
                    for k in ("recall@10", "all@20", "mrr")}
    (OUT / f"{args.out}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:12000])


if __name__ == "__main__":
    main()
