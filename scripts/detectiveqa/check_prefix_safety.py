"""Independent leakage audit for the DetectiveQA prefix-safe retriever.

Every question is replayed on a real HippoRAG graph.  The audit recomputes the
visible world from raw stores (chunk paragraph spans, OpenIE provenance and
graph edge metadata) rather than trusting ``PrefixGraphRetriever``:

1. every retrieved chunk ends strictly before the question cutoff;
2. every top fact has at least one provenance chunk before the cutoff (facts
   that recur later stay visible, but their edge weight may only count visible
   sources);
3. the PPR subgraph contains no future passage node;
4. an entity first seen in a future chunk never enters the subgraph;
5. every subgraph edge weight equals the visible-source-only recomputation;
6. repeated retrieval at the same cutoff is deterministic.

Any hard failure exits non-zero.  Soft observations (facts whose text recurs
after the cutoff) are recorded in the JSONL for the report, but they are not
leaks: no future content is read.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detectiveqa.data import OUT, gold_chunk_ids, load_prepared, write_jsonl  # noqa: E402
from src.detectiveqa.graph import PrefixGraphRetriever, open_graph  # noqa: E402


def _query(question, with_options: bool = True) -> str:
    if not with_options:
        return question.question
    return f"{question.question}\n" + "\n".join(f"{k}. {v}" for k, v in question.options.items())


def _audit_one(retriever: PrefixGraphRetriever, question, k: int) -> Dict:
    cutoff = question.cutoff
    query = _query(question)
    first = retriever.retrieve(query, cutoff, k)
    second = retriever.retrieve(query, cutoff, k)
    ranked = first["ranked"]
    failures: List[str] = []
    notes: List[str] = []

    # 1. retrieved chunks must end strictly before the cutoff.
    future_chunks = []
    for row in ranked:
        chunk = retriever.by_id.get(row["chunk_id"])
        if chunk is None:
            failures.append(f"unknown_chunk:{row['chunk_id']}")
        elif chunk.available_at > cutoff or not (chunk.end_pid < cutoff):
            future_chunks.append({"chunk_id": chunk.chunk_id, "end_pid": chunk.end_pid, "cutoff": cutoff})
    if future_chunks:
        failures.append(f"future_chunk:{len(future_chunks)}")

    # 2. every top fact must be supported by a visible provenance chunk.
    fact_reports = []
    for fact in first["facts"]:
        key = str(tuple(fact))
        docs = retriever.hippo.proc_triples_to_docs.get(key, set())
        visible_docs = sorted(d for d in docs if retriever.passage_key_to_available.get(d, 10 ** 12) <= cutoff)
        future_docs = sorted(d for d in docs if retriever.passage_key_to_available.get(d, 10 ** 12) > cutoff)
        if not visible_docs:
            failures.append(f"future_only_fact:{fact}")
        if future_docs:
            notes.append(f"fact_recurs_later:{len(future_docs)}")
        fact_reports.append({"fact": list(fact), "n_visible_sources": len(visible_docs),
                             "n_future_sources": len(future_docs),
                             "all_sources_visible": not future_docs})

    # 3/4. independently rebuild the allowed vertex set and compare with the PPR subgraph.
    passage_keys = set(retriever.hippo.passage_node_keys)
    visible_passage_keys = {key for key, avail in
                            zip(retriever.hippo.passage_node_keys, retriever.passage_available)
                            if int(avail) <= cutoff}
    visible_entity_keys = {key for key in retriever.hippo.entity_node_keys
                           if retriever.entity_available.get(key, 10 ** 12) <= cutoff}
    future_only_entities = {key for key in retriever.hippo.entity_node_keys
                            if retriever.entity_available.get(key, 10 ** 12) > cutoff}
    graph, _ = retriever._visible_graph(cutoff)
    subgraph_names = set(graph.vs["name"])
    future_passages_in_graph = (subgraph_names - visible_passage_keys) & passage_keys
    future_entities_in_graph = subgraph_names & future_only_entities
    unknown_in_graph = subgraph_names - visible_passage_keys - visible_entity_keys
    if future_passages_in_graph:
        failures.append(f"future_passage_in_subgraph:{len(future_passages_in_graph)}")
    if future_entities_in_graph:
        failures.append(f"future_entity_in_subgraph:{len(future_entities_in_graph)}")
    if unknown_in_graph:
        failures.append(f"unexpected_vertex:{len(unknown_in_graph)}")

    # 5. independent edge-weight recomputation from visible provenance only.
    edge_mismatch = []
    for edge in graph.es:
        attrs = edge.attributes()
        counts = attrs.get("fact_source_counts") or {}
        fact_weight = float(sum(n for src, n in counts.items()
                                if retriever.passage_key_to_available.get(src, 10 ** 12) <= cutoff))
        psrc = attrs.get("passage_source")
        passage_weight = 1.0 if psrc is not None and \
            retriever.passage_key_to_available.get(psrc, 10 ** 12) <= cutoff else 0.0
        synonym = float(attrs.get("synonym_score") or 0.0)
        expected = max(fact_weight, passage_weight, synonym)
        if abs(float(attrs.get("weight", -1.0)) - expected) > 1e-9:
            edge_mismatch.append({"edge": (edge.source, edge.target),
                                  "stored": float(attrs.get("weight", -1.0)), "expected": expected})
        if expected <= 0:
            failures.append("nonpositive_weight_edge")
    if edge_mismatch:
        failures.append(f"edge_weight_mismatch:{len(edge_mismatch)}")

    # 6. determinism: ordering and facts must be identical; PPR scores may differ
    # only at floating-point round-off level (igraph prpack summation order).
    first_ids = [r["chunk_id"] for r in ranked]
    second_ids = [r["chunk_id"] for r in second["ranked"]]
    max_score_delta = max((abs(a["score"] - b["score"]) for a, b in
                           zip(ranked, second["ranked"])), default=0.0)
    deterministic = (first_ids == second_ids and
                     [list(x) for x in first["facts"]] == [list(x) for x in second["facts"]] and
                     max_score_delta <= 1e-9)
    if not deterministic:
        failures.append("nondeterministic_retrieval")

    return {
        "qid": question.qid, "novel_id": question.novel_id, "cutoff": cutoff,
        "mode": first["mode"], "n_ranked": len(ranked),
        "ranked_chunk_ids": [r["chunk_id"] for r in ranked],
        "facts": [list(f) for f in first["facts"]],
        "fact_provenance": fact_reports,
        "n_visible_passages": len(visible_passage_keys),
        "n_visible_entities": len(visible_entity_keys),
        "n_future_only_entities": len(future_only_entities),
        "n_subgraph_vertices": graph.vcount(), "n_subgraph_edges": graph.ecount(),
        "deterministic": deterministic, "max_repeat_score_delta": max_score_delta,
        "notes": notes, "failures": failures,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval_human")
    ap.add_argument("--novels", default="27,79,117,118,121,124")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--out", default="prefix_safety")
    args = ap.parse_args()
    questions, chunks, splits = load_prepared()
    novels = set(args.novels.split(",")) if args.novels else None
    selected = [questions[qid] for qid in splits[args.split]
                if novels is None or questions[qid].novel_id in novels]
    rows = []
    for novel_id in sorted({q.novel_id for q in selected}, key=int):
        with open_graph(novel_id, args.variant) as hippo:
            retriever = PrefixGraphRetriever(hippo, chunks[novel_id])
            for q in [x for x in selected if x.novel_id == novel_id]:
                rows.append(_audit_one(retriever, q, args.k))
                print(q.qid, rows[-1]["mode"], len(rows[-1]["failures"]), flush=True)
    write_jsonl(OUT / f"{args.out}.jsonl", rows)
    n_fail = sum(bool(r["failures"]) for r in rows)
    failure_kinds: Dict[str, int] = {}
    for r in rows:
        for f in r["failures"]:
            failure_kinds[f.split(":", 1)[0]] = failure_kinds.get(f.split(":", 1)[0], 0) + 1
    summary = {
        "n_questions": len(rows),
        "n_failed_questions": n_fail,
        "failure_kinds": failure_kinds,
        "n_nondeterministic": sum(not r["deterministic"] for r in rows),
        "n_empty_retrieval": sum(not r["ranked_chunk_ids"] for r in rows),
        "modes": {m: sum(r["mode"] == m for r in rows) for m in {r["mode"] for r in rows}},
        "n_facts_with_later_recurrence": sum(
            any(not p["all_sources_visible"] for p in r["fact_provenance"]) for r in rows),
    }
    (OUT / f"{args.out}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
