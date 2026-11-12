"""Retrieval with the intermediate state that ``HippoRAG.retrieve`` discards.

The official implementation already computes ``facts_before_rerank`` and
``facts_after_rerank`` inside ``rerank_facts`` but drops the log.  We call the
same official methods in the same order and keep the log, so we can tell whether
a ranking change came from PPR or from a dense fallback.
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import PATHS

_HIPPO_SRC = str(PATHS["third_party"] / "src")
if _HIPPO_SRC not in sys.path:
    sys.path.insert(0, _HIPPO_SRC)

from hipporag.utils.misc_utils import QuerySolution  # noqa: E402


def norm_fact_part(text) -> str:
    """HippoRAG lower-cases facts and drops apostrophes; mirror that for lookup."""
    import re
    s = str(text).lower().replace("'", " ").replace("’", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


def norm_fact(triple) -> tuple:
    return tuple(norm_fact_part(x) for x in triple)


def embedding_top_facts(hippo, query_fact_scores: np.ndarray, k: int):
    """Replacement for the official LLM reranker: keep top-k facts by embedding score."""
    import ast
    n = len(hippo.fact_node_keys)
    if len(query_fact_scores) == 0 or n == 0:
        return [], []
    k = min(k, n)
    candidate_indices = np.argsort(query_fact_scores)[::-1][:k].tolist()
    fact_ids = [hippo.fact_node_keys[i] for i in candidate_indices]
    rows = hippo.fact_embedding_store.get_rows(fact_ids)
    facts = [ast.literal_eval(rows[fid]["content"]) for fid in fact_ids]
    return candidate_indices, facts


def retrieve_with_diag(hippo, queries: Sequence[str], num_to_retrieve: int = 20,
                       fact_trace: Optional[Dict[tuple, str]] = None,
                       use_llm_rerank: bool = True,
                       fact_top_k: Optional[int] = None,
                       link_top_k: Optional[int] = None) -> List[Dict]:
    """Return one diagnostic dict per query.

    ``fact_trace`` maps a (s,p,o) tuple to a human-readable provenance note
    (e.g. the comprehension relation it came from) so that top facts can be
    traced back to injected knowledge.

    ``use_llm_rerank=False`` replaces the official per-query LLM fact filter
    with a pure embedding top-k selection (the ablation requested in the review).
    """
    queries = list(queries)
    if not hippo.ready_to_retrieve:
        hippo.prepare_retrieval_objects()
    hippo.get_query_embeddings(queries)
    k = fact_top_k or hippo.global_config.linking_top_k
    out = []
    for query in queries:
        qfs = hippo.get_fact_scores(query)
        if use_llm_rerank:
            indices, facts, log = hippo.rerank_facts(query, qfs)
            n_before = len(log.get("facts_before_rerank", []))
        else:
            indices, facts = embedding_top_facts(hippo, qfs, k)
            n_before = len(indices)
            log = {"facts_before_rerank": facts, "facts_after_rerank": facts}
        fallback = len(facts) == 0
        if fallback:
            sorted_doc_ids, sorted_doc_scores = hippo.dense_passage_retrieval(query)
        else:
            sorted_doc_ids, sorted_doc_scores = hippo.graph_search_with_fact_entities(
                query=query,
                link_top_k=(link_top_k if link_top_k is not None else hippo.global_config.linking_top_k),
                query_fact_scores=qfs,
                top_k_facts=facts,
                top_k_fact_indices=indices,
                passage_node_weight=hippo.global_config.passage_node_weight,
            )
        result = hippo._build_retrieval_result(query, sorted_doc_ids, sorted_doc_scores,
                                               num_to_retrieve, facts)
        traced = []
        if fact_trace:
            for f in facts:
                note = fact_trace.get(norm_fact(f)) or fact_trace.get(tuple(str(x) for x in f))
                if note:
                    traced.append({"fact": list(f), "origin": note})
        out.append({
            "query": query,
            "fallback_dense": fallback,
            "n_fact_candidates_before_rerank": len(log.get("facts_before_rerank", [])),
            "n_facts_after_rerank": len(facts),
            "top_facts": [list(f) for f in facts],
            "top_facts_traced": traced,
            "docs": list(result.docs),
            "doc_scores": np.asarray(result.scores, dtype=float).tolist(),
            "doc_metadata": [dict(m) for m in result.doc_metadata],
        })
    return out
