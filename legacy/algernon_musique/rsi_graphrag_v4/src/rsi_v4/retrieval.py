"""HippoRAG2-style PPR retrieval over G_raw plus an optional comprehension layer.

Fact reranking is embedding top-k. The bundled Llama-3.3 DSPy filter is not used.
Comprehension chunks are document nodes, and comprehension edges are graph edges,
so a new node can change Personalized PageRank rather than only the text index.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import igraph as ig
import numpy as np

HIPPO_SRC = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")
if str(HIPPO_SRC) not in sys.path:
    sys.path.insert(0, str(HIPPO_SRC))

from hipporag.prompts.linking import get_query_instruction  # noqa: E402
from hipporag.utils.misc_utils import min_max_normalize  # noqa: E402

from .comprehension_store import ComprehensionStore


class GraphRetriever:
    def __init__(self, raw_graph: dict, store: ComprehensionStore, embedder, config: dict) -> None:
        self.raw_graph = raw_graph
        self.store = store
        self.embedder = embedder
        self.config = config
        self._key = None
        self.prepared = None

    def prepare(self) -> dict:
        key = (self.raw_graph["graph_hash"], self.store.version, id(self.embedder))
        if self.prepared is not None and self._key == key:
            return self.prepared
        entities = self.raw_graph["entities"]
        entity_ids = sorted(entities)
        docs = []
        for passage in self.raw_graph["passages"].values():
            docs.append(
                {
                    "doc_id": passage["passage_id"],
                    "kind": "passage",
                    "content": passage["content"],
                    "title": passage.get("title", ""),
                    "text": passage.get("text", ""),
                }
            )
        comp_facts = []
        active = {node["comprehension_id"]: node for node in self.store.active_nodes()}
        for node in active.values():
            docs.append(
                {
                    "doc_id": node["comprehension_id"],
                    "kind": "comprehension",
                    "content": node["chunk_text"],
                    "title": node["comprehension_id"],
                    "text": node["chunk_text"],
                    "comprehension_id": node["comprehension_id"],
                }
            )
        facts = []
        for triple in self.raw_graph["triples"].values():
            text = str((triple["s"], triple["p"], triple["o"]))
            facts.append(
                {
                    "text": text,
                    "s_id": triple["s_id"],
                    "o_id": triple["o_id"],
                    "s": triple["s"],
                    "o": triple["o"],
                    "kind": "raw",
                }
            )
        for edge in self.store.active_edges():
            if edge["src"] not in active:
                continue
            if edge["dst"] not in entities:
                continue
            comp = active[edge["src"]]
            obj = comp["semantic_content"][:240]
            text = str((entities[edge["dst"]]["content"], edge["relation"], obj))
            facts.append(
                {
                    "text": text,
                    "s_id": edge["dst"],
                    "o_id": edge["src"],
                    "s": entities[edge["dst"]]["content"],
                    "o": obj,
                    "kind": "comprehension",
                    "comprehension_id": edge["src"],
                }
            )
            comp_facts.append(facts[-1])

        doc_ids = [doc["doc_id"] for doc in docs]
        node_ids = entity_ids + doc_ids
        index = {node_id: position for position, node_id in enumerate(node_ids)}
        graph = ig.Graph(n=len(node_ids), directed=False)
        graph.vs["name"] = node_ids
        edge_weights = {}

        def add_edge(left: str, right: str, weight: float) -> None:
            if left not in index or right not in index or left == right:
                return
            key = tuple(sorted((left, right)))
            edge_weights[key] = max(edge_weights.get(key, 0.0), float(weight))

        for triple in self.raw_graph["triples"].values():
            add_edge(triple["s_id"], triple["o_id"], 1.0)
        for edge in self.raw_graph["context_edges"]:
            add_edge(edge["passage_id"], edge["entity_id"], 1.0)
        for edge in self.raw_graph.get("synonymy_edges") or []:
            add_edge(edge["a"], edge["b"], float(edge["score"]))
        for edge in self.store.active_edges():
            if edge["src"] in active and edge["dst"] in entities:
                add_edge(edge["src"], edge["dst"], 1.0)

        if edge_weights:
            endpoints = [(index[left], index[right]) for left, right in edge_weights]
            graph.add_edges(endpoints)
            graph.es["weight"] = [edge_weights[key] for key in edge_weights]

        entity_degree = {entity_id: 0 for entity_id in entity_ids}
        for edge in self.raw_graph["context_edges"]:
            if edge["entity_id"] in entity_degree:
                entity_degree[edge["entity_id"]] += 1

        fact_matrix = (
            self._normalize(self.embedder.encode([fact["text"] for fact in facts], instruction=""))
            if facts
            else np.zeros((0, 1), dtype=np.float32)
        )
        doc_matrix = self._normalize(self.embedder.encode([doc["content"] for doc in docs], instruction=""))
        self.prepared = {
            "graph": graph,
            "node_ids": node_ids,
            "index": index,
            "entity_ids": entity_ids,
            "docs": docs,
            "doc_ids": doc_ids,
            "facts": facts,
            "fact_matrix": fact_matrix,
            "doc_matrix": doc_matrix,
            "entity_degree": entity_degree,
            "entities": entities,
        }
        self._key = key
        return self.prepared

    def retrieve(self, queries: Sequence[str], top_k: int | None = None) -> dict:
        prepared = self.prepare()
        if top_k is None:
            top_k = int(self.config.get("retrieval_top_k") or len(prepared["docs"]))
        pooled = None
        traces = []
        for query in queries:
            scores, trace = self._retrieve_one(query, prepared)
            traces.append(trace)
            pooled = scores if pooled is None else np.maximum(pooled, scores)
        order = np.argsort(-pooled)
        ranked_docs = []
        for position in order[: max(top_k, len(prepared["docs"]))]:
            doc = prepared["docs"][int(position)]
            ranked_docs.append({**doc, "score": float(pooled[int(position)])})
        return {
            "queries": list(queries),
            "docs": ranked_docs,
            "traces": traces,
            "comprehension_ids_in_top10": [
                doc["doc_id"] for doc in ranked_docs[:10] if doc["kind"] == "comprehension"
            ],
        }

    def _retrieve_one(self, query: str, prepared: dict):
        linking_top_k = int(self.config.get("linking_top_k", 5))
        if len(prepared["facts"]) == 0:
            doc_scores = self._dpr(query, prepared)
            return doc_scores, {"query": query, "facts": [], "mode": "dpr_only"}
        query_fact = self._normalize(
            self.embedder.encode([query], instruction=get_query_instruction("query_to_fact"))
        )[0]
        fact_scores = prepared["fact_matrix"] @ query_fact
        fact_scores = np.squeeze(fact_scores)
        if fact_scores.ndim == 0:
            fact_scores = np.array([float(fact_scores)])
        fact_scores = min_max_normalize(fact_scores)
        order = np.argsort(-fact_scores)[:linking_top_k]
        chosen = [prepared["facts"][int(index)] for index in order]
        reset = np.zeros(len(prepared["node_ids"]), dtype=np.float64)
        phrase_scores: Dict[str, List[float]] = {}
        for rank, fact in enumerate(chosen):
            score = float(fact_scores[int(order[rank])])
            for endpoint in (fact["s_id"], fact["o_id"]):
                if endpoint in prepared["index"] and endpoint in prepared["entity_degree"]:
                    degree = max(1, prepared["entity_degree"][endpoint])
                    position = prepared["index"][endpoint]
                    reset[position] = max(reset[position], score / degree)
                    phrase_scores.setdefault(fact["s"] if endpoint == fact["s_id"] else fact["o"], []).append(score)
            if fact["kind"] == "comprehension":
                position = prepared["index"].get(fact["comprehension_id"])
                if position is not None:
                    reset[position] = max(reset[position], score)
        if phrase_scores:
            averaged = {phrase: float(np.mean(values)) for phrase, values in phrase_scores.items()}
            keep = set(sorted(averaged, key=averaged.get, reverse=True)[:linking_top_k])
            for entity_id in prepared["entity_ids"]:
                content = prepared["entities"][entity_id]["content"]
                if content not in keep:
                    reset[prepared["index"][entity_id]] = 0.0
        dpr_scores = self._dpr(query, prepared)
        weight = float(self.config.get("passage_node_weight", 0.05))
        for doc_index, doc in enumerate(prepared["docs"]):
            reset[prepared["index"][doc["doc_id"]]] += float(dpr_scores[doc_index]) * weight
        if float(reset.sum()) <= 0:
            return dpr_scores, {"query": query, "facts": [fact["text"] for fact in chosen], "mode": "dpr_fallback"}
        damping = float(self.config.get("damping", 0.5))
        try:
            pagerank = prepared["graph"].personalized_pagerank(
                damping=damping,
                directed=False,
                weights="weight" if prepared["graph"].ecount() else None,
                reset=reset.tolist(),
                implementation="prpack",
            )
        except Exception:
            pagerank = prepared["graph"].personalized_pagerank(
                damping=damping,
                directed=False,
                reset=reset.tolist(),
            )
        doc_scores = np.array(
            [pagerank[prepared["index"][doc["doc_id"]]] for doc in prepared["docs"]], dtype=np.float64
        )
        return doc_scores, {
            "query": query,
            "facts": [fact["text"] for fact in chosen],
            "mode": "ppr",
            "comprehension_facts": [fact["text"] for fact in chosen if fact["kind"] == "comprehension"],
        }

    def _dpr(self, query: str, prepared: dict) -> np.ndarray:
        query_vec = self._normalize(
            self.embedder.encode([query], instruction=get_query_instruction("query_to_passage"))
        )[0]
        scores = prepared["doc_matrix"] @ query_vec
        scores = np.squeeze(scores)
        if np.ndim(scores) == 0:
            scores = np.array([float(scores)])
        return min_max_normalize(scores)

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
