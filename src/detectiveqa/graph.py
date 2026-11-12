"""One HippoRAG2 graph per novel, queried with a hard story-prefix mask.

The graph is built once from every chunk of a novel.  For a question at cutoff
``c`` all future passage nodes, future facts and phrase nodes that only occur in
future chunks are removed from retrieval.  PPR runs on the induced visible
subgraph, so future text cannot act as a bridge.  This keeps indexing cost
linear in the number of novels while matching the incremental-reading setting.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..adapters.hy4_client import load_env
from ..config import NVEMBED_PATH, PATHS
from .data import Chunk, OUT

load_env()
from ..graph.hipporag_adapter import make_base_config  # noqa: E402
from ..graph.runtime import EmbeddingProvider, open_hipporag  # noqa: E402
from ..graph.variants import clone_state, inject_triples, working_dir_of  # noqa: E402

GRAPH_ROOT = Path(os.environ.get("DETECTIVEQA_GRAPH_ROOT", OUT / "graphs"))
INDEX_IDENTITY = "rsigraph-detectiveqa|hy4-no_think|nvembed-v2|prefix-mask-v1"


def graph_dir(novel_id: str, variant: Optional[str] = None) -> Path:
    base = GRAPH_ROOT / "raw" / str(novel_id)
    return base if not variant else base / "variants" / variant


def hippo_chunks(chunks: Sequence[Chunk]):
    from hipporag.utils.misc_utils import Chunk as HippoChunk

    return [HippoChunk(content=c.text, source_id=c.chunk_id,
                       metadata={"chunk_id": c.chunk_id, "novel_id": c.novel_id,
                                 "start_pid": c.start_pid, "end_pid": c.end_pid})
            for c in chunks]


def _config_kwargs() -> Dict:
    return {"dataset": "detectiveqa", "retrieval_top_k": 200, "linking_top_k": 5, "qa_top_k": 5}


def _install_tolerant_openie(hippo) -> List[str]:
    from hipporag.utils.misc_utils import NerRawOutput, TripleRawOutput

    openie = hippo.openie
    orig_ner, orig_triple = openie.ner, openie.triple_extraction
    patched: List[str] = []

    def ner(chunk_key, passage):
        out = orig_ner(chunk_key, passage)
        if out.metadata.get("error"):
            patched.append(chunk_key)
            return NerRawOutput(chunk_id=chunk_key, response=out.response, unique_entities=[],
                                metadata={**out.metadata, "error": None, "patched": True})
        return out

    def triple(chunk_key, passage, named_entities):
        out = orig_triple(chunk_key, passage, named_entities)
        if out.metadata.get("error"):
            patched.append(chunk_key)
            return TripleRawOutput(chunk_id=chunk_key, response=out.response, triples=[],
                                   metadata={**out.metadata, "error": None, "patched": True})
        return out

    openie.ner = ner
    openie.triple_extraction = triple
    return patched


def build_raw_graph(novel_id: str, chunks: Sequence[Chunk], *, provider=None, force: bool = False) -> Dict:
    dst = graph_dir(novel_id)
    marker = dst / "detectiveqa_graph_info.json"
    if marker.exists() and not force:
        return json.loads(marker.read_text())
    if dst.exists() and force:
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    provider = provider or EmbeddingProvider.get(make_base_config(dst / "_cfg"))
    with open_hipporag(str(dst), conv_id=f"detectiveqa-{novel_id}", embedding_provider=provider,
                       force_index_from_scratch=True, **_config_kwargs()) as hippo:
        patched = _install_tolerant_openie(hippo)
        hippo.index(hippo_chunks(chunks))
        info = hippo.get_graph_info()
    info.update({"novel_id": novel_id, "variant": "raw", "n_chunks": len(chunks),
                 "patched_chunks": patched, "build_seconds": round(time.time() - t0, 1)})
    marker.write_text(json.dumps(info, indent=1, default=str))
    return info


def build_variant_graph(novel_id: str, chunks: Sequence[Chunk], variant: str,
                        extra_by_chunk: Dict[str, List[Tuple[str, str, str]]], *,
                        provider=None, force: bool = False, base_variant: Optional[str] = None) -> Dict:
    src = graph_dir(novel_id, base_variant)
    dst = graph_dir(novel_id, variant)
    marker = dst / "detectiveqa_graph_info.json"
    if marker.exists() and not force:
        return json.loads(marker.read_text())
    if dst.exists() and force:
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    clone_state(src, dst)
    state = working_dir_of(dst, "hy4", NVEMBED_PATH) / "openie_state.json"
    stats = inject_triples(state, extra_by_chunk, {c.chunk_id: c.text for c in chunks},
                           provenance_note=variant)
    provider = provider or EmbeddingProvider.get(make_base_config(dst / "_cfg"))
    with open_hipporag(str(dst), conv_id=f"detectiveqa-{novel_id}", embedding_provider=provider,
                       force_index_from_scratch=True, **_config_kwargs()) as hippo:
        hippo.index(hippo_chunks(chunks))
        info = hippo.get_graph_info()
    info.update({"novel_id": novel_id, "variant": variant, "base_variant": base_variant or "raw",
                 "injection": stats, "n_chunks": len(chunks), "build_seconds": round(time.time() - t0, 1)})
    marker.write_text(json.dumps(info, indent=1, default=str))
    return info


@contextmanager
def open_graph(novel_id: str, variant: Optional[str] = None, provider=None, llm=None):
    save_dir = graph_dir(novel_id, variant)
    if not (save_dir / "detectiveqa_graph_info.json").exists():
        raise FileNotFoundError(f"graph missing: {save_dir}")
    with open_hipporag(str(save_dir), conv_id=f"detectiveqa-{novel_id}", embedding_provider=provider,
                       llm=llm, **_config_kwargs()) as hippo:
        yield hippo


def _min_max(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low, high = float(np.min(values)), float(np.max(values))
    return np.ones_like(values) if high == low else (values - low) / (high - low)


class PrefixGraphRetriever:
    """Prefix-safe HippoRAG retrieval with optional official LLM fact reranking."""

    def __init__(self, hippo, chunks: Sequence[Chunk]):
        from hipporag.utils.misc_utils import compute_mdhash_id

        self.hippo = hippo
        if not hippo.ready_to_retrieve:
            hippo.prepare_retrieval_objects()
        self.compute_id = compute_mdhash_id
        self.chunks = list(chunks)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.by_text = {c.text: c for c in self.chunks}
        self.passage_chunk: List[Optional[Chunk]] = []
        for key in hippo.passage_node_keys:
            meta = hippo.chunk_metadata.get(key, {})
            chunk = self.by_id.get(meta.get("chunk_id"))
            if chunk is None:
                chunk = self.by_text.get(hippo.chunk_embedding_store.get_row(key)["content"])
            self.passage_chunk.append(chunk)
        self.passage_available = np.asarray([c.available_at if c else 10 ** 12 for c in self.passage_chunk])
        self.passage_key_to_available = {k: int(a) for k, a in zip(hippo.passage_node_keys, self.passage_available)}
        self.fact_available = self._fact_availability()
        self.entity_available = self._entity_availability()
        self._graph_cache: Dict[Tuple[int, bool], Tuple[object, Dict[int, int]]] = {}

    def _fact_availability(self) -> np.ndarray:
        rows = self.hippo.fact_embedding_store.get_rows(self.hippo.fact_node_keys)
        out = []
        for key in self.hippo.fact_node_keys:
            fact = ast.literal_eval(rows[key]["content"])
            docs = self.hippo.proc_triples_to_docs.get(str(tuple(fact)), set())
            times = [self.passage_key_to_available[d] for d in docs if d in self.passage_key_to_available]
            out.append(min(times) if times else 10 ** 12)
        return np.asarray(out)

    def _entity_availability(self) -> Dict[str, int]:
        out = {}
        for key, docs in self.hippo.ent_node_to_chunk_ids.items():
            times = [self.passage_key_to_available[d] for d in docs if d in self.passage_key_to_available]
            out[key] = min(times) if times else 10 ** 12
        return out

    def _visible_nodes(self, cutoff: int) -> Tuple[List[int], List[int]]:
        passages = [self.hippo.node_name_to_vertex_idx[k]
                    for k, a in zip(self.hippo.passage_node_keys, self.passage_available) if a <= cutoff]
        entities = [self.hippo.node_name_to_vertex_idx[k] for k in self.hippo.entity_node_keys
                    if self.entity_available.get(k, 10 ** 12) <= cutoff]
        return passages, entities

    def _visible_graph(self, cutoff: int, use_synonym: bool = True):
        """Induced graph whose fact-edge weights only count visible provenance."""
        cache_key = (cutoff, use_synonym)
        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]
        visible_passages, visible_entities = self._visible_nodes(cutoff)
        vertices = sorted(set(visible_passages) | set(visible_entities))
        graph = self.hippo.graph.induced_subgraph(vertices)
        keep_weights = []
        for edge in graph.es:
            attrs = edge.attributes()
            counts = attrs.get("fact_source_counts") or {}
            fact_weight = float(sum(n for src, n in counts.items()
                                    if self.passage_key_to_available.get(src, 10 ** 12) <= cutoff))
            passage_source = attrs.get("passage_source")
            passage_weight = 1.0 if passage_source is not None and \
                self.passage_key_to_available.get(passage_source, 10 ** 12) <= cutoff else 0.0
            synonym_weight = float(attrs.get("synonym_score") or 0.0) if use_synonym else 0.0
            keep_weights.append(max(fact_weight, passage_weight, synonym_weight))
        graph.es["weight"] = keep_weights
        graph.delete_edges([i for i, w in enumerate(keep_weights) if w <= 0])
        mapping = {g: i for i, g in enumerate(vertices)}
        if len(self._graph_cache) >= 16:
            self._graph_cache.pop(next(iter(self._graph_cache)))
        self._graph_cache[cache_key] = (graph, mapping)
        return graph, mapping

    def _query_vectors(self, query: str):
        self.hippo.get_query_embeddings([query])
        return self.hippo.query_to_embedding["triple"].get(query), self.hippo.query_to_embedding["passage"][query]

    def _dense(self, query: str, cutoff: int) -> Tuple[np.ndarray, np.ndarray]:
        _, qv = self._query_vectors(query)
        visible = np.flatnonzero(self.passage_available <= cutoff)
        if visible.size == 0:
            return visible, np.asarray([], dtype=float)
        scores = _min_max(np.dot(self.hippo.passage_embeddings[visible], qv.T).reshape(-1))
        order = np.argsort(scores)[::-1]
        return visible[order], scores[order]

    def _fact_candidates(self, query: str, cutoff: int, k: int) -> Tuple[List[int], List[Tuple], np.ndarray]:
        qv, _ = self._query_vectors(query)
        visible = np.flatnonzero(self.fact_available <= cutoff)
        all_scores = np.zeros(len(self.hippo.fact_node_keys), dtype=float)
        if qv is None or visible.size == 0:
            return [], [], all_scores
        scores = _min_max(np.dot(self.hippo.fact_embeddings[visible], qv.T).reshape(-1))
        all_scores[visible] = scores
        indices = visible[np.argsort(scores)[::-1][:k]].tolist()
        rows = self.hippo.fact_embedding_store.get_rows([self.hippo.fact_node_keys[i] for i in indices])
        facts = [ast.literal_eval(rows[self.hippo.fact_node_keys[i]]["content"]) for i in indices]
        return indices, facts, all_scores

    def retrieve(self, query: str, cutoff: int, k: int = 20, *, use_llm_rerank: bool = False,
                 fact_top_k: int = 5, link_top_k: int = 5, use_synonym: bool = True) -> Dict:
        indices, facts, fact_scores = self._fact_candidates(query, cutoff, fact_top_k)
        before = list(facts)
        if use_llm_rerank and facts:
            indices, facts, _ = self.hippo.rerank_filter(query, facts, indices, len_after_rerank=fact_top_k)
        if not facts:
            ids, scores = self._dense(query, cutoff)
            mode = "dense_fallback"
        else:
            ids, scores = self._ppr(query, cutoff, facts, indices, fact_scores, link_top_k, use_synonym)
            mode = "prefix_ppr"
        ranked = []
        for idx, score in zip(ids[:k], scores[:k]):
            chunk = self.passage_chunk[int(idx)]
            if chunk is not None and chunk.available_at <= cutoff:
                ranked.append({"chunk_id": chunk.chunk_id, "score": float(score)})
        return {"mode": mode, "facts_before_rerank": [list(x) for x in before],
                "facts": [list(x) for x in facts], "ranked": ranked}

    def _ppr(self, query: str, cutoff: int, facts: Sequence[Tuple], indices: Sequence[int],
             fact_scores: np.ndarray, link_top_k: int,
             use_synonym: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        graph, global_to_local = self._visible_graph(cutoff, use_synonym)
        weights = np.zeros(graph.vcount(), dtype=float)
        phrase_scores: Dict[int, List[float]] = {}
        for rank, fact in enumerate(facts):
            score = float(fact_scores[indices[rank]])
            for phrase in (fact[0].lower(), fact[2].lower()):
                key = self.compute_id(phrase, prefix="entity-")
                global_id = self.hippo.node_name_to_vertex_idx.get(key)
                if global_id not in global_to_local:
                    continue
                n_docs = sum(1 for d in self.hippo.ent_node_to_chunk_ids.get(key, set())
                             if self.passage_key_to_available.get(d, 10 ** 12) <= cutoff)
                phrase_scores.setdefault(global_to_local[global_id], []).append(score / max(1, n_docs))
        top_phrases = sorted(phrase_scores, key=lambda i: float(np.mean(phrase_scores[i])), reverse=True)[:link_top_k]
        for local in top_phrases:
            weights[local] = float(np.mean(phrase_scores[local]))
        dense_ids, dense_scores = self._dense(query, cutoff)
        for passage_idx, score in zip(dense_ids, dense_scores):
            global_id = self.hippo.passage_node_idxs[int(passage_idx)]
            weights[global_to_local[global_id]] = score * self.hippo.global_config.passage_node_weight
        if weights.sum() <= 0:
            return dense_ids, dense_scores
        pr = graph.personalized_pagerank(vertices=range(graph.vcount()), damping=self.hippo.global_config.damping,
                                         directed=False, weights="weight", reset=weights,
                                         implementation="prpack")
        passage_local = [global_to_local[self.hippo.passage_node_idxs[i]]
                         for i in np.flatnonzero(self.passage_available <= cutoff)]
        passage_global = np.flatnonzero(self.passage_available <= cutoff)
        scores = np.asarray([pr[i] for i in passage_local])
        order = np.argsort(scores)[::-1]
        return passage_global[order], scores[order]
