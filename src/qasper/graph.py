"""HippoRAG2 graph construction and retrieval for QASPER papers.

G_raw is built first from every paragraph of a paper and then frozen. Variants
(M2 atomic facts, M3 + comprehension triples) are produced exactly like the
LoCoMo pipeline: copy the raw embedding/OpenIE state, inject extra triples into
the OpenIE rows of their source paragraphs, re-run the official ``index()`` in a
new directory. No QA label is ever used to choose paragraphs or groups.
"""
from __future__ import annotations

import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import NVEMBED_PATH
from ..adapters.hy4_client import load_env
from ..graph.hipporag_adapter import make_base_config

load_env()
from ..graph.runtime import EmbeddingProvider, open_hipporag
from ..graph.variants import chunk_id_for, clone_state, inject_triples, working_dir_of

OUT = Path(__file__).resolve().parents[2] / "outputs" / "qasper_readtime"
GRAPH_ROOT = OUT / "graphs"


def graph_dir(paper_id: str, variant: Optional[str] = None) -> Path:
    base = GRAPH_ROOT / "raw" / paper_id
    return base if not variant else base / "variants" / variant


def _chunks(paper):
    from hipporag.utils.misc_utils import Chunk

    out = []
    for p in paper.paragraphs:
        out.append(Chunk(
            content=p.text,
            source_id=p.passage_id,
            metadata={
                "passage_id": p.passage_id,
                "paper_id": paper.paper_id,
                "section_index": p.section_index,
                "paragraph_index": p.paragraph_index,
                "section_name": p.section_name,
            },
        ))
    return out


def _install_tolerant_openie(hippo) -> List[str]:
    """Let the official OpenIE continue when a single chunk is unparseable.

    The official ``batch_openie`` raises if any chunk's (cached) response cannot
    be parsed as NER/triples, which aborts an entire paper. Retrying cannot help
    because the bad response is cached. We wrap the two per-chunk methods so a
    failed chunk yields an empty result and the paper still gets a graph; the
    passage node remains retrievable by dense scoring. Patched chunk ids are
    recorded on the graph marker.
    """
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
    hippo._qasper_patched_chunks = patched
    return patched


def build_raw_graph(paper, *, provider=None, force: bool = False, verbose: bool = True) -> Dict:
    dst = graph_dir(paper.paper_id)
    marker = dst / "qasper_graph_info.json"
    if marker.exists() and not force:
        return json.loads(marker.read_text())
    if dst.exists() and force:
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    prov = provider or EmbeddingProvider.get(make_base_config(dst / "_cfg"))
    patched: List[str] = []
    info = {}
    with open_hipporag(str(dst), conv_id=paper.paper_id, embedding_provider=prov,
                       force_index_from_scratch=True, dataset="qasper",
                       retrieval_top_k=50, linking_top_k=5) as hippo:
        patched = _install_tolerant_openie(hippo)
        hippo.index(_chunks(paper))
        info = hippo.get_graph_info()
        patched = list(getattr(hippo, "_qasper_patched_chunks", patched))
    info.update({"paper_id": paper.paper_id, "variant": "raw",
                 "n_paragraphs": len(paper.paragraphs),
                 "patched_chunks": patched,
                 "build_seconds": round(time.time() - t0, 1)})
    marker.write_text(json.dumps(info, indent=1, default=str))
    if verbose:
        print(f"[raw] {paper.paper_id}: {info.get('num_total_nodes')} nodes / "
              f"{info.get('num_total_triples')} triples in {info['build_seconds']}s"
              + (f" (patched {len(patched)} chunks)" if patched else ""), flush=True)
    return info


def build_variant_graph(paper, variant: str, extra_by_passage: Dict[str, List[Tuple[str, str, str]]],
                        *, provider=None, force: bool = False, base_variant: Optional[str] = None,
                        verbose: bool = True) -> Dict:
    """Clone a source graph's state, inject triples, rebuild the graph.

    base_variant: None -> clone G_raw; otherwise clone that variant (used when a
    method is defined as "previous variant + new knowledge").
    """
    src_dir = graph_dir(paper.paper_id, base_variant)
    dst = graph_dir(paper.paper_id, variant)
    marker = dst / "qasper_graph_info.json"
    if marker.exists() and not force:
        return json.loads(marker.read_text())
    if dst.exists() and force:
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    clone_state(src_dir, dst)

    doc_text = {p.passage_id: p.text for p in paper.paragraphs}
    state_path = working_dir_of(dst, "hy4", NVEMBED_PATH) / "openie_state.json"
    stats = inject_triples(state_path, extra_by_passage, doc_text, provenance_note=variant)

    prov = provider or EmbeddingProvider.get(make_base_config(dst / "_cfg"))
    with open_hipporag(str(dst), conv_id=paper.paper_id, embedding_provider=prov,
                       force_index_from_scratch=True, dataset="qasper",
                       retrieval_top_k=50, linking_top_k=5) as hippo:
        hippo.index(_chunks(paper))
        info = hippo.get_graph_info()
    info.update({"paper_id": paper.paper_id, "variant": variant,
                 "base_variant": base_variant or "raw",
                 "injection": stats,
                 "n_paragraphs": len(paper.paragraphs),
                 "build_seconds": round(time.time() - t0, 1)})
    marker.write_text(json.dumps(info, indent=1, default=str))
    if verbose:
        print(f"[{variant}] {paper.paper_id}: +{stats['added']} triples, "
              f"{info.get('num_total_nodes')} nodes in {info['build_seconds']}s", flush=True)
    return info


@contextmanager
def open_graph(paper_id: str, variant: Optional[str] = None, provider=None, llm=None):
    save_dir = graph_dir(paper_id, variant)
    if not (save_dir / "qasper_graph_info.json").exists():
        raise FileNotFoundError(f"graph missing: {save_dir}")
    with open_hipporag(str(save_dir), conv_id=paper_id, embedding_provider=provider, llm=llm,
                       dataset="qasper", retrieval_top_k=50, linking_top_k=5) as hippo:
        yield hippo


def retrieve(hippo, question: str, k: int = 20) -> List[Dict]:
    """Return ranked passages with source passage_id and the seed facts."""
    sol = hippo.retrieve([question], num_to_retrieve=k)[0]
    out = []
    for doc, score, meta in zip(sol.docs, sol.doc_scores, sol.doc_metadata or [{}] * len(sol.docs)):
        pid = (meta or {}).get("passage_id") or (meta or {}).get("source_id")
        out.append({"passage_id": pid, "score": float(score)})
    facts = []
    for f in getattr(sol, "graph_seeds", []) or []:
        try:
            facts.append({"subject": f[0], "predicate": f[1], "object": f[2]})
        except Exception:  # noqa: BLE001
            facts.append({"raw": str(f)})
    if out:
        out[0]["graph_seeds"] = facts
    return out
