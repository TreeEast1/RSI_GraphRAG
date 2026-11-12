"""Build graph variants (G_raw + extra read-time knowledge) without touching G_raw.

Mechanism
---------
1. Copy the raw working directory's *state* (embedding parquet stores, chunk
   metadata, OpenIE state) into a variant directory. The graph pickle and the
   index manifest are deliberately NOT copied.
2. Merge the extra triples into the copied `openie_state.json`, attributing each
   triple to the source session's chunk id.
3. Re-run the official `HippoRAG.index()` with `force_index_from_scratch=True`.
   Because the OpenIE state already covers every chunk, **no LLM extraction is
   invoked**; only embeddings for genuinely new entities/facts are computed and
   the official graph builder runs.
4. Save the variant graph.

This keeps G_raw frozen and keeps graph construction entirely inside the
official implementation.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ..config import NVEMBED_PATH, PATHS

import sys

_HIPPO_SRC = str(PATHS["third_party"] / "src")
if _HIPPO_SRC not in sys.path:
    sys.path.insert(0, _HIPPO_SRC)

from hipporag.embedding_store import compute_mdhash_id  # noqa: E402

TripleStr = Tuple[str, str, str]


def working_dir_of(save_dir: str | Path, llm_name: str, embedding_name: str) -> Path:
    return Path(save_dir) / f"{llm_name.replace('/', '_')}_{embedding_name.replace('/', '_')}"


def raw_save_dir(conv_id: str) -> Path:
    return PATHS["raw_graphs"] / conv_id


def clone_state(src_save_dir: Path, dst_save_dir: Path, llm_name: str = "hy4",
                embedding_name: str = NVEMBED_PATH) -> Path:
    """Copy reusable index state from raw -> variant; skip graph + manifest."""
    src = working_dir_of(src_save_dir, llm_name, embedding_name)
    dst = working_dir_of(dst_save_dir, llm_name, embedding_name)
    if not src.exists():
        raise FileNotFoundError(f"raw working dir missing: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    # Keep the manifest (identical embedding/LLM identity so the copied stores
    # validate), drop only the graph so index() rebuilds it from the injected OpenIE state.
    skip = {"graph.pickle"}
    for item in src.iterdir():
        if item.name in skip:
            continue
        target = dst / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    # share the HY4 response cache with the raw run
    src_cache = Path(src_save_dir) / "llm_cache"
    dst_cache = Path(dst_save_dir) / "llm_cache"
    if src_cache.exists() and not dst_cache.exists():
        dst_cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst_cache.symlink_to(src_cache, target_is_directory=True)
        except OSError:
            shutil.copytree(src_cache, dst_cache)
    # share the human-readable openie export location inside the working dir
    return dst


def chunk_id_for(text: str) -> str:
    return compute_mdhash_id(text, prefix="chunk-")


def load_openie_rows(state_path: Path) -> Dict:
    with open(state_path, encoding="utf-8") as f:
        return json.load(f)


def inject_triples(
    state_path: Path,
    extra_by_doc: Dict[str, List[TripleStr]],
    doc_text: Dict[str, str],
    provenance_note: str = "read_time",
) -> Dict[str, int]:
    """Append extra triples to the OpenIE rows of their source chunks.

    Returns counts: {added, chunks_touched, skipped_duplicate}.
    """
    payload = load_openie_rows(state_path)
    rows = payload.get("docs", [])
    by_idx = {r["idx"]: r for r in rows}
    added = dup = 0
    touched = set()
    for doc_id, triples in extra_by_doc.items():
        cid = chunk_id_for(doc_text[doc_id])
        row = by_idx.get(cid)
        if row is None:
            continue
        existing = {tuple(t) for t in row.get("extracted_triples", [])}
        for t in triples:
            t = tuple(str(x) for x in t)
            if t in existing:
                dup += 1
                continue
            existing.add(t)
            row["extracted_triples"].append(list(t))
            for e in (t[0], t[2]):
                if e not in row.setdefault("extracted_entities", []):
                    row["extracted_entities"].append(e)
            added += 1
        if triples:
            touched.add(doc_id)
    # keep provenance; add a non-validated annotation for traceability
    payload.setdefault("read_time_injection", []).append(
        {"note": provenance_note, "n_added": added, "docs": sorted(touched)}
    )
    tmp = state_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, state_path)
    return {"added": added, "chunks_touched": len(touched), "skipped_duplicate": dup}


def build_variant_graph(
    conv,
    variant_name: str,
    extra_by_doc: Dict[str, List[TripleStr]],
    *,
    provider,
    llm,
    force: bool = False,
) -> Dict:
    """Create outputs/raw_graphs/{conv}/variants/{name} and return build info."""
    from .runtime import open_hipporag

    raw_dir = raw_save_dir(conv.conversation_id)
    dst_dir = raw_dir / "variants" / variant_name
    if dst_dir.exists() and not force:
        return {"variant": variant_name, "reused": True, "save_dir": str(dst_dir)}

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    clone_state(raw_dir, dst_dir)
    doc_text = {s.document_id: s.text for s in conv.sessions}
    state_path = working_dir_of(dst_dir, "hy4", NVEMBED_PATH) / "openie_state.json"
    stats = inject_triples(state_path, extra_by_doc, doc_text, provenance_note=variant_name)

    from hipporag.utils.misc_utils import Chunk
    chunks = [Chunk(content=s.text, source_id=s.document_id,
                    metadata={"document_id": s.document_id, "session_id": s.session_id,
                              "date_time": s.date_time_raw, "conversation_id": conv.conversation_id})
              for s in conv.sessions]

    import time
    t = time.time()
    with open_hipporag(str(dst_dir), conv_id=conv.conversation_id, embedding_provider=provider,
                       llm=llm, force_index_from_scratch=True) as hippo:
        hippo.index(chunks)
        info = hippo.get_graph_info()
    return {
        "variant": variant_name,
        "reused": False,
        "save_dir": str(dst_dir),
        "injection": stats,
        "graph_info": info,
        "build_seconds": round(time.time() - t, 1),
    }
