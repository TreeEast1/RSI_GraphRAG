"""HippoRAG2-compatible OpenIE and immutable G_raw construction.

Official NER and triple prompts are loaded from the local HippoRAG checkout.
This module never writes into that checkout.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

HIPPO_SRC = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")
if str(HIPPO_SRC) not in sys.path:
    sys.path.insert(0, str(HIPPO_SRC))

from hipporag.prompts.prompt_template_manager import PromptTemplateManager  # noqa: E402
from hipporag.utils.llm_utils import filter_invalid_triples  # noqa: E402
from hipporag.utils.misc_utils import compute_mdhash_id, text_processing  # noqa: E402

from .hashing import file_sha256, sha256_obj, sha256_text

HIPPO_ROOT = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG")
PROMPT_FILES = [
    HIPPO_ROOT / "src/hipporag/prompts/templates/ner.py",
    HIPPO_ROOT / "src/hipporag/prompts/templates/triple_extraction.py",
    HIPPO_ROOT / "src/hipporag/prompts/templates/rag_qa_musique.py",
]


def hipporag_commit() -> str:
    head = HIPPO_ROOT / ".git/HEAD"
    if not head.exists():
        return "unknown"
    ref = head.read_text(encoding="utf-8").strip()
    if ref.startswith("ref:"):
        pointer = HIPPO_ROOT / ".git" / ref.split(" ", 1)[1]
        if pointer.exists():
            return pointer.read_text(encoding="utf-8").strip()
    return ref


def official_prompt_hash() -> str:
    return sha256_text("\n".join(path.read_text(encoding="utf-8") for path in PROMPT_FILES))


def prompt_manager() -> PromptTemplateManager:
    return PromptTemplateManager(
        role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
    )


def extract_json_object(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def parse_entities(text: str) -> List[str]:
    try:
        payload = extract_json_object(text)
    except Exception:
        match = re.search(r"\{[^{}]*\"named_entities\"\s*:\s*\[[^\]]*\][^{}]*\}", text, re.DOTALL)
        if not match:
            return []
        payload = json.loads(match.group())
    entities = payload.get("named_entities", []) if isinstance(payload, dict) else payload
    if not isinstance(entities, list):
        return []
    return [item.strip() for item in entities if isinstance(item, str) and item.strip()]


def parse_triples(text: str) -> List[List[str]]:
    try:
        payload = extract_json_object(text)
    except Exception:
        match = re.search(r"\{[^{}]*\"triples\"\s*:\s*\[[^\]]*\][^{}]*\}", text, re.DOTALL)
        if not match:
            return []
        payload = json.loads(match.group())
    triples = payload.get("triples", []) if isinstance(payload, dict) else payload
    if not isinstance(triples, list):
        return []
    return filter_invalid_triples(triples)


def openie_one(client, passage: dict, manager: PromptTemplateManager, reasoning_effort: str = "low") -> dict:
    content = passage["content"]
    ner_messages = manager.render(name="ner", passage=content)
    ner = client.complete(
        stage="openie_ner",
        messages=ner_messages,
        max_completion_tokens=1200,
        reasoning_effort=reasoning_effort,
    )
    entities = parse_entities(ner.text)
    triple_messages = manager.render(
        name="triple_extraction",
        passage=content,
        named_entity_json=json.dumps({"named_entities": entities}, ensure_ascii=False),
    )
    triple = client.complete(
        stage="openie_triple",
        messages=triple_messages,
        max_completion_tokens=2500,
        reasoning_effort=reasoning_effort,
    )
    return {
        "passage_id": passage["passage_id"],
        "title": passage.get("title", ""),
        "text": passage.get("text", ""),
        "content": content,
        "entities": entities,
        "triples": parse_triples(triple.text),
        "ner_finish_reason": ner.metadata.get("finish_reason"),
        "triple_finish_reason": triple.metadata.get("finish_reason"),
        "response_model": triple.metadata.get("response_model") or ner.metadata.get("response_model"),
        "requested_deployment": triple.metadata.get("requested_deployment"),
        "prompt_hash": official_prompt_hash(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def processed_triple(triple: Sequence[str]) -> Optional[tuple]:
    if len(triple) != 3:
        return None
    parts = tuple(text_processing(part) for part in triple)
    if any(not part for part in parts):
        return None
    return parts


def entity_id_for(surface: str) -> str:
    return compute_mdhash_id(text_processing(surface), prefix="entity-")


def triple_id_for(triple: Sequence[str]) -> str:
    return compute_mdhash_id(" | ".join(triple), prefix="triple-")


def build_raw_graph(
    records: Sequence[dict],
    *,
    status: str,
    model_name: str,
    corpus_hash: str,
    embeddings: Optional[Dict[str, np.ndarray]] = None,
    synonymy_threshold: float = 0.8,
    synonymy_topk: int = 100,
) -> dict:
    entities: Dict[str, dict] = {}
    passages: Dict[str, dict] = {}
    triples: Dict[str, dict] = {}
    context = {}
    uncovered = []
    linked = set()
    for record in records:
        pid = record["passage_id"]
        passages[pid] = {
            "passage_id": pid,
            "title": record.get("title", ""),
            "text": record.get("text", ""),
            "content": record["content"],
        }
        valid = []
        for triple in record.get("triples") or []:
            processed = processed_triple(triple)
            if processed is not None:
                valid.append(processed)
        if not valid:
            uncovered.append(pid)
        for subject, predicate, obj in valid:
            sid = entity_id_for(subject)
            oid = entity_id_for(obj)
            linked.add(sid)
            linked.add(oid)
            entities.setdefault(sid, {"entity_id": sid, "content": subject})
            entities.setdefault(oid, {"entity_id": oid, "content": obj})
            tid = triple_id_for((subject, predicate, obj))
            row = triples.setdefault(
                tid,
                {
                    "triple_id": tid,
                    "s": subject,
                    "p": predicate,
                    "o": obj,
                    "s_id": sid,
                    "o_id": oid,
                    "passage_ids": [],
                },
            )
            if pid not in row["passage_ids"]:
                row["passage_ids"].append(pid)
            context[(pid, sid)] = {"passage_id": pid, "entity_id": sid, "kind": "context"}
            context[(pid, oid)] = {"passage_id": pid, "entity_id": oid, "kind": "context"}
        for surface in record.get("entities") or []:
            processed = text_processing(surface)
            if not processed:
                continue
            eid = entity_id_for(processed)
            entities.setdefault(eid, {"entity_id": eid, "content": processed})
    for entity in entities.values():
        entity["ner_only"] = entity["entity_id"] not in linked

    if embeddings:
        synonymy, synonymy_status = _synonymy_edges(entities, embeddings, synonymy_threshold, synonymy_topk)
    else:
        synonymy, synonymy_status = [], "skipped_no_embedding"
    graph = {
        "schema_version": 1,
        "kind": "g_raw",
        "immutable": True,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hipporag_commit": hipporag_commit(),
        "official_prompt_hash": official_prompt_hash(),
        "model_name": model_name,
        "corpus_hash": corpus_hash,
        "synonymy_status": synonymy_status,
        "synonymy_threshold": synonymy_threshold,
        "passages": passages,
        "entities": entities,
        "triples": triples,
        "context_edges": list(context.values()),
        "synonymy_edges": synonymy,
        "uncovered_passage_ids": sorted(set(uncovered)),
    }
    graph["graph_hash"] = graph_hash(graph)
    return graph


def graph_hash(graph: dict) -> str:
    payload = {
        "passages": sorted(graph["passages"]),
        "entities": sorted((item["entity_id"], item["content"]) for item in graph["entities"].values()),
        "triples": sorted(
            (
                item["triple_id"],
                item["s"],
                item["p"],
                item["o"],
                tuple(sorted(item["passage_ids"])),
            )
            for item in graph["triples"].values()
        ),
        "context_edges": sorted((item["passage_id"], item["entity_id"]) for item in graph["context_edges"]),
        "synonymy_edges": sorted(
            (item["a"], item["b"], round(float(item["score"]), 6)) for item in graph["synonymy_edges"]
        ),
    }
    return sha256_obj(payload)


def save_raw_graph(graph: dict, path: str) -> str:
    target = Path(path)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("graph_hash") != graph.get("graph_hash"):
            raise RuntimeError(f"refusing to overwrite a different frozen graph at {path}")
        return existing["graph_hash"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    sidecar = target.with_suffix(target.suffix + ".sha256")
    sidecar.write_text(file_sha256(str(target)) + "\n", encoding="utf-8")
    return graph["graph_hash"]


def load_raw_graph(path: str) -> dict:
    target = Path(path)
    graph = json.loads(target.read_text(encoding="utf-8"))
    if graph_hash(graph) != graph.get("graph_hash"):
        raise RuntimeError("G_raw hash mismatch; the snapshot was modified")
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if sidecar.exists() and sidecar.read_text(encoding="utf-8").strip() != file_sha256(str(target)):
        raise RuntimeError("G_raw file bytes do not match the sidecar hash")
    return graph


def _synonymy_edges(entities, embeddings, threshold: float, topk: int):
    ids = [
        eid
        for eid, entity in entities.items()
        if eid in embeddings and len(re.sub(r"[^A-Za-z0-9]", "", entity["content"])) > 2
    ]
    if len(ids) < 2:
        return [], "too_few_entities"
    matrix = np.stack([np.asarray(embeddings[eid], dtype=np.float32) for eid in ids])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    edges = []
    seen = set()
    for start in range(0, len(ids), 256):
        scores = matrix[start : start + 256] @ matrix.T
        for row_index, row in enumerate(scores):
            global_index = start + row_index
            row = row.copy()
            row[global_index] = -1.0
            if topk < len(row):
                chosen = np.argpartition(row, -topk)[-topk:]
            else:
                chosen = np.arange(len(row))
            order = sorted(chosen, key=lambda index: float(row[index]), reverse=True)
            kept = 0
            for index in order:
                score = float(row[index])
                if score < threshold or kept >= 100:
                    break
                key = tuple(sorted((ids[global_index], ids[int(index)])))
                if key[0] == key[1] or key in seen:
                    continue
                seen.add(key)
                edges.append({"a": key[0], "b": key[1], "score": score, "kind": "synonymy"})
                kept += 1
    return edges, "computed"


def load_openie_dir(directory: str) -> List[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(directory).glob("*.json"))
    ]


def write_openie_record(directory: str, record: dict) -> None:
    target = Path(directory) / f"{record['passage_id']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)


def historical_records(path: str) -> Iterable[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    docs = payload["docs"] if isinstance(payload, dict) else payload
    for doc in docs:
        content = doc.get("passage") or f"{doc.get('title', '')}\n{doc.get('text', '')}"
        yield {
            "passage_id": doc.get("idx") or compute_mdhash_id(content, prefix="chunk-"),
            "title": doc.get("title") or "",
            "text": doc.get("text") or "",
            "content": content,
            "entities": doc.get("extracted_entities") or [],
            "triples": doc.get("extracted_triples") or [],
            "model": "gpt-4o-mini",
            "official_idx": doc.get("idx"),
        }
