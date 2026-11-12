"""Graph patches against G_comp. G_raw is only read."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple

from .comprehension_store import ComprehensionStore
from .hashing import sha256_text
from .hipporag_adapter import graph_hash
from .validation import near_duplicate, span_supported


def assert_raw_unchanged(graph: dict, expected_hash: str) -> None:
    actual = graph_hash(graph)
    if actual != expected_hash or actual != graph.get("graph_hash"):
        raise RuntimeError("G_raw changed during a comprehension or question-time update")


class GraphPatchManager:
    def __init__(self, raw_graph: dict, store: ComprehensionStore) -> None:
        self.raw_graph = raw_graph
        self.store = store
        self.raw_hash = graph_hash(raw_graph)
        self.rejected: List[dict] = []
        self.accepted: List[dict] = []

    def existing_texts(self) -> List[str]:
        texts = [f"{item['s']} {item['p']} {item['o']}" for item in self.raw_graph["triples"].values()]
        texts.extend(node["semantic_content"] for node in self.store.active_nodes())
        return texts

    def passages(self) -> Dict[str, dict]:
        return self.raw_graph["passages"]

    def known_entity_ids(self) -> set:
        return set(self.raw_graph["entities"])

    def validate_and_apply(self, ops: Sequence[dict], *, created_by: str) -> Tuple[List[dict], List[dict]]:
        accepted, rejected = [], []
        for op in ops:
            reason = self._reject_reason(op)
            if reason:
                rejected.append({**op, "reason": reason, "created_by": created_by})
                continue
            self._apply(op, created_by)
            accepted.append({**op, "created_by": created_by})
        self.accepted.extend(accepted)
        self.rejected.extend(rejected)
        assert_raw_unchanged(self.raw_graph, self.raw_hash)
        return accepted, rejected

    def _reject_reason(self, op: dict) -> str:
        kind = op.get("op")
        if kind == "NO_OP":
            return ""
        if kind == "DEACTIVATE":
            if op.get("comprehension_id") not in self.store.nodes:
                return "unknown comprehension node"
            return ""
        if kind not in {"ADD", "UPDATE"}:
            return "unsupported op"
        passages = self.passages()
        spans = op.get("evidence_spans") or []
        if not spans:
            return "missing evidence span"
        source_passages = op.get("source_passage_ids") or []
        if not source_passages:
            return "missing source passage"
        for passage_id in source_passages:
            if passage_id not in passages:
                return f"unknown passage {passage_id}"
        supported = False
        for span in spans:
            passage_id = span.get("passage_id")
            if passage_id not in passages:
                return f"evidence passage {passage_id} is not in G_raw"
            if span_supported(span.get("span", ""), passages[passage_id]["content"]):
                supported = True
        if not supported:
            return "no evidence span is a substring of its source passage"
        for entity_id in op.get("source_entity_ids") or []:
            if entity_id not in self.known_entity_ids():
                return f"invented entity {entity_id}"
        for triple_id in op.get("source_triple_ids") or []:
            if triple_id not in self.raw_graph["triples"]:
                return f"unknown triple {triple_id}"
        for link in op.get("links") or []:
            target = link.get("target_id")
            if target not in self.known_entity_ids() and target not in self.store.nodes:
                return f"link target {target} is not in the graph"
            if not link.get("relation"):
                return "link missing relation"
        content = (op.get("semantic_content") or "").strip()
        if len(content) < 20:
            return "semantic content too short"
        if kind == "ADD" and near_duplicate(content, self.existing_texts()):
            return "duplicate of an existing triple or comprehension"
        if kind == "UPDATE" and op.get("comprehension_id") not in self.store.nodes:
            return "update target does not exist"
        return ""

    def _apply(self, op: dict, created_by: str) -> None:
        if op["op"] == "NO_OP":
            self.store.version += 0
            return
        if op["op"] == "DEACTIVATE":
            self.store.deactivate(op["comprehension_id"], op.get("reason") or "deactivated")
            return
        chunk_text = _chunk_text(op)
        if op["op"] == "ADD":
            comprehension_id = "comp-" + sha256_text(op["semantic_content"] + "|" + created_by)[:16]
            node = {
                "comprehension_id": comprehension_id,
                "semantic_content": op["semantic_content"].strip(),
                "node_type": op.get("node_type") or "implicit_relation",
                "source_entity_ids": list(op.get("source_entity_ids") or []),
                "source_triple_ids": list(op.get("source_triple_ids") or []),
                "source_passage_ids": list(op.get("source_passage_ids") or []),
                "evidence_spans": list(op.get("evidence_spans") or []),
                "reasoning_description": op.get("reasoning_description") or "",
                "why_retrieval_useful": op.get("why_retrieval_useful") or "",
                "confidence": float(op.get("confidence") or 0.0),
                "version": 1,
                "status": "active",
                "created_by": created_by,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "chunk_id": "cchunk-" + sha256_text(chunk_text)[:16],
                "chunk_text": chunk_text,
                "primary_unit_id": op.get("primary_unit_id") or "",
            }
            self.store.add_node(node)
            for link in op.get("links") or []:
                self.store.add_edge(
                    {
                        "edge_id": "cedge-" + sha256_text(comprehension_id + link["target_id"] + link["relation"])[:16],
                        "src": comprehension_id,
                        "dst": link["target_id"],
                        "relation": link["relation"],
                        "status": "active",
                        "created_by": created_by,
                    }
                )
            return
        self.store.update_node(
            op["comprehension_id"],
            {
                "semantic_content": op["semantic_content"].strip(),
                "source_entity_ids": list(op.get("source_entity_ids") or []),
                "source_triple_ids": list(op.get("source_triple_ids") or []),
                "source_passage_ids": list(op.get("source_passage_ids") or []),
                "evidence_spans": list(op.get("evidence_spans") or []),
                "reasoning_description": op.get("reasoning_description") or "",
                "chunk_text": chunk_text,
                "status": "active",
                "created_by": created_by,
            },
        )


def _chunk_text(op: dict) -> str:
    spans = "\n".join(
        f"{span.get('passage_id')}: {span.get('span')}" for span in (op.get("evidence_spans") or [])
    )
    return (
        f"{op.get('semantic_content', '').strip()}\n\nEvidence:\n{spans}\n\n"
        f"Reasoning:\n{op.get('reasoning_description') or ''}"
    )
