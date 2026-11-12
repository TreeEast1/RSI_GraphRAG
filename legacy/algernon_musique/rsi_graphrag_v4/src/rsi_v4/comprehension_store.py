"""Versioned comprehension layer. It never stores raw-graph nodes."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Dict, List


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ComprehensionStore:
    def __init__(self) -> None:
        self.nodes: Dict[str, dict] = {}
        self.edges: List[dict] = []
        self.version = 0
        self.events: List[dict] = []

    def snapshot(self) -> dict:
        return {
            "nodes": copy.deepcopy(self.nodes),
            "edges": copy.deepcopy(self.edges),
            "version": self.version,
            "events": copy.deepcopy(self.events),
        }

    def restore(self, snapshot: dict) -> None:
        self.nodes = copy.deepcopy(snapshot["nodes"])
        self.edges = copy.deepcopy(snapshot["edges"])
        self.version = snapshot["version"]
        self.events = copy.deepcopy(snapshot["events"])

    def active_nodes(self) -> List[dict]:
        return [node for node in self.nodes.values() if node["status"] == "active"]

    def active_edges(self) -> List[dict]:
        return [edge for edge in self.edges if edge["status"] == "active"]

    def add_node(self, node: dict) -> None:
        self.nodes[node["comprehension_id"]] = node
        self.version += 1
        self.events.append({"op": "ADD", "id": node["comprehension_id"], "version": self.version, "at": _now()})

    def update_node(self, comprehension_id: str, fields: dict) -> None:
        node = self.nodes[comprehension_id]
        node.update(fields)
        node["version"] = int(node.get("version", 1)) + 1
        node["updated_at"] = _now()
        self.version += 1
        self.events.append({"op": "UPDATE", "id": comprehension_id, "version": self.version, "at": _now()})

    def deactivate(self, comprehension_id: str, reason: str) -> None:
        node = self.nodes[comprehension_id]
        node["status"] = "inactive"
        node["deactivated_reason"] = reason
        self.version += 1
        self.events.append({"op": "DEACTIVATE", "id": comprehension_id, "version": self.version, "at": _now()})

    def add_edge(self, edge: dict) -> None:
        self.edges.append(edge)
        self.version += 1

    def to_dict(self) -> dict:
        return self.snapshot()

    def load_dict(self, payload: dict) -> None:
        self.restore(payload)
