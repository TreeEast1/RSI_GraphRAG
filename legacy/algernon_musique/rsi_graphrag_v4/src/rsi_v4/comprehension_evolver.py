"""Read-time comprehension over one reading unit."""

from __future__ import annotations

from pathlib import Path

from .graph_partitioner import format_unit_input
from .hipporag_adapter import extract_json_object

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "comprehension_system.txt"
SUMMARY_PATH = Path(__file__).resolve().parents[2] / "prompts" / "summary_control_system.txt"


def evolve_unit(client, cluster: dict, unit: dict, manager, *, mode: str = "comprehension", reasoning_effort: str = "medium") -> dict:
    if unit.get("over_budget"):
        return {"unit_id": unit.get("unit_id"), "status": "skipped_over_budget", "accepted": [], "rejected": []}
    nodes = {node["id"]: node for node in cluster["nodes"]}
    triples = {triple["triple_id"]: triple for triple in cluster["triples"]}
    passages = {passage["passage_id"]: passage for passage in cluster["passages"]}
    missing = [pid for pid in unit["passage_ids"] if pid not in passages]
    if missing:
        return {
            "unit_id": unit.get("unit_id"),
            "status": "skipped_missing_passages",
            "missing": missing,
            "accepted": [],
            "rejected": [],
        }
    system = (SUMMARY_PATH if mode == "summary" else PROMPT_PATH).read_text(encoding="utf-8")
    user = format_unit_input(unit, nodes, triples, passages)
    related = []
    core = set(unit["core_node_ids"])
    for node in manager.store.active_nodes():
        if core.intersection(node.get("source_entity_ids") or []):
            related.append({"id": node["comprehension_id"], "content": node["semantic_content"]})
    if related:
        user += "\nEXISTING COMPREHENSION\n" + "\n".join(
            f"{item['id']}\t{item['content']}" for item in related[:8]
        )
    response = client.complete(
        stage="read_time_summary" if mode == "summary" else "read_time_comprehension",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_completion_tokens=4000,
        reasoning_effort=reasoning_effort,
    )
    try:
        payload = extract_json_object(response.text)
        ops = payload.get("operations") or []
    except Exception as exc:  # noqa: BLE001
        return {
            "unit_id": unit.get("unit_id"),
            "status": "parse_error",
            "error": str(exc)[:300],
            "accepted": [],
            "rejected": [],
        }
    for op in ops:
        op["primary_unit_id"] = unit.get("unit_id")
        if mode == "summary":
            op["node_type"] = "summary_control"
    accepted, rejected = manager.validate_and_apply(ops, created_by="read_time_" + mode)
    return {
        "unit_id": unit.get("unit_id"),
        "status": "completed",
        "mode": mode,
        "accepted": accepted,
        "rejected": rejected,
        "response_model": response.metadata.get("response_model"),
        "prompt_tokens": response.metadata.get("prompt_tokens"),
        "completion_tokens": response.metadata.get("completion_tokens"),
    }
