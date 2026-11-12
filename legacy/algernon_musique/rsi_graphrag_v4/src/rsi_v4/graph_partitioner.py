"""Graph-aware reading-unit partition with boundary overlap.

Core nodes keep a single owning unit. Overlap copies an id into another unit's
context; it does not create a second graph node.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Set

import igraph as ig


TokenCounter = Callable[[str], int]


def format_unit_input(unit: dict, nodes: Dict[str, dict], triples: Dict[str, dict], passages: Dict[str, dict]) -> str:
    lines = ["READING UNIT", unit["unit_id"], "CORE ENTITIES"]
    for node_id in unit["core_node_ids"]:
        lines.append(f"{node_id}\t{nodes[node_id]['content']}")
    lines.append("OVERLAP ENTITIES")
    for node_id in unit["overlap_node_ids"]:
        lines.append(f"{node_id}\t{nodes[node_id]['content']}")
    lines.append("CORE TRIPLES")
    for triple_id in unit["core_triple_ids"]:
        triple = triples[triple_id]
        lines.append(f"{triple_id}\t{triple['s']} | {triple['p']} | {triple['o']}")
    lines.append("BOUNDARY TRIPLES")
    for triple_id in unit["context_triple_ids"]:
        triple = triples[triple_id]
        lines.append(f"{triple_id}\t{triple['s']} | {triple['p']} | {triple['o']}")
    lines.append("PASSAGES")
    for passage_id in unit["passage_ids"]:
        passage = passages[passage_id]
        lines.append(f"PASSAGE {passage_id}")
        lines.append(passage["content"])
    return "\n".join(lines)


def _split_nodes(node_ids: Sequence[str], adjacency: Dict[str, Set[str]]) -> List[List[str]]:
    ordered = sorted(node_ids)
    if len(ordered) <= 1:
        return [ordered]
    if len(ordered) == 2:
        return [[ordered[0]], [ordered[1]]]
    index = {node: position for position, node in enumerate(ordered)}
    graph = ig.Graph(n=len(ordered), directed=False)
    edges = []
    for node in ordered:
        for other in adjacency.get(node, ()):
            if other in index and index[node] < index[other]:
                edges.append((index[node], index[other]))
    groups = None
    if edges:
        graph.add_edges(edges)
        try:
            parts = graph.community_leiden(objective_function="modularity", n_iterations=2)
            groups = [[ordered[vertex] for vertex in community] for community in parts if len(community)]
        except Exception:
            groups = None
    if not groups or len(groups) <= 1:
        return _bisect(ordered, adjacency)
    return groups


def _bisect(node_ids: Sequence[str], adjacency: Dict[str, Set[str]]) -> List[List[str]]:
    ordered = list(node_ids)
    if len(ordered) <= 1:
        return [ordered]
    degree = {node: len(adjacency.get(node, ()) & set(ordered)) for node in ordered}
    start = min(ordered, key=lambda node: (degree[node], node))
    seen = [start]
    seen_set = {start}
    frontier = [start]
    target = max(1, len(ordered) // 2)
    while frontier and len(seen) < target:
        current = frontier.pop(0)
        neighbors = sorted(adjacency.get(current, ()) & set(ordered))
        for neighbor in neighbors:
            if neighbor not in seen_set:
                seen_set.add(neighbor)
                seen.append(neighbor)
                frontier.append(neighbor)
            if len(seen) >= target:
                break
    rest = [node for node in ordered if node not in seen_set]
    if not rest:
        rest = [seen.pop()]
        seen_set.discard(rest[0])
    return [seen, rest]


def _khop(core: Set[str], adjacency: Dict[str, Set[str]], hops: int, universe: Set[str]) -> Dict[str, int]:
    distances = {}
    frontier = set(core)
    visited = set(core)
    for hop in range(1, hops + 1):
        nxt = set()
        for node in frontier:
            nxt |= adjacency.get(node, set())
        nxt = (nxt & universe) - visited
        for node in nxt:
            distances[node] = hop
        visited |= nxt
        frontier = nxt
        if not frontier:
            break
    return distances


def partition_cluster(
    cluster: dict,
    *,
    input_token_budget: int,
    overlap_hops: int,
    count_tokens: TokenCounter,
    overhead_tokens: int = 0,
) -> List[dict]:
    nodes = {node["id"]: node for node in cluster["nodes"]}
    triples = {triple["triple_id"]: triple for triple in cluster["triples"]}
    passages = {passage["passage_id"]: passage for passage in cluster["passages"]}
    universe = set(nodes)
    adjacency = defaultdict(set)
    for triple in triples.values():
        if triple["s_id"] in universe and triple["o_id"] in universe and triple["s_id"] != triple["o_id"]:
            adjacency[triple["s_id"]].add(triple["o_id"])
            adjacency[triple["o_id"]].add(triple["s_id"])

    def measure(core_ids: Sequence[str], overlap_ids: Sequence[str]) -> int:
        draft = _draft_unit(cluster, core_ids, overlap_ids, triples, adjacency)
        text = format_unit_input(draft, nodes, triples, passages)
        return overhead_tokens + count_tokens(text)

    cores: List[List[str]] = []

    def rec(node_ids: List[str], depth: int) -> None:
        node_ids = sorted(set(node_ids))
        if not node_ids:
            return
        if len(node_ids) == 1 or depth >= 24 or measure(node_ids, []) <= input_token_budget:
            cores.append(node_ids)
            return
        parts = _split_nodes(node_ids, adjacency)
        if len(parts) <= 1 or any(len(part) == len(node_ids) for part in parts):
            parts = _bisect(node_ids, adjacency)
        if len(parts) <= 1 or any(len(part) == len(node_ids) for part in parts):
            cores.append(node_ids)
            return
        for part in parts:
            rec(part, depth + 1)

    rec(sorted(nodes), 0)
    units = []
    for index, core_ids in enumerate(cores):
        distances = _khop(set(core_ids), adjacency, overlap_hops, universe) if overlap_hops else {}
        overlap_ids = sorted(distances)
        while overlap_ids and measure(core_ids, overlap_ids) > input_token_budget:
            farthest = max(overlap_ids, key=lambda node: (distances[node], node))
            overlap_ids.remove(farthest)
        unit = _draft_unit(cluster, core_ids, overlap_ids, triples, adjacency)
        unit["unit_id"] = f"{cluster['cluster_id']}-u{index:03d}"
        unit["input_tokens"] = measure(core_ids, unit["overlap_node_ids"])
        unit["over_budget"] = unit["input_tokens"] > input_token_budget
        unit["overlap_hops"] = overlap_hops
        units.append(unit)
    return units


def _draft_unit(cluster, core_ids, overlap_ids, triples, adjacency) -> dict:
    core = set(core_ids)
    overlap = set(overlap_ids) - core
    core_triples = []
    context_triples = []
    passage_ids = set()
    for triple in triples.values():
        ends = {triple["s_id"], triple["o_id"]}
        if ends <= core:
            core_triples.append(triple["triple_id"])
            passage_ids.update(triple["passage_ids"])
        elif ends & core and ends & overlap:
            context_triples.append(triple["triple_id"])
            passage_ids.update(triple["passage_ids"])
        elif ends <= overlap and any(adjacency[node] & core for node in ends):
            context_triples.append(triple["triple_id"])
            passage_ids.update(triple["passage_ids"])
    return {
        "unit_id": cluster["cluster_id"],
        "parent_cluster_id": cluster["cluster_id"],
        "parent_component_id": cluster["parent_component_id"],
        "graph_id": cluster.get("graph_id", ""),
        "core_node_ids": sorted(core),
        "overlap_node_ids": sorted(overlap),
        "core_triple_ids": sorted(core_triples),
        "context_triple_ids": sorted(context_triples),
        "passage_ids": sorted(passage_ids),
    }


def audit_partition(cluster: dict, units: Sequence[dict], input_token_budget: int) -> dict:
    cores = [set(unit["core_node_ids"]) for unit in units]
    covered = set()
    overlap_cores = []
    for core in cores:
        overlap_cores.append(covered & core)
        covered |= core
    cluster_nodes = {node["id"] for node in cluster["nodes"]}
    triple_map = {triple["triple_id"]: triple for triple in cluster["triples"]}
    passage_ids = {passage["passage_id"] for passage in cluster["passages"]}
    missing_core = sorted(cluster_nodes - covered)
    cross_edges = []
    owner = {}
    for unit in units:
        for node_id in unit["core_node_ids"]:
            owner[node_id] = unit["unit_id"]
    retained = 0
    for triple in cluster["triples"]:
        left, right = triple["s_id"], triple["o_id"]
        if left not in owner or right not in owner or owner[left] == owner[right]:
            continue
        cross_edges.append(triple["triple_id"])
        left_unit = next(unit for unit in units if unit["unit_id"] == owner[left])
        right_unit = next(unit for unit in units if unit["unit_id"] == owner[right])
        if right in left_unit["overlap_node_ids"] or left in right_unit["overlap_node_ids"]:
            retained += 1
    invalid_passages = []
    lost_triples = []
    for unit in units:
        for passage_id in unit["passage_ids"]:
            if passage_id not in passage_ids:
                invalid_passages.append(passage_id)
        for triple_id in unit["core_triple_ids"] + unit["context_triple_ids"]:
            if triple_id not in triple_map:
                lost_triples.append(triple_id)
    over = [unit["unit_id"] for unit in units if unit.get("input_tokens", 0) > input_token_budget and not unit.get("over_budget")]
    flagged = [unit["unit_id"] for unit in units if unit.get("over_budget")]
    overlap_mentions = sum(len(unit["overlap_node_ids"]) for unit in units)
    core_mentions = sum(len(unit["core_node_ids"]) for unit in units) or 1
    return {
        "n_units": len(units),
        "core_coverage_complete": not missing_core and not any(overlap_cores),
        "missing_core_nodes": missing_core,
        "core_intersections": [sorted(items) for items in overlap_cores if items],
        "invalid_passages": invalid_passages,
        "lost_triple_ids": lost_triples,
        "silent_over_budget_units": over,
        "flagged_over_budget_units": flagged,
        "max_input_tokens": max((unit.get("input_tokens", 0) for unit in units), default=0),
        "mean_input_tokens": (
            sum(unit.get("input_tokens", 0) for unit in units) / len(units) if units else 0
        ),
        "overlap_mention_ratio": overlap_mentions / core_mentions,
        "cross_unit_edges": len(cross_edges),
        "boundary_edges_retained": retained,
        "boundary_retention_rate": (retained / len(cross_edges)) if cross_edges else 1.0,
        "graph_ids": sorted({unit.get("graph_id", "") for unit in units}),
    }


def fallback_passage_units(passages: Sequence[dict], count_tokens: TokenCounter, budget: int, graph_id: str) -> List[dict]:
    units = []
    bucket = []
    bucket_tokens = 0
    for passage in passages:
        tokens = count_tokens(passage["content"]) + 16
        if bucket and bucket_tokens + tokens > budget:
            units.append(_fallback(bucket, graph_id, len(units), bucket_tokens, budget))
            bucket = []
            bucket_tokens = 0
        bucket.append(passage)
        bucket_tokens += tokens
    if bucket:
        units.append(_fallback(bucket, graph_id, len(units), bucket_tokens, budget))
    return units


def _fallback(passages: Sequence[dict], graph_id: str, index: int, tokens: int, budget: int) -> dict:
    return {
        "unit_id": f"{graph_id}-fallback-{index:04d}",
        "parent_cluster_id": "uncovered_passages",
        "parent_component_id": "uncovered_passages",
        "graph_id": graph_id,
        "core_node_ids": [],
        "overlap_node_ids": [],
        "core_triple_ids": [],
        "context_triple_ids": [],
        "passage_ids": [passage["passage_id"] for passage in passages],
        "input_tokens": tokens,
        "over_budget": tokens > budget,
        "overlap_hops": 0,
        "kind": "uncovered_passage",
    }
