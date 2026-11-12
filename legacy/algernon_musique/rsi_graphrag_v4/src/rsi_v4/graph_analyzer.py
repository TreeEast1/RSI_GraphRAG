"""Relation-graph and full-graph statistics, plus Leiden communities.

Synonymy is not invented here. A full-graph view in this module is relation
edges plus passage-context edges. Historical gpt-4o-mini OpenIE can be
analysed, but those numbers are not a Luna graph.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import igraph as ig

HIPPO_SRC = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")
if str(HIPPO_SRC) not in sys.path:
    sys.path.insert(0, str(HIPPO_SRC))

from hipporag.utils.misc_utils import compute_mdhash_id, text_processing  # noqa: E402

from .hipporag_adapter import entity_id_for, processed_triple, triple_id_for


def _find(parent: Dict[str, str], node: str) -> str:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _union(parent: Dict[str, str], left: str, right: str) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def _histogram(sizes: Sequence[int]) -> Dict[str, int]:
    buckets = {"1": 0, "2-5": 0, "6-20": 0, "21-100": 0, "101-1000": 0, "1000+": 0}
    for size in sizes:
        if size <= 1:
            buckets["1"] += 1
        elif size <= 5:
            buckets["2-5"] += 1
        elif size <= 20:
            buckets["6-20"] += 1
        elif size <= 100:
            buckets["21-100"] += 1
        elif size <= 1000:
            buckets["101-1000"] += 1
        else:
            buckets["1000+"] += 1
    return buckets


def _components(nodes: Sequence[str], edges: Sequence[tuple]) -> List[List[str]]:
    parent = {node: node for node in nodes}
    for left, right in edges:
        if left not in parent:
            parent[left] = left
        if right not in parent:
            parent[right] = right
        _union(parent, left, right)
    groups = defaultdict(list)
    for node in parent:
        groups[_find(parent, node)].append(node)
    return list(groups.values())


def _leiden_parts(node_ids: Sequence[str], adjacency: Dict[str, set], resolution: float) -> List[List[str]]:
    ordered = sorted(node_ids)
    if len(ordered) <= 2:
        return [ordered]
    index = {node: position for position, node in enumerate(ordered)}
    graph = ig.Graph(n=len(ordered), directed=False)
    edges = []
    for node in ordered:
        for other in adjacency.get(node, ()):
            if other in index and index[node] < index[other]:
                edges.append((index[node], index[other]))
    if not edges:
        return [[node] for node in ordered]
    graph.add_edges(edges)
    try:
        parts = graph.community_leiden(
            objective_function="modularity",
            resolution=resolution,
            n_iterations=2,
        )
    except TypeError:
        parts = graph.community_leiden(objective_function="modularity", n_iterations=2)
    groups = [[ordered[vertex] for vertex in community] for community in parts if len(community)]
    return groups or [ordered]


def analyze_openie_records(
    records: Sequence[dict],
    *,
    token_by_passage: Dict[str, int],
    input_token_budget: int,
    leiden_resolution: float = 1.0,
    source_label: str,
) -> dict:
    relation_nodes = set()
    relation_edges = set()
    full_nodes = set()
    full_edges = set()
    adjacency = defaultdict(set)
    entity_passages = defaultdict(set)
    passage_entities = defaultdict(set)
    triples = {}
    predicate_counts = Counter()
    ner_entities = set()
    n_records = 0
    idx_mismatches = 0
    empty_triple_passages = []
    for record in records:
        n_records += 1
        content = record["content"]
        recomputed = compute_mdhash_id(content, prefix="chunk-")
        if record.get("official_idx") and record["official_idx"] != recomputed:
            idx_mismatches += 1
        pid = recomputed
        full_nodes.add(pid)
        valid = []
        for raw in record.get("triples") or []:
            processed = processed_triple(raw)
            if processed is not None:
                valid.append(processed)
        if not valid:
            empty_triple_passages.append(pid)
        for subject, predicate, obj in valid:
            sid = entity_id_for(subject)
            oid = entity_id_for(obj)
            relation_nodes.add(sid)
            relation_nodes.add(oid)
            full_nodes.add(sid)
            full_nodes.add(oid)
            if sid != oid:
                relation_edges.add(tuple(sorted((sid, oid))))
                adjacency[sid].add(oid)
                adjacency[oid].add(sid)
                full_edges.add(tuple(sorted((sid, oid))))
            full_edges.add((pid, sid))
            full_edges.add((pid, oid))
            entity_passages[sid].add(pid)
            entity_passages[oid].add(pid)
            passage_entities[pid].add(sid)
            passage_entities[pid].add(oid)
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
            predicate_counts[predicate] += 1
        for surface in record.get("entities") or []:
            processed = text_processing(surface)
            if processed:
                ner_entities.add(entity_id_for(processed))

    relation_components = _components(sorted(relation_nodes), list(relation_edges))
    full_components = _components(sorted(full_nodes), [(a, b) for a, b in full_edges])
    relation_sizes = sorted((len(component) for component in relation_components), reverse=True)
    full_sizes = sorted((len(component) for component in full_components), reverse=True)
    isolated_ner = ner_entities - relation_nodes
    max_relation = relation_sizes[0] if relation_sizes else 0
    n_relation_nodes = len(relation_nodes) or 1

    clusters = []
    skipped_large_leiden = 0
    leiden_errors = 0
    for component_index, component in enumerate(relation_components):
        passages = set()
        for node in component:
            passages.update(entity_passages.get(node, ()))
        estimated = sum(token_by_passage.get(pid, 0) for pid in passages)
        parent_id = f"cc-{component_index:05d}"
        communities = [component]
        if estimated > input_token_budget and 8 < len(component) <= 8000:
            try:
                communities = _leiden_parts(component, adjacency, leiden_resolution)
            except Exception:
                leiden_errors += 1
                communities = [component]
        elif estimated > input_token_budget and len(component) > 8000:
            skipped_large_leiden += 1
        for community_index, community in enumerate(communities):
            community_passages = set()
            for node in community:
                community_passages.update(entity_passages.get(node, ()))
            community_tokens = sum(token_by_passage.get(pid, 0) for pid in community_passages)
            clusters.append(
                {
                    "cluster_id": f"{parent_id}-k{community_index:03d}",
                    "parent_component_id": parent_id,
                    "n_nodes": len(community),
                    "n_passages": len(community_passages),
                    "estimated_tokens": community_tokens,
                    "node_ids": community,
                }
            )

    passage_lookup = {
        compute_mdhash_id(record["content"], prefix="chunk-"): {
            "passage_id": compute_mdhash_id(record["content"], prefix="chunk-"),
            "title": record.get("title", ""),
            "text": record.get("text", ""),
            "content": record["content"],
        }
        for record in records
    }
    relation_component_rows = []
    for index, component in enumerate(relation_components):
        passages = set()
        for node in component:
            passages.update(entity_passages.get(node, ()))
        estimated = sum(token_by_passage.get(pid, 0) for pid in passages)
        relation_component_rows.append(
            {
                "cluster_id": f"cc-{index:05d}",
                "parent_component_id": f"cc-{index:05d}",
                "n_nodes": len(component),
                "n_passages": len(passages),
                "estimated_tokens": estimated,
                "node_ids": component,
            }
        )
    summary = {
        "source_label": source_label,
        "n_records": n_records,
        "official_idx_mismatches": idx_mismatches,
        "n_relation_nodes": len(relation_nodes),
        "n_relation_edges": len(relation_edges),
        "n_ner_entities": len(ner_entities),
        "n_isolated_ner_entities": len(isolated_ner),
        "isolated_ner_ratio": (len(isolated_ner) / len(ner_entities)) if ner_entities else 0.0,
        "n_triples": len(triples),
        "n_uncovered_passages": len(empty_triple_passages),
        "uncovered_passage_ratio": len(empty_triple_passages) / n_records if n_records else 0.0,
        "relation_component_count": len(relation_components),
        "relation_component_size_histogram": _histogram(relation_sizes),
        "relation_largest_component_ratio": max_relation / n_relation_nodes,
        "relation_largest_component_nodes": max_relation,
        "full_graph_definition": "relation edges plus passage-context edges; synonymy not included",
        "full_component_count": len(full_components),
        "full_component_size_histogram": _histogram(full_sizes),
        "full_largest_component_nodes": full_sizes[0] if full_sizes else 0,
        "top_predicates": predicate_counts.most_common(30),
        "n_clusters_after_leiden": len(clusters),
        "clusters_over_budget": sum(1 for cluster in clusters if cluster["estimated_tokens"] > input_token_budget),
        "leiden_skipped_components_over_8000_nodes": skipped_large_leiden,
        "leiden_errors": leiden_errors,
    }
    bundle = {
        "summary": summary,
        "clusters": clusters,
        "relation_components": relation_component_rows,
        "adjacency": adjacency,
        "triples": triples,
        "entity_passages": entity_passages,
        "passage_lookup": passage_lookup,
    }
    return bundle


def materialize_cluster(bundle: dict, cluster: dict) -> dict:
    node_ids = set(cluster["node_ids"])
    nodes = []
    seen_content = {}
    triples = []
    passage_ids = set()
    for triple in bundle["triples"].values():
        if triple["s_id"] in node_ids or triple["o_id"] in node_ids:
            if triple["s_id"] in node_ids and triple["o_id"] in node_ids:
                triples.append(triple)
                passage_ids.update(triple["passage_ids"])
                seen_content[triple["s_id"]] = triple["s"]
                seen_content[triple["o_id"]] = triple["o"]
    for node_id in sorted(node_ids):
        nodes.append({"id": node_id, "content": seen_content.get(node_id, node_id)})
        passage_ids.update(bundle["entity_passages"].get(node_id, ()))
    passages = [bundle["passage_lookup"][pid] for pid in sorted(passage_ids) if pid in bundle["passage_lookup"]]
    return {
        "cluster_id": cluster["cluster_id"],
        "parent_component_id": cluster["parent_component_id"],
        "nodes": nodes,
        "triples": triples,
        "passages": passages,
    }


def select_representative_clusters(clusters: Sequence[dict], n: int = 25) -> List[dict]:
    eligible = [cluster for cluster in clusters if cluster["n_nodes"] >= 4 and cluster["n_passages"] > 0]
    if not eligible:
        eligible = list(clusters)
    eligible = sorted(eligible, key=lambda cluster: (cluster["n_nodes"], cluster["cluster_id"]))
    if len(eligible) <= n:
        return eligible
    small = eligible[:8]
    large = eligible[-9:]
    middle_start = max(0, len(eligible) // 2 - 4)
    middle = eligible[middle_start : middle_start + 8]
    chosen = {}
    for cluster in small + middle + large:
        chosen[cluster["cluster_id"]] = cluster
    ordered = list(chosen.values())
    if len(ordered) < n:
        for cluster in eligible:
            if cluster["cluster_id"] not in chosen:
                chosen[cluster["cluster_id"]] = cluster
                ordered.append(cluster)
            if len(ordered) >= n:
                break
    return ordered[:n]
