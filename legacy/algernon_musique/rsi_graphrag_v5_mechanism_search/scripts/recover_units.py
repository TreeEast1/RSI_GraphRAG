#!/usr/bin/env python3
"""Rebuild reading units from the frozen Luna OpenIE graph and freeze a sample.

Questions are not loaded. G_raw files are only read. Leiden community ids are
reassigned by minimum node id so this snapshot does not depend on igraph's
return order. Historical smoke ids are matched by passage anchors and by the
LLM cache key when the formatted unit is identical.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
V4 = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/experiments/rsi_graphrag_v4")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V4 / "src"))
sys.path.insert(0, "/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")

from hipporag.utils.misc_utils import compute_mdhash_id  # noqa: E402
from rsi_v4.graph_analyzer import _leiden_parts, analyze_openie_records, materialize_cluster  # noqa: E402
from rsi_v4.graph_partitioner import format_unit_input, partition_cluster  # noqa: E402
from rsi_v4.hashing import sha256_obj  # noqa: E402
from rsi_v4.hipporag_adapter import load_openie_dir  # noqa: E402

from v5_mech.client import load_llm_config, request_payload  # noqa: E402
from v5_mech.paths import ART, OPENIE, V4_PROMPTS  # noqa: E402

ANCHOR_MEDIUM = {
    "chunk-56664711c1c4de235b820a051107fb2f",
    "chunk-1a2f9e9e4d0a9f1c7cdb872540133c08",
    "chunk-bdff4e9a69ee9bf8c09cf8574391ba71",
    "chunk-9498c40a3a45abafaadf924367b5e688",
    "chunk-ed22e931f32904f9af5775fa847dedd6",
    "chunk-c926f13fe5e0380cb9cc662ed5f246df",
    "chunk-5d3d787bca10ff123105226827a4e2e2",
}
ANCHOR_LARGE = {
    "chunk-11b375ea69e06d1107ef5ec409d4cb32",
    "chunk-c48d38af96b2654659443d02f43dc84f",
    "chunk-c30c674749748e4a1b36f06ffa6bed8a",
    "chunk-d4e0062d70b935f8f0f9e503b45bdf7e",
}
SMOKE_KEYS = {
    "07af7c5bc5163b05b479ddf7a7361a8c573a879174c8e566afe0e9660bca95e6": "small_A",
    "46fc23a6d7cdf0ec3eea0bb7fcdc33eab332adf4516bd17b135e7f5e1aed33a3": "small_B",
    "3bb8d79ac4f443505067590c9d079b0c082a4e4055dc5076aadef8614719ca05": "medium_A",
    "7c619babc06c804633ffd3f6109d666f5500cf3e4fd42ddbd81ea49a84d08554": "medium_B",
    "bc123f4abf989ecebbcf3dc0c469aa91b558f53737e9756a54fd05d68b1bfb37": "large_A",
    "52c00614c50f60fb8a8f2c1678a6edf315cad8a9a2577684bf9db1e49cb2761a": "large_B",
}


def cache_keys(unit_text: str, prompt: str, max_tokens: int, effort: str) -> dict:
    config = load_llm_config()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": unit_text},
    ]
    found = {}
    for disabled in ([], ["seed"], ["reasoning_effort"], ["seed", "reasoning_effort"]):
        payload = request_payload(config, messages, max_tokens, effort, disabled)
        key = sha256_obj(payload)
        if key in SMOKE_KEYS:
            found[SMOKE_KEYS[key]] = {"disabled_params": disabled, "key": key}
    return found


def community_stats(node_ids, bundle, token_by_passage) -> dict:
    passages = set()
    for node in node_ids:
        passages.update(bundle["entity_passages"].get(node, ()))
    tokens = sum(token_by_passage.get(pid, 0) for pid in passages)
    node_set = set(node_ids)
    triple_count = 0
    for triple in bundle["triples"].values():
        if triple["s_id"] in node_set and triple["o_id"] in node_set:
            triple_count += 1
    return {
        "n_nodes": len(node_ids),
        "n_passages": len(passages),
        "estimated_tokens": tokens,
        "n_triples": triple_count,
        "passage_ids": sorted(passages),
        "node_ids": list(node_ids),
    }


def prepare_unit(cluster, encoder, overhead: int) -> tuple:
    units = partition_cluster(
        cluster,
        input_token_budget=16000,
        overlap_hops=1,
        count_tokens=lambda text: len(encoder.encode(text or "")),
        overhead_tokens=overhead,
    )
    usable = [unit for unit in units if not unit.get("over_budget") and unit["passage_ids"]]
    nodes = {node["id"]: node for node in cluster["nodes"]}
    triples = {triple["triple_id"]: triple for triple in cluster["triples"]}
    passages = {passage["passage_id"]: passage for passage in cluster["passages"]}
    packed = []
    for unit in units:
        text = format_unit_input(unit, nodes, triples, passages)
        packed.append({
            "unit_id": unit["unit_id"],
            "n_core": len(unit["core_node_ids"]),
            "n_overlap": len(unit["overlap_node_ids"]),
            "n_passages": len(unit["passage_ids"]),
            "input_tokens": unit["input_tokens"],
            "over_budget": unit["over_budget"],
            "core_node_ids": unit["core_node_ids"],
            "overlap_node_ids": unit["overlap_node_ids"],
            "core_triple_ids": unit["core_triple_ids"],
            "context_triple_ids": unit["context_triple_ids"],
            "passage_ids": unit["passage_ids"],
            "unit_text": text,
            "text_tokens": len(encoder.encode(text)),
        })
    return packed, usable


def dump_unit(spec: dict, packed_unit: dict, cluster: dict, out_dir: Path) -> None:
    passages = {row["passage_id"]: row["content"] for row in cluster["passages"] if row["passage_id"] in set(packed_unit["passage_ids"])}
    triple_ids = set(packed_unit["core_triple_ids"]) | set(packed_unit["context_triple_ids"])
    triples = [triple for triple in cluster["triples"] if triple["triple_id"] in triple_ids]
    payload = {
        "role": spec["role"],
        "cluster_id": spec["cluster_id"],
        "parent_component_id": spec.get("parent_component_id"),
        "n_nodes": spec["n_nodes"],
        "n_passages_cluster": spec["n_passages"],
        "estimated_tokens": spec["estimated_tokens"],
        "historical_match": spec.get("historical_match"),
        "cache_key_hits": spec.get("cache_key_hits"),
        "unit": {key: value for key, value in packed_unit.items() if key != "unit_text"},
        "unit_text": packed_unit["unit_text"],
        "passages": passages,
        "triples": triples,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{spec['role']}__{packed_unit['unit_id']}.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    out = ART / "units"
    out.mkdir(parents=True, exist_ok=True)
    print("loading openie", flush=True)
    records = load_openie_dir(str(OPENIE))
    encoder = tiktoken.get_encoding("o200k_base")
    print("counting tokens", len(records), flush=True)
    token_by_passage = {
        compute_mdhash_id(record["content"], prefix="chunk-"): len(encoder.encode(record["content"]))
        for record in records
    }
    print("analyzing components", flush=True)
    bundle = analyze_openie_records(
        records,
        token_by_passage=token_by_passage,
        input_token_budget=16000,
        source_label="luna_openie_g_raw",
    )
    raw_manifest = sha256_obj(sorted(bundle["triples"]))
    (out / "g_raw_triple_manifest.sha256").write_text(raw_manifest + "\n", encoding="utf-8")
    components = bundle["relation_components"]
    largest = max(components, key=lambda row: row["n_nodes"])
    print("largest", largest["cluster_id"], largest["n_nodes"], "components", len(components), flush=True)
    small = components[29]
    print("cc-00029 check", small["cluster_id"], small["n_nodes"], small["n_passages"], flush=True)

    prompt_a = (V4_PROMPTS / "comprehension_protocol_a.txt").read_text(encoding="utf-8")
    prompt_b = (V4_PROMPTS / "comprehension_protocol_b.txt").read_text(encoding="utf-8")
    overhead = len(encoder.encode(prompt_a))
    effort = "medium"

    def score_cluster(spec):
        cluster = materialize_cluster(bundle, spec)
        cluster["graph_id"] = "v5-frozen"
        packed, usable = prepare_unit(cluster, encoder, overhead)
        return cluster, packed, usable

    historical = []
    small_spec = dict(small)
    small_spec["role"] = "hist_small"
    cluster, packed, usable = score_cluster(small_spec)
    usable_ids = {unit["unit_id"] for unit in usable}
    ready = [unit for unit in packed if unit["unit_id"] in usable_ids] or packed
    chosen = ready[0]
    hits = {}
    hits.update(cache_keys(chosen["unit_text"], prompt_a, 3500, effort))
    hits.update(cache_keys(chosen["unit_text"], prompt_b, 3500, effort))
    small_spec["historical_match"] = small["cluster_id"] == "cc-00029"
    small_spec["cache_key_hits"] = hits
    dump_unit(small_spec, chosen, cluster, out)
    historical.append({
        "role": "hist_small",
        "cluster_id": small["cluster_id"],
        "n_nodes": small["n_nodes"],
        "n_passages": small["n_passages"],
        "unit_id": chosen["unit_id"],
        "cache_key_hits": hits,
        "passage_ids": chosen["passage_ids"],
    })
    print("small cache hits", hits, flush=True)

    print("leiden on giant", flush=True)
    communities = _leiden_parts(largest["node_ids"], bundle["adjacency"], 1.0)
    # Stable ids: sort by minimum node id, not by igraph's return order.
    communities = sorted((sorted(community) for community in communities), key=lambda row: (row[0], len(row)))
    print("communities", len(communities), flush=True)
    scored = []
    for index, community in enumerate(communities):
        stats = community_stats(community, bundle, token_by_passage)
        stats["cluster_id"] = f"{largest['cluster_id']}-stable-{index:04d}"
        stats["parent_component_id"] = largest["cluster_id"]
        scored.append(stats)
    (out / "leiden_stable_stats.json").write_text(
        json.dumps(
            [{key: value for key, value in row.items() if key != "node_ids"} | {"n_node_ids": row["n_nodes"]} for row in scored],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def anchor_hits(row, anchors):
        return len(anchors & set(row["passage_ids"]))

    medium_pool = sorted(scored, key=lambda row: (anchor_hits(row, ANCHOR_MEDIUM), row["n_nodes"] == 99, row["n_passages"] == 24), reverse=True)
    large_pool = sorted(scored, key=lambda row: (anchor_hits(row, ANCHOR_LARGE), row["n_nodes"] == 670, row["n_passages"] == 196), reverse=True)
    medium = medium_pool[0]
    large = large_pool[0]
    print(
        "medium anchors", anchor_hits(medium, ANCHOR_MEDIUM), medium["n_nodes"], medium["n_passages"], medium["cluster_id"],
        "large anchors", anchor_hits(large, ANCHOR_LARGE), large["n_nodes"], large["n_passages"], large["cluster_id"],
        flush=True,
    )

    for spec, anchors, label in ((medium, ANCHOR_MEDIUM, "hist_medium"), (large, ANCHOR_LARGE, "hist_large")):
        spec = dict(spec)
        spec["role"] = label
        cluster, packed, usable = score_cluster(spec)
        # Prefer a unit whose formatted text hits the historical cache. Otherwise the smoke pick rule.
        best = None
        best_hits = {}
        usable_ids = {unit["unit_id"] for unit in usable}
        ready = [unit for unit in packed if unit["unit_id"] in usable_ids and not unit["over_budget"]] or packed
        for unit in ready:
            hits = {}
            hits.update(cache_keys(unit["unit_text"], prompt_a, 3500, effort))
            hits.update(cache_keys(unit["unit_text"], prompt_b, 3500, effort))
            if hits and (best is None or len(hits) > len(best_hits)):
                best = unit
                best_hits = hits
        if best is None:
            best = max(
                ready,
                key=lambda unit: (
                    len(set(unit["passage_ids"]) & anchors),
                    unit["n_overlap"],
                    len(unit["core_triple_ids"]),
                ),
            )
            best_hits = {}
            best_hits.update(cache_keys(best["unit_text"], prompt_a, 3500, effort))
            best_hits.update(cache_keys(best["unit_text"], prompt_b, 3500, effort))
        spec["cache_key_hits"] = best_hits
        spec["historical_match"] = {
            "anchor_passages": anchor_hits(spec, anchors),
            "anchor_total": len(anchors),
            "n_nodes": spec["n_nodes"],
            "n_passages": spec["n_passages"],
            "exact_cache": bool(best_hits),
            "unit_anchor_passages": len(set(best["passage_ids"]) & anchors),
            "unit_passages": len(best["passage_ids"]),
        }
        dump_unit(spec, best, cluster, out)
        historical.append({
            "role": label,
            "cluster_id": spec["cluster_id"],
            "n_nodes": spec["n_nodes"],
            "n_passages": spec["n_passages"],
            "estimated_tokens": spec["estimated_tokens"],
            "unit_id": best["unit_id"],
            "n_core": best["n_core"],
            "n_overlap": best["n_overlap"],
            "cache_key_hits": best_hits,
            "anchors": spec["historical_match"],
        })
        print(
            label, "hits", best_hits, "core", best["n_core"], "overlap", best["n_overlap"],
            "unit_anchors", len(set(best["passage_ids"]) & anchors),
            flush=True,
        )

    historical_passages = set()
    for row in historical:
        # passage ids are inside the dumped file; collect anchors so new units avoid them
        historical_passages |= ANCHOR_MEDIUM | ANCHOR_LARGE
        historical_passages.update(row.get("passage_ids") or [])

    rng = random.Random(20261030)
    non_giant = [row for row in components if row["cluster_id"] != largest["cluster_id"] and row["cluster_id"] != "cc-00029"]
    small_pool = [
        row for row in non_giant
        if 6 <= row["n_nodes"] <= 40 and 2 <= row["n_passages"] <= 4 and 0 < row["estimated_tokens"] <= 4000
    ]
    small_pool = sorted(small_pool, key=lambda row: row["cluster_id"])
    picked_small = rng.sample(small_pool, 3) if len(small_pool) >= 3 else small_pool

    fresh_communities = [row for row in scored if not (set(row["passage_ids"]) & (ANCHOR_MEDIUM | ANCHOR_LARGE))]
    medium_new = [
        row for row in fresh_communities
        if 20 <= row["n_nodes"] <= 120 and 4 <= row["n_passages"] <= 15 and 1500 <= row["estimated_tokens"] <= 12000
    ]
    medium_new = sorted(medium_new, key=lambda row: row["cluster_id"])
    picked_medium = rng.sample(medium_new, 3) if len(medium_new) >= 3 else medium_new
    large_new = [
        row for row in fresh_communities
        if row["estimated_tokens"] > 16000 and 80 <= row["n_nodes"] <= 700 and row["n_passages"] >= 8
    ]
    large_new = sorted(large_new, key=lambda row: row["cluster_id"])
    picked_large = rng.sample(large_new, 1) if large_new else []
    single_extra = [
        row for row in fresh_communities
        if 8 <= row["n_passages"] <= 30 and 4000 < row["estimated_tokens"] <= 14000 and row["n_triples"] >= 8
        and row["cluster_id"] not in {item["cluster_id"] for item in picked_medium}
    ]
    single_extra = sorted(single_extra, key=lambda row: row["cluster_id"])
    picked_extra = rng.sample(single_extra, 1) if single_extra else []

    new_rows = []
    for offset, spec in enumerate(picked_small + picked_medium + picked_extra):
        spec = dict(spec)
        spec["role"] = f"new_{offset:02d}"
        cluster, packed, usable = score_cluster(spec)
        usable_ids = {unit["unit_id"] for unit in usable}
        ready = [unit for unit in packed if unit["unit_id"] in usable_ids] or packed
        chosen = sorted(ready, key=lambda unit: (unit["n_overlap"], unit["n_core"]))[0]
        spec["cache_key_hits"] = {}
        spec["historical_match"] = None
        dump_unit(spec, chosen, cluster, out)
        new_rows.append({
            "role": spec["role"],
            "cluster_id": spec["cluster_id"],
            "n_nodes": spec["n_nodes"],
            "n_passages": spec["n_passages"],
            "estimated_tokens": spec["estimated_tokens"],
            "n_triples": spec.get("n_triples"),
            "unit_id": chosen["unit_id"],
            "n_core": chosen["n_core"],
            "n_overlap": chosen["n_overlap"],
            "has_overlap": chosen["n_overlap"] > 0,
        })
        print("selected", spec["role"], spec["cluster_id"], "nodes", spec["n_nodes"], "passages", spec["n_passages"], flush=True)
    for spec in picked_large:
        spec = dict(spec)
        spec["role"] = "new_overlap"
        cluster, packed, usable = score_cluster(spec)
        usable_ids = {unit["unit_id"] for unit in usable}
        ready = [unit for unit in packed if unit["unit_id"] in usable_ids] or packed
        with_overlap = [unit for unit in ready if unit["n_overlap"] > 0]
        chosen = sorted(with_overlap or ready, key=lambda unit: unit["n_overlap"], reverse=True)[0]
        spec["cache_key_hits"] = {}
        dump_unit(spec, chosen, cluster, out)
        new_rows.append({
            "role": spec["role"],
            "cluster_id": spec["cluster_id"],
            "n_nodes": spec["n_nodes"],
            "n_passages": spec["n_passages"],
            "estimated_tokens": spec["estimated_tokens"],
            "unit_id": chosen["unit_id"],
            "n_core": chosen["n_core"],
            "n_overlap": chosen["n_overlap"],
            "has_overlap": chosen["n_overlap"] > 0,
            "n_units": len(packed),
        })
        print("selected overlap", chosen["unit_id"], "overlap", chosen["n_overlap"], "units", len(packed), flush=True)

    manifest = {
        "g_raw_triple_manifest": raw_manifest,
        "seed": 20261030,
        "leiden_order": "sorted by minimum node id; not the historical Leiden index",
        "n_communities": len(scored),
        "historical": historical,
        "new_units": new_rows,
        "questions_used_for_selection": False,
    }
    (out / "selection.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out / "selection.json", flush=True)


if __name__ == "__main__":
    main()
