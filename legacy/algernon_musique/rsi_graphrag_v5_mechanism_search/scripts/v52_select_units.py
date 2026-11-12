#!/usr/bin/env python3
"""Select new reading units from the frozen Leiden passage sets and stable components.

Questions, gold labels, answers, and previous comprehension outputs are not read.
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
from rsi_v4.graph_analyzer import analyze_openie_records, materialize_cluster  # noqa: E402
from rsi_v4.graph_partitioner import format_unit_input, partition_cluster  # noqa: E402
from rsi_v4.hipporag_adapter import entity_id_for, load_openie_dir, processed_triple, triple_id_for  # noqa: E402

from v5_mech.paths import ART, OPENIE  # noqa: E402

USED_CLUSTERS = {
    "cc-00029",
    "cc-00172",
    "cc-00083",
    "cc-00646",
    "cc-00000-stable-0059",
    "cc-00000-stable-0074",
    "cc-00000-stable-0026",
    "cc-00000-stable-0068",
}
SEED = 20261030


def unit_from_passages(cluster_id: str, passage_ids, records_by_id, encoder):
    nodes = {}
    triples = {}
    passages = []
    for pid in sorted(passage_ids):
        record = records_by_id.get(pid)
        if record is None:
            continue
        passages.append({
            "passage_id": pid,
            "title": record.get("title") or "",
            "text": record.get("text") or "",
            "content": record.get("content") or "",
        })
        for raw in record.get("triples") or []:
            processed = processed_triple(raw)
            if processed is None:
                continue
            subject, predicate, obj = processed
            sid = entity_id_for(subject)
            oid = entity_id_for(obj)
            nodes[sid] = {"id": sid, "content": subject}
            nodes[oid] = {"id": oid, "content": obj}
            tid = triple_id_for((subject, predicate, obj))
            row = triples.setdefault(tid, {
                "triple_id": tid,
                "s": subject,
                "p": predicate,
                "o": obj,
                "s_id": sid,
                "o_id": oid,
                "passage_ids": [],
            })
            if pid not in row["passage_ids"]:
                row["passage_ids"].append(pid)
    cluster = {
        "cluster_id": cluster_id,
        "parent_component_id": "frozen-leiden-passages",
        "graph_id": "v52",
        "nodes": [nodes[key] for key in sorted(nodes)],
        "triples": [triples[key] for key in sorted(triples)],
        "passages": passages,
    }
    prompt = (ROOT / "prompts" / "method_c_compact.txt").read_text(encoding="utf-8")
    overhead = len(encoder.encode(prompt))
    units = partition_cluster(
        cluster,
        input_token_budget=16000,
        overlap_hops=1,
        count_tokens=lambda text: len(encoder.encode(text or "")),
        overhead_tokens=overhead,
    )
    usable = [unit for unit in units if unit.get("passage_ids") and not unit.get("over_budget")]
    if not usable:
        usable = units
    # Structural choice: the unit with the most overlap, then the most core triples.
    chosen = sorted(usable, key=lambda unit: (len(unit["overlap_node_ids"]), len(unit["core_triple_ids"])), reverse=True)[0]
    node_map = {node["id"]: node for node in cluster["nodes"]}
    triple_map = {triple["triple_id"]: triple for triple in cluster["triples"]}
    passage_map = {passage["passage_id"]: passage for passage in cluster["passages"]}
    text = format_unit_input(chosen, node_map, triple_map, passage_map)
    kept_passages = {pid: passage_map[pid]["content"] for pid in chosen["passage_ids"] if pid in passage_map}
    kept_triples = [triple_map[tid] for tid in chosen["core_triple_ids"] + chosen["context_triple_ids"] if tid in triple_map]
    return {
        "role": cluster_id,
        "cluster_id": cluster_id,
        "parent_component_id": cluster["parent_component_id"],
        "n_nodes": len(cluster["nodes"]),
        "n_passages_cluster": len(passages),
        "historical_match": None,
        "cache_key_hits": {},
        "unit": {
            "unit_id": chosen["unit_id"],
            "n_core": len(chosen["core_node_ids"]),
            "n_overlap": len(chosen["overlap_node_ids"]),
            "n_passages": len(chosen["passage_ids"]),
            "input_tokens": chosen.get("input_tokens"),
            "over_budget": chosen.get("over_budget"),
            "core_node_ids": chosen["core_node_ids"],
            "overlap_node_ids": chosen["overlap_node_ids"],
            "core_triple_ids": chosen["core_triple_ids"],
            "context_triple_ids": chosen["context_triple_ids"],
            "passage_ids": chosen["passage_ids"],
            "text_tokens": len(encoder.encode(text)),
        },
        "unit_text": text,
        "passages": kept_passages,
        "triples": kept_triples,
        "n_partition_units": len(units),
    }


def main() -> None:
    stats = json.loads((ART / "units" / "leiden_stable_stats.json").read_text(encoding="utf-8"))
    small = [
        row for row in stats
        if row["cluster_id"] not in USED_CLUSTERS
        and 2 <= row["n_passages"] <= 3
        and row["estimated_tokens"] <= 800
        and row["n_nodes"] <= 40
    ]
    small = sorted(small, key=lambda row: row["cluster_id"])
    rng = random.Random(SEED)
    picked = rng.sample(small, 8) if len(small) >= 8 else small
    print("loading openie for components", flush=True)
    records = load_openie_dir(str(OPENIE))
    by_id = {}
    encoder = tiktoken.get_encoding("o200k_base")
    token_by_passage = {}
    for record in records:
        pid = record.get("passage_id") or compute_mdhash_id(record["content"], prefix="chunk-")
        record = dict(record)
        record["passage_id"] = pid
        by_id[pid] = record
        token_by_passage[pid] = len(encoder.encode(record.get("content") or ""))
    bundle = analyze_openie_records(
        records,
        token_by_passage=token_by_passage,
        input_token_budget=16000,
        source_label="luna_openie_g_raw",
    )
    components = [
        row for row in bundle["relation_components"]
        if row["cluster_id"] not in USED_CLUSTERS
        and row["cluster_id"] != "cc-00000"
        and 2 <= row["n_passages"] <= 4
        and 6 <= row["n_nodes"] <= 30
        and 0 < row["estimated_tokens"] <= 2500
    ]
    components = sorted(components, key=lambda row: row["cluster_id"])
    picked_components = rng.sample(components, 2) if len(components) >= 2 else components
    out_dir = ART / "v52" / "units"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, row in enumerate(picked):
        built = unit_from_passages(row["cluster_id"], row["passage_ids"], by_id, encoder)
        built["role"] = f"v52_{index:02d}"
        built["selection_bin"] = "frozen_leiden_small"
        path = out_dir / f"{built['role']}.json"
        path.write_text(json.dumps(built, ensure_ascii=False), encoding="utf-8")
        manifest.append({
            "role": built["role"],
            "cluster_id": row["cluster_id"],
            "bin": "frozen_leiden_small",
            "n_nodes_community": row["n_nodes"],
            "n_passages_community": row["n_passages"],
            "estimated_tokens": row["estimated_tokens"],
            "unit_id": built["unit"]["unit_id"],
            "n_core": built["unit"]["n_core"],
            "n_overlap": built["unit"]["n_overlap"],
            "n_passages_unit": built["unit"]["n_passages"],
            "text_tokens": built["unit"]["text_tokens"],
            "passage_ids": built["unit"]["passage_ids"],
        })
        print("leiden", built["role"], row["cluster_id"], "passages", built["unit"]["n_passages"], "tokens", built["unit"]["text_tokens"], flush=True)
    for offset, row in enumerate(picked_components):
        cluster = materialize_cluster(bundle, row)
        cluster["graph_id"] = "v52"
        built = unit_from_passages(row["cluster_id"], [passage["passage_id"] for passage in cluster["passages"]], by_id, encoder)
        built["role"] = f"v52_{len(picked)+offset:02d}"
        built["selection_bin"] = "stable_component"
        (out_dir / f"{built['role']}.json").write_text(json.dumps(built, ensure_ascii=False), encoding="utf-8")
        manifest.append({
            "role": built["role"],
            "cluster_id": row["cluster_id"],
            "bin": "stable_component",
            "n_nodes_community": row["n_nodes"],
            "n_passages_community": row["n_passages"],
            "estimated_tokens": row["estimated_tokens"],
            "unit_id": built["unit"]["unit_id"],
            "n_core": built["unit"]["n_core"],
            "n_overlap": built["unit"]["n_overlap"],
            "n_passages_unit": built["unit"]["n_passages"],
            "text_tokens": built["unit"]["text_tokens"],
            "passage_ids": built["unit"]["passage_ids"],
        })
        print("component", built["role"], row["cluster_id"], "passages", built["unit"]["n_passages"], flush=True)
    payload = {
        "seed": SEED,
        "questions_read": False,
        "gold_read": False,
        "comprehension_outputs_read": False,
        "leiden_source": "artifacts/units/leiden_stable_stats.json",
        "n_units": len(manifest),
        "units": manifest,
    }
    (ART / "v52" / "unit_selection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", len(manifest), "units", flush=True)


if __name__ == "__main__":
    main()
