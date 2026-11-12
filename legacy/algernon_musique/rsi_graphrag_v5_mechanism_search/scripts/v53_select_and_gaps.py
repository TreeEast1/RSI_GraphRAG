#!/usr/bin/env python3
"""Pick unused units by graph structure and list missing cross-entity links.

No questions, gold answers, or model outputs are read.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
V4 = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/experiments/rsi_graphrag_v4")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V4 / "src"))
sys.path.insert(0, "/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")

from hipporag.utils.misc_utils import compute_mdhash_id  # noqa: E402
from rsi_v4.graph_analyzer import analyze_openie_records  # noqa: E402
from rsi_v4.graph_partitioner import format_unit_input, partition_cluster  # noqa: E402
from rsi_v4.hipporag_adapter import entity_id_for, load_openie_dir, processed_triple, triple_id_for  # noqa: E402

from v5_mech.paths import ART, OPENIE  # noqa: E402

USED = {
    "cc-00029", "cc-00172", "cc-00083", "cc-00646", "cc-00402", "cc-00107",
    "cc-00000-stable-0059", "cc-00000-stable-0074", "cc-00000-stable-0026",
    "cc-00000-stable-0068", "cc-00000-stable-0112", "cc-00000-stable-0099",
    "cc-00000-stable-0103", "cc-00000-stable-0089", "cc-00000-stable-0110",
    "cc-00000-stable-0046", "cc-00000-stable-0108", "cc-00000-stable-0051",
}


def build_slice(passage_ids, by_id):
    nodes, triples, passages = {}, {}, []
    entity_passages = defaultdict(set)
    for pid in sorted(passage_ids):
        record = by_id.get(pid)
        if record is None:
            continue
        passages.append({"passage_id": pid, "content": record.get("content") or ""})
        for raw in record.get("triples") or []:
            processed = processed_triple(raw)
            if processed is None:
                continue
            subject, predicate, obj = processed
            sid, oid = entity_id_for(subject), entity_id_for(obj)
            nodes[sid] = subject
            nodes[oid] = obj
            entity_passages[sid].add(pid)
            entity_passages[oid].add(pid)
            tid = triple_id_for((subject, predicate, obj))
            row = triples.setdefault(tid, {
                "triple_id": tid, "s": subject, "p": predicate, "o": obj,
                "s_id": sid, "o_id": oid, "passage_ids": [],
            })
            if pid not in row["passage_ids"]:
                row["passage_ids"].append(pid)
    return nodes, triples, passages, entity_passages


def gaps_for(nodes, triples, entity_passages):
    neighbors = defaultdict(set)
    direct = set()
    labeled = defaultdict(list)
    for triple in triples.values():
        a, b = triple["s_id"], triple["o_id"]
        direct.add(tuple(sorted((a, b))))
        if a != b:
            neighbors[a].add(b)
            neighbors[b].add(a)
        labeled[(a, b)].append(triple)
        labeled[(b, a)].append(triple)
    found = []
    ids = sorted(nodes)
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            if tuple(sorted((left, right))) in direct:
                continue
            shared = neighbors[left] & neighbors[right]
            left_pass = entity_passages[left]
            right_pass = entity_passages[right]
            if not left_pass or not right_pass or left_pass == right_pass and not shared:
                continue
            cross = not left_pass <= right_pass and not right_pass <= left_pass and left_pass != right_pass
            if not shared and not cross:
                continue
            if not shared and cross:
                # Same unit, different passages, no common neighbor and no edge: weak co-presence.
                if len(found) > 12:
                    continue
            if not shared:
                continue
            mid = sorted(shared)[0]
            path = []
            for src, dst in ((left, mid), (mid, right)):
                options = labeled.get((src, dst)) or []
                if options:
                    path.append(f"{nodes[src]} | {options[0]['p']} | {nodes[dst]}")
            found.append({
                "entities": [
                    {"id": left, "name": nodes[left]},
                    {"id": mid, "name": nodes[mid]},
                    {"id": right, "name": nodes[right]},
                ],
                "passages": sorted(left_pass | right_pass | entity_passages[mid]),
                "cross_passage": len(left_pass | right_pass) > 1,
                "path": path,
            })
    found.sort(key=lambda row: (not row["cross_passage"], row["entities"][0]["name"], row["entities"][2]["name"]))
    kept = []
    for index, row in enumerate(found[:4], start=1):
        row["gap_id"] = f"g{index}"
        kept.append(row)
    return kept


def format_gap_input(cluster_id, gaps, passages, triples):
    lines = [f"GAPS FOR {cluster_id}"]
    for gap in gaps:
        lines.append(f"GAP {gap['gap_id']} cross_passage={gap['cross_passage']}")
        lines.append("ENTITIES")
        for entity in gap["entities"]:
            lines.append(f"{entity['id']}\t{entity['name']}")
        lines.append("PATH")
        for hop in gap["path"]:
            lines.append(hop)
        lines.append("PASSAGES")
        for pid in gap["passages"]:
            lines.append(f"PASSAGE {pid}")
            lines.append(passages[pid])
    lines.append("TRIPLES")
    for triple in triples.values():
        lines.append(f"{triple['triple_id']}\t{triple['s']} | {triple['p']} | {triple['o']}")
    return "\n".join(lines)


def main() -> None:
    print("loading", flush=True)
    records = load_openie_dir(str(OPENIE))
    encoder = tiktoken.get_encoding("o200k_base")
    by_id = {}
    token_by_passage = {}
    for record in records:
        pid = record.get("passage_id") or compute_mdhash_id(record["content"], prefix="chunk-")
        record = dict(record)
        record["passage_id"] = pid
        by_id[pid] = record
        token_by_passage[pid] = len(encoder.encode(record.get("content") or ""))
    stats = json.loads((ART / "units" / "leiden_stable_stats.json").read_text(encoding="utf-8"))
    leiden = [
        row for row in stats
        if row["cluster_id"] not in USED and 2 <= row["n_passages"] <= 4 and row["n_nodes"] <= 50 and row["estimated_tokens"] <= 1200
    ]
    bundle = analyze_openie_records(records, token_by_passage=token_by_passage, input_token_budget=16000, source_label="luna_openie_g_raw")
    components = [
        row for row in bundle["relation_components"]
        if row["cluster_id"] not in USED and row["cluster_id"] != "cc-00000"
        and 2 <= row["n_passages"] <= 4 and 6 <= row["n_nodes"] <= 40 and row["estimated_tokens"] <= 2000
    ]
    candidates = []
    for row in leiden:
        candidates.append({"cluster_id": row["cluster_id"], "passage_ids": row["passage_ids"], "source": "frozen_leiden"})
    for row in components:
        passages = set()
        for node in row["node_ids"]:
            passages.update(bundle["entity_passages"].get(node, ()))
        candidates.append({"cluster_id": row["cluster_id"], "passage_ids": sorted(passages), "source": "stable_component"})
    scored = []
    for row in candidates:
        nodes, triples, passages, entity_passages = build_slice(row["passage_ids"], by_id)
        if len(passages) < 2 or len(triples) < 4:
            continue
        gap_rows = gaps_for(nodes, triples, entity_passages)
        cross = sum(1 for gap in gap_rows if gap["cross_passage"])
        density = len(triples) / max(1, len(nodes))
        scored.append({**row, "n_nodes": len(nodes), "n_triples": len(triples), "n_passages": len(passages), "n_cross_gaps": cross, "n_gaps": len(gap_rows), "density": density})
    sparse = sorted([row for row in scored if row["n_passages"] >= 2], key=lambda row: (row["density"], row["cluster_id"]))[:2]
    linked = sorted(scored, key=lambda row: (-row["n_cross_gaps"], row["cluster_id"]))[:2]
    dense = sorted(scored, key=lambda row: (-row["density"], row["cluster_id"]))[:1]
    multi = sorted([row for row in scored if row["n_gaps"] >= 1], key=lambda row: (-row["n_gaps"], row["density"], row["cluster_id"]))[:1]
    chosen = []
    seen = set()
    for row in sparse + linked + multi + dense:
        if row["cluster_id"] in seen:
            continue
        seen.add(row["cluster_id"])
        chosen.append(row)
        if len(chosen) >= 6:
            break
    if len(chosen) < 6:
        for row in sorted(scored, key=lambda item: item["cluster_id"]):
            if row["cluster_id"] in seen:
                continue
            chosen.append(row)
            seen.add(row["cluster_id"])
            if len(chosen) >= 6:
                break
    out_dir = ART / "v53" / "units"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    prompt = (ROOT / "prompts" / "method_c_compact.txt").read_text(encoding="utf-8")
    overhead = len(encoder.encode(prompt))
    for index, row in enumerate(chosen):
        nodes, triples, passages, entity_passages = build_slice(row["passage_ids"], by_id)
        cluster = {
            "cluster_id": row["cluster_id"],
            "parent_component_id": row["source"],
            "graph_id": "v53",
            "nodes": [{"id": key, "content": nodes[key]} for key in sorted(nodes)],
            "triples": [triples[key] for key in sorted(triples)],
            "passages": passages,
        }
        units = partition_cluster(cluster, input_token_budget=16000, overlap_hops=1, count_tokens=lambda text: len(encoder.encode(text or "")), overhead_tokens=overhead)
        usable = [unit for unit in units if unit.get("passage_ids") and not unit.get("over_budget")] or units
        unit = sorted(usable, key=lambda item: (len(item["overlap_node_ids"]), len(item["core_triple_ids"])), reverse=True)[0]
        node_map = {node["id"]: node for node in cluster["nodes"]}
        triple_map = {triple["triple_id"]: triple for triple in cluster["triples"]}
        passage_map = {passage["passage_id"]: passage for passage in passages}
        text = format_unit_input(unit, node_map, triple_map, passage_map)
        gap_rows = gaps_for(nodes, triples, entity_passages)
        gap_text = format_gap_input(row["cluster_id"], gap_rows, {passage["passage_id"]: passage["content"] for passage in passages}, triples)
        role = f"v53_{index:02d}"
        payload = {
            "role": role,
            "cluster_id": row["cluster_id"],
            "parent_component_id": row["source"],
            "selection_bin": row["source"],
            "n_cross_gaps": row["n_cross_gaps"],
            "density": row["density"],
            "cache_key_hits": {},
            "unit": {
                "unit_id": unit["unit_id"],
                "n_core": len(unit["core_node_ids"]),
                "n_overlap": len(unit["overlap_node_ids"]),
                "passage_ids": unit["passage_ids"],
                "text_tokens": len(encoder.encode(text)),
                "input_tokens": unit.get("input_tokens"),
            },
            "unit_text": text,
            "gap_text": gap_text,
            "gaps": gap_rows,
            "passages": {passage["passage_id"]: passage["content"] for passage in passages if passage["passage_id"] in set(unit["passage_ids"])},
            "triples": [triple_map[tid] for tid in unit["core_triple_ids"] + unit["context_triple_ids"] if tid in triple_map],
        }
        (out_dir / f"{role}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        manifest.append({
            "role": role,
            "cluster_id": row["cluster_id"],
            "source": row["source"],
            "n_nodes": row["n_nodes"],
            "n_triples": row["n_triples"],
            "n_passages": row["n_passages"],
            "density": round(row["density"], 3),
            "n_cross_gaps": row["n_cross_gaps"],
            "n_gaps_sent": len(gap_rows),
            "text_tokens": payload["unit"]["text_tokens"],
            "passage_ids": payload["unit"]["passage_ids"],
        })
        print(role, row["cluster_id"], "density", round(row["density"], 3), "cross", row["n_cross_gaps"], "gaps", len(gap_rows), flush=True)
    (ART / "v53" / "unit_selection.json").write_text(json.dumps({
        "questions_read": False,
        "gold_read": False,
        "comprehension_outputs_read": False,
        "units": manifest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("selected", len(manifest), flush=True)


if __name__ == "__main__":
    main()
