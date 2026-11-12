#!/usr/bin/env python3
"""Leiden on the largest historical relation component. Not a Luna graph."""

import json
import sys
from collections import Counter
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")

from hipporag.utils.misc_utils import compute_mdhash_id  # noqa: E402

from rsi_v4.graph_analyzer import (  # noqa: E402
    _leiden_parts,
    analyze_openie_records,
    materialize_cluster,
    select_representative_clusters,
)
from rsi_v4.graph_partitioner import audit_partition, partition_cluster  # noqa: E402
from rsi_v4.hipporag_adapter import historical_records  # noqa: E402

CONFIG = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
OUT = ROOT / "artifacts" / "giant_leiden.json"


def main() -> None:
    encoder = tiktoken.get_encoding(CONFIG["reading"]["tokenizer"])
    records = list(historical_records(CONFIG["historical_openie"]))
    token_by_passage = {
        compute_mdhash_id(record["content"], prefix="chunk-"): len(encoder.encode(record["content"]))
        for record in records
    }
    bundle = analyze_openie_records(
        records,
        token_by_passage=token_by_passage,
        input_token_budget=CONFIG["reading"]["input_token_budget"],
        source_label="historical_gpt-4o-mini_openie_not_luna",
    )
    largest = max(bundle["relation_components"], key=lambda row: row["n_nodes"])
    print("largest", largest["cluster_id"], largest["n_nodes"], flush=True)
    communities = _leiden_parts(largest["node_ids"], bundle["adjacency"], CONFIG["reading"]["leiden_resolution"])
    sizes = sorted((len(community) for community in communities), reverse=True)
    histogram = Counter()
    for size in sizes:
        if size <= 1:
            histogram["1"] += 1
        elif size <= 5:
            histogram["2-5"] += 1
        elif size <= 20:
            histogram["6-20"] += 1
        elif size <= 100:
            histogram["21-100"] += 1
        elif size <= 1000:
            histogram["101-1000"] += 1
        else:
            histogram["1000+"] += 1
    community_rows = []
    for index, community in enumerate(communities):
        community_rows.append(
            {
                "cluster_id": f"{largest['cluster_id']}-leiden-{index:04d}",
                "parent_component_id": largest["cluster_id"],
                "n_nodes": len(community),
                "n_passages": 0,
                "estimated_tokens": 0,
                "node_ids": community,
            }
        )
    eligible = [row for row in community_rows if 20 <= row["n_nodes"] <= 400]
    chosen = select_representative_clusters(eligible, 12)
    overhead = len(encoder.encode((ROOT / "prompts" / "comprehension_system.txt").read_text(encoding="utf-8")))

    def count(text: str) -> int:
        return len(encoder.encode(text or ""))

    audits = []
    for row in chosen:
        cluster = materialize_cluster(bundle, row)
        cluster["graph_id"] = "historical-giant"
        entry = {"cluster_id": row["cluster_id"], "n_nodes": row["n_nodes"], "budgets": {}}
        for name, budget in (("budget_16000", 16000), ("stress_800", 800)):
            entry["budgets"][name] = {}
            for hops in (0, 1):
                units = partition_cluster(
                    cluster,
                    input_token_budget=budget,
                    overlap_hops=hops,
                    count_tokens=count,
                    overhead_tokens=overhead,
                )
                entry["budgets"][name][str(hops)] = audit_partition(cluster, units, budget)
        audits.append(entry)
        print("partitioned community", row["cluster_id"], row["n_nodes"], flush=True)
    payload = {
        "source_label": "historical_gpt-4o-mini_relation_graph_largest_component",
        "warning": "Not a Luna graph. Synonymy edges are absent.",
        "parent_component_nodes": largest["n_nodes"],
        "n_communities": len(communities),
        "largest_community_nodes": sizes[0] if sizes else 0,
        "largest_community_fraction_of_component": (sizes[0] / largest["n_nodes"]) if sizes and largest["n_nodes"] else 0,
        "community_size_histogram": dict(histogram),
        "n_communities_20_to_400": len(eligible),
        "partition_audits": [
            {
                "cluster_id": row["cluster_id"],
                "n_nodes": row["n_nodes"],
                "budgets": {
                    budget: {
                        hop: {
                            "n_units": audit["n_units"],
                            "core_coverage_complete": audit["core_coverage_complete"],
                            "boundary_retention_rate": audit["boundary_retention_rate"],
                            "cross_unit_edges": audit["cross_unit_edges"],
                            "max_input_tokens": audit["max_input_tokens"],
                            "overlap_mention_ratio": audit["overlap_mention_ratio"],
                            "flagged_over_budget": len(audit["flagged_over_budget_units"]),
                        }
                        for hop, audit in hops.items()
                    }
                    for budget, hops in row["budgets"].items()
                },
            }
            for row in audits
        ],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT, "communities", len(communities), "largest", sizes[0] if sizes else None)


if __name__ == "__main__":
    main()
