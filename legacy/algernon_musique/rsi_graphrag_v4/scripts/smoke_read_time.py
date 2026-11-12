#!/usr/bin/env python3
"""Smoke-test read-time comprehension on three structure-selected clusters.

Clusters are chosen from the Luna relation graph. Questions are not read.
Question-time updates are not run. At most one reading unit per cluster is sent
to Luna, and the fresh-call cap is 20.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")

from rsi_v4.comprehension_store import ComprehensionStore  # noqa: E402
from rsi_v4.experiment_runner import _safe, _write_json, resolve_luna_key  # noqa: E402
from rsi_v4.graph_analyzer import _leiden_parts, analyze_openie_records, materialize_cluster  # noqa: E402
from rsi_v4.graph_partitioner import audit_partition, format_unit_input, partition_cluster  # noqa: E402
from rsi_v4.graph_patch_manager import GraphPatchManager  # noqa: E402
from rsi_v4.hashing import file_sha256, sha256_obj  # noqa: E402
from rsi_v4.hipporag_adapter import build_raw_graph, extract_json_object, load_openie_dir  # noqa: E402
from rsi_v4.llm_client import BudgetExceeded, LLMClient, UsageLedger  # noqa: E402

FRESH_CAP = 20
PROMPT_A = ROOT / "prompts" / "comprehension_protocol_a.txt"
PROMPT_B = ROOT / "prompts" / "comprehension_protocol_b.txt"
PROMPT_V = ROOT / "prompts" / "comprehension_verifier.txt"


def community_stats(node_ids, bundle, token_by_passage):
    passages = set()
    for node in node_ids:
        passages.update(bundle["entity_passages"].get(node, ()))
    tokens = sum(token_by_passage.get(pid, 0) for pid in passages)
    triple_count = 0
    node_set = set(node_ids)
    for triple in bundle["triples"].values():
        if triple["s_id"] in node_set and triple["o_id"] in node_set:
            triple_count += 1
    return {
        "n_nodes": len(node_ids),
        "n_passages": len(passages),
        "estimated_tokens": tokens,
        "n_triples": triple_count,
        "node_ids": list(node_ids),
    }


def choose_clusters(bundle):
    encoder = tiktoken.get_encoding("o200k_base")
    token_by_passage = {
        pid: len(encoder.encode(row["content"]))
        for pid, row in bundle["passage_lookup"].items()
    }
    components = bundle["relation_components"]
    largest = max(components, key=lambda row: row["n_nodes"])
    small_pool = [
        row for row in components
        if row["cluster_id"] != largest["cluster_id"]
        and 2 <= row["n_passages"] <= 4
        and 6 <= row["n_nodes"] <= 40
        and 0 < row["estimated_tokens"] <= 4000
    ]
    small_pool.sort(key=lambda row: (row["n_passages"], row["n_nodes"], row["cluster_id"]))
    if not small_pool:
        raise RuntimeError("no small component matched the structural rule")
    small = dict(small_pool[len(small_pool) // 2])
    small["role"] = "small"

    print(f"leiden on largest component {largest['n_nodes']} nodes", flush=True)
    communities = _leiden_parts(largest["node_ids"], bundle["adjacency"], 1.0)
    scored = []
    for index, community in enumerate(communities):
        stats = community_stats(community, bundle, token_by_passage)
        stats["cluster_id"] = f"{largest['cluster_id']}-leiden-{index:04d}"
        stats["parent_component_id"] = largest["cluster_id"]
        scored.append(stats)
    medium_pool = [
        row for row in scored
        if 5 <= row["n_passages"] <= 20 and 4000 < row["estimated_tokens"] <= 14000 and row["n_triples"] >= 8
    ]
    medium_pool.sort(key=lambda row: (row["estimated_tokens"], row["cluster_id"]))
    if not medium_pool:
        medium_pool = [
            row for row in scored
            if row["n_passages"] >= 3 and row["n_triples"] >= 8 and row["estimated_tokens"] <= 14000
        ]
        medium_pool.sort(key=lambda row: (row["estimated_tokens"], row["cluster_id"]))
    if not medium_pool:
        raise RuntimeError("no medium community matched the structural rule")
    medium = dict(medium_pool[len(medium_pool) // 2])
    medium["role"] = "medium"

    large_pool = [
        row for row in scored
        if row["estimated_tokens"] > 16000 and row["n_nodes"] <= 800 and row["n_passages"] >= 8
    ]
    large_pool.sort(key=lambda row: (row["estimated_tokens"], row["cluster_id"]))
    if large_pool:
        large = dict(large_pool[len(large_pool) // 2])
    else:
        capped = [row for row in scored if row["n_nodes"] <= 500 and row["n_passages"] >= 8]
        large = dict(max(capped or scored, key=lambda row: row["estimated_tokens"]))
    large["role"] = "large"
    return [small, medium, large], {
        "n_leiden_communities": len(scored),
        "leiden_size_max": max(row["n_nodes"] for row in scored),
        "communities_over_16000_tokens": sum(1 for row in scored if row["estimated_tokens"] > 16000),
    }


def pick_unit(cluster, encoder, overhead):
    units = partition_cluster(
        cluster,
        input_token_budget=16000,
        overlap_hops=1,
        count_tokens=lambda text: len(encoder.encode(text or "")),
        overhead_tokens=overhead,
    )
    audit = audit_partition(cluster, units, 16000)
    usable = [unit for unit in units if not unit.get("over_budget") and unit["passage_ids"]]
    if not usable:
        return None, units, audit
    usable.sort(key=lambda unit: (len(unit["overlap_node_ids"]), len(unit["core_triple_ids"])), reverse=True)
    return usable[0], units, audit


def candidate_to_op(update):
    operation = (update.get("operation") or update.get("op") or "").upper()
    if operation == "NO_OP":
        return {"op": "NO_OP"}, ""
    if update.get("derivation_type") == "uncertain_hypothesis":
        return None, "uncertain_hypothesis_not_committable"
    links = []
    relation = (update.get("new_semantic_relation") or "").strip() or "related to"
    for entity_id in update.get("linked_entity_ids") or []:
        links.append({"target_id": entity_id, "relation": relation})
    return {
        "op": operation or "ADD",
        "semantic_content": update.get("comprehension_text") or "",
        "source_entity_ids": list(update.get("linked_entity_ids") or []),
        "source_triple_ids": list(update.get("source_triple_ids") or []),
        "source_passage_ids": list(update.get("source_passage_ids") or []),
        "evidence_spans": list(update.get("evidence_spans") or []),
        "reasoning_description": update.get("reasoning_summary") or "",
        "why_retrieval_useful": update.get("novelty_justification") or "",
        "node_type": update.get("derivation_type") or "derived",
        "links": links[:4],
        "candidate_id": update.get("candidate_id"),
    }, ""


def verify(client, unit_text, updates, effort):
    if not updates:
        return []
    payload = json.dumps(updates, ensure_ascii=False)
    response = client.complete(
        stage="smoke_verify",
        messages=[
            {"role": "system", "content": PROMPT_V.read_text(encoding="utf-8")},
            {"role": "user", "content": "UNIT AND TRIPLES\n" + unit_text[:12000] + "\n\nCANDIDATES\n" + payload},
        ],
        max_completion_tokens=2500,
        reasoning_effort=effort,
    )
    try:
        parsed = extract_json_object(response.text)
    except Exception as exc:  # noqa: BLE001
        return [{"candidate_id": item.get("candidate_id"), "decision": "UNCERTAIN", "reason": f"verifier_parse_error: {exc}"} for item in updates]
    return parsed.get("judgments") or []


def run_protocol(client, cluster, unit, graph, protocol, system_prompt, effort):
    nodes = {node["id"]: node for node in cluster["nodes"]}
    triples = {triple["triple_id"]: triple for triple in cluster["triples"]}
    passages = {passage["passage_id"]: passage for passage in cluster["passages"]}
    unit_text = format_unit_input(unit, nodes, triples, passages)
    response = client.complete(
        stage=f"smoke_{protocol}",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": unit_text},
        ],
        max_completion_tokens=3500,
        reasoning_effort=effort,
    )
    try:
        parsed = extract_json_object(response.text)
        updates = parsed.get("updates") or parsed.get("operations") or []
    except Exception as exc:  # noqa: BLE001
        return {"protocol": protocol, "status": "parse_error", "error": str(exc)[:300], "accepted": 0, "rejected": 0, "uncertain": 0}
    store = ComprehensionStore()
    manager = GraphPatchManager(graph, store)
    raw_hash = graph["graph_hash"]
    judgments = verify(client, unit_text, [item for item in updates if (item.get("operation") or "").upper() == "ADD"], effort)
    by_id = {item.get("candidate_id"): item for item in judgments}
    accepted_rows = []
    rejected_rows = []
    uncertain_rows = []
    for update in updates:
        op, blocked = candidate_to_op(update)
        if blocked:
            uncertain_rows.append({**update, "reason": blocked})
            continue
        if op["op"] == "NO_OP":
            continue
        judgment = by_id.get(update.get("candidate_id")) or {}
        decision = (judgment.get("decision") or "UNCERTAIN").upper()
        if decision != "ACCEPT":
            uncertain_rows.append({**update, "verifier": judgment, "reason": "verifier_" + decision.lower()})
            continue
        taken, refused = manager.validate_and_apply([op], created_by=f"smoke_{protocol}")
        if taken:
            accepted_rows.append({"update": update, "verifier": judgment, "stored": taken[0]})
        else:
            rejected_rows.append({"update": update, "verifier": judgment, "reason": refused[0].get("reason") if refused else "rejected"})
    snapshot = store.snapshot()
    if store.active_nodes():
        store.deactivate(store.active_nodes()[0]["comprehension_id"], "smoke_rollback_check")
        rolled = len(store.active_nodes()) == len(snapshot["nodes"]) - 1 or store.active_nodes()[0]["status"] != "active"
        store.restore(snapshot)
        rollback_ok = len(store.active_nodes()) == len(snapshot["nodes"]) and store.active_nodes()[0]["status"] == "active" if store.active_nodes() else True
    else:
        rolled = True
        rollback_ok = True
    rejections = []
    for row in (rejected_rows + uncertain_rows)[:3]:
        update = row.get("update") or row
        rejections.append({
            "text": update.get("comprehension_text"),
            "reason": row.get("reason"),
            "verifier": row.get("verifier"),
        })
    return {
        "protocol": protocol,
        "status": "completed",
        "n_updates": len(updates),
        "accepted": len(accepted_rows),
        "rejected": len(rejected_rows),
        "uncertain": len(uncertain_rows),
        "raw_hash_unchanged": graph["graph_hash"] == raw_hash,
        "rollback_ok": rollback_ok and rolled,
        "active_nodes": len(store.active_nodes()),
        "examples": [
            {
                "text": row["update"].get("comprehension_text"),
                "derivation_type": row["update"].get("derivation_type"),
                "evidence": row["update"].get("evidence_spans"),
                "verifier": row["verifier"],
            }
            for row in accepted_rows[:2]
        ],
        "rejections": rejections,
        "prompt_tokens": response.metadata.get("prompt_tokens"),
        "completion_tokens": response.metadata.get("completion_tokens"),
    }


def main() -> None:
    config = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    art = ROOT / "artifacts"
    out = art / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    openie_dir = art / "openie" / "luna"
    print("loading openie", flush=True)
    records = load_openie_dir(str(openie_dir))
    encoder = tiktoken.get_encoding("o200k_base")
    from hipporag.utils.misc_utils import compute_mdhash_id

    token_by_passage = {
        compute_mdhash_id(record["content"], prefix="chunk-"): len(encoder.encode(record["content"]))
        for record in records
    }
    print("analyzing relation graph", len(records), flush=True)
    bundle = analyze_openie_records(
        records,
        token_by_passage=token_by_passage,
        input_token_budget=16000,
        source_label="luna_openie_g_raw",
    )
    summary = bundle["summary"]
    _write_json(art / "luna_graph_stats.json", summary)
    print(json.dumps({k: summary[k] for k in ("n_relation_nodes", "n_triples", "relation_component_count", "relation_largest_component_nodes", "relation_largest_component_ratio")}, ensure_ascii=False), flush=True)
    chosen, leiden_meta = choose_clusters(bundle)
    overhead = len(encoder.encode(PROMPT_A.read_text(encoding="utf-8")))
    secret, auth = resolve_luna_key(config, art)
    if not secret:
        raise SystemExit(f"Luna auth failed: {auth}")
    ledger = UsageLedger(max_fresh_calls=FRESH_CAP)
    client = LLMClient(
        cache_path=str(art / "llm_cache.sqlite"),
        deployment=config["llm"]["deployment"],
        endpoint=config["llm"]["endpoint"],
        api_version=config["llm"]["api_version"],
        api_key=secret,
        ledger=ledger,
        seed=config["llm"]["seed"],
    )
    results = []
    for spec in chosen:
        cluster = materialize_cluster(bundle, spec)
        cluster["graph_id"] = "luna-smoke"
        file_hashes = []
        for passage in cluster["passages"]:
            path = openie_dir / f"{passage['passage_id']}.json"
            if path.exists():
                file_hashes.append((passage["passage_id"], file_sha256(str(path))))
        unit, units, audit = pick_unit(cluster, encoder, overhead)
        graph = build_raw_graph(
            [record for record in records if record["passage_id"] in {passage["passage_id"] for passage in cluster["passages"]}],
            status="smoke_cluster_view",
            model_name=config["llm"]["deployment"],
            corpus_hash=summary.get("source_label", ""),
        )
        before_hash = graph["graph_hash"]
        row = {
            "role": spec["role"],
            "cluster_id": spec["cluster_id"],
            "n_nodes": spec["n_nodes"],
            "n_passages": spec["n_passages"],
            "estimated_tokens": spec["estimated_tokens"],
            "n_triples": len(cluster["triples"]),
            "n_units": audit["n_units"],
            "core_coverage_complete": audit["core_coverage_complete"],
            "boundary_retention_rate": audit["boundary_retention_rate"],
            "cross_unit_edges": audit["cross_unit_edges"],
            "selected_unit": None if unit is None else {
                "unit_id": unit["unit_id"],
                "n_core": len(unit["core_node_ids"]),
                "n_overlap": len(unit["overlap_node_ids"]),
                "input_tokens": unit["input_tokens"],
                "over_budget": unit["over_budget"],
            },
        }
        print("cluster", spec["role"], spec["cluster_id"], "units", audit["n_units"], "passages", spec["n_passages"], flush=True)
        if unit is None:
            row["protocols"] = []
            row["skipped"] = "no unit under budget"
        else:
            protocols = []
            for name, prompt in (("A", PROMPT_A), ("B", PROMPT_B)):
                if ledger.fresh_calls >= FRESH_CAP - 2:
                    protocols.append({"protocol": name, "status": "stopped_budget"})
                    break
                try:
                    outcome = run_protocol(
                        client, cluster, unit, graph, name, prompt.read_text(encoding="utf-8"),
                        config["llm"]["comprehension_reasoning_effort"],
                    )
                except BudgetExceeded:
                    outcome = {"protocol": name, "status": "stopped_budget"}
                except Exception as exc:  # noqa: BLE001
                    outcome = {"protocol": name, "status": "error", "error": _safe(exc, secret)}
                protocols.append(outcome)
                print(" protocol", name, outcome.get("status"), "accepted", outcome.get("accepted"), flush=True)
            row["protocols"] = protocols
        after_files = []
        for passage_id, digest in file_hashes:
            after_files.append(file_sha256(str(openie_dir / f"{passage_id}.json")) == digest)
        row["openie_files_unchanged"] = all(after_files) if after_files else False
        row["cluster_graph_hash_unchanged"] = graph["graph_hash"] == before_hash
        results.append(row)
    summary_out = {
        "graph": {key: summary[key] for key in summary if key != "top_predicates"},
        "top_predicates": summary["top_predicates"][:15],
        "leiden": leiden_meta,
        "selection_rule": "median structural component/community; questions were not loaded",
        "fresh_cap": FRESH_CAP,
        "ledger": ledger.snapshot(),
        "response_model": auth.get("response_model"),
        "clusters": results,
        "question_time": "not_run",
    }
    _write_json(out / "smoke_summary.json", summary_out)
    print("smoke done", ledger.fresh_calls, flush=True)


if __name__ == "__main__":
    main()
