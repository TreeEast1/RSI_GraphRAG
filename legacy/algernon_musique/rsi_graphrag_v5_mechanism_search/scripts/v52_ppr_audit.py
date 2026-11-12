#!/usr/bin/env python3
"""Audit whether stored comprehension nodes actually move Personalized PageRank.

The controlled bridge is an engineering probe. It is not a MuSiQue result.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval_v51 import (  # noqa: E402
    build_graph,
    load_comps,
    pagerank,
    prepare_graph,
    rank_passages,
)
from rsi_v4.hipporag_adapter import load_openie_dir  # noqa: E402
from v5_mech.paths import ART, OPENIE, V4  # noqa: E402


def top_changed(left, right, k):
    return left[:k] != right[:k]


def main() -> None:
    records = load_openie_dir(str(OPENIE))
    entities, edges, entity_passages, passage_ids = build_graph(records)
    comps = load_comps()
    base_nodes = list(entities) + list(passage_ids)
    base_edges = set(edges)
    for eid, pids in entity_passages.items():
        for pid in pids:
            base_edges.add((eid, pid))
    graph0, index0 = prepare_graph(base_nodes, base_edges)
    extra = []
    nodes = list(base_nodes)
    matched = 0
    declared = 0
    for comp in comps:
        nodes.append(comp["id"])
        for eid in comp["entities"]:
            declared += 1
            if eid in entities:
                matched += 1
                extra.append((comp["id"], eid))
        for pid in comp["passages"]:
            if pid in passage_ids:
                extra.append((comp["id"], pid))
    graph3, index3 = prepare_graph(nodes, base_edges, extra)
    degrees = {}
    for comp in comps:
        vid = index3.get(comp["id"])
        degrees[comp["id"]] = graph3.degree(vid) if vid is not None else 0

    rows = []
    for comp in comps:
        seeds = [eid for eid in comp["entities"] if eid in entities]
        if not seeds:
            # Fall back to entities that touch the source passage.
            seeds = []
            for pid in comp["passages"]:
                for eid, pids in entity_passages.items():
                    if pid in pids:
                        seeds.append(eid)
            seeds = seeds[:4]
        if not seeds:
            rows.append({"id": comp["id"], "reachable": False, "reason": "no_seed"})
            continue
        score0 = pagerank(graph0, index0, seeds, passage_ids)
        score3 = pagerank(graph3, index3, seeds, passage_ids)
        # Comp-node mass is not in the passage dict. Recompute a one-node lookup.
        reset = [0.0] * graph3.vcount()
        ids = [index3[eid] for eid in seeds if eid in index3]
        share = 1.0 / len(ids)
        for vid in ids:
            reset[vid] = share
        full = graph3.personalized_pagerank(damping=0.5, reset=reset, implementation="prpack")
        comp_mass = full[index3[comp["id"]]] if comp["id"] in index3 else 0.0
        rank0 = rank_passages(score0, passage_ids)
        rank3 = rank_passages(score3, passage_ids)
        deltas = []
        for pid in comp["passages"]:
            if pid not in score0:
                continue
            deltas.append({
                "passage_id": pid,
                "score_r0": score0[pid],
                "score_r3": score3[pid],
                "delta": score3[pid] - score0[pid],
                "rank_r0": rank0.index(pid) + 1,
                "rank_r3": rank3.index(pid) + 1,
            })
        rows.append({
            "id": comp["id"],
            "value_type": comp["value_type"],
            "degree": degrees.get(comp["id"], 0),
            "n_declared_entities": len(comp["entities"]),
            "n_matched_seeds": len([eid for eid in comp["entities"] if eid in entities]),
            "comp_mass": comp_mass,
            "reachable": comp_mass > 0,
            "top5_changed": top_changed(rank0, rank3, 5),
            "top20_changed": top_changed(rank0, rank3, 20),
            "top50_changed": top_changed(rank0, rank3, 50),
            "source_passages": deltas,
        })

    # Controlled bridge. Pick the first evaluated dev question whose gold is outside top 20.
    questions = []
    with (V4 / "data" / "questions_eval.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("split_role") == "dev":
                questions.append(item)
    from retrieval_v51 import link_entities

    probe = None
    for item in questions:
        linked = link_entities(item["question"], entities)
        gold = [pid for pid in item.get("gold_passage_ids") or [] if pid in passage_ids]
        if not linked or not gold:
            continue
        scores = pagerank(graph0, index0, linked, passage_ids)
        ranked = rank_passages(scores, passage_ids)
        outside = [pid for pid in gold if ranked.index(pid) + 1 > 20]
        if not outside:
            continue
        target = outside[0]
        seed = linked[0]
        bridge_id = "probe-bridge"
        nodes_b = list(base_nodes) + [bridge_id]
        extra_b = [(bridge_id, seed), (bridge_id, target)]
        graph_b, index_b = prepare_graph(nodes_b, base_edges, extra_b)
        scores_b = pagerank(graph_b, index_b, linked, passage_ids)
        ranked_b = rank_passages(scores_b, passage_ids)
        rng = random.Random(20261030)
        random_target = rng.choice([pid for pid in passage_ids if pid not in gold])
        rand_id = "probe-random"
        graph_r, index_r = prepare_graph(list(base_nodes) + [rand_id], base_edges, [(rand_id, seed), (rand_id, random_target)])
        scores_r = pagerank(graph_r, index_r, linked, passage_ids)
        ranked_r = rank_passages(scores_r, passage_ids)
        probe = {
            "label": "engineering_probe_not_musique_result",
            "question_id": item["question_id"],
            "seed_entity": entities.get(seed, seed),
            "target_was_outside_top20": True,
            "rank_before": ranked.index(target) + 1,
            "rank_after_correct_bridge": ranked_b.index(target) + 1,
            "score_before": scores[target],
            "score_after_correct_bridge": scores_b[target],
            "random_passage_rank_before": ranked.index(random_target) + 1,
            "random_passage_rank_after_random_edge": ranked_r.index(random_target) + 1,
            "top5_changed_by_correct_bridge": top_changed(ranked, ranked_b, 5),
            "top20_changed_by_correct_bridge": top_changed(ranked, ranked_b, 20),
        }
        break

    summary = {
        "n_comps": len(comps),
        "declared_entity_slots": declared,
        "matched_entity_slots": matched,
        "match_rate": matched / declared if declared else 0.0,
        "comps_with_positive_mass": sum(1 for row in rows if row.get("comp_mass", 0) > 0),
        "comps_top5_changed_when_seeded_on_themselves": sum(1 for row in rows if row.get("top5_changed")),
        "comps_top20_changed": sum(1 for row in rows if row.get("top20_changed")),
        "comps_top50_changed": sum(1 for row in rows if row.get("top50_changed")),
        "mean_degree": sum(degrees.values()) / len(degrees) if degrees else 0.0,
        "probe": probe,
    }
    out = ART / "v52" / "ppr_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
