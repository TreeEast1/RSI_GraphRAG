#!/usr/bin/env python3
"""Full-corpus graph retrieval on frozen Luna G_raw.

Passage NV-Embed vectors exist for the MuSiQue chunks. Entity and fact
vectors in case_study_musique belong to the gpt-4o-mini OpenIE graph, not
Luna G_raw, so they are not used as the graph. Ranking here is personalized
PageRank over the Luna relation graph. A comprehension node changes the
ranking only when its edges are in that graph. Dev questions are the fixed
split; gold is used after ranking.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import igraph as ig

ROOT = Path(__file__).resolve().parents[1]
V4 = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/experiments/rsi_graphrag_v4")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V4 / "src"))
sys.path.insert(0, "/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")

from hipporag.utils.misc_utils import text_processing  # noqa: E402
from rsi_v4.hipporag_adapter import load_openie_dir  # noqa: E402

from v5_mech.novelty import classify_novelty  # noqa: E402
from v5_mech.paths import ART, OPENIE, V4 as V4_PATH  # noqa: E402
from v5_mech.verifier import verify_candidate  # noqa: E402

STORE_TYPES = {"EXTRACTION_REPAIR", "COMPOSITIONAL_UNDERSTANDING", "SEMANTIC_ABSTRACTION"}
SEED = 20261030


def build_graph(records):
    entities = {}
    edges = set()
    entity_passages = defaultdict(set)
    passage_ids = set()
    from rsi_v4.hipporag_adapter import entity_id_for, processed_triple

    for record in records:
        pid = record.get("passage_id")
        if not pid:
            from hipporag.utils.misc_utils import compute_mdhash_id
            pid = compute_mdhash_id(record["content"], prefix="chunk-")
        passage_ids.add(pid)
        for raw in record.get("triples") or []:
            processed = processed_triple(raw)
            if processed is None:
                continue
            subject, predicate, obj = processed
            sid = entity_id_for(subject)
            oid = entity_id_for(obj)
            entities[sid] = subject
            entities[oid] = obj
            if sid != oid:
                edges.add(tuple(sorted((sid, oid))))
            entity_passages[sid].add(pid)
            entity_passages[oid].add(pid)
    return entities, edges, entity_passages, passage_ids


def load_comps():
    comps = []
    for folder in ("r1_hist", "r3_new_ac", "r51_effort_low"):
        directory = ART / "runs" / folder
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*__C.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            judgments = {row.get("candidate_id"): row for row in data.get("judgments") or []}
            for update in data.get("updates") or []:
                if (update.get("operation") or "").upper() != "ADD":
                    continue
                if (judgments.get(update.get("candidate_id")) or {}).get("decision") != "ACCEPT":
                    continue
                pids = []
                passages = {}
                triples = {}
                for span in update.get("evidence_spans") or []:
                    pid = span.get("passage_id")
                    if not pid or pid in passages:
                        continue
                    file = OPENIE / f"{pid}.json"
                    if not file.exists():
                        continue
                    record = json.loads(file.read_text(encoding="utf-8"))
                    passages[pid] = record.get("content") or ""
                    triples[pid] = record.get("triples") or []
                    pids.append(pid)
                if not passages:
                    continue
                base = verify_candidate(update, passages, triples)
                if base.get("faithfulness") != "supported":
                    continue
                flat = [triple for rows in triples.values() for triple in rows]
                evidence = " ".join(base.get("evidence_sentences") or [])
                novelty = classify_novelty(update.get("comprehension_text") or "", flat, evidence, len(pids))
                if novelty["value_type"] not in STORE_TYPES:
                    continue
                comps.append({
                    "id": f"{data.get('role')}-{update.get('candidate_id')}",
                    "text": update.get("comprehension_text") or "",
                    "value_type": novelty["value_type"],
                    "passages": pids,
                    "entities": [eid for eid in (update.get("linked_entity_ids") or []) if isinstance(eid, str)],
                    "role": data.get("role"),
                })
    return comps


def dev_questions():
    rows = []
    path = V4_PATH / "data" / "questions_eval.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("split_role") != "dev":
                continue
            rows.append({
                "question_id": item["question_id"],
                "question": item.get("question") or "",
                "gold": list(item.get("gold_passage_ids") or []),
            })
    return rows


def link_entities(question: str, entities: dict) -> list:
    lowered = question.lower()
    found = []
    for eid, name in entities.items():
        surface = (name or "").strip()
        if len(surface) < 8 or " " not in surface:
            continue
        if surface.lower() in lowered:
            found.append(eid)
    return found


def prepare_graph(nodes, edges, extra_edges=()):
    index = {node: i for i, node in enumerate(nodes)}
    graph = ig.Graph(n=len(nodes), directed=False)
    pairs = []
    seen = set()
    for left, right in list(edges) + list(extra_edges):
        if left not in index or right not in index or left == right:
            continue
        pair = (index[left], index[right]) if index[left] < index[right] else (index[right], index[left])
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    if pairs:
        graph.add_edges(pairs)
    return graph, index


def pagerank(graph, index, reset, passage_ids):
    reset_ids = [index[node] for node in reset if node in index]
    if not reset_ids:
        return {}
    weights = [0.0] * graph.vcount()
    share = 1.0 / len(reset_ids)
    for vid in reset_ids:
        weights[vid] = share
    scores = graph.personalized_pagerank(damping=0.5, reset=weights, implementation="prpack")
    return {pid: scores[index[pid]] for pid in passage_ids if pid in index}


def rank_passages(ppr, passage_ids):
    return sorted(passage_ids, key=lambda pid: (-ppr.get(pid, 0.0), pid))


def metrics(ranked, gold):
    if not gold:
        return {"evaluable": False}
    gold_set = set(gold)
    return {
        "evaluable": True,
        "recall_at_1": len(gold_set & set(ranked[:1])) / len(gold_set),
        "recall_at_3": len(gold_set & set(ranked[:3])) / len(gold_set),
        "recall_at_5": len(gold_set & set(ranked[:5])) / len(gold_set),
        "complete_at_5": gold_set <= set(ranked[:5]),
    }


def mean(rows, key):
    vals = [row[key] for row in rows if row.get("evaluable")]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    print("loading openie", flush=True)
    records = load_openie_dir(str(OPENIE))
    entities, edges, entity_passages, passage_ids = build_graph(records)
    print("graph", len(entities), "entities", len(edges), "edges", len(passage_ids), "passages", flush=True)
    comps = load_comps()
    print("stored comps", len(comps), flush=True)
    questions = dev_questions()
    print("dev", len(questions), flush=True)
    base_nodes = list(entities) + list(passage_ids)
    base_edges = set(edges)
    for eid, pids in entity_passages.items():
        for pid in pids:
            base_edges.add((eid, pid))
    def arm_edges(selected, rewire: bool):
        extra = []
        nodes = list(base_nodes)
        rng = random.Random(SEED)
        pool = list(passage_ids)
        for comp in selected:
            nodes.append(comp["id"])
            targets = comp["passages"]
            if rewire and pool:
                targets = [rng.choice(pool) for _ in comp["passages"]]
            for eid in comp["entities"]:
                if eid in entities:
                    extra.append((comp["id"], eid))
            for pid in targets:
                if pid in passage_ids:
                    extra.append((comp["id"], pid))
        return nodes, extra

    repairs = [comp for comp in comps if comp["value_type"] == "EXTRACTION_REPAIR"]
    print("preparing graphs", len(repairs), "repairs", flush=True)

    def pack(selected, rewire: bool):
        nodes, extra = arm_edges(selected, rewire)
        return prepare_graph(nodes, base_edges, extra)

    prepared = {
        "R0": prepare_graph(base_nodes, base_edges),
        "R1": pack(repairs, False),
        "R3": pack(comps, False),
        "R3_control": pack(comps, True),
    }
    summaries = {}
    per_question = []
    linked_count = 0
    outside = 0
    for question in questions:
        linked = link_entities(question["question"], entities)
        gold = [pid for pid in question["gold"] if pid in passage_ids]
        missing = [pid for pid in question["gold"] if pid not in passage_ids]
        if missing:
            outside += 1
        if linked:
            linked_count += 1
        row = {"question_id": question["question_id"], "n_linked": len(linked), "n_gold": len(question["gold"]), "n_gold_in_corpus": len(gold)}
        if not linked or not gold:
            row["evaluable"] = False
            row["reason"] = "no_entity_link" if not linked else "gold_not_in_corpus"
            per_question.append(row)
            continue
        r0_graph, r0_index = prepared["R0"]
        r0_scores = pagerank(r0_graph, r0_index, linked, passage_ids)
        for name in ("R0", "R1", "R3", "R3_control"):
            graph, index = prepared[name]
            scores = r0_scores if name == "R0" else pagerank(graph, index, linked, passage_ids)
            row[name] = metrics(rank_passages(scores, passage_ids), gold)
        bonus = {pid: 0.0 for pid in passage_ids}
        qset = set(text_processing(question["question"]).split())
        hit_comps = []
        for comp in comps:
            cset = set(text_processing(comp["text"]).split())
            if not qset or not cset:
                continue
            overlap = len(qset & cset) / len(qset)
            if overlap < 0.25 and not any(eid in linked for eid in comp["entities"]):
                continue
            if overlap >= 0.25 or any(eid in linked for eid in comp["entities"]):
                hit_comps.append(comp["id"])
            if overlap < 0.25:
                continue
            for pid in comp["passages"]:
                if pid in bonus:
                    bonus[pid] += overlap
        if any(bonus.values()):
            mixed = {pid: r0_scores.get(pid, 0.0) + bonus[pid] for pid in passage_ids}
            row["R2"] = metrics(rank_passages(mixed, passage_ids), gold)
        else:
            row["R2"] = row["R0"]
        row["comp_hits"] = hit_comps
        row["evaluable"] = True
        per_question.append(row)
    evaluated = [row for row in per_question if row.get("evaluable")]
    for name in ("R0", "R1", "R2", "R3", "R3_control"):
        summaries[name] = {
            "n": len(evaluated),
            "recall_at_1": mean([row[name] for row in evaluated], "recall_at_1"),
            "recall_at_3": mean([row[name] for row in evaluated], "recall_at_3"),
            "recall_at_5": mean([row[name] for row in evaluated], "recall_at_5"),
            "complete_at_5": mean([{**row[name], "complete_at_5": float(row[name]["complete_at_5"])} for row in evaluated], "complete_at_5"),
        }
    comp_passages = {pid for comp in comps for pid in comp["passages"]}
    gold_overlap = 0
    for question in questions:
        if comp_passages & set(question["gold"]):
            gold_overlap += 1
    payload = {
        "index_audit": {
            "luna_openie_records": len(records),
            "luna_entities": len(entities),
            "luna_relation_edges": len(edges),
            "luna_synonymy": "not_built",
            "passage_nvembed": "case_study_musique/traditional/graph/chunk_embeddings/vdb_chunk.parquet",
            "passage_nvembed_rows": 11656,
            "passage_nvembed_dim": 4096,
            "entity_fact_nvembed": "present for gpt-4o-mini OpenIE in case_study_musique, not for Luna G_raw",
            "gpu_used_for_this_run": False,
            "ranking": "personalized PageRank on Luna G_raw; dense query vectors were not computed",
        },
        "n_dev": len(questions),
        "n_entity_linked": linked_count,
        "n_evaluated": len(evaluated),
        "n_gold_outside_corpus": outside,
        "n_dev_whose_gold_intersects_comp_passages": gold_overlap,
        "n_comps": len(comps),
        "comp_types": {},
        "summaries": summaries,
    }
    for comp in comps:
        payload["comp_types"][comp["value_type"]] = payload["comp_types"].get(comp["value_type"], 0) + 1
    out = ART / "v51" / "retrieval_graph.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Drop per-question detail to a sidecar, keep it.
    (ART / "v51" / "retrieval_questions.json").write_text(json.dumps(per_question, ensure_ascii=False), encoding="utf-8")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
