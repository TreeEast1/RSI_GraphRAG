#!/usr/bin/env python3
"""Score the v52 diagnostic queries and the frozen 100-question dev set.

Gold labels are read only at scoring time. Unit selection did not use them.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval_v51 import (  # noqa: E402
    build_graph,
    link_entities,
    pagerank,
    prepare_graph,
    rank_passages,
)
from rsi_v4.hipporag_adapter import load_openie_dir  # noqa: E402

from hipporag.utils.misc_utils import text_processing  # noqa: E402
from v5_mech.novelty import classify_novelty  # noqa: E402
from v5_mech.paths import ART, OPENIE, V4  # noqa: E402
from v5_mech.verifier import verify_candidate  # noqa: E402

STORE = {"EXTRACTION_REPAIR", "COMPOSITIONAL_UNDERSTANDING", "SEMANTIC_ABSTRACTION"}


def load_v52_comps():
    comps = []
    directory = ART / "runs" / "r52_units"
    for path in sorted(directory.glob("v52_*__Ccompact.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        judgments = {row.get("candidate_id"): row for row in data.get("judgments") or []}
        for update in data.get("updates") or []:
            if (update.get("operation") or "").upper() != "ADD":
                continue
            passages, triples, pids = {}, {}, []
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
                comps.append({
                    "id": f"{data.get('role')}-{update.get('candidate_id')}",
                    "text": update.get("comprehension_text") or "",
                    "kept": False,
                    "reason": base.get("failure_stage") or base.get("decision"),
                    "passages": pids,
                    "entities": update.get("linked_entity_ids") or [],
                    "value_type": "UNSUPPORTED",
                })
                continue
            flat = [triple for rows in triples.values() for triple in rows]
            novelty = classify_novelty(update.get("comprehension_text") or "", flat, " ".join(base.get("evidence_sentences") or []), len(pids))
            kept = novelty["value_type"] in STORE
            comps.append({
                "id": f"{data.get('role')}-{update.get('candidate_id')}",
                "text": update.get("comprehension_text") or "",
                "kept": kept,
                "value_type": novelty["value_type"],
                "reason": novelty["reason"],
                "passages": pids,
                "entities": [eid for eid in (update.get("linked_entity_ids") or []) if isinstance(eid, str)],
                "role": data.get("role"),
            })
    return comps


def metrics(ranked, gold):
    gold = [pid for pid in gold if pid]
    if not gold:
        return {"evaluable": False}
    gold_set = set(gold)
    return {
        "evaluable": True,
        "recall_at_1": len(gold_set & set(ranked[:1])) / len(gold_set),
        "recall_at_3": len(gold_set & set(ranked[:3])) / len(gold_set),
        "recall_at_5": len(gold_set & set(ranked[:5])) / len(gold_set),
        "complete_at_5": set(gold) <= set(ranked[:5]),
        "ranks": {pid: ranked.index(pid) + 1 if pid in ranked else None for pid in gold},
    }


def mean(rows, arm, key):
    vals = []
    for row in rows:
        cell = row.get(arm) or {}
        if cell.get("evaluable") and key in cell:
            vals.append(float(cell[key]))
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    records = load_openie_dir(str(OPENIE))
    entities, edges, entity_passages, passage_ids = build_graph(records)
    base_nodes = list(entities) + list(passage_ids)
    base_edges = set(edges)
    for eid, pids in entity_passages.items():
        for pid in pids:
            base_edges.add((eid, pid))
    comps = [comp for comp in load_v52_comps() if comp["kept"]]
    repairs = [comp for comp in comps if comp["value_type"] == "EXTRACTION_REPAIR"]
    compositional = [comp for comp in comps if comp["value_type"] != "EXTRACTION_REPAIR"]

    def pack(selected, rewire=False):
        import random
        nodes = list(base_nodes)
        extra = []
        rng = random.Random(20261030)
        pool = list(passage_ids)
        for comp in selected:
            nodes.append(comp["id"])
            targets = comp["passages"]
            if rewire:
                targets = [rng.choice(pool) for _ in comp["passages"]] or [rng.choice(pool)]
            for eid in comp["entities"]:
                if eid in entities:
                    extra.append((comp["id"], eid))
            for pid in targets:
                if pid in passage_ids:
                    extra.append((comp["id"], pid))
        return prepare_graph(nodes, base_edges, extra)

    prepared = {
        "R0": prepare_graph(base_nodes, base_edges),
        "R1": pack(repairs, False),
        "R3": pack(comps, False),
        "R4": pack(comps, True),
    }

    def score_query(query, targets):
        linked = link_entities(query, entities)
        row = {"n_linked": len(linked), "linked": linked[:6]}
        if not linked:
            row["evaluable"] = False
            row["reason"] = "no_entity_link"
            for name in ("R0", "R1", "R2", "R3", "R4"):
                row[name] = {"evaluable": False}
            return row
        r0_graph, r0_index = prepared["R0"]
        r0_scores = pagerank(r0_graph, r0_index, linked, passage_ids)
        for name in ("R0", "R1", "R3", "R4"):
            graph, index = prepared[name]
            scores = r0_scores if name == "R0" else pagerank(graph, index, linked, passage_ids)
            row[name] = metrics(rank_passages(scores, passage_ids), targets)
        bonus = defaultdict(float)
        qset = set(text_processing(query).split())
        hits = []
        for comp in comps:
            cset = set(text_processing(comp["text"]).split())
            overlap = len(qset & cset) / len(qset) if qset and cset else 0.0
            entity_hit = any(eid in linked for eid in comp["entities"])
            if overlap >= 0.2 or entity_hit:
                hits.append({"id": comp["id"], "overlap": round(overlap, 3), "type": comp["value_type"]})
            if overlap >= 0.2:
                for pid in comp["passages"]:
                    bonus[pid] += overlap
        mixed = {pid: r0_scores.get(pid, 0.0) + bonus.get(pid, 0.0) for pid in passage_ids}
        row["R2"] = metrics(rank_passages(mixed, passage_ids), targets)
        row["comp_hits"] = hits
        row["evaluable"] = True
        # propagation: max absolute score change on the target passages
        if targets and row["R3"].get("evaluable"):
            graph, index = prepared["R3"]
            scores3 = pagerank(graph, index, linked, passage_ids)
            row["target_score_delta"] = {
                pid: scores3.get(pid, 0.0) - r0_scores.get(pid, 0.0) for pid in targets if pid in passage_ids
            }
        return row

    diagnostic = json.loads((ART / "v52" / "diagnostic_queries.json").read_text(encoding="utf-8"))
    diag_rows = []
    for query in diagnostic["queries"]:
        scored = score_query(query["query"], query["support_passage_ids"])
        scored["qid"] = query["qid"]
        scored["kind"] = query["kind"]
        diag_rows.append(scored)

    dev_rows = []
    with (V4 / "data" / "questions_eval.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("split_role") != "dev":
                continue
            gold = [pid for pid in item.get("gold_passage_ids") or [] if pid in passage_ids]
            scored = score_query(item["question"], gold)
            scored["question_id"] = item["question_id"]
            scored["n_gold"] = len(item.get("gold_passage_ids") or [])
            scored["n_gold_in_corpus"] = len(gold)
            dev_rows.append(scored)

    comp_passages = {pid for comp in comps for pid in comp["passages"]}
    covered = []
    uncovered = []
    for row in dev_rows:
        # gold ids are inside the metrics ranks keys when evaluable; recover from file by re-read is heavy.
        # Use target ranks if present.
        gold_ids = set((row.get("R0") or {}).get("ranks") or {})
        (covered if gold_ids & comp_passages else uncovered).append(row)

    def table(rows):
        return {
            name: {
                "n": sum(1 for row in rows if (row.get(name) or {}).get("evaluable")),
                "recall_at_1": mean(rows, name, "recall_at_1"),
                "recall_at_3": mean(rows, name, "recall_at_3"),
                "recall_at_5": mean(rows, name, "recall_at_5"),
                "complete_at_5": mean(rows, name, "complete_at_5"),
            }
            for name in ("R0", "R1", "R2", "R3", "R4")
        }

    # All-100 table counts unlinkable as zero recall.
    def table_all(rows):
        out = {}
        for name in ("R0", "R1", "R2", "R3", "R4"):
            recalls = {key: [] for key in ("recall_at_1", "recall_at_3", "recall_at_5", "complete_at_5")}
            for row in rows:
                cell = row.get(name) or {}
                if cell.get("evaluable"):
                    for key in recalls:
                        recalls[key].append(float(cell[key]))
                else:
                    for key in recalls:
                        recalls[key].append(0.0)
            out[name] = {key: (sum(vals) / len(vals) if vals else None) for key, vals in recalls.items()}
            out[name]["n"] = len(rows)
        return out

    changed = []
    for row in dev_rows:
        if not row.get("evaluable"):
            continue
        if row["R0"]["recall_at_5"] != row["R3"]["recall_at_5"] or row["R0"]["recall_at_1"] != row["R3"]["recall_at_1"]:
            changed.append({
                "question_id": row["question_id"],
                "r0": row["R0"]["recall_at_5"],
                "r3": row["R3"]["recall_at_5"],
                "r0_at_1": row["R0"]["recall_at_1"],
                "r3_at_1": row["R3"]["recall_at_1"],
            })
    kind_tables = {}
    for kind in sorted({row["kind"] for row in diag_rows}):
        kind_tables[kind] = table([row for row in diag_rows if row["kind"] == kind])
    payload = {
        "n_comps_kept": len(comps),
        "comp_types": {},
        "kept_texts": [{"id": comp["id"], "type": comp["value_type"], "text": comp["text"], "passages": comp["passages"]} for comp in comps],
        "diagnostic": {
            "by_kind": kind_tables,
            "rows": diag_rows,
        },
        "dev_all_100_unlinkable_as_zero": table_all(dev_rows),
        "dev_linked_only": table([row for row in dev_rows if row.get("evaluable")]),
        "n_dev": len(dev_rows),
        "n_linked": sum(1 for row in dev_rows if row.get("evaluable")),
        "n_unlinked": sum(1 for row in dev_rows if not row.get("evaluable")),
        "n_dev_gold_intersects_comp_passages": len(covered),
        "covered_linked": table([row for row in covered if row.get("evaluable")]),
        "uncovered_linked": table([row for row in uncovered if row.get("evaluable")]),
        "r0_r3_changes": changed,
        "n_repairs": len(repairs),
        "n_compositional_or_abstract": len(compositional),
    }
    for comp in comps:
        payload["comp_types"][comp["value_type"]] = payload["comp_types"].get(comp["value_type"], 0) + 1
    out = ART / "v52" / "eval.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "kept": payload["n_comps_kept"],
        "types": payload["comp_types"],
        "diag_kinds": {kind: kind_tables[kind]["R0"]["recall_at_5"] for kind in kind_tables},
        "dev100_r0": payload["dev_all_100_unlinkable_as_zero"]["R0"],
        "dev100_r3": payload["dev_all_100_unlinkable_as_zero"]["R3"],
        "linked_r0": payload["dev_linked_only"]["R0"],
        "linked_r3": payload["dev_linked_only"]["R3"],
        "changes": len(changed),
        "gold_overlap_questions": payload["n_dev_gold_intersects_comp_passages"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
