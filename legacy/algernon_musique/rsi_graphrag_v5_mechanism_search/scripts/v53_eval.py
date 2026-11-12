#!/usr/bin/env python3
"""Compare evidence-chain and structure-guided nodes on frozen diagnostic queries and dev questions."""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval_v51 import build_graph, link_entities, pagerank, prepare_graph, rank_passages  # noqa: E402
from hipporag.utils.misc_utils import text_processing  # noqa: E402
from rsi_v4.hipporag_adapter import load_openie_dir  # noqa: E402

from v5_mech.novelty import classify_novelty  # noqa: E402
from v5_mech.paths import ART, OPENIE, V4  # noqa: E402
from v5_mech.verifier import verify_candidate  # noqa: E402

STORE = {"EXTRACTION_REPAIR", "COMPOSITIONAL_UNDERSTANDING", "SEMANTIC_ABSTRACTION"}


def load_arm(run_id: str, suffix: str):
    kept = []
    raw_accept = 0
    proposed = 0
    for path in sorted((ART / "runs" / run_id).glob(f"v53_*__{suffix}.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        judgments = {row.get("candidate_id"): row for row in data.get("judgments") or []}
        for update in data.get("updates") or []:
            if (update.get("operation") or "").upper() != "ADD":
                continue
            proposed += 1
            if (judgments.get(update.get("candidate_id")) or {}).get("decision") != "ACCEPT":
                continue
            raw_accept += 1
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
                continue
            flat = [triple for rows in triples.values() for triple in rows]
            novelty = classify_novelty(update.get("comprehension_text") or "", flat, " ".join(base.get("evidence_sentences") or []), len(pids))
            if novelty["value_type"] not in STORE:
                continue
            kept.append({
                "id": f"{data.get('role')}-{update.get('candidate_id')}",
                "text": update.get("comprehension_text") or "",
                "value_type": novelty["value_type"],
                "passages": pids,
                "entities": [eid for eid in (update.get("linked_entity_ids") or []) if isinstance(eid, str)],
                "cross_passage": len(pids) > 1,
                "role": data.get("role"),
            })
    return {"proposed": proposed, "verifier_accept": raw_accept, "kept": kept}


def metrics(ranked, gold):
    gold = [pid for pid in gold if pid]
    if not gold:
        return {"evaluable": False}
    gold_set = set(gold)
    def rec(k):
        return len(gold_set & set(ranked[:k])) / len(gold_set)
    return {
        "evaluable": True,
        "recall_at_1": rec(1),
        "recall_at_3": rec(3),
        "recall_at_5": rec(5),
        "recall_at_20": rec(20),
        "complete_at_5": gold_set <= set(ranked[:5]),
        "ranks": {pid: (ranked.index(pid) + 1 if pid in ranked else None) for pid in gold},
    }


def pack(base_nodes, base_edges, passage_ids, entities, comps, rewire=False):
    nodes = list(base_nodes)
    extra = []
    rng = random.Random(20261030)
    pool = list(passage_ids)
    for comp in comps:
        nodes.append(comp["id"])
        targets = [rng.choice(pool) for _ in comp["passages"]] if rewire and pool else comp["passages"]
        for eid in comp["entities"]:
            if eid in entities:
                extra.append((comp["id"], eid))
        for pid in targets:
            if pid in passage_ids:
                extra.append((comp["id"], pid))
    return prepare_graph(nodes, base_edges, extra)


def score(prepared, passage_ids, entities, linked, targets):
    out = {}
    if not linked:
        for name in prepared:
            out[name] = {"evaluable": False}
        return out
    for name, (graph, index) in prepared.items():
        scores = pagerank(graph, index, linked, passage_ids)
        out[name] = metrics(rank_passages(scores, passage_ids), targets)
    return out


def mean_table(rows):
    table = {}
    for name in ("R0", "R1", "R2", "R3"):
        vals = defaultdict(list)
        for row in rows:
            cell = row.get(name) or {}
            if not cell.get("evaluable"):
                continue
            for key in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_20", "complete_at_5"):
                vals[key].append(float(cell[key]))
        table[name] = {key: (sum(v) / len(v) if v else None) for key, v in vals.items()}
        table[name]["n"] = len(vals["recall_at_5"])
    return table


def main() -> None:
    records = load_openie_dir(str(OPENIE))
    entities, edges, entity_passages, passage_ids = build_graph(records)
    base_nodes = list(entities) + list(passage_ids)
    base_edges = set(edges)
    for eid, pids in entity_passages.items():
        for pid in pids:
            base_edges.add((eid, pid))
    chain = load_arm("r53_chain", "Ccompact")
    struct = load_arm("r53_struct", "S")
    prepared = {
        "R0": prepare_graph(base_nodes, base_edges),
        "R1": pack(base_nodes, base_edges, passage_ids, entities, chain["kept"], False),
        "R2": pack(base_nodes, base_edges, passage_ids, entities, struct["kept"], False),
        "R3": pack(base_nodes, base_edges, passage_ids, entities, struct["kept"], True),
    }
    diagnostic = json.loads((ART / "v53" / "diagnostic_queries.json").read_text(encoding="utf-8"))
    diag_rows = []
    for query in diagnostic["queries"]:
        linked = link_entities(query["query"], entities)
        row = {"qid": query["qid"], "kind": query["kind"], "n_linked": len(linked)}
        row.update(score(prepared, passage_ids, entities, linked, query["support_passage_ids"]))
        diag_rows.append(row)
    dev_rows = []
    with (V4 / "data" / "questions_eval.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("split_role") != "dev":
                continue
            gold = [pid for pid in item.get("gold_passage_ids") or [] if pid in passage_ids]
            linked = link_entities(item["question"], entities)
            row = {"question_id": item["question_id"], "n_linked": len(linked), "n_gold_in_corpus": len(gold)}
            row.update(score(prepared, passage_ids, entities, linked, gold))
            dev_rows.append(row)
    struct_passages = {pid for comp in struct["kept"] for pid in comp["passages"]}
    chain_passages = {pid for comp in chain["kept"] for pid in comp["passages"]}
    def overlap_count(passages):
        n = 0
        for row in dev_rows:
            ranks = (row.get("R0") or {}).get("ranks") or {}
            if set(ranks) & passages:
                n += 1
        return n
    changes = []
    for row in dev_rows:
        if not (row.get("R0") or {}).get("evaluable"):
            continue
        if row["R0"]["recall_at_5"] != row["R2"]["recall_at_5"] or row["R0"]["recall_at_1"] != row["R2"]["recall_at_1"]:
            changes.append(row["question_id"])
    payload = {
        "chain": {**{k: chain[k] for k in ("proposed", "verifier_accept")}, "kept": len(chain["kept"]), "cross_passage": sum(1 for c in chain["kept"] if c["cross_passage"]), "texts": chain["kept"]},
        "struct": {**{k: struct[k] for k in ("proposed", "verifier_accept")}, "kept": len(struct["kept"]), "cross_passage": sum(1 for c in struct["kept"] if c["cross_passage"]), "texts": struct["kept"]},
        "diagnostic_linked": mean_table([row for row in diag_rows if row.get("n_linked")]),
        "diagnostic_rows": diag_rows,
        "dev_linked": mean_table([row for row in dev_rows if row.get("n_linked")]),
        "n_dev": len(dev_rows),
        "n_linked": sum(1 for row in dev_rows if row.get("n_linked")),
        "n_unlinked": sum(1 for row in dev_rows if not row.get("n_linked")),
        "dev_gold_overlap_struct": overlap_count(struct_passages),
        "dev_gold_overlap_chain": overlap_count(chain_passages),
        "dev_r0_r2_changes": changes,
    }
    (ART / "v53" / "eval.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "chain_kept": payload["chain"]["kept"],
        "struct_kept": payload["struct"]["kept"],
        "struct_cross": payload["struct"]["cross_passage"],
        "chain_cross": payload["chain"]["cross_passage"],
        "diag": payload["diagnostic_linked"],
        "dev": payload["dev_linked"],
        "overlap_struct": payload["dev_gold_overlap_struct"],
        "changes": len(changes),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
