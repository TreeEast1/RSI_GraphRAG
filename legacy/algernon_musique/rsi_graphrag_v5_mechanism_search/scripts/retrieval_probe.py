#!/usr/bin/env python3
"""Local structural retrieval probe. Not a full HippoRAG2 / NV-Embed run.

NV-Embed-v2 is on disk, but every GPU had under 4 GB free, below the encoder's
12 GB threshold. This probe ranks the pilot passages with lexical overlap plus
entity links. R3 adds accepted comprehension nodes as graph nodes. Dev questions
are chosen only when an entity string from the pilot appears in the question
text. Gold passages are used after selection, only for scoring.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")

from hipporag.utils.misc_utils import text_processing  # noqa: E402

from v5_mech.paths import ART, V4  # noqa: E402

DIAGNOSTIC = [
    {
        "qid": "diag-kremer",
        "query": "Who formed the first all-women booth to call a major men's team sport?",
        "targets": ["chunk-c30c674749748e4a1b36f06ffa6bed8a"],
    },
    {
        "qid": "diag-tristan",
        "query": "Which character appeared earlier, Tristan von Lahnstein or Ludwig von Lahnstein?",
        "targets": [
            "chunk-5531a128df4a1d94732e682316b73430",
            "chunk-429ba54878382d30bf5cb2e822a01def",
        ],
    },
    {
        "qid": "diag-teeth",
        "query": "When do permanent teeth begin to form before birth?",
        "targets": [],
    },
    {
        "qid": "diag-kreindler",
        "query": "Who is considered the founder of air disaster law?",
        "targets": [],
    },
]


def toks(text: str) -> set:
    return {tok for tok in text_processing(text).split() if len(tok) > 2}


def lexical(query: str, text: str) -> float:
    left, right = toks(query), toks(text)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def load_pilot():
    passages = {}
    entities = {}
    triples = []
    entity_passages = defaultdict(set)
    for path in sorted((ART / "units").glob("*.json")):
        if path.name in {"selection.json", "leiden_stable_stats.json"} or path.suffix != ".json":
            continue
        if not path.name.startswith(("hist_", "new_")):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for pid, content in (data.get("passages") or {}).items():
            passages[pid] = content
        for triple in data.get("triples") or []:
            triples.append(triple)
            entities[triple["s_id"]] = triple["s"]
            entities[triple["o_id"]] = triple["o"]
            for pid in triple.get("passage_ids") or []:
                if pid in passages or True:
                    entity_passages[triple["s_id"]].add(pid)
                    entity_passages[triple["o_id"]].add(pid)
    comps = []
    for run in ("r1_hist", "r3_new_ac"):
        directory = ART / "runs" / run
        if not directory.exists():
            continue
        for path in directory.glob("*__C.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            judgments = {row["candidate_id"]: row for row in data.get("judgments") or []}
            for update in data.get("updates") or []:
                if (update.get("operation") or "").upper() != "ADD":
                    continue
                judged = judgments.get(update.get("candidate_id")) or {}
                if judged.get("decision") != "ACCEPT":
                    continue
                comps.append({
                    "text": update.get("comprehension_text") or "",
                    "relation": update.get("new_semantic_relation") or "",
                    "passages": list(update.get("source_passage_ids") or []),
                    "entities": list(update.get("linked_entity_ids") or []),
                    "audit_type": judged.get("audit_type"),
                    "role": data.get("role"),
                })
    return passages, entities, triples, entity_passages, comps


def fill_targets(comps, queries):
    for query in queries:
        if query["qid"] == "diag-teeth":
            for comp in comps:
                if "permanent teeth" in comp["text"].lower() and comp["passages"]:
                    query["targets"] = comp["passages"][:1]
        if query["qid"] == "diag-kreindler":
            for comp in comps:
                if "air disaster" in comp["text"].lower() and comp["passages"]:
                    query["targets"] = comp["passages"][:1]


def linked_entities(query: str, entities: dict) -> list:
    found = []
    lowered = " " + re.sub(r"\s+", " ", query.lower()) + " "
    for eid, name in entities.items():
        surface = (name or "").strip()
        if len(surface) < 6:
            continue
        if surface.lower() in lowered:
            found.append(eid)
    return found


def rank(query, passages, entities, entity_passages, comps, mode: str):
    scores = {pid: lexical(query, text) for pid, text in passages.items()}
    linked = linked_entities(query, entities)
    for eid in linked:
        for pid in entity_passages.get(eid, ()):
            if pid in scores:
                scores[pid] += 0.4
    use = comps
    if mode == "R1":
        use = [comp for comp in comps if comp["audit_type"] == "EXTRACTION_REPAIR"]
    elif mode == "R0":
        use = []
    for comp in use:
        overlap = lexical(query, comp["text"])
        if mode == "R2":
            if overlap <= 0:
                continue
            for pid in comp["passages"]:
                if pid in scores:
                    scores[pid] += 0.6 * overlap
        elif mode in {"R1", "R3"}:
            entity_hit = any(eid in linked for eid in comp["entities"])
            if overlap <= 0 and not entity_hit:
                continue
            bonus = 0.6 * overlap + (0.35 if entity_hit else 0.0)
            for pid in comp["passages"]:
                if pid in scores:
                    scores[pid] += bonus
    ordered = sorted(scores, key=lambda pid: (-scores[pid], pid))
    return ordered, scores, linked


def recall_at(ordered, targets, k: int) -> float:
    if not targets:
        return 0.0
    hit = len(set(ordered[:k]) & set(targets))
    return hit / len(set(targets))


def dev_questions(entities: dict):
    path = V4 / "data" / "questions_eval.jsonl"
    names = [name for name in entities.values() if isinstance(name, str) and len(name) >= 10 and " " in name]
    # Long unique names only, so "United States" style strings still match, but single tokens do not.
    names = sorted(set(names), key=len, reverse=True)[:400]
    chosen = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split_role") != "dev":
                continue
            question = row.get("question") or ""
            lowered = question.lower()
            hits = [name for name in names if name.lower() in lowered]
            if not hits:
                continue
            chosen.append({
                "question_id": row["question_id"],
                "question": question,
                "gold": row.get("gold_passage_ids") or [],
                "matched_entities": hits[:5],
            })
    return chosen


def main() -> None:
    passages, entities, triples, entity_passages, comps = load_pilot()
    fill_targets(comps, DIAGNOSTIC)
    modes = ("R0", "R1", "R2", "R3")
    rows = []
    for query in DIAGNOSTIC:
        row = {"qid": query["qid"], "query": query["query"], "targets": query["targets"], "set": "diagnostic"}
        for mode in modes:
            ordered, scores, linked = rank(query["query"], passages, entities, entity_passages, comps, mode)
            row[mode] = {
                "recall_at_5": recall_at(ordered, query["targets"], 5),
                "top5": ordered[:5],
                "target_ranks": {pid: ordered.index(pid) + 1 if pid in ordered else None for pid in query["targets"]},
                "n_linked_entities": len(linked),
            }
        rows.append(row)
    dev = dev_questions(entities)
    dev_rows = []
    pilot_ids = set(passages)
    for item in dev:
        gold_in_pilot = [pid for pid in item["gold"] if pid in pilot_ids]
        row = {
            "qid": item["question_id"],
            "set": "musique_dev_entity_match",
            "matched_entities": item["matched_entities"],
            "n_gold": len(item["gold"]),
            "n_gold_in_pilot": len(gold_in_pilot),
        }
        for mode in modes:
            ordered, _, _ = rank(item["question"], passages, entities, entity_passages, comps, mode)
            row[mode] = {"recall_at_5_in_pilot": recall_at(ordered, gold_in_pilot, 5) if gold_in_pilot else None}
        dev_rows.append(row)
    summary = {
        "label": "local structural probe, not HippoRAG2 and not NV-Embed",
        "n_passages": len(passages),
        "n_entities": len(entities),
        "n_triples": len(triples),
        "n_accepted_c_nodes": len(comps),
        "type_counts": {},
        "diagnostic": rows,
        "n_dev_entity_matches": len(dev_rows),
        "dev": dev_rows,
    }
    for comp in comps:
        summary["type_counts"][comp["audit_type"]] = summary["type_counts"].get(comp["audit_type"], 0) + 1
    out = ART / "runs" / "r4_retrieval.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "passages": summary["n_passages"],
        "comps": summary["n_accepted_c_nodes"],
        "types": summary["type_counts"],
        "dev_matches": summary["n_dev_entity_matches"],
        "diagnostic": [
            {
                "qid": row["qid"],
                "ranks": {mode: row[mode]["target_ranks"] for mode in modes},
                "r5": {mode: row[mode]["recall_at_5"] for mode in modes},
            }
            for row in rows
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
