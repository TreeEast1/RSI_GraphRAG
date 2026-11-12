#!/usr/bin/env python3
"""Compare the old residue verifier with semantic entailment.

The dev split contains the paraphrase failures named in the v5.1 brief.
Rules may be read against that split. The held-out split is scored once
and is not used to add synonyms.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v5_mech.novelty import classify_novelty  # noqa: E402
from v5_mech.paths import ART, OPENIE  # noqa: E402
from v5_mech.verifier import verify_candidate  # noqa: E402

# expect_value is the information label when the claim is supported.
# UNSUPPORTED means the claim should not be treated as factually established.
CASES = [
    # --- dev: the failures the rules were written to catch ---
    {"id": "dev-kpbd", "split": "dev", "file": "round0", "run": "medium_A", "candidate_id": "local-004", "expect": "REDUNDANT"},
    {"id": "dev-shades", "split": "dev", "file": "round0", "run": "large_A", "candidate_id": "candidate-1", "expect": "REDUNDANT"},
    {"id": "dev-sisters", "split": "dev", "file": "round0", "run": "medium_A", "candidate_id": "local-003", "expect": "REDUNDANT"},
    {"id": "dev-lipshitz-broadcast", "split": "dev", "file": "round0", "run": "large_B", "candidate_id": "cand-3", "expect": "REDUNDANT"},
    {"id": "dev-gemma-appearance", "split": "dev", "file": "r1", "name": "hist_medium__C.json", "candidate_id": "add-1", "expect": "REDUNDANT"},
    {"id": "dev-tristan", "split": "dev", "file": "round0", "run": "small_B", "candidate_id": "local-1", "expect": "COMPOSITIONAL_UNDERSTANDING"},
    {"id": "dev-kremer", "split": "dev", "file": "r1", "name": "hist_large__C.json", "candidate_id": "candidate-1", "expect": "EXTRACTION_REPAIR"},
    {"id": "dev-azawad", "split": "dev", "file": "round0", "run": "medium_B", "candidate_id": "local-004", "expect": "UNSUPPORTED"},
    {"id": "dev-plainsman-interval", "split": "dev", "file": "c2", "name": "hist_large__C2.json", "candidate_id": "candidate-1", "expect": "TRIVIAL_DERIVATION"},
    {"id": "dev-golovko", "split": "dev", "file": "round0", "run": "medium_B", "candidate_id": "local-003", "expect": "REDUNDANT"},
    # --- held-out: not used to choose synonyms ---
    {"id": "hold-glockner", "split": "heldout", "file": "round0", "run": "medium_A", "candidate_id": "local-002", "expect": "EXTRACTION_REPAIR"},
    {"id": "hold-morley-chris", "split": "heldout", "file": "round0", "run": "medium_B", "candidate_id": "local-002", "expect": "REDUNDANT"},
    {"id": "hold-lipshitz-exact", "split": "heldout", "file": "round0", "run": "large_A", "candidate_id": "candidate-2", "expect": "REDUNDANT"},
    {"id": "hold-sajak", "split": "heldout", "file": "round0", "run": "large_B", "candidate_id": "cand-2", "expect": "SEMANTIC_ABSTRACTION"},
    {"id": "hold-gemma-madge", "split": "heldout", "file": "round0", "run": "medium_B", "candidate_id": "local-001", "expect": "UNSUPPORTED"},
    {"id": "hold-bobby-contract", "split": "heldout", "file": "round0", "run": "medium_A", "candidate_id": "local-001", "expect": "UNSUPPORTED"},
    {"id": "hold-next-caller", "split": "heldout", "file": "r1", "name": "hist_large__C.json", "candidate_id": "candidate-3", "expect": "EXTRACTION_REPAIR"},
    {"id": "hold-ten-week", "split": "heldout", "file": "r1", "name": "hist_medium__C.json", "candidate_id": "add-2", "expect": "EXTRACTION_REPAIR"},
    {"id": "hold-aidan-dates", "split": "heldout", "file": "c2", "name": "hist_medium__C2.json", "candidate_id": "candidate-1", "expect": "COMPOSITIONAL_UNDERSTANDING"},
    {"id": "hold-teeth", "split": "heldout", "file": "new", "name": "new_00__C.json", "candidate_id": "candidate-2", "expect": "EXTRACTION_REPAIR"},
    {"id": "hold-gao", "split": "heldout", "file": "new", "name": "new_03__C.json", "candidate_id": "local-002", "expect": "EXTRACTION_REPAIR"},
    {"id": "hold-kreindler", "split": "heldout", "file": "new", "name": "new_overlap__A.json", "candidate_id": "cand-2", "expect": "EXTRACTION_REPAIR"},
    {"id": "hold-cristine", "split": "heldout", "file": "c2", "name": "hist_large__C2.json", "candidate_id": "candidate-2", "expect": "REDUNDANT"},
    {"id": "hold-wang", "split": "heldout", "file": "new", "name": "new_03__C.json", "candidate_id": "local-001", "expect": "UNSUPPORTED"},
]


def _load_round0():
    payload = json.loads((ART / "round0" / "audit.json").read_text(encoding="utf-8"))
    found = {}
    for row in payload["rows"]:
        if not row.get("text"):
            continue
        found[(row["run"], row["candidate_id"])] = row
    return found


def _load_run(folder: str, name: str, candidate_id: str):
    data = json.loads((ART / "runs" / folder / name).read_text(encoding="utf-8"))
    updates = {item.get("candidate_id"): item for item in data.get("updates") or []}
    judgments = {item.get("candidate_id"): item for item in data.get("judgments") or []}
    update = updates[candidate_id]
    return update, judgments.get(candidate_id) or {}


def _passages(update: dict):
    pids = []
    for span in update.get("evidence_spans") or []:
        pid = span.get("passage_id")
        if pid and pid not in pids:
            pids.append(pid)
    passages, triples = {}, {}
    for pid in pids:
        path = OPENIE / f"{pid}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        passages[pid] = record.get("content") or ""
        triples[pid] = record.get("triples") or []
    return passages, triples


def _old_label(judgment: dict) -> str:
    if not judgment:
        return "MISSING"
    if judgment.get("decision") != "ACCEPT":
        if judgment.get("faithfulness") == "supported" and judgment.get("audit_type") == "REDUNDANT":
            return "REDUNDANT"
        return "UNSUPPORTED" if judgment.get("decision") == "REJECT" else judgment.get("decision")
    return judgment.get("audit_type") or "ACCEPT"


def _new_label(update: dict, passages: dict, triples: dict) -> dict:
    base = verify_candidate(update, passages, triples)
    if base.get("faithfulness") not in {"supported"} or base.get("decision") == "EVIDENCE_MISMATCH":
        if base.get("decision") == "ACCEPT":
            pass
        else:
            value = "UNCERTAIN" if base.get("decision") == "UNCERTAIN" else "UNSUPPORTED"
            if base.get("audit_type") == "REDUNDANT":
                value = "REDUNDANT"
            return {"value": value, "store": False, "detail": base.get("reason"), "stage": base.get("failure_stage")}
    if base.get("decision") != "ACCEPT" and base.get("audit_type") == "REDUNDANT":
        # Old V4 already saw exact cover. Still run entailment for the report.
        pass
    if base.get("faithfulness") != "supported" and base.get("decision") != "ACCEPT":
        return {"value": "UNSUPPORTED", "store": False, "detail": base.get("reason"), "stage": base.get("failure_stage")}
    evidence = " ".join(base.get("evidence_sentences") or [])
    n_passages = len({span.get("passage_id") for span in update.get("evidence_spans") or []})
    flat = []
    for pid in triples:
        flat.extend(triples[pid])
    novelty = classify_novelty(update.get("comprehension_text") or "", flat, evidence, n_passages)
    store = novelty["value_type"] in {"EXTRACTION_REPAIR", "COMPOSITIONAL_UNDERSTANDING", "SEMANTIC_ABSTRACTION"}
    return {"value": novelty["value_type"], "store": store, "detail": novelty["reason"], "stage": "V4-entail" if not store else ""}


def main() -> None:
    round0 = _load_round0()
    rows = []
    for case in CASES:
        if case["file"] == "round0":
            src = round0[(case["run"], case["candidate_id"])]
            update = {
                "operation": "ADD",
                "candidate_id": case["candidate_id"],
                "comprehension_text": src["text"],
                "new_semantic_relation": src.get("relation") or "",
                "evidence_spans": src["spans"],
            }
            old = _old_label(src["new_verifier"])
        else:
            folder = {"r1": "r1_hist", "c2": "r3_c2_hist", "new": "r3_new_ac"}[case["file"]]
            update, judgment = _load_run(folder, case["name"], case["candidate_id"])
            old = _old_label(judgment)
        passages, triples = _passages(update)
        new = _new_label(update, passages, triples)
        rows.append({
            "id": case["id"],
            "split": case["split"],
            "text": update.get("comprehension_text"),
            "expect": case["expect"],
            "old": old,
            "new": new["value"],
            "old_match": old == case["expect"] or (case["expect"] == "UNSUPPORTED" and old not in {"EXTRACTION_REPAIR", "COMPOSITIONAL_UNDERSTANDING", "SEMANTIC_ABSTRACTION", "CROSS_PASSAGE_REASONING", "SINGLE_PASSAGE_CONSOLIDATION", "REDUNDANT", "ACCEPT"}),
            "new_match": new["value"] == case["expect"],
            "detail": new["detail"],
        })
    # Old match for UNSUPPORTED: old rejected or labeled unsupported.
    for row, case in zip(rows, CASES):
        if case["expect"] == "UNSUPPORTED":
            row["old_match"] = row["old"] in {"UNSUPPORTED", "REJECT", "EVIDENCE_MISMATCH", "UNCERTAIN", "INCOMPLETE_CITATION", "ENTITY_BINDING_ERROR", "UNSUPPORTED_HYPOTHESIS"}
        elif case["expect"] == "REDUNDANT":
            row["old_match"] = row["old"] in {"REDUNDANT"}
        elif case["expect"] == "TRIVIAL_DERIVATION":
            row["old_match"] = False
        else:
            row["old_match"] = row["old"] in {case["expect"], "EXTRACTION_REPAIR", "COMPOSITIONAL_REASONING", "CROSS_PASSAGE_REASONING", "SINGLE_PASSAGE_CONSOLIDATION"} and case["expect"] != "REDUNDANT"
            if case["expect"] == "COMPOSITIONAL_UNDERSTANDING":
                row["old_match"] = row["old"] in {"CROSS_PASSAGE_REASONING", "COMPOSITIONAL_REASONING", "COMPOSITIONAL_UNDERSTANDING"}
            if case["expect"] == "EXTRACTION_REPAIR":
                row["old_match"] = row["old"] in {"EXTRACTION_REPAIR"}
            if case["expect"] == "SEMANTIC_ABSTRACTION":
                row["old_match"] = row["old"] == "SEMANTIC_ABSTRACTION"
    summary = {}
    for split in ("dev", "heldout"):
        part = [row for row in rows if row["split"] == split]
        summary[split] = {
            "n": len(part),
            "old_correct": sum(1 for row in part if row["old_match"]),
            "new_correct": sum(1 for row in part if row["new_match"]),
        }
    out = ART / "v51" / "novelty_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for row in rows:
        flag = "OK" if row["new_match"] else "MISS"
        print(f"{flag:4} {row['split']:7} {row['id']:28} old={row['old'][:28]:28} new={row['new']:28} expect={row['expect']}")


if __name__ == "__main__":
    main()
