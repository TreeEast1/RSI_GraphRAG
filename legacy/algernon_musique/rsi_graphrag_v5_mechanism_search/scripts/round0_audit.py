#!/usr/bin/env python3
"""Audit the 14 historical smoke candidates from the Luna cache and source passages.

No new Luna calls. The gold labels below are the manual reading of those passages.
They are not model outputs.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v5_mech.paths import ART, OPENIE, REPORTS, V4_CACHE  # noqa: E402
from v5_mech.verifier import verify_candidate  # noqa: E402

# Keys are the final smoke responses, matched by prompt-token counts in smoke_summary.json.
GENERATORS = {
    "small_A": "07af7c5bc5163b05b479ddf7a7361a8c573a879174c8e566afe0e9660bca95e6",
    "small_B": "46fc23a6d7cdf0ec3eea0bb7fcdc33eab332adf4516bd17b135e7f5e1aed33a3",
    "medium_A": "3bb8d79ac4f443505067590c9d079b0c082a4e4055dc5076aadef8614719ca05",
    "medium_B": "7c619babc06c804633ffd3f6109d666f5500cf3e4fd42ddbd81ea49a84d08554",
    "large_A": "bc123f4abf989ecebbcf3dc0c469aa91b558f53737e9756a54fd05d68b1bfb37",
    "large_B": "52c00614c50f60fb8a8f2c1678a6edf315cad8a9a2577684bf9db1e49cb2761a",
}

# Manual labels after reading the OpenIE passage files. faithful_passage means the
# full passage supports the claim. citation_ok means the cited sentences are enough.
# novel means the claim is not already covered by that passage's OpenIE triples.
GOLD = {
    "small_B:local-1": {
        "audit_class": "Correct & Novel",
        "type": "CROSS_PASSAGE_REASONING",
        "faithful_passage": True,
        "citation_ok": True,
        "novel": True,
        "old_decision": "ACCEPT",
        "note": "Tristan's passage says he first appeared on 23 October 2009. Ludwig's passage says he appeared from 22 December 2009 until his death. The comparison is not a stored triple. It is not a causal claim. OpenIE already inferred 'first appeared' for Ludwig from 'appeared from'.",
    },
    "medium_A:local-001": {
        "audit_class": "Extraction Repair",
        "type": "EXTRACTION_REPAIR",
        "faithful_passage": True,
        "citation_ok": False,
        "novel": True,
        "old_decision": "ACCEPT",
        "note": "The contract sentence states the six-month term and says Morley, but it does not say Aidan Foster. The role is explicit elsewhere on the page. The full claim 'contracted to play Aidan Foster' cites only the duration sentence. No OpenIE triple stores the contract length. The old verifier accepted the role join from a casting triple.",
    },
    "medium_A:local-002": {
        "audit_class": "Extraction Repair",
        "type": "EXTRACTION_REPAIR",
        "faithful_passage": True,
        "citation_ok": True,
        "novel": True,
        "old_decision": "REJECT",
        "note": "The contract sentence continues 'and he began filming his first scenes as Mason'. The passage opens by saying Mason Turner is played by Taylor Glockner. The old verifier rejected because the short quote omitted Mason. The full sentence includes Mason.",
    },
    "medium_A:local-003": {
        "audit_class": "Correct but Redundant",
        "type": "REDUNDANT",
        "faithful_passage": True,
        "citation_ok": True,
        "novel": False,
        "old_decision": "REJECT",
        "note": "The opening sentence names the Sisters of Saint Joseph of Nazareth and says they have their main centre in Whanganui, New Zealand. OpenIE already has that triple. The old verifier treated the subject as unbound and pointed at Sisters of St Joseph Whanganui, which is a later fusion sentence.",
    },
    "medium_A:local-004": {
        "audit_class": "Correct but Redundant",
        "type": "REDUNDANT",
        "faithful_passage": True,
        "citation_ok": True,
        "novel": False,
        "old_decision": "ACCEPT",
        "note": "OpenIE already stores KPBD began broadcasting in June 2005. The old verifier accepted it as new. The passage identifies KPBD as the station in the same short article.",
    },
    "medium_B:local-001": {
        "audit_class": "Correct & Novel",
        "type": "SINGLE_PASSAGE_CONSOLIDATION",
        "faithful_passage": True,
        "citation_ok": False,
        "novel": True,
        "old_decision": "ACCEPT",
        "note": "The cited sentence says she moves to Erinsborough to live with her aunt following her mother's death. 'Following' supports 'after', not the relation label 'motivated by'. Madge Bishop is named only in the previous sentence. The old verifier accepted the Madge binding from a separate casting triple.",
    },
    "medium_B:local-002": {
        "audit_class": "Correct but Redundant",
        "type": "REDUNDANT",
        "faithful_passage": True,
        "citation_ok": True,
        "novel": False,
        "old_decision": "ACCEPT",
        "note": "OpenIE already has both 'Aidan Foster played by Bobby Morley' and 'Aidan Foster introduced as a love interest for Chris Pappas'. The candidate verbalizes a two-hop path.",
    },
    "medium_B:local-003": {
        "audit_class": "Correct & Novel",
        "type": "SINGLE_PASSAGE_CONSOLIDATION",
        "faithful_passage": True,
        "citation_ok": True,
        "novel": True,
        "old_decision": "ACCEPT",
        "note": "The passage says the branch was first commanded by Ostapenko, who was promoted in November 2012, and that on 24 December 2012 Golovko was appointed the new commander. 'The new commander' plus 'first commanded by' supports succession. It is stronger than a bare pair of dates. The quoted span cuts off the 'first commanded' clause, but that clause is in the same sentence as the promotion.",
    },
    "medium_B:local-004": {
        "audit_class": "Uncertain",
        "type": "UNSUPPORTED_HYPOTHESIS",
        "faithful_passage": False,
        "citation_ok": False,
        "novel": False,
        "old_decision": "ACCEPT",
        "note": "The full sentence is 'northern Mali, which Tuareg rebels took control of by April and declared the secession of a new state, Azawad.' The quote drops 'northern Mali, which', so the quoted span has no object. Both events are already separate triples. 'Before' is not entailed by 'and'. The old verifier accepted the broken quote.",
    },
    "large_A:candidate-1": {
        "audit_class": "Correct but Redundant",
        "type": "REDUNDANT",
        "faithful_passage": True,
        "citation_ok": True,
        "novel": False,
        "old_decision": "ACCEPT",
        "note": "OpenIE already stores that NBC renewed Shades of Blue for a third season and that the third season was renewed on March 17, 2017. The old verifier said the triples only mentioned the second season.",
    },
    "large_A:candidate-2": {
        "audit_class": "Correct but Redundant",
        "type": "REDUNDANT",
        "faithful_passage": True,
        "citation_ok": True,
        "novel": False,
        "old_decision": "REJECT",
        "note": "The passage says 'The show was not picked up for air.' OpenIE already has that triple. The old verifier said the span was absent. The smoke verifier truncates the unit to 12,000 characters, and passages are placed after the entity and triple lists.",
    },
    "large_B:cand-1": {
        "audit_class": "Correct & Novel",
        "type": "SINGLE_PASSAGE_CONSOLIDATION",
        "faithful_passage": True,
        "citation_ok": True,
        "novel": True,
        "old_decision": "REJECT",
        "note": "One sentence says Kremer currently calls Thursday Night Football for Amazon Prime Video, making sports history with Hannah Storm as the first all-women booth to call any major men's team sport. OpenIE has the co-calling facts and not the 'first all-women booth' status. The old verifier said no supplied passage supported it.",
    },
    "large_B:cand-2": {
        "audit_class": "Correct & Novel",
        "type": "SINGLE_PASSAGE_CONSOLIDATION",
        "faithful_passage": True,
        "citation_ok": True,
        "novel": True,
        "old_decision": "REJECT",
        "note": "Woolery hosted until 25 December 1981 and his contract was not renewed. Sajak made his debut as host on 28 December 1981. That is the next host of the same show, not two unrelated dates. The old verifier said no quoted support was supplied.",
    },
    "large_B:cand-3": {
        "audit_class": "Correct but Redundant",
        "type": "REDUNDANT",
        "faithful_passage": True,
        "citation_ok": True,
        "novel": False,
        "old_decision": "REJECT",
        "note": "Both sentences are explicit and both are already triples: NBC approved the pilot in July 2006, and the show was not picked up for air. The candidate paraphrases 'air' as 'broadcast'. The old verifier rejected the second half as absent.",
    },
}


def load_cache(key: str) -> str:
    conn = sqlite3.connect(f"file:{V4_CACHE}?mode=ro", uri=True)
    row = conn.execute("select message from cache where key = ?", (key,)).fetchone()
    conn.close()
    if row is None:
        raise RuntimeError(f"missing cache key {key}")
    return row[0]


def load_passage(pid: str) -> dict:
    path = OPENIE / f"{pid}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def synthetic_controls() -> list:
    tristan = load_passage("chunk-5531a128df4a1d94732e682316b73430")
    ludwig = load_passage("chunk-429ba54878382d30bf5cb2e822a01def")
    passages = {
        tristan["passage_id"] if "passage_id" in tristan else "chunk-5531a128df4a1d94732e682316b73430": tristan["content"],
        "chunk-429ba54878382d30bf5cb2e822a01def": ludwig["content"],
    }
    # passage_id may only live in the filename.
    passages = {
        "chunk-5531a128df4a1d94732e682316b73430": tristan["content"],
        "chunk-429ba54878382d30bf5cb2e822a01def": ludwig["content"],
    }
    triples = {
        "chunk-5531a128df4a1d94732e682316b73430": tristan["triples"],
        "chunk-429ba54878382d30bf5cb2e822a01def": ludwig["triples"],
    }
    controls = [
        {
            "id": "SYN-DUP",
            "expect": "REJECT",
            "candidate": {
                "operation": "ADD",
                "candidate_id": "SYN-DUP",
                "comprehension_text": "Tristan von Lahnstein first appeared on 23 October 2009.",
                "new_semantic_relation": "first appeared on",
                "evidence_spans": [{"passage_id": "chunk-5531a128df4a1d94732e682316b73430", "span": "He first appeared on 23 October 2009."}],
            },
        },
        {
            "id": "SYN-CAUSE",
            "expect": "REJECT",
            "candidate": {
                "operation": "ADD",
                "candidate_id": "SYN-CAUSE",
                "comprehension_text": "Ludwig von Lahnstein died because Tristan von Lahnstein had already appeared.",
                "new_semantic_relation": "caused by",
                "evidence_spans": [
                    {"passage_id": "chunk-5531a128df4a1d94732e682316b73430", "span": "He first appeared on 23 October 2009."},
                    {"passage_id": "chunk-429ba54878382d30bf5cb2e822a01def", "span": "He appeared from 22 December 2009 to the character's death"},
                ],
            },
        },
        {
            "id": "SYN-MISS",
            "expect": "EVIDENCE_MISMATCH",
            "candidate": {
                "operation": "ADD",
                "candidate_id": "SYN-MISS",
                "comprehension_text": "Tristan von Lahnstein won a prize in Berlin.",
                "new_semantic_relation": "won",
                "evidence_spans": [{"passage_id": "chunk-5531a128df4a1d94732e682316b73430", "span": "He won a prize in Berlin in 2010."}],
            },
        },
    ]
    rows = []
    for item in controls:
        judged = verify_candidate(item["candidate"], passages, triples)
        judged["synthetic"] = True
        judged["expect"] = item["expect"]
        judged["expect_met"] = judged["decision"] == item["expect"]
        rows.append(judged)
    return rows


def main() -> None:
    out = ART / "round0"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_name, key in GENERATORS.items():
        payload = json.loads(load_cache(key))
        (out / f"{run_name}_generator.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        for update in payload.get("updates") or []:
            if (update.get("operation") or "").upper() != "ADD":
                rows.append({
                    "run": run_name,
                    "candidate_id": update.get("candidate_id"),
                    "operation": "NO_OP",
                    "text": "",
                    "gold": None,
                    "new_verifier": None,
                })
                continue
            pids = []
            for span in update.get("evidence_spans") or []:
                if span.get("passage_id") and span["passage_id"] not in pids:
                    pids.append(span["passage_id"])
            for pid in update.get("source_passage_ids") or []:
                if pid not in pids:
                    pids.append(pid)
            passages = {}
            triples = {}
            sources = {}
            for pid in pids:
                record = load_passage(pid)
                passages[pid] = record["content"]
                triples[pid] = record.get("triples") or []
                sources[pid] = {"content": record["content"], "triples": record.get("triples") or []}
            judged = verify_candidate(update, passages, triples)
            gold_key = f"{run_name}:{update.get('candidate_id')}"
            gold = GOLD[gold_key]
            rows.append({
                "run": run_name,
                "candidate_id": update.get("candidate_id"),
                "text": update.get("comprehension_text"),
                "relation": update.get("new_semantic_relation"),
                "generator_type": update.get("derivation_type"),
                "spans": update.get("evidence_spans"),
                "gold": gold,
                "new_verifier": judged,
                "passages": sources,
                "old_matches_faithfulness": (gold["old_decision"] == "ACCEPT") == gold["faithful_passage"],
                "new_accepts": judged["decision"] == "ACCEPT",
            })
    synthetic = synthetic_controls()
    summary = {
        "n_add": sum(1 for row in rows if row.get("text")),
        "new_accept": sum(1 for row in rows if row.get("new_verifier") and row["new_verifier"]["decision"] == "ACCEPT"),
        "new_reject": sum(1 for row in rows if row.get("new_verifier") and row["new_verifier"]["decision"] == "REJECT"),
        "new_other": sum(1 for row in rows if row.get("new_verifier") and row["new_verifier"]["decision"] not in {"ACCEPT", "REJECT"}),
        "synthetic_pass": sum(1 for row in synthetic if row["expect_met"]),
        "synthetic_n": len(synthetic),
    }
    # Faithfulness: old ACCEPT vs gold faithful. New ACCEPT vs gold faithful AND citation_ok AND novel is the store rule.
    faith_errors_old = []
    faith_errors_new = []
    for row in rows:
        if not row.get("gold"):
            continue
        gold = row["gold"]
        old_accept = gold["old_decision"] == "ACCEPT"
        if old_accept != gold["faithful_passage"]:
            faith_errors_old.append(row["run"] + ":" + row["candidate_id"])
        new_accept = row["new_verifier"]["decision"] == "ACCEPT"
        should_store = gold["faithful_passage"] and gold["citation_ok"] and gold["novel"]
        if new_accept != should_store:
            faith_errors_new.append({
                "id": row["run"] + ":" + row["candidate_id"],
                "new": row["new_verifier"]["decision"],
                "stage": row["new_verifier"].get("failure_stage"),
                "should_store": should_store,
                "type": row["new_verifier"].get("audit_type"),
                "reason": row["new_verifier"].get("reason"),
            })
    summary["old_faithfulness_disagreements"] = faith_errors_old
    summary["new_store_disagreements"] = faith_errors_new
    payload = {"summary": summary, "rows": rows, "synthetic": synthetic}
    (out / "audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
