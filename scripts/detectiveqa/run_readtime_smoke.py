"""Read-time smoke test: Joint OpenIE vs Comprehension Triples on identical packets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.adapters.hy4_client import HY4Client  # noqa: E402
from src.config import NVEMBED_PATH  # noqa: E402
from src.detectiveqa.data import OUT, load_prepared, write_jsonl  # noqa: E402
from src.detectiveqa.graph import graph_dir  # noqa: E402
from src.detectiveqa.prompts import (COMPREHENSION, EVIDENCE_SYSTEM, JOINT_OPENIE,  # noqa: E402
                                     PROMPT_VERSION, VERIFY_RELATIONS)
from src.detectiveqa.readtime import (build_packets, load_openie_entities,  # noqa: E402
                                      render_packet, validate_relation)
from src.graph.variants import working_dir_of  # noqa: E402


def openie_state(novel_id: str) -> Path:
    return working_dir_of(graph_dir(novel_id), "hy4", NVEMBED_PATH) / "openie_state.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--novels", default="27,79,117,118,121,124")
    ap.add_argument("--packets-per-novel", type=int, default=5)
    ap.add_argument("--chunks-per-packet", type=int, default=6)
    ap.add_argument("--max-words", type=int, default=1800)
    ap.add_argument("--tag", default="readtime_smoke")
    args = ap.parse_args()
    questions, chunks, splits = load_prepared()
    client = HY4Client(tag=args.tag)
    rows: List[Dict] = []
    partial = open(OUT / "readtime_smoke_raw.partial.jsonl", "w", encoding="utf-8")
    for novel_id in args.novels.split(","):
        state = openie_state(novel_id)
        if not state.exists():
            print(f"SKIP novel {novel_id}: missing {state}", flush=True)
            continue
        entities = load_openie_entities(state, chunks[novel_id])
        packets = build_packets(chunks[novel_id], entities,
                                max_packets=args.packets_per_novel,
                                chunks_per_packet=args.chunks_per_packet,
                                max_words=args.max_words)
        for packet in packets:
            rendered = render_packet(packet, chunks[novel_id])
            joint_raw, joint_err = {}, ""
            try:
                # Atomic facts from a 6-chunk packet can exceed 4k completion tokens;
                # the first run was truncated there and silently produced no facts.
                joint_raw = client.complete_json(JOINT_OPENIE.format(packet=rendered),
                                                 system=EVIDENCE_SYSTEM, max_tokens=8192,
                                                 tag=args.tag)
            except Exception as exc:  # noqa: BLE001
                joint_err = f"{type(exc).__name__}: {exc}"[:300]
            joint_facts = joint_raw.get("facts", []) if isinstance(joint_raw, dict) else []
            if isinstance(joint_facts, dict):
                joint_facts = []
            comp_raw, comp_err = {}, ""
            try:
                comp_raw = client.complete_json(
                    COMPREHENSION.format(facts=json.dumps(joint_facts, ensure_ascii=False),
                                         packet=rendered),
                    system=EVIDENCE_SYSTEM, max_tokens=4096, tag=args.tag)
            except Exception as exc:  # noqa: BLE001
                comp_err = f"{type(exc).__name__}: {exc}"[:300]
            relations_raw = comp_raw.get("relations", []) if isinstance(comp_raw, dict) else []
            validated, rejected = [], []
            existing = [(str(f.get("subject", "")), str(f.get("predicate", "")),
                         str(f.get("object", ""))) for f in joint_facts if isinstance(f, dict)]
            for item in relations_raw:
                if not isinstance(item, dict):
                    rejected.append({"item": item, "reason": "not_object"})
                    continue
                relation, reason = validate_relation(item, packet, chunks[novel_id],
                                                     existing_facts=existing)
                if relation:
                    validated.append(relation)
                else:
                    rejected.append({"item": item, "reason": reason})
            verdicts: List[Dict] = []
            verify_err = ""
            if validated:
                candidates = [{**r, "id": f"c{i}"} for i, r in enumerate(validated)]
                try:
                    out = client.complete_json(
                        VERIFY_RELATIONS.format(candidates=json.dumps(candidates, ensure_ascii=False),
                                                packet=rendered),
                        system=EVIDENCE_SYSTEM, max_tokens=2048, tag=args.tag)
                    verdicts = out.get("verdicts", []) if isinstance(out, dict) else []
                except Exception as exc:  # noqa: BLE001
                    verify_err = f"{type(exc).__name__}: {exc}"[:300]
            verdict_by_id = {str(v.get("id")): v for v in verdicts if isinstance(v, dict)}
            kept, dropped = [], []
            for i, relation in enumerate(validated):
                verdict = verdict_by_id.get(f"c{i}")
                if verdict is None:
                    dropped.append({"relation": relation, "reason": "no_verdict"})
                elif verdict.get("keep"):
                    kept.append({**relation, "verifier_reason": verdict.get("reason", ""),
                                 "fixed_status": verdict.get("fixed_status", relation["status"])})
                else:
                    dropped.append({"relation": relation,
                                    "reason": str(verdict.get("reason", "rejected"))})
            rows.append({
                "packet_id": packet.packet_id, "novel_id": novel_id, "anchor": packet.anchor,
                "chunk_ids": packet.chunk_ids, "packet_words": packet.n_words,
                "prompt_version": PROMPT_VERSION,
                "joint_facts": joint_facts, "joint_error": joint_err,
                "relations_raw": relations_raw, "comprehension_error": comp_err,
                "validated": validated, "rejected": rejected,
                "verifier_verdicts": verdicts, "verifier_error": verify_err,
                "kept": kept, "dropped_by_verifier": dropped,
            })
            partial.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
            partial.flush()
            print(packet.packet_id, "joint", len(joint_facts), "raw", len(relations_raw),
                  "valid", len(validated), "kept", len(kept), flush=True)
    partial.close()
    write_jsonl(OUT / "readtime_smoke_raw.jsonl", rows)
    all_kept = [r for row in rows for r in row["kept"]]
    gaps = []
    for row in rows:
        order = {c.chunk_id: i for i, c in enumerate(chunks[row["novel_id"]])}
        for relation in row["kept"]:
            gaps.append(max(order[c] for c in relation["source_chunk_ids"]) -
                        min(order[c] for c in relation["source_chunk_ids"]))
    summary = {
        "prompt_version": PROMPT_VERSION,
        "n_packets": len(rows),
        "n_joint_facts": sum(len(r["joint_facts"]) for r in rows),
        "n_relations_raw": sum(len(r["relations_raw"]) for r in rows),
        "n_validated": sum(len(r["validated"]) for r in rows),
        "n_kept": len(all_kept),
        "reject_reasons": {},
        "kept_relation_stats": {
            "n": len(all_kept),
            "types": {},
            "status": {},
            "median_chunk_gap": sorted(gaps)[len(gaps) // 2] if gaps else None,
        },
        "joint_errors": sum(bool(r["joint_error"]) for r in rows),
        "comprehension_errors": sum(bool(r["comprehension_error"]) for r in rows),
        "verifier_errors": sum(bool(r["verifier_error"]) for r in rows),
        "client_stats": client.summary(),
    }
    for relation in all_kept:
        summary["kept_relation_stats"]["types"][relation["relation_type"]] = \
            summary["kept_relation_stats"]["types"].get(relation["relation_type"], 0) + 1
        summary["kept_relation_stats"]["status"][relation["status"]] = \
            summary["kept_relation_stats"]["status"].get(relation["status"], 0) + 1
    for row in rows:
        for rej in row["rejected"]:
            reason = rej["reason"]
            summary["reject_reasons"][reason] = summary["reject_reasons"].get(reason, 0) + 1
        for drop in row["dropped_by_verifier"]:
            reason = "verifier_" + str(drop["reason"])[:40]
            summary["reject_reasons"][reason] = summary["reject_reasons"].get(reason, 0) + 1
    (OUT / "readtime_smoke_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
