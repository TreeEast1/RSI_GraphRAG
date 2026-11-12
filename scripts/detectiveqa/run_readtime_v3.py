"""Question-agnostic read-time v3 with deterministic quotes and semantic-only verification."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.adapters.hy4_client import HY4Client  # noqa: E402
from src.config import NVEMBED_PATH  # noqa: E402
from src.detectiveqa.data import OUT, load_prepared, write_jsonl  # noqa: E402
from src.detectiveqa.graph import graph_dir  # noqa: E402
from src.detectiveqa.prompts import (COMPREHENSION_V3, EVIDENCE_SYSTEM, JOINT_OPENIE,  # noqa: E402
                                     VERIFY_RELATIONS_V3)
from src.detectiveqa.readtime import (build_question_agnostic_packets, load_openie_entities,  # noqa: E402
                                      relation_stats, render_packet, validate_relation)
from src.graph.variants import working_dir_of  # noqa: E402


def state_path(novel_id: str) -> Path:
    return working_dir_of(graph_dir(novel_id), "hy4", NVEMBED_PATH) / "openie_state.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--novels", default="27,79,117,118,121,124")
    ap.add_argument("--packets-per-novel", type=int, default=20)
    ap.add_argument("--chunks-per-packet", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=1500)
    ap.add_argument("--max-total-packets", type=int, default=0, help="smoke-test only")
    ap.add_argument("--tag", default="readtime_v3")
    ap.add_argument("--out", default="readtime_v3")
    args = ap.parse_args()
    _, chunks, _ = load_prepared()
    client = HY4Client(tag=args.tag)
    rows: List[Dict] = []
    partial = open(OUT / f"{args.out}.partial.jsonl", "w", encoding="utf-8")
    for novel_id in args.novels.split(","):
        entities = load_openie_entities(state_path(novel_id), chunks[novel_id])
        packets = build_question_agnostic_packets(chunks[novel_id], entities,
                                                  max_packets=args.packets_per_novel,
                                                  chunks_per_packet=args.chunks_per_packet,
                                                  max_words=args.max_words)
        if args.max_total_packets:
            packets = packets[:max(0, args.max_total_packets - len(rows))]
        for packet in packets:
            rendered = render_packet(packet, chunks[novel_id])
            errors = {}
            try:
                joint = client.complete_json(JOINT_OPENIE.format(packet=rendered), system=EVIDENCE_SYSTEM,
                                             max_tokens=6144, tag=args.tag)
                joint_facts = joint.get("facts", []) if isinstance(joint, dict) else []
            except Exception as exc:  # noqa: BLE001
                joint_facts, errors["joint"] = [], f"{type(exc).__name__}: {exc}"[:300]
            try:
                comp = client.complete_json(COMPREHENSION_V3.format(
                    facts=json.dumps(joint_facts[:40], ensure_ascii=False), packet=rendered),
                    system=EVIDENCE_SYSTEM, max_tokens=4096, tag=args.tag)
                raw_relations = comp.get("relations", []) if isinstance(comp, dict) else []
            except Exception as exc:  # noqa: BLE001
                raw_relations, errors["comprehension"] = [], f"{type(exc).__name__}: {exc}"[:300]
            existing = [(str(x.get("subject", "")), str(x.get("predicate", "")), str(x.get("object", "")))
                        for x in joint_facts if isinstance(x, dict)]
            validated, rule_rejected = [], []
            for item in raw_relations:
                if not isinstance(item, dict):
                    rule_rejected.append({"item": item, "reason": "not_object"})
                    continue
                relation, reason = validate_relation(item, packet, chunks[novel_id], existing)
                if relation:
                    relation["relation_id"] = f"{packet.packet_id}:r{len(validated):02d}"
                    validated.append(relation)
                else:
                    rule_rejected.append({"item": item, "reason": reason})
            verdicts = []
            if validated:
                candidates = [{**r, "id": f"c{i}"} for i, r in enumerate(validated)]
                try:
                    out = client.complete_json(VERIFY_RELATIONS_V3.format(
                        candidates=json.dumps(candidates, ensure_ascii=False), packet=rendered),
                        system=EVIDENCE_SYSTEM, max_tokens=3072, tag=args.tag)
                    verdicts = out.get("verdicts", []) if isinstance(out, dict) else []
                except Exception as exc:  # noqa: BLE001
                    errors["verifier"] = f"{type(exc).__name__}: {exc}"[:300]
            by_id = {str(v.get("id")): v for v in verdicts if isinstance(v, dict)}
            all_candidates, kept, dropped = [], [], []
            for i, relation in enumerate(validated):
                verdict = by_id.get(f"c{i}", {})
                relation = {**relation, "verifier": verdict,
                            "fixed_status": verdict.get("fixed_status") or relation["status"]}
                all_candidates.append(relation)
                good = (verdict.get("keep") is True and verdict.get("semantic_support") in {"strong", "partial"}
                        and verdict.get("cross_chunk_necessary") is True and verdict.get("specificity") != "low")
                (kept if good else dropped).append(relation)
            row = {"packet_id": packet.packet_id, "novel_id": novel_id, "anchor": packet.anchor,
                   "chunk_ids": packet.chunk_ids, "packet_words": packet.n_words,
                   "joint_facts": joint_facts, "raw_relations": raw_relations,
                   "rule_rejected": rule_rejected, "validated": all_candidates,
                   "kept": kept, "dropped_by_verifier": dropped, "errors": errors}
            rows.append(row)
            partial.write(json.dumps(row, ensure_ascii=False) + "\n")
            partial.flush()
            print(packet.packet_id, "joint", len(joint_facts), "raw", len(raw_relations),
                  "valid", len(validated), "kept", len(kept), errors, flush=True)
    partial.close()
    write_jsonl(OUT / f"{args.out}.jsonl", rows)
    validated = [r for row in rows for r in row["validated"]]
    kept = [r for row in rows for r in row["kept"]]
    summary = {
        "n_packets": len(rows), "n_joint_facts": sum(len(r["joint_facts"]) for r in rows),
        "n_raw": sum(len(r["raw_relations"]) for r in rows), "n_validated": len(validated),
        "n_kept": len(kept), "rule_reject_reasons": dict(Counter(x["reason"] for r in rows for x in r["rule_rejected"])),
        "verifier_reasons": dict(Counter(str(x.get("verifier", {}).get("reason", "no_verdict"))[:50]
                                         for r in rows for x in r["dropped_by_verifier"])),
        "validated_stats": {n: relation_stats([x for x in validated if x["source_chunk_ids"][0].startswith(f"dq-{n}-")],
                                              chunks[n]) for n in args.novels.split(",")},
        "kept_stats": {n: relation_stats([x for x in kept if x["source_chunk_ids"][0].startswith(f"dq-{n}-")],
                                         chunks[n]) for n in args.novels.split(",")},
        "errors": dict(Counter(k for r in rows for k in r["errors"])), "client": client.summary(),
    }
    (OUT / f"{args.out}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:12000])


if __name__ == "__main__":
    main()
