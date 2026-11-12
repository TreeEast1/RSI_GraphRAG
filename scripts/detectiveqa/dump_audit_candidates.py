"""Prepare comprehension candidates for manual audit.

Sampling frame: first all verifier-kept relations, then validated-but-dropped
relations, then raw relations rejected by rule checks, until the requested
number is reached.  Each row carries the full cited chunks so a human can judge
whether the relation truly needs cross-chunk evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detectiveqa.data import OUT, load_prepared, write_jsonl  # noqa: E402


def context_for(chunks, chunk_ids: List[str], limit: int = 2400) -> List[Dict]:
    by_id = {c.chunk_id: c for c in chunks}
    out = []
    for cid in chunk_ids:
        chunk = by_id.get(cid)
        if chunk is None:
            out.append({"chunk_id": cid, "text": "<missing>"})
        else:
            out.append({"chunk_id": cid, "start_pid": chunk.start_pid, "end_pid": chunk.end_pid,
                        "text": chunk.text[:limit]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="readtime_smoke_raw.jsonl")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--kept-max", type=int, default=20,
                    help="at most this many verifier-kept relations; the rest are verifier-dropped")
    ap.add_argument("--out", default="readtime_audit_candidates.jsonl")
    args = ap.parse_args()
    _, chunks, _ = load_prepared()
    rows = [json.loads(line) for line in (OUT / args.input).read_text().splitlines() if line.strip()]
    candidates = []
    # 1. verifier-kept
    for row in rows:
        for relation in row["kept"]:
            candidates.append(("kept", row, relation, ""))
    kept = candidates[:args.kept_max]
    # 2. validated but dropped by verifier
    for row in rows:
        for drop in row["dropped_by_verifier"]:
            candidates.append(("verifier_dropped", row, drop["relation"], str(drop.get("reason", ""))))
    dropped = [c for c in candidates if c[0] == "verifier_dropped"]
    # 3. raw relations rejected by rule checks (last resort)
    for row in rows:
        for rej in row["rejected"]:
            item = rej.get("item")
            if isinstance(item, dict):
                candidates.append(("rule_rejected", row, item, str(rej.get("reason", ""))))
    rejected = [c for c in candidates if c[0] == "rule_rejected"]
    ordered = kept + dropped + rejected
    out = []
    for stage, row, relation, reason in ordered[:args.n]:
        cited = [str(x) for x in (relation.get("source_chunk_ids") or [])]
        if not cited:
            cited = [e.get("chunk_id") for e in (relation.get("evidence") or [])
                     if isinstance(e, dict) and e.get("chunk_id")]
        out.append({
            "audit_id": f"a{len(out):02d}", "stage": stage, "stage_reason": reason,
            "packet_id": row["packet_id"], "novel_id": row["novel_id"], "anchor": row["anchor"],
            "relation": relation, "source_chunk_ids": cited,
            "chunk_context": context_for(chunks[row["novel_id"]], cited),
            "joint_facts_same_packet": row["joint_facts"],
            "manual_label": "", "manual_note": "",
        })
    write_jsonl(OUT / args.out, out)
    print(json.dumps({"n_candidates_total": len(candidates), "n_sampled": len(out),
                      "stages": {s: sum(1 for x in out if x["stage"] == s)
                                 for s in {x["stage"] for x in out}}}, indent=1))


if __name__ == "__main__":
    main()
