"""Sample read-time v3 candidates for the required 60-item manual audit.

Sampling frame: 30 verifier-kept, 20 verifier-dropped, 10 rule-rejected, round-robin
across the six pilot novels so every novel is represented in every stage where
possible.  Gold answers/clues are never read here.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detectiveqa.data import OUT, load_prepared, write_jsonl  # noqa: E402

NOVELS = ["27", "79", "117", "118", "121", "124"]


def context_for(chunks, chunk_ids: List[str], limit: int = 2500) -> List[Dict]:
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


def round_robin(candidates: Dict[str, List[Dict]], n: int) -> List[Dict]:
    chosen: List[Dict] = []
    while len(chosen) < n:
        progressed = False
        for novel in NOVELS:
            pool = candidates.get(novel, [])
            if pool:
                chosen.append(pool.pop(0))
                progressed = True
                if len(chosen) >= n:
                    break
        if not progressed:
            break
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readtime", default="readtime_v3.jsonl")
    ap.add_argument("--out", default="readtime_v3_audit_candidates.jsonl")
    ap.add_argument("--kept", type=int, default=30)
    ap.add_argument("--dropped", type=int, default=20)
    ap.add_argument("--rejected", type=int, default=10)
    args = ap.parse_args()
    _, chunks, _ = load_prepared()
    rows = [json.loads(line) for line in (OUT / args.readtime).read_text().splitlines() if line.strip()]
    pools: Dict[str, Dict[str, List[Dict]]] = {
        "kept": defaultdict(list), "verifier_dropped": defaultdict(list), "rule_rejected": defaultdict(list)}
    for row in rows:
        novel = row["novel_id"]
        for relation in row.get("kept", []):
            pools["kept"][novel].append({"row": row, "relation": relation, "reason": ""})
        for relation in row.get("dropped_by_verifier", []):
            pools["verifier_dropped"][novel].append({
                "row": row, "relation": relation,
                "reason": str((relation.get("verifier") or {}).get("reason", ""))})
        for rej in row.get("rule_rejected", []):
            item = rej.get("item")
            if isinstance(item, dict):
                pools["rule_rejected"][novel].append({
                    "row": row, "relation": item, "reason": str(rej.get("reason", ""))})
    out: List[Dict] = []
    for stage, n in (("kept", args.kept), ("verifier_dropped", args.dropped),
                     ("rule_rejected", args.rejected)):
        for candidate in round_robin(pools[stage], n):
            row, relation = candidate["row"], candidate["relation"]
            cited = [str(x) for x in (relation.get("source_chunk_ids") or [])]
            if not cited:
                cited = [str(e.get("chunk_id")) for e in (relation.get("evidence") or [])
                         if isinstance(e, dict) and e.get("chunk_id")]
            out.append({
                "audit_id": f"a{len(out):02d}", "stage": stage, "stage_reason": candidate["reason"],
                "packet_id": row["packet_id"], "novel_id": row["novel_id"], "anchor": row["anchor"],
                "relation": relation, "source_chunk_ids": cited,
                "chunk_context": context_for(chunks[row["novel_id"]], cited),
                "joint_facts_same_packet": row.get("joint_facts", []),
                "manual_label": "", "manual_note": "",
            })
    write_jsonl(OUT / args.out, out)
    from collections import Counter
    print(json.dumps({"n_sampled": len(out), "stages": dict(Counter(x["stage"] for x in out)),
                      "novels": dict(Counter(x["novel_id"] for x in out))}, indent=1))


if __name__ == "__main__":
    main()
