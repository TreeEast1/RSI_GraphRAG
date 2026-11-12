"""Build HippoRAG graph variants from read-time v3 relations without changing raw graphs."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detectiveqa.data import OUT, load_prepared, read_jsonl  # noqa: E402
from src.detectiveqa.graph import build_variant_graph  # noqa: E402
from src.detectiveqa.readtime import extra_by_latest_chunk  # noqa: E402


def joint_by_latest(row, chunks):
    order = {c.chunk_id: c.available_at for c in chunks}
    out = defaultdict(list)
    for fact in row.get("joint_facts", []):
        ids = [x for x in fact.get("source_chunk_ids", []) if x in order]
        if ids and all(fact.get(k) for k in ("subject", "predicate", "object")):
            latest = max(ids, key=order.get)
            out[latest].append((str(fact["subject"]), str(fact["predicate"]), str(fact["object"])))
    return out


def flatten(mapping):
    return [(cid, triple) for cid, triples in mapping.items() for triple in triples]


def regroup(items):
    out = defaultdict(list)
    for cid, triple in items:
        out[cid].append(triple)
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readtime", default="readtime_v3.jsonl")
    ap.add_argument("--novels", default="27,79,117,118,121,124")
    ap.add_argument("--variants", default="r2_joint,r3_validated,r4_kept,r5_joint_count")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="relation_variant_builds.json")
    args = ap.parse_args()
    _, chunks, _ = load_prepared()
    rows = read_jsonl(OUT / args.readtime)
    variants = args.variants.split(",")
    infos = []
    for novel_id in args.novels.split(","):
        novel_rows = [r for r in rows if r["novel_id"] == novel_id]
        joint = defaultdict(list)
        for row in novel_rows:
            for cid, triples in joint_by_latest(row, chunks[novel_id]).items():
                joint[cid].extend(triples)
        validated = extra_by_latest_chunk(x for r in novel_rows for x in r["validated"])
        kept = extra_by_latest_chunk(x for r in novel_rows for x in r["kept"])
        joint_items = flatten(joint)
        rng = random.Random(f"r5|{novel_id}")
        rng.shuffle(joint_items)
        n_kept = sum(len(v) for v in kept.values())
        payloads = {"r2_joint": dict(joint), "r3_validated": validated, "r4_kept": kept,
                    "r5_joint_count": regroup(joint_items[:n_kept])}
        for variant in variants:
            info = build_variant_graph(novel_id, chunks[novel_id], variant, payloads[variant], force=args.force)
            info["requested_triples"] = sum(len(v) for v in payloads[variant].values())
            infos.append(info)
            print(novel_id, variant, info.get("injection"), info.get("requested_triples"), flush=True)
    (OUT / args.out).write_text(json.dumps(infos, indent=1, default=str))


if __name__ == "__main__":
    main()
