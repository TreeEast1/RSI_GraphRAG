#!/usr/bin/env python3
"""Timed traditional graph build with the current local model. Does not touch the 9B triple store."""
from __future__ import annotations

import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import abspath, append_jsonl, load_json, load_llm, parse_json_object, read_jsonl, save_json  # noqa: E402
from run_readtime import EXTRACT_SYSTEM, entity_in_docs  # noqa: E402


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    out = abspath("predictions", "speed_27b", "triples.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    docs = load_json(abspath("data", "documents.json"))
    read_ids = load_json(abspath("data", "read_doc_ids.json"))[:limit]
    client = load_llm()
    done = {r["doc_id"] for r in read_jsonl(out)}
    t0 = time.time()
    fresh_n = fresh_lat = fresh_in = fresh_out = fresh_kept = 0
    for i, did in enumerate(read_ids):
        if did in done:
            continue
        doc = docs[did]
        passage = f"Document {did}\nTitle: {doc['title']}\n{doc['text']}"
        resp = client.chat([
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": passage},
        ], max_output_tokens=700)
        obj = parse_json_object(resp["visible_text"])
        triples = []
        if obj:
            for item in obj.get("triples") or []:
                if isinstance(item, dict):
                    head, relation, tail = item.get("head"), item.get("relation"), item.get("tail")
                elif isinstance(item, list) and len(item) >= 3:
                    head, relation, tail = item[0], item[1], item[2]
                else:
                    continue
                head, relation, tail = str(head or "").strip(), str(relation or "").strip(), str(tail or "").strip()
                if not (head and relation and tail):
                    continue
                status = "accepted" if entity_in_docs(head, docs, [did]) and entity_in_docs(tail, docs, [did]) else "rejected"
                tid = "t-" + hashlib.sha1(f"{did}|{head}|{relation}|{tail}".encode()).hexdigest()[:12]
                triples.append({
                    "triple_id": tid, "head": head, "relation": relation, "tail": tail,
                    "source_ids": [did], "status": status, "origin": "speed_27b",
                })
        accepted = [t for t in triples if t["status"] == "accepted"]
        append_jsonl(out, {
            "doc_id": did, "json_ok": obj is not None, "triples": accepted,
            "n_rejected_unsupported": sum(1 for t in triples if t["status"] == "rejected"),
            "latency_s": resp["latency_s"], "prompt_tokens": resp["prompt_tokens"],
            "completion_tokens": resp["completion_tokens"],
        })
        fresh_n += 1
        fresh_lat += resp["latency_s"]
        fresh_in += resp["prompt_tokens"] or 0
        fresh_out += resp["completion_tokens"] or 0
        fresh_kept += len(accepted)
        tps = (fresh_out / fresh_lat) if fresh_lat else 0
        print(
            f"{i+1}/{len(read_ids)} {did} json={obj is not None} kept={len(accepted)} "
            f"sec={resp['latency_s']:.1f} out={resp['completion_tokens']} "
            f"avg_sec={fresh_lat/fresh_n:.1f} tok/s={tps:.1f}",
            flush=True,
        )
    rows = read_jsonl(out)
    summary = {
        "model": client.model,
        "docs_requested": len(read_ids),
        "docs_written": len({r["doc_id"] for r in rows}),
        "fresh_docs": fresh_n,
        "fresh_latency_s": fresh_lat,
        "fresh_prompt_tokens": fresh_in,
        "fresh_completion_tokens": fresh_out,
        "fresh_tok_per_s": (fresh_out / fresh_lat) if fresh_lat else 0,
        "fresh_sec_per_doc": (fresh_lat / fresh_n) if fresh_n else 0,
        "fresh_triples": fresh_kept,
        "wall_s": time.time() - t0,
        "json_ok": sum(1 for r in rows if r.get("json_ok")),
        "triples": sum(len(r.get("triples") or []) for r in rows),
    }
    save_json(abspath("predictions", "speed_27b", "summary.json"), summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
