"""Aggregate per-novel graph build cost, size and OpenIE failure statistics."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import NVEMBED_PATH  # noqa: E402
from src.detectiveqa.data import OUT, load_prepared  # noqa: E402
from src.detectiveqa.graph import graph_dir  # noqa: E402
from src.graph.variants import working_dir_of  # noqa: E402


def llm_usage(novel_id: str) -> Dict:
    cache = graph_dir(novel_id) / "llm_cache"
    out = {"cache_rows": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
           "cached_tokens": 0, "reasoning_tokens": 0}
    for db in sorted(cache.glob("*.sqlite")):
        conn = sqlite3.connect(db)
        try:
            rows = conn.execute("SELECT metadata FROM cache").fetchall()
        finally:
            conn.close()
        for (meta,) in rows:
            try:
                data = json.loads(meta)
            except (TypeError, ValueError):
                continue
            out["cache_rows"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                        "cached_tokens", "reasoning_tokens"):
                out[key] += int(data.get(key) or 0)
    return out


def openie_stats(novel_id: str) -> Dict:
    state = working_dir_of(graph_dir(novel_id), "hy4", NVEMBED_PATH) / "openie_state.json"
    if not state.exists():
        return {"state_exists": False}
    payload = json.loads(state.read_text(encoding="utf-8"))
    docs = payload.get("docs", [])
    empty_entities = empty_triples = errors = 0
    token_like = 0
    for row in docs:
        ents = row.get("extracted_entities") or []
        triples = row.get("extracted_triples") or []
        meta = row.get("metadata") or {}
        if not ents:
            empty_entities += 1
        if not triples:
            empty_triples += 1
        if meta.get("error") or meta.get("patched"):
            errors += 1
        token_like += len(ents) + len(triples)
    return {"state_exists": True, "openie_rows": len(docs),
            "empty_entity_rows": empty_entities, "empty_triple_rows": empty_triples,
            "rows_with_error_metadata": errors, "mean_entities_plus_triples_per_chunk":
            round(token_like / len(docs), 2) if docs else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--novels", default="27,79,117,118,121,124")
    ap.add_argument("--out", default="raw_graph_build_stats.json")
    args = ap.parse_args()
    _, chunks, _ = load_prepared()
    rows: List[Dict] = []
    for novel_id in args.novels.split(","):
        marker = graph_dir(novel_id) / "detectiveqa_graph_info.json"
        if not marker.exists():
            print(f"SKIP {novel_id}: no marker")
            continue
        info = json.loads(marker.read_text())
        row = {k: info.get(k) for k in
               ("novel_id", "n_chunks", "num_phrase_nodes", "num_passage_nodes",
                "num_total_nodes", "num_extracted_triples", "num_triples_with_passage_node",
                "num_synonymy_triples", "num_total_triples", "build_seconds")}
        row["patched_chunks"] = len(info.get("patched_chunks") or [])
        row["actual_chunks"] = len(chunks[novel_id])
        row.update(llm_usage(novel_id))
        row.update(openie_stats(novel_id))
        rows.append(row)
        print(json.dumps(row), flush=True)
    totals = {k: sum(r.get(k) or 0 for r in rows) for k in
              ("n_chunks", "num_total_nodes", "num_extracted_triples", "num_total_triples",
               "build_seconds", "prompt_tokens", "completion_tokens", "total_tokens",
               "cache_rows", "patched_chunks")}
    report = {"per_novel": rows, "totals": totals}
    (OUT / args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(totals, indent=1))


if __name__ == "__main__":
    main()
