#!/usr/bin/env python
"""P0 smoke test: HY4 -> NV-Embed -> HippoRAG2 index -> retrieve -> QA."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapters.hy4_client import HY4Client  # noqa: E402
from src.config import PATHS  # noqa: E402
from src.dataset.locomo import load_locomo  # noqa: E402
from src.graph.runtime import EmbeddingProvider, open_hipporag  # noqa: E402


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="conv-30")
    ap.add_argument("--n-sessions", type=int, default=3)
    ap.add_argument("--save-dir", default=str(PATHS["outputs"] / "smoke"))
    args = ap.parse_args()

    t0 = time.time()
    print("[1] HY4 client test")
    client = HY4Client(tag="smoke")
    out = client.chat([{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=16)
    print("    content:", out["content"][:80], "| usage:", out["usage"], "| cached:", out["cached"])

    print("[2] load LoCoMo")
    convs = {c.sample_id: c for c in load_locomo(ROOT / "data" / "raw" / "locomo" / "locomo10.json")}
    conv = convs[args.conv]
    sessions = conv.sessions[: args.n_sessions]
    docs = [s.text for s in sessions]
    print(f"    {conv.sample_id}: using {len(docs)} of {len(conv.sessions)} sessions, "
          f"{sum(len(s.turns) for s in sessions)} turns")

    from src.graph.runtime import make_base_config
    cfg = make_base_config(args.save_dir, conv_id=conv.sample_id)
    print("[3] load NV-Embed-v2")
    provider = EmbeddingProvider.get(cfg)
    print("    dim", provider.model.embedding_dim)

    print("[4] build HiPPoRAG2 index")
    t = time.time()
    with open_hipporag(args.save_dir, conv_id=conv.sample_id, embedding_provider=provider) as hippo:
        hippo.index(docs)
        print(f"    indexed in {time.time()-t:.1f}s")
        info = hippo.get_graph_info()
        print("    graph info:", json.dumps(info, ensure_ascii=False)[:400])
        print("    chunks:", len(hippo.chunk_embedding_store.get_all_ids()),
              "entities:", len(hippo.entity_embedding_store.get_all_ids()),
              "facts:", len(hippo.fact_embedding_store.get_all_ids()))

        # show a few triples
        infos = hippo._openie_info or []
        for row in infos[:2]:
            print("    sample triples from", row["idx"][:12], ":",
                  json.dumps(row["extracted_triples"][:4], ensure_ascii=False))

        print("[5] retrieve")
        qs = [q for q in conv.qa if q.evidence][:3]
        t = time.time()
        results = hippo.retrieve([q.question for q in qs], num_to_retrieve=5)
        print(f"    retrieved in {time.time()-t:.1f}s")
        for q, r in zip(qs, results):
            print(f"    Q: {q.question}")
            print(f"       gold sessions: {q.evidence_sessions} | top docs: {[m.get('document_id') for m in r.doc_metadata][:5]}")

        print("[6] QA with HippoRAG's official reader")
        sols, raws, metas = hippo.qa(results)
        for s, raw in zip(sols, raws):
            print("    answer:", s.answer[:100], "| raw:", raw[:80].replace("\n", " "))

    print(f"done in {time.time()-t0:.1f}s")
    print("HY4 stats:", client.summary())


if __name__ == "__main__":
    main()
