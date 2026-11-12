#!/usr/bin/env python
"""Experiment B: entity-first unsupervised grouping of session documents."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import NVEMBED_PATH, PATHS  # noqa: E402
from src.dataset.locomo import load_locomo, num_tokens  # noqa: E402
from src.grouping.evaluate import grouping_metrics  # noqa: E402
from src.grouping.methods import GroupingConfig, build_grouping  # noqa: E402

sys.path.insert(0, str(PATHS["third_party"] / "src"))


def load_entity_sets(conv_id: str) -> dict:
    wd = PATHS["raw_graphs"] / conv_id / f"hy4_{NVEMBED_PATH.replace('/', '_')}"
    meta_p = wd / "chunk_metadata.json"
    openie_p = wd / "openie_state.json"
    if not openie_p.exists():
        raise FileNotFoundError(f"build G_raw first: {openie_p} missing")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    payload = json.loads(openie_p.read_text())
    doc_to_ents = {}
    for row in payload.get("docs", []):
        m = meta.get(row["idx"], {})
        doc_id = m.get("document_id")
        if not doc_id:
            continue
        doc_to_ents[doc_id] = set(str(e) for e in row.get("extracted_entities", []))
    return doc_to_ents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--convs", default=None)
    ap.add_argument("--budgets", default="4096,8192")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--methods", default=("random,temporal,entity_jaccard,entity_idf,"
                                          "embedding,hybrid_0.25,hybrid_0.5,hybrid_0.75,community"))
    args = ap.parse_args()

    convs = {c.sample_id: c for c in load_locomo(ROOT / "data" / "raw" / "locomo" / "locomo10.json")}
    split = json.loads((PATHS["splits"] / "split.json").read_text())
    if args.convs:
        conv_ids = args.convs.split(",")
    elif args.split == "all":
        conv_ids = sorted(convs)
    else:
        conv_ids = split[args.split]

    from src.embedding.nvembed import NVEmbedder
    from src.graph.runtime import EmbeddingProvider, make_base_config

    cfg = make_base_config(PATHS["raw_graphs"] / "_warmup")
    provider = EmbeddingProvider.get(cfg)
    import os as _os
    _shard = _os.environ.get("HY4_CACHE_SHARD", "")
    embedder = NVEmbedder(provider.model, PATHS["cache"] / f"doc_embeddings{_shard}.pkl")

    methods = args.methods.split(",")
    results = []
    for conv_id in conv_ids:
        conv = convs[conv_id]
        docs = {s.document_id: s for s in conv.sessions}
        doc_ids = [s.document_id for s in conv.sessions]
        token_sizes = {d: docs[d].n_tokens for d in doc_ids}
        t = time.time()
        emb = embedder.encode_documents([docs[d].text for d in doc_ids])
        print(f"{conv_id}: embeddings {emb.shape} in {time.time()-t:.1f}s")
        entity_sets = load_entity_sets(conv_id)
        missing = [d for d in doc_ids if d not in entity_sets]
        if missing:
            print(f"  WARNING: {len(missing)} docs missing entity sets")

        for budget in [int(b) for b in args.budgets.split(",")]:
            for method in methods:
                seeds = [int(s) for s in args.seeds.split(",")] if method == "random" else [0]
                for seed in seeds:
                    cfg_g = GroupingConfig(budget_tokens=budget, seed=seed)
                    t = time.time()
                    res = build_grouping(method, conv_id, doc_ids, token_sizes,
                                         entity_sets, emb, cfg_g)
                    m = grouping_metrics(res.groups, conv.qa, token_sizes, {})
                    rec = {"conv_id": conv_id, "method": method, "seed": seed,
                           "budget": budget, "n_groups": len(res.groups),
                           "build_s": round(time.time() - t, 2), **m,
                           **{f"param_{k}": v for k, v in res.params.items()}}
                    results.append(rec)
                    out = PATHS["groupings"] / conv_id / f"{method}_b{budget}_s{seed}.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps({"result": rec, "groups": res.groups}, indent=1))
                    print(f"  {method:16s} b={budget} s={seed} groups={len(res.groups):2d} "
                          f"full={m['gold_full_coverage']:.3f} partial={m['gold_partial_coverage']:.3f} "
                          f"cross={m['cross_session_colocation']:.3f} cost={m['reading_cost_ratio']:.2f}")

    out = PATHS["reports"] / "grouping_runs.json"
    out.write_text(json.dumps(results, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
