#!/usr/bin/env python
"""Read-time extraction/comprehension (experiment C) — LLM stage only.

Produces, per conversation and scope:
  outputs/comprehension/{conv}/{scope}.json
with per-group triples (Explicit), relations (Composed/Inferred), rejects and
token accounting.  Graphs are built separately by run_read_time_eval.py.

Scopes
------
cluster_<method>_b<budget>_triple      : C2 joint extraction over groups
cluster_<method>_b<budget>_comp        : C3 = C2 + comprehension
cluster_<method>_b<budget>_comp_ver    : C3 + LLM verification
full_conv_triple / full_conv_comp      : C5 / C6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.comprehension.extract import run_scope  # noqa: E402
from src.config import PATHS  # noqa: E402
from src.dataset.locomo import load_locomo  # noqa: E402


def load_groups(conv_id: str, method: str, budget: int, seed: int = 0):
    p = PATHS["groupings"] / conv_id / f"{method}_b{budget}_s{seed}.json"
    if not p.exists():
        raise FileNotFoundError(f"grouping missing: {p} (run run_grouping.py first)")
    return json.loads(p.read_text())["groups"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--scopes", default="cluster_hybrid_0.5_b8192")
    ap.add_argument("--modes", default="triple,comp")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--grouping-method", default="hybrid_0.5")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-groups", type=int, default=None)
    args = ap.parse_args()

    from src.adapters.hy4_client import HY4Client

    convs = {c.sample_id: c for c in load_locomo(ROOT / "data" / "raw" / "locomo" / "locomo10.json")}
    split = json.loads((PATHS["splits"] / "split.json").read_text())
    conv_ids = args.convs.split(",") if args.convs else (
        sorted(convs) if args.split == "all" else split[args.split])

    client = HY4Client(tag="read_time")
    for conv_id in conv_ids:
        conv = convs[conv_id]
        all_ids = [s.document_id for s in conv.sessions]
        for scope in args.scopes.split(","):
            if scope.startswith("full_conv"):
                groups = [all_ids]
                tag = "full"
            else:
                # scope name is cluster_{method}_b{budget}; parse it so every scope
                # really uses its own grouping instead of one global flag.
                method, budget = args.grouping_method, args.budget
                head = scope.split("_b")
                if len(head) == 2 and head[0].startswith("cluster_"):
                    method = head[0][len("cluster_"):]
                    budget = int(head[1])
                groups = load_groups(conv_id, method, budget)
                tag = "cluster"
            if args.max_groups:
                groups = groups[: args.max_groups]
            for mode in args.modes.split(","):
                do_comp = mode.startswith("comp")
                do_ver = mode.endswith("_ver")
                name = f"{scope}_{mode}" if tag == "cluster" else f"full_conv_{mode}"
                out_p = PATHS["comprehension"] / conv_id / f"{name}.json"
                if out_p.exists():
                    print(f"  skip existing {out_p.name}")
                    continue
                print(f"\n=== {conv_id} / {name}: {len(groups)} groups ===")
                t = time.time()
                res = run_scope(client, conv, groups, do_comprehension=do_comp,
                                do_verification=do_ver, max_tokens=args.max_tokens,
                                tag_prefix=f"{conv_id}_{name}")
                res["scope_name"] = name
                res["elapsed_s"] = round(time.time() - t, 1)
                res["grouping_method"] = args.grouping_method if tag == "cluster" else "full"
                res["budget_tokens"] = args.budget if tag == "cluster" else None
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text(json.dumps(res, ensure_ascii=False, indent=1))
                print(f"  done {time.time()-t:.0f}s triples={res['n_triples']} relations={res['n_relations']}")
    print("HY4 stats:", client.summary())


if __name__ == "__main__":
    main()
