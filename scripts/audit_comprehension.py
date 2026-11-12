#!/usr/bin/env python
"""Audit comprehension output (report section 11.1).

Classifies every candidate relation into:
  A  already in raw OpenIE (exact triple)
  B  paraphrase/synonym of a raw triple (char n-gram similarity)
  C  single-session relation (no cross-session evidence)
  D  cross-session relation supported by >=2 sessions  <- the target class
  E  cross-session but low-confidence / unverified inference
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import NVEMBED_PATH, PATHS  # noqa: E402
from src.dataset.locomo import load_locomo  # noqa: E402


def load_raw_triples(conv_id: str):
    wd = PATHS["raw_graphs"] / conv_id / f"hy4_{NVEMBED_PATH.replace('/', '_')}"
    p = wd / "openie_state.json"
    payload = json.loads(p.read_text())
    triples = set()
    for row in payload["docs"]:
        for t in row.get("extracted_triples", []):
            triples.add(tuple(str(x) for x in t))
    return triples


def norm_triple(t):
    return tuple(" ".join(str(x).lower().split()) for x in t)


def best_similarity(triple, raw_norm, raw_list, threshold=0.85):
    a = " ".join(norm_triple(triple))
    best = 0.0
    for r in raw_list:
        s = difflib.SequenceMatcher(None, a, " ".join(r)).ratio()
        if s > best:
            best = s
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--scopes", default=None, help="comma separated scope names")
    args = ap.parse_args()

    convs = {c.sample_id: c for c in load_locomo(ROOT / "data" / "raw" / "locomo" / "locomo10.json")}
    split = json.loads((PATHS["splits"] / "split.json").read_text())
    conv_ids = args.convs.split(",") if args.convs else (
        sorted(convs) if args.split == "all" else split[args.split])

    report = []
    for conv_id in conv_ids:
        raw = load_raw_triples(conv_id)
        raw_norm = {norm_triple(t) for t in raw}
        raw_list = list(raw_norm)
        scope_files = sorted((PATHS["comprehension"] / conv_id).glob("*.json"))
        if args.scopes:
            wanted = set(args.scopes.split(","))
            scope_files = [p for p in scope_files if p.stem in wanted]
        for p in scope_files:
            data = json.loads(p.read_text())
            rels = data.get("relations", [])
            if not rels:
                continue
            counts = Counter()
            examples = {"D": [], "E": [], "B": []}
            for r in rels:
                t = (r["subject"], r["predicate"], r["object"])
                sessions = r.get("source_sessions", [])
                if t in raw or norm_triple(t) in raw_norm:
                    counts["A"] += 1
                    continue
                sim = best_similarity(t, raw_norm, raw_list)
                if sim >= 0.85:
                    counts["B"] += 1
                    if len(examples["B"]) < 3:
                        examples["B"].append({"relation": t, "sim": round(sim, 3)})
                    continue
                if len(sessions) < 2:
                    counts["C"] += 1
                    continue
                conf = float(r.get("confidence", 0))
                if conf < 0.6 or r.get("verification_status") == "rejected":
                    counts["E"] += 1
                    if len(examples["E"]) < 3:
                        examples["E"].append({"relation": t, "confidence": conf,
                                              "sessions": sessions})
                    continue
                counts["D"] += 1
                if len(examples["D"]) < 5:
                    examples["D"].append({"relation": t, "sessions": sessions,
                                          "derivation": r.get("derivation"),
                                          "evidence": r.get("evidence", "")[:160]})
            total = sum(counts.values())
            rec = {"conv_id": conv_id, "scope": p.stem, "n_relations": len(rels),
                   "counts": dict(counts),
                   "pct_D": counts["D"] / total if total else 0.0,
                   "pct_A_dup": counts["A"] / total if total else 0.0,
                   "examples": examples}
            report.append(rec)
            print(f"{conv_id} {p.stem:45s} n={len(rels):3d} A={counts['A']:3d} B={counts['B']:3d} "
                  f"C={counts['C']:3d} D={counts['D']:3d} E={counts['E']:3d}")

    out = PATHS["reports"] / "comprehension_audit.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
