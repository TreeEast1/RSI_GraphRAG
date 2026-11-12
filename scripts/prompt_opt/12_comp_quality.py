#!/usr/bin/env python
"""Phase 3.3 — audit comprehension relation quality without any LLM call.

For the original C3/C6 relations and every COMP-Px extraction it reports:
  * counts by relation_type / derivation
  * subject/object/predicate length
  * whether both endpoints already exist in the C2 entity set (so the relation
    attaches to query-matchable graph entities)
  * lexical redundancy against the C2 atomic facts (stemmed Jaccard)
  * citation validity and cross-session count
  * confidence distribution
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from nltk.stem import PorterStemmer  # noqa: E402

from src.config import PATHS  # noqa: E402

OUT = ROOT / "outputs" / "prompt_optimization_conv30" / "comprehension"
PS = PorterStemmer()
ORDER_RE = re.compile(r"_before|_after|preceded|followed|before_|after_")


def stems(text):
    return {PS.stem(w) for w in re.findall(r"[a-z0-9']+", str(text).lower()) if len(w) > 1}


def jacc(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def fact_keys(triples):
    return [(t["subject"].lower(), t["predicate"].lower(), t["object"].lower()) for t in triples]


def audit(name, relations, triples, source):
    ents = set()
    for t in triples:
        ents.add(t["subject"].lower())
        ents.add(t["object"].lower())
    fact_stems = [stems(f"{t['subject']} {t['predicate']} {t['object']}") for t in triples]
    rows = []
    for r in relations:
        s, p, o = str(r["subject"]), str(r["predicate"]), str(r["object"])
        rs = stems(f"{s} {p} {o}")
        max_j = max((jacc(rs, fs) for fs in fact_stems), default=0.0)
        ss = [x for x in r.get("source_sessions", [])]
        rows.append({
            "subject": s, "predicate": p, "object": o,
            "relation_type": r.get("relation_type"), "derivation": r.get("derivation"),
            "confidence": r.get("confidence"),
            "subject_words": len(s.split()), "object_words": len(o.split()),
            "predicate_words": len(p.replace("_", " ").split()),
            "both_existing": s.lower() in ents and o.lower() in ents,
            "subject_existing": s.lower() in ents, "object_existing": o.lower() in ents,
            "max_fact_jaccard": round(max_j, 3),
            "near_duplicate_of_fact": max_j >= 0.8,
            "n_sessions": len(ss), "sessions": ss,
            "ordering_predicate": bool(ORDER_RE.search(f"{p} {o}".lower())),
            "source_dia_ids": r.get("source_dia_ids", []),
        })
    n = len(rows) or 1
    summary = {
        "name": name, "source": str(source), "n_relations": len(rows),
        "by_type": dict(Counter(r["relation_type"] for r in rows)),
        "by_derivation": dict(Counter(r["derivation"] for r in rows)),
        "mean_predicate_words": round(float(np.mean([r["predicate_words"] for r in rows])), 2) if rows else None,
        "max_predicate_words": max([r["predicate_words"] for r in rows], default=None),
        "predicate_ge5": sum(r["predicate_words"] >= 5 for r in rows),
        "object_gt6": sum(r["object_words"] > 6 for r in rows),
        "both_existing": sum(r["both_existing"] for r in rows),
        "any_existing_endpoint": sum(r["both_existing"] or r["subject_existing"] or r["object_existing"] for r in rows),
        "near_duplicate_of_fact": sum(r["near_duplicate_of_fact"] for r in rows),
        "ordering_predicate": sum(r["ordering_predicate"] for r in rows),
        "n_sessions_mean": round(float(np.mean([r["n_sessions"] for r in rows])), 2) if rows else None,
        "confidence_mean": round(float(np.mean([r["confidence"] for r in rows if r["confidence"] is not None])), 3) if rows else None,
        "fraction_both_existing": round(sum(r["both_existing"] for r in rows) / n, 3),
        "fraction_ordering": round(sum(r["ordering_predicate"] for r in rows) / n, 3),
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", default="COMP-P1,COMP-P2,COMP-P3,COMP-P4,COMP-P5")
    args = ap.parse_args()

    c2 = json.loads((PATHS["comprehension"] / "conv-30" / "cluster_hybrid_0.5_b8192_triple.json").read_text())
    triples = c2["triples"]
    items = []
    c3 = json.loads((PATHS["comprehension"] / "conv-30" / "cluster_hybrid_0.5_b8192_comp.json").read_text())
    items.append(audit("COMP-P0 (original C3)", c3["relations"], triples,
                       PATHS["comprehension"] / "conv-30" / "cluster_hybrid_0.5_b8192_comp.json"))
    c6 = json.loads((PATHS["comprehension"] / "conv-30" / "full_conv_comp.json").read_text())
    items.append(audit("C6 (full-conv original)", c6["relations"], c6["triples"],
                       PATHS["comprehension"] / "conv-30" / "full_conv_comp.json"))
    for pid in args.pids.split(","):
        p = OUT / "extractions" / f"{pid}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        items.append(audit(f"{pid} — {d.get('name','')}", d["relations"], triples, p))

    summaries = [x[0] for x in items]
    detail = {s["name"]: rows for s, rows in items}
    (OUT / "quality_report.json").write_text(json.dumps(
        {"summaries": summaries, "relations": detail}, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Comprehension relation quality audit (conv-30)", "",
             "| source | n | temporal_ordering | both endpoints existing | pred>=5 words | ordering predicate | near-dup of fact | mean conf |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['name']} | {s['n_relations']} | {s['by_type'].get('temporal_ordering', 0)} | "
                     f"{s['both_existing']} ({s['fraction_both_existing']:.0%}) | {s['predicate_ge5']} | "
                     f"{s['ordering_predicate']} | {s['near_duplicate_of_fact']} | {s['confidence_mean']} |")
    for s in summaries:
        lines += ["", f"## {s['name']}", ""]
        lines.append(f"- types: {s['by_type']}")
        lines.append(f"- derivation: {s['by_derivation']}")
        lines.append(f"- predicate words mean/max: {s['mean_predicate_words']}/{s['max_predicate_words']}; "
                     f"object >6 words: {s['object_gt6']}; any existing endpoint: {s['any_existing_endpoint']}")
        lines.append(f"- sessions per relation: {s['n_sessions_mean']}")
        lines += ["", "| subject | predicate | object | type | both-exist | pred-w | near-dup | sessions |",
                  "|---|---|---|---|---:|---:|---:|---|"]
        for r in detail[s["name"]]:
            lines.append(f"| {r['subject']} | {r['predicate']} | {r['object']} | {r['relation_type']} | "
                         f"{'Y' if r['both_existing'] else '-'} | {r['predicate_words']} | "
                         f"{'Y' if r['near_duplicate_of_fact'] else '-'} | {r['sessions']} |")
    (OUT / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
