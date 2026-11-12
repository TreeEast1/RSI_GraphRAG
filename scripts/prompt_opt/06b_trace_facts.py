#!/usr/bin/env python
"""Post-hoc fact provenance tracing for the comprehension variants.

The diagnostics already store ``top_facts`` for every question.  This script
matches those facts back to (a) the fixed C2 triples and (b) the new relations,
using the same normalisation HippoRAG applies, and reports how often each new
relation actually entered the top-5 fact candidates.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import PATHS  # noqa: E402
from src.prompt_optimization.retrieval_diag import norm_fact  # noqa: E402

OUT = ROOT / "outputs" / "prompt_optimization_conv30" / "comprehension"
PID2VAR = {"COMP-P1": "cmpo_compp1", "COMP-P2": "cmpo_compp2", "COMP-P3": "cmpo_compp3",
           "COMP-P4": "cmpo_compp4", "COMP-P5": "cmpo_compp5"}


def main():
    c2 = json.loads((PATHS["comprehension"] / "conv-30" / "cluster_hybrid_0.5_b8192_triple.json").read_text())
    c2_map = {}
    for t in c2["triples"]:
        c2_map[norm_fact((t["subject"], t["predicate"], t["object"]))] = "C2_explicit"

    summary = {}
    for pid, var in PID2VAR.items():
        p = OUT / "extractions" / f"{pid}.json"
        rp = OUT / "retrieval" / var / "records.jsonl"
        if not p.exists() or not rp.exists():
            continue
        rels = json.loads(p.read_text())["relations"]
        rel_map = {}
        for i, r in enumerate(rels):
            key = norm_fact((r["subject"], r["predicate"], r["object"]))
            rel_map[key] = {"idx": i, "subject": r["subject"], "predicate": r["predicate"],
                            "object": r["object"], "sessions": r["source_sessions"],
                            "relation_type": r["relation_type"]}
        records = [json.loads(l) for l in open(rp, encoding="utf-8")]
        usage = Counter()
        per_q = []
        for rec in records:
            facts = rec.get("diag", {}).get("top_facts", [])
            new_hits, c2_hits = [], []
            for f in facts:
                key = norm_fact(f)
                if key in rel_map:
                    info = rel_map[key]
                    new_hits.append({"fact": f, **info})
                    usage[(info["idx"], f"{info['subject']} | {info['predicate']} | {info['object']}")] += 1
                elif key in c2_map:
                    c2_hits.append(f)
            if new_hits:
                per_q.append({"qid": rec["qid"], "question": rec["question"],
                              "category": rec["category"], "new_fact_hits": new_hits})
        by_idx = {v["idx"]: v for v in rel_map.values()}
        summary[pid] = {
            "variant": var, "n_relations": len(rels),
            "n_questions_with_new_fact_in_top5": len(per_q),
            "n_new_fact_hits": int(sum(usage.values())),
            "relation_usage": [
                {"idx": idx, "count": count, "subject": by_idx[idx]["subject"],
                 "predicate": by_idx[idx]["predicate"], "object": by_idx[idx]["object"],
                 "sessions": by_idx[idx]["sessions"], "relation_type": by_idx[idx]["relation_type"]}
                for (idx, _), count in usage.most_common()
            ],
            "questions": per_q,
        }
        print(f"{pid}: {len(rels)} relations, new facts in top-5 for "
              f"{len(per_q)}/105 questions, {sum(usage.values())} hits; "
              f"used relations: {len(usage)}")
        for (idx, label), count in usage.most_common(5):
            print(f"    {count:>3}x  {label}")
    (OUT / "fact_trace_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
