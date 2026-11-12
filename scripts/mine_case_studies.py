#!/usr/bin/env python
"""Mine real case studies by diffing two per-question result files.

Usage:
  python scripts/mine_case_studies.py --base outputs/qa/baselines_hipporag_conv-30.jsonl \
      --new outputs/qa/readtime_<variant>_conv-30.jsonl --out outputs/reports/cases_conv-30.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    return {r["qid"]: r for r in (json.loads(l) for l in Path(p).read_text().splitlines() if l.strip())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-type", type=int, default=5)
    args = ap.parse_args()

    base, new = load(args.base), load(args.new)
    qids = sorted(set(base) & set(new))
    cases = defaultdict(list)
    for qid in qids:
        b, n = base[qid], new[qid]
        rb = (b.get("retrieval") or {}).get("recall@5")
        rn = (n.get("retrieval") or {}).get("recall@5")
        eb, en = b.get("em"), n.get("em")
        rec = {
            "qid": qid, "conv_id": b["conv_id"], "question": b["question"],
            "gold_answer": b.get("gold_answer"), "category": b.get("category"),
            "gold_sessions": b.get("gold_sessions"), "gold_turns": b.get("gold_turns"),
            "base_topk": b.get("ranked_docs", [])[:10],
            "new_topk": n.get("ranked_docs", [])[:10],
            "base_answer": b.get("pred_answer"), "new_answer": n.get("pred_answer"),
            "base_recall5": rb, "new_recall5": rn,
            "base_em": eb, "new_em": en,
            "base_evidence4096": ((b.get("budget") or {}).get("4096") or {}).get("evidence_full"),
            "new_evidence4096": ((n.get("budget") or {}).get("4096") or {}).get("evidence_full"),
        }
        if (rb == 0) and rn and rn > 0:
            cases["retrieval_gain"].append(rec)
        if (eb == 0) and en == 1:
            cases["qa_gain"].append(rec)
        if (eb == 1) and en == 0:
            cases["qa_regression"].append(rec)
        if (rb and rb > 0) and (rn == 0):
            cases["retrieval_regression"].append(rec)

    out = {k: v[: args.max_per_type] for k, v in cases.items()}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k, v in out.items():
        print(f"{k}: {len(v)}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
