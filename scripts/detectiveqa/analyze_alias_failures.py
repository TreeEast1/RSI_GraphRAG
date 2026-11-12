"""Diagnose English name/alias mismatches between questions and story text.

This is a proxy analysis, not a proof: DetectiveQA questions sometimes use a
different romanisation or a diminutive than the novel body.  For every question
we take capitalised name candidates from the question and options, then mark
each as exact, a near-miss variant of some visible token (edit similarity >=
0.82), or absent from all text visible before the cutoff.  We report how these
categories relate to retrieval failure and keep samples for manual review.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detectiveqa.data import OUT, load_prepared  # noqa: E402

NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?\b")
STOP = {"The", "What", "Who", "Why", "How", "When", "Where", "Which", "According",
        "Based", "Does", "Did", "Was", "Were", "Is", "Are", "In", "On", "At", "After",
        "Before", "During", "From", "With", "This", "That", "There", "Then", "Their",
        "She", "He", "They", "It", "His", "Her", "Him", "Them", "Its", "We", "You", "I"}


def visible_text(chunks, cutoff: int) -> str:
    return "\n".join(c.text for c in chunks if c.end_pid < cutoff)


def classify(names: Set[str], text: str) -> Dict[str, str]:
    lowered = text.lower()
    words = set(re.findall(r"[a-z]+", lowered))
    out = {}
    for name in names:
        key = name.lower()
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            out[name] = "exact"
            continue
        tokens = key.split()
        best = max((difflib.SequenceMatcher(None, tokens[-1], w).ratio() for w in words),
                   default=0.0)
        out[name] = "variant" if best >= 0.82 else "absent"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", default="retrieval_eval_human_raw_both.jsonl")
    ap.add_argument("--novels", default="27,79,117,118,121,124")
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()
    questions, chunks, splits = load_prepared()
    novels = set(args.novels.split(","))
    retrieval = {}
    path = OUT / args.retrieval
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                retrieval[row["qid"]] = row
    rows = []
    for qid, q in questions.items():
        if q.novel_id not in novels or q.source != "human_anno":
            continue
        if splits["eval_human"] and qid not in splits["eval_human"]:
            continue
        text = visible_text(chunks[q.novel_id], q.cutoff)
        names = {n for n in NAME_RE.findall(q.question + " " + " ".join(q.options.values()))
                 if n not in STOP}
        labels = classify(names, text)
        row = retrieval.get(qid, {})
        dense_fail = row.get("qo_dense_gold_rank", -1) == 0 if row else None
        graph_fail = row.get("qo_graph_gold_rank", -1) == 0 if row else None
        rows.append({"qid": qid, "novel_id": q.novel_id, "names": labels,
                     "n_absent": sum(v == "absent" for v in labels.values()),
                     "n_variant": sum(v == "variant" for v in labels.values()),
                     "dense_fail": dense_fail, "graph_fail": graph_fail,
                     "question": q.question, "options": q.options})
    summary = {
        "n_questions": len(rows),
        "n_with_absent_name": sum(r["n_absent"] > 0 for r in rows),
        "n_with_variant_name": sum(r["n_variant"] > 0 for r in rows),
        "n_names": sum(len(r["names"]) for r in rows),
        "name_labels": dict(Counter(v for r in rows for v in r["names"].values())),
        "questions_with_retrieval": sum(r["dense_fail"] is not None for r in rows),
    }
    for method in ("dense", "graph"):
        sub = [r for r in rows if r[f"{method}_fail"] is not None]
        fail = [r for r in sub if r[f"{method}_fail"]]
        ok = [r for r in sub if not r[f"{method}_fail"]]
        summary[method] = {
            "n": len(sub), "n_fail": len(fail),
            "absent_name_rate_fail": round(mean(r["n_absent"] > 0 for r in fail), 4) if fail else None,
            "absent_name_rate_ok": round(mean(r["n_absent"] > 0 for r in ok), 4) if ok else None,
        }
    summary["examples_absent_or_variant"] = [
        r for r in rows if r["n_absent"] or r["n_variant"]][:args.examples]
    (OUT / "alias_mismatch_analysis.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    with open(OUT / "alias_mismatch_questions.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
