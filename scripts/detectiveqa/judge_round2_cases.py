"""Post-hoc LLM case audit for every round-two question; diagnostic only."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.adapters.hy4_client import HY4Client  # noqa: E402
from src.detectiveqa.data import OUT, gold_chunk_ids, load_prepared, read_jsonl, write_jsonl  # noqa: E402
from src.detectiveqa.prompts import CASE_JUDGE, CASE_JUDGE_SYSTEM  # noqa: E402
from src.detectiveqa.reader import render_chunk  # noqa: E402
from src.detectiveqa.round2 import pack_ranked, recent_chunks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="round2_reader.jsonl")
    ap.add_argument("--conditions", default="options_only_memory,q_dense,q_graph,q_rrf,recent,oracle")
    ap.add_argument("--tag", default="round2_case_judge")
    ap.add_argument("--out", default="round2_case_judge")
    args = ap.parse_args()
    questions, chunks, _ = load_prepared()
    wanted = set(args.conditions.split(","))
    by_q = defaultdict(dict)
    for row in read_jsonl(OUT / args.reader):
        if row["option_seed"] == 0 and row["condition"] in wanted:
            by_q[row["qid"]][row["condition"]] = row
    client = HY4Client(tag=args.tag)
    out = []
    for qid, rows in sorted(by_q.items()):
        q = questions[qid]
        novel_chunks = chunks[q.novel_id]
        gold, _ = pack_ranked(novel_chunks, gold_chunk_ids(q, novel_chunks), 3500)
        recent = recent_chunks(novel_chunks, q.cutoff, 2200)
        compact = {}
        for name, row in rows.items():
            compact[name] = {"answer": row["prediction"]["answer"], "correct": row["scores"]["correct"],
                             "reasoning": row["prediction"]["reasoning"][:500],
                             "context_chunk_ids": row["context_chunk_ids"],
                             "gold_recall_in_context": row["scores"]["gold_recall_in_context"]}
        prompt = CASE_JUDGE.format(
            question=q.question, options="\n".join(f"{k}. {v}" for k, v in q.options.items()),
            gold_answer=q.answer, cutoff_ratio=next(iter(rows.values()))["cutoff_ratio"],
            gold_context="\n\n".join(render_chunk(c) for c in gold),
            recent_context="\n\n".join(render_chunk(c) for c in recent),
            conditions=json.dumps(compact, ensure_ascii=False, indent=1))
        try:
            verdict = client.complete_json(prompt, system=CASE_JUDGE_SYSTEM, max_tokens=2048, tag=args.tag)
            error = ""
        except Exception as exc:  # noqa: BLE001
            verdict, error = {}, f"{type(exc).__name__}: {exc}"[:300]
        out.append({"qid": qid, "novel_id": q.novel_id, "verdict": verdict, "error": error,
                    "condition_correct": {k: v["scores"]["correct"] for k, v in rows.items()}})
        print(qid, verdict.get("item_labels"), error, flush=True)
    write_jsonl(OUT / f"{args.out}.jsonl", out)
    summary = {
        "n": len(out), "errors": sum(bool(x["error"]) for x in out),
        "item_labels": dict(Counter(label for x in out for label in x["verdict"].get("item_labels", []))),
        "explicit_reveal_before_cutoff": sum(x["verdict"].get("explicit_reveal_before_cutoff") is True for x in out),
        "gold_sufficient_false": sum(x["verdict"].get("gold_sufficient") is False for x in out),
        "condition_labels": dict(Counter(f"{name}:{value.get('label')}" for x in out
                                         for name, value in (x["verdict"].get("conditions") or {}).items())),
    }
    (OUT / f"{args.out}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
