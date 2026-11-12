"""Round-two Reader controls: deployable q-only retrieval, locality and option-order robustness."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.adapters.hy4_client import HY4Client, parse_json_loose  # noqa: E402
from src.detectiveqa.data import OUT, gold_chunk_ids, load_prepared, read_jsonl, write_jsonl  # noqa: E402
from src.detectiveqa.reader import make_prompt, parse_answer, render_chunk, score_reader, system_prompt  # noqa: E402
from src.detectiveqa.round2 import (bootstrap_paired_by_novel, gold_reach, pack_ranked, permute_options,  # noqa: E402
                                    recent_chunks, relation_note, tokens)

DEFAULT_CONDITIONS = ("options_only,options_only_memory,q_dense,q_graph,q_rrf,qo_dense,recent,"
                      "nonrecent_q_dense,recent_plus_q_dense,oracle")


def relation_lookup(path: str) -> Dict[str, List[Dict]]:
    if not path:
        return {}
    return {row["qid"]: row.get("relations", []) for row in read_jsonl(Path(path))}


def ranked(row: Dict, name: str) -> List[str]:
    return row["conditions"][name]["ranked"]


def context_for(condition: str, q, chunks, retrieval: Dict, budget: int, relations: Dict[str, List[Dict]]):
    gold = gold_chunk_ids(q, chunks)
    if condition in {"options_only", "options_only_memory"}:
        return [], {"n_selected": 0}, "No story excerpts are provided."
    if condition == "recent":
        chosen = recent_chunks(chunks, q.cutoff, budget)
        meta = {"n_selected": len(chosen)}
    elif condition == "oracle":
        chosen, meta = pack_ranked(chunks, gold, budget)
    elif condition == "nonrecent_q_dense":
        recent = {c.chunk_id for c in recent_chunks(chunks, q.cutoff, budget)}
        chosen, meta = pack_ranked(chunks, ranked(retrieval, "q_dense"), budget, exclude=recent)
    elif condition == "recent_plus_q_dense":
        recent = recent_chunks(chunks, q.cutoff, budget // 2)
        used = {c.chunk_id for c in recent}
        extra, _ = pack_ranked(chunks, ranked(retrieval, "q_dense"), budget - budget // 2, exclude=used)
        chosen = sorted(recent + extra, key=lambda c: c.index)
        meta = {"n_selected": len(chosen), "recent": len(recent), "retrieved": len(extra)}
    elif condition == "q_rrf":
        chosen, meta = pack_ranked(chunks, ranked(retrieval, "q_rrf_dense_graph"), budget)
    elif condition == "q_graph":
        chosen, meta = pack_ranked(chunks, ranked(retrieval, "q_graph_f5"), budget)
    elif condition.startswith("relations_"):
        base = condition.removeprefix("relations_")
        source = {"q_dense": "q_dense", "q_graph": "q_graph_f5", "q_rrf": "q_rrf_dense_graph"}[base]
        notes = relations.get(q.qid, [])[:8]
        note_text = "\n".join(relation_note(x) for x in notes)
        note_budget = min(700, budget // 4)
        while notes and tokens(note_text) > note_budget:
            notes = notes[:-1]
            note_text = "\n".join(relation_note(x) for x in notes)
        chosen, meta = pack_ranked(chunks, ranked(retrieval, source), budget - note_budget)
        meta.update({"n_relation_notes": len(notes)})
        context = "\n\n".join(render_chunk(c) for c in chosen)
        if note_text:
            context = "Precomputed cross-chunk notes; verify them against excerpts:\n" + note_text + "\n\n" + context
        return chosen, meta, context
    else:
        chosen, meta = pack_ranked(chunks, ranked(retrieval, condition), budget)
    return chosen, meta, "\n\n".join(render_chunk(c) for c in chosen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", default="round2_retrieval.jsonl")
    ap.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    ap.add_argument("--option-seeds", default="0,1")
    ap.add_argument("--relations", default="")
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test only: first N retrieval rows")
    ap.add_argument("--tag", default="round2_reader")
    ap.add_argument("--out", default="round2_reader")
    args = ap.parse_args()
    questions, chunks, _ = load_prepared()
    retrieval_rows = read_jsonl(OUT / args.retrieval)
    if args.limit:
        retrieval_rows = retrieval_rows[:args.limit]
    relations = relation_lookup(args.relations)
    client = HY4Client(tag=args.tag)
    conditions = [x for x in args.conditions.split(",") if x]
    seeds = [int(x) for x in args.option_seeds.split(",") if x]
    rows = []
    partial = open(OUT / f"{args.out}.partial.jsonl", "w", encoding="utf-8")
    for retrieval in retrieval_rows:
        original = questions[retrieval["qid"]]
        novel_chunks = chunks[original.novel_id]
        gold = gold_chunk_ids(original, novel_chunks)
        for seed in seeds:
            q, to_original = permute_options(original, seed)
            for condition in conditions:
                chosen, meta, context = context_for(condition, q, novel_chunks, retrieval,
                                                    args.token_budget, relations)
                t0 = time.time()
                error, parsed, usage = "", {}, {}
                try:
                    out = client.chat([{"role": "system", "content": system_prompt(condition)},
                                       {"role": "user", "content": make_prompt(q, context)}],
                                      json_mode=True, max_tokens=1024, tag=args.tag)
                    usage = out.get("usage") or {}
                    parsed = parse_json_loose(out["content"]) or {}
                    error = "" if parsed else "unparseable_json"
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"[:300]
                pred = parse_answer(parsed, q)
                pred["answer_original"] = to_original.get(pred["answer"], "")
                ids = [c.chunk_id for c in chosen]
                scores = score_reader(pred, q, gold, ids)
                scores.update(gold_reach(ids, gold))
                row = {
                    "qid": q.qid, "novel_id": q.novel_id, "condition": condition, "option_seed": seed,
                    "cutoff_bucket": retrieval["cutoff_bucket"], "cutoff_ratio": retrieval["cutoff_ratio"],
                    "question": q.question, "options": q.options, "gold_answer": q.answer,
                    "gold_answer_original": original.answer, "option_to_original": to_original,
                    "gold_chunk_ids": gold, "context_chunk_ids": ids, "context_meta": meta,
                    "prediction": pred, "scores": scores, "usage": usage,
                    "seconds": round(time.time() - t0, 3), "error": error,
                }
                rows.append(row)
                partial.write(json.dumps(row, ensure_ascii=False) + "\n")
                partial.flush()
                print(q.qid, seed, condition, pred["answer"], scores["correct"], error, flush=True)
    partial.close()
    write_jsonl(OUT / f"{args.out}.jsonl", rows)
    summary = {"n_rows": len(rows), "conditions": {}, "paired_seed0": {}, "client": client.summary()}
    for condition in conditions:
        for seed in seeds:
            sub = [r for r in rows if r["condition"] == condition and r["option_seed"] == seed]
            if sub:
                summary["conditions"][f"{condition}@{seed}"] = {
                    "n": len(sub), "accuracy": round(mean(r["scores"]["correct"] for r in sub), 4),
                    "gold_any": round(mean(r["scores"]["gold_any_in_context"] for r in sub), 4),
                    "gold_all": round(mean(r["scores"]["gold_all_in_context"] for r in sub), 4),
                    "invalid": round(mean(r["scores"]["invalid"] for r in sub), 4),
                    "total_tokens": sum((r["usage"] or {}).get("total_tokens", 0) for r in sub),
                }
    paired = defaultdict(dict)
    for row in rows:
        if row["option_seed"] == seeds[0]:
            paired[row["qid"]][row["condition"]] = row["scores"]
            paired[row["qid"]]["novel_id"] = row["novel_id"]
    paired_rows = list(paired.values())
    comparisons = (("q_dense", "q_graph"), ("q_dense", "q_rrf"), ("q_dense", "recent"),
                   ("recent", "recent_plus_q_dense"), ("q_dense", "qo_dense"),
                   ("q_dense", "relations_q_dense"), ("q_rrf", "relations_q_rrf"))
    for left, right in comparisons:
        if left in conditions and right in conditions:
            summary["paired_seed0"][f"{left}->{right}"] = bootstrap_paired_by_novel(paired_rows, left, right)
    (OUT / f"{args.out}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:12000])


if __name__ == "__main__":
    main()
