#!/usr/bin/env python
"""Phase 2.2 — Oracle-evidence and full-conversation diagnostics.

Conditions
----------
R1  normal G_raw retrieval context (reused from reader experiments; no new calls)
R2  Gold-Evidence Oracle: the full source sessions that contain the gold evidence
R3  Full Conversation: all sessions, chronological, untruncated

R2 is a diagnostic only. Gold answers are never placed in any prompt; gold
evidence is used solely to choose which source sessions to show.

Outputs:
  reader/oracle/{condition}_{pid}.jsonl
  reader/oracle/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import PATHS  # noqa: E402
from src.dataset.locomo import num_tokens  # noqa: E402
from src.prompt_optimization import reader_prompts as RP  # noqa: E402
from src.prompt_optimization.runner import (OUT_ROOT, evaluate, load_conv, mean,  # noqa: E402
                                            parse_answer, render, session_docs, sha,
                                            summarize_prompt, write_jsonl)
from src.retrieval.reader import format_context  # noqa: E402

MAX_TOKENS = 256


def context_for(conv, docs, qa, condition):
    if condition == "R2":
        sids = sorted(set(qa.evidence_sessions))
        selected = [docs[f"{conv.conversation_id}::s{sid:02d}"] for sid in sids]
    elif condition == "R3":
        selected = [docs[s.document_id] for s in conv.sessions]
    else:
        raise ValueError(condition)
    return format_context(selected), [d["document_id"] for d in selected]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="R2,R3")
    ap.add_argument("--pids", required=True)
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--shard", default="or")
    args = ap.parse_args()

    import os
    os.environ["HY4_CACHE_SHARD"] = args.shard
    from src.adapters.hy4_client import HY4Client

    conv = load_conv("conv-30")
    docs = session_docs(conv)
    base = [json.loads(l) for l in open(PATHS["qa"] / "baselines_hipporag_conv-30.jsonl", encoding="utf-8")]
    client = HY4Client(tag="oracle")
    outdir = OUT_ROOT / "reader" / "oracle"
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    for condition in args.conditions.split(","):
        qas = [q for q in conv.qa if (condition != "R2" or q.category != 5)]
        for pid in args.pids.split(","):
            spec = RP.get(pid)
            # R3 + P0 exactly matches the existing full-context baseline; reuse it.
            if condition == "R3" and pid == "P0":
                rows = []
                for qa, r in zip(conv.qa, base):
                    ctx, sel = context_for(conv, docs, qa, condition)
                    rows.append({
                        "pid": "P0", "condition": condition, "qid": r["qid"],
                        "qidx": int(r["qid"].split("q")[-1]), "question": r["question"],
                        "category": int(r["category"]),
                        "context_docs": sel, "context_tokens": num_tokens(ctx),
                        "pred_answer": r.get("pred_answer"), "raw": None,
                        "usage": r.get("reader_usage"), "finish_reason": None,
                        "cached": True, "error": None, "reused_baseline": True,
                        "context": ctx,
                    })
                results[f"{condition}_{pid}"] = rows
                print(f"  R3/P0 reused from baselines_fullcontext ({len(rows)} rows)")
                continue

            def one(qa):
                ctx, sel = context_for(conv, docs, qa, condition)
                system, user = render(spec, qa.question, ctx)
                t0 = time.time()
                try:
                    out = client.chat([{"role": "system", "content": system},
                                       {"role": "user", "content": user}],
                                      max_tokens=MAX_TOKENS, temperature=0.0,
                                      tag=f"oracle_{condition}_{pid}")
                    return {
                        "pid": pid, "condition": condition, "qid": qa.qid,
                        "qidx": int(qa.qid.split("q")[-1]), "question": qa.question,
                        "category": int(qa.category),
                        "context_docs": sel, "context_tokens": num_tokens(ctx),
                        "system": system, "user": user,
                        "prompt_sha256": sha(system + "\n" + user),
                        "raw": out["content"], "pred_answer": parse_answer(pid, out["content"]),
                        "usage": out.get("usage"), "finish_reason": out.get("finish_reason"),
                        "cached": out.get("cached"), "latency_s": out.get("latency_s"),
                        "error": None, "context": ctx,
                    }
                except Exception as exc:  # noqa: BLE001
                    return {
                        "pid": pid, "condition": condition, "qid": qa.qid,
                        "qidx": int(qa.qid.split("q")[-1]), "question": qa.question,
                        "category": int(qa.category),
                        "context_docs": sel, "context_tokens": num_tokens(ctx),
                        "pred_answer": "", "raw": "", "usage": {}, "finish_reason": "error",
                        "cached": False, "latency_s": round(time.time() - t0, 2),
                        "error": f"{type(exc).__name__}: {exc}", "context": ctx,
                    }

            t0 = time.time()
            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                rows = list(ex.map(one, qas))
            print(f"  {condition}/{pid}: {len(rows)} in {time.time()-t0:.0f}s "
                  f"errors={sum(1 for r in rows if r['error'])}")
            results[f"{condition}_{pid}"] = rows
            write_jsonl(outdir / f"{condition}_{pid}.jsonl", rows)

    summary = {}
    gold_by_q = {int(r["qid"].split("q")[-1]): r["gold_answer"] for r in base}
    cat_by_q = {int(r["qid"].split("q")[-1]): int(r["category"]) for r in base}
    for key, rows in results.items():
        for r in rows:
            r.update(evaluate(r["pred_answer"], gold_by_q[r["qidx"]], cat_by_q[r["qidx"]]))
        summary[key] = summarize_prompt(rows)
        summary[key]["mean_context_tokens"] = mean(r["context_tokens"] for r in rows)
        summary[key]["n_context_docs"] = mean(len(r["context_docs"]) for r in rows)
    (outdir / "summary.json").write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(args),
        "summaries": summary, "hy4": client.summary(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== oracle summary ===")
    for k, s in summary.items():
        print(f"{k:10s} n={s['n']:3d} EM={s['project_em']:.3f} F1={s['project_f1']:.3f} "
              f"offF1={s['official_f1']:.3f} refusal={s['refusal_rate_answerable']:.3f} "
              f"ctx={s['mean_context_tokens']:.0f}")


if __name__ == "__main__":
    main()
