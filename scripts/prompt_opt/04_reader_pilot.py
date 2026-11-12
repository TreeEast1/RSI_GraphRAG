#!/usr/bin/env python
"""Phase 1.1 — freeze G_raw Reader contexts, then run QA-P0..P6 on the 40 dev questions.

P0 is the saved G_raw answer (same frozen context, same prompt) — no LLM call.
P1-P6 each get one call per dev question, same context, same max_tokens=256,
temperature=0, reasoning_effort=no_think.

Outputs:
  reader/contexts/graw/qNNN.json
  reader/prompts/{pid}.txt
  reader/pilot/results_{pid}.jsonl
  reader/pilot/summary.json
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
from src.prompt_optimization import reader_prompts as RP  # noqa: E402
from src.prompt_optimization.runner import (OUT_ROOT, evaluate, freeze_context,  # noqa: E402
                                            load_conv, mean, parse_answer, render,
                                            session_docs, sha, summarize_prompt,
                                            write_jsonl)

MAX_TOKENS = 256


def freeze_all(conv, docs, rows):
    ctx_dir = OUT_ROOT / "reader" / "contexts" / "graw"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    frozen, bad = {}, []
    for r in rows:
        fz = freeze_context(r, docs, 4096)
        if fz["context_tokens"] != r.get("extra", {}).get("context_tokens") or \
                fz["n_docs"] != r.get("extra", {}).get("n_context"):
            bad.append((fz["qidx"], fz["context_tokens"], r.get("extra", {}).get("context_tokens")))
        (ctx_dir / f"q{fz['qidx']:03d}.json").write_text(json.dumps(fz, ensure_ascii=False, indent=1),
                                                        encoding="utf-8")
        frozen[fz["qidx"]] = fz
    return frozen, bad


def run_one(client, spec, fz):
    system, user = render(spec, fz["question"], fz["context"])
    t0 = time.time()
    try:
        out = client.chat([{"role": "system", "content": system},
                           {"role": "user", "content": user}],
                          max_tokens=MAX_TOKENS, temperature=0.0, tag=f"reader_{spec.pid}")
        raw = out["content"]
        rec = {
            "pid": spec.pid, "qid": fz["qid"], "qidx": fz["qidx"], "question": fz["question"],
            "category": int(fz["category"]),
            "context_docs": fz["selected_docs"], "context_tokens": fz["context_tokens"],
            "system": system, "user": user, "prompt_sha256": sha(system + "\n" + user),
            "raw": raw, "pred_answer": parse_answer(spec.pid, raw),
            "finish_reason": out.get("finish_reason"), "usage": out.get("usage"),
            "cached": out.get("cached"), "latency_s": out.get("latency_s"),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        rec = {
            "pid": spec.pid, "qid": fz["qid"], "qidx": fz["qidx"], "question": fz["question"],
            "category": int(fz["category"]),
            "context_docs": fz["selected_docs"], "context_tokens": fz["context_tokens"],
            "system": system, "user": user, "prompt_sha256": sha(system + "\n" + user),
            "raw": "", "pred_answer": "", "finish_reason": "error", "usage": {},
            "cached": False, "latency_s": round(time.time() - t0, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "val", "all"])
    ap.add_argument("--pids", default="P1,P2,P3,P4,P5,P6")
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--shard", default="po")
    ap.add_argument("--out-subdir", default="pilot")
    args = ap.parse_args()

    import os
    if args.shard:
        os.environ["HY4_CACHE_SHARD"] = args.shard
    from src.adapters.hy4_client import HY4Client

    conv = load_conv("conv-30")
    docs = session_docs(conv)
    base_rows = [json.loads(l) for l in open(PATHS["qa"] / "baselines_hipporag_conv-30.jsonl", encoding="utf-8")]
    frozen, bad = freeze_all(conv, docs, base_rows)
    if bad:
        raise SystemExit(f"context reconstruction mismatch: {bad[:5]}")

    split = json.loads((OUT_ROOT / "splits" / "split_dev40_val65.json").read_text())
    qids = split[f"{'dev' if args.split == 'dev' else 'val'}_qids"] if args.split != "all" else list(frozen)
    qids = [q for q in qids if q in frozen]

    # prompt artifacts
    outdir = OUT_ROOT / "reader" / args.out_subdir
    prom_dir = OUT_ROOT / "reader" / "prompts"
    prom_dir.mkdir(parents=True, exist_ok=True)
    for spec in RP.ALL_PROMPTS:
        (prom_dir / f"{spec.pid}.txt").write_text(
            f"# {spec.pid} — {spec.name}\n# {spec.notes}\n# system_sha256={sha(spec.system)}\n"
            f"# template_sha256={sha(spec.template)}\n\n--- SYSTEM ---\n{spec.system}\n\n"
            f"--- TEMPLATE ---\n{spec.template}\n", encoding="utf-8")

    results = {}

    # P0: reuse saved baseline answers (identical context + prompt)
    p0 = []
    for q in qids:
        r = base_rows[q]
        p0.append({
            "pid": "P0", "qid": r["qid"], "qidx": q, "question": r["question"],
            "category": int(r["category"]),
            "context_docs": frozen[q]["selected_docs"], "context_tokens": frozen[q]["context_tokens"],
            "system": RP.P0.system, "user": RP.P0.template.format(
                context=frozen[q]["context"], question=frozen[q]["question"]),
            "raw": None, "pred_answer": r.get("pred_answer"),
            "finish_reason": None, "usage": r.get("reader_usage"), "cached": True,
            "latency_s": None, "error": None, "reused_baseline": True,
        })
    results["P0"] = p0

    client = HY4Client(tag="reader_pilot")
    for pid in args.pids.split(","):
        spec = RP.get(pid)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            rows = list(ex.map(lambda q: run_one(client, spec, frozen[q]), qids))
        results[pid] = rows
        n_err = sum(1 for r in rows if r["error"])
        print(f"  {pid} done {len(rows)} in {time.time()-t0:.0f}s errors={n_err} "
              f"cache_hits={sum(1 for r in rows if r['cached'])}")
        write_jsonl(outdir / f"results_{pid}.jsonl", rows)
        (outdir / f"results_{pid}.jsonl.meta.json").write_text(json.dumps({
            "pid": pid, "split": args.split, "qids": qids, "max_tokens": MAX_TOKENS,
            "temperature": 0.0, "reasoning_effort": "no_think",
            "system_sha256": sha(spec.system), "template_sha256": sha(spec.template),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "elapsed_s": round(time.time()-t0, 1),
        }, indent=2), encoding="utf-8")

    summary = {}
    for pid, rows in results.items():
        for r in rows:
            r.update(evaluate(r["pred_answer"], base_rows[r["qidx"]]["gold_answer"],
                              int(base_rows[r["qidx"]]["category"])))
        summary[pid] = summarize_prompt(rows)
        summary[pid]["name"] = RP.get(pid).name
    (outdir / "summary.json").write_text(json.dumps({
        "split": args.split, "n": len(qids), "hy4": client.summary(),
        "summaries": summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== dev summary ===")
    for pid, s in summary.items():
        print(f"{pid} {s['name'][:34]:34s} EM={s['project_em']:.3f} F1={s['project_f1']:.3f} "
              f"offF1={s['official_f1']:.3f} refusal={s['refusal_rate_answerable']:.3f} "
              f"adv={s['adversarial_refusal_project']:.2f} words={s['mean_answer_words']:.1f} "
              f"trunc={s['truncated']}")
    print("HY4:", client.summary())


if __name__ == "__main__":
    main()
