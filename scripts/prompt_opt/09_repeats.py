#!/usr/bin/env python
"""Phase 2.4 — measure HY4 output nondeterminism with identical prompts.

Calls are made with ``use_cache=False`` so repeated identical requests really hit
the endpoint.  For each question we run N times, then report:
  * how many questions change wording at all
  * how many change the scored answer (EM)
  * mean pairwise token Jaccard
  * majority-vote EM (self-consistency diagnostic)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import PATHS  # noqa: E402
from src.prompt_optimization import reader_prompts as RP  # noqa: E402
from src.prompt_optimization.official_metrics import normalize_answer  # noqa: E402
from src.prompt_optimization.runner import (OUT_ROOT, evaluate, freeze_context,  # noqa: E402
                                            load_conv, parse_answer, render,
                                            session_docs, write_jsonl)


def jaccard(a, b):
    ta, tb = set(normalize_answer(a).split()), set(normalize_answer(b).split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", default="P0,P7")
    ap.add_argument("--qids", default="4,9,16,17,24,29,31,33,39,47,54,63,74,85")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--shard", default="rp")
    args = ap.parse_args()

    import os
    os.environ["HY4_CACHE_SHARD"] = args.shard
    from src.adapters.hy4_client import HY4Client

    conv = load_conv("conv-30")
    docs = session_docs(conv)
    base = [json.loads(l) for l in open(PATHS["qa"] / "baselines_hipporag_conv-30.jsonl", encoding="utf-8")]
    qids = [int(x) for x in args.qids.split(",")]
    frozen = {q: freeze_context(base[q], docs, 4096) for q in qids}
    client = HY4Client(tag="repeat")
    outdir = OUT_ROOT / "reader" / "repeats"
    outdir.mkdir(parents=True, exist_ok=True)

    all_out = {}
    for pid in args.pids.split(","):
        spec = RP.get(pid)
        runs = []
        for k in range(args.repeats):
            def one(q):
                fz = frozen[q]
                system, user = render(spec, fz["question"], fz["context"])
                t0 = time.time()
                try:
                    out = client.chat([{"role": "system", "content": system},
                                       {"role": "user", "content": user}],
                                      max_tokens=256, temperature=0.0, use_cache=False,
                                      tag=f"repeat_{pid}_{k}")
                    return {"pid": pid, "run": k, "qidx": q, "qid": base[q]["qid"],
                            "category": int(base[q]["category"]), "raw": out["content"],
                            "pred_answer": parse_answer(pid, out["content"]),
                            "usage": out.get("usage"), "finish_reason": out.get("finish_reason"),
                            "latency_s": out.get("latency_s"), "error": None}
                except Exception as exc:  # noqa: BLE001
                    return {"pid": pid, "run": k, "qidx": q, "qid": base[q]["qid"],
                            "category": int(base[q]["category"]), "raw": "", "pred_answer": "",
                            "usage": {}, "finish_reason": "error",
                            "error": f"{type(exc).__name__}: {exc}"}
            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                rows = list(ex.map(one, qids))
            runs.append(rows)
            write_jsonl(outdir / f"{pid}_run{k}.jsonl", rows)
            print(f"  {pid} run {k} done ({len(rows)})")
        all_out[pid] = runs

    summary = {}
    for pid, runs in all_out.items():
        per_q = []
        for i, q in enumerate(qids):
            answers = [runs[k][i]["pred_answer"] for k in range(len(runs))]
            gold = base[q]["gold_answer"]
            cat = int(base[q]["category"])
            ems = [evaluate(a, gold, cat)["em"] for a in answers]
            f1s = [evaluate(a, gold, cat)["f1"] for a in answers]
            pair_j = [jaccard(answers[a], answers[b])
                      for a in range(len(answers)) for b in range(a + 1, len(answers))]
            norm = [normalize_answer(a) for a in answers]
            majority = Counter(norm).most_common(1)[0][0]
            maj_answer = next(a for a, n in zip(answers, norm) if n == majority)
            per_q.append({
                "qidx": q, "category": cat, "gold": gold, "answers": answers,
                "ems": ems, "f1s": f1s, "em_stable": len(set(ems)) == 1,
                "wording_stable": len(set(norm)) == 1,
                "mean_pair_jaccard": float(np.mean(pair_j)),
                "majority_em": evaluate(maj_answer, gold, cat)["em"],
            })
        summary[pid] = {
            "n": len(qids), "repeats": args.repeats,
            "wording_unstable_qids": [r["qidx"] for r in per_q if not r["wording_stable"]],
            "scored_unstable_qids": [r["qidx"] for r in per_q if not r["em_stable"]],
            "wording_unstable_rate": float(np.mean([not r["wording_stable"] for r in per_q])),
            "scored_unstable_rate": float(np.mean([not r["em_stable"] for r in per_q])),
            "mean_pair_jaccard": float(np.mean([r["mean_pair_jaccard"] for r in per_q])),
            "em_per_run": [float(np.mean([runs[k][i]["pred_answer"] and evaluate(
                runs[k][i]["pred_answer"], base[q]["gold_answer"], int(base[q]["category"]))["em"]
                for i, q in enumerate(qids)])) for k in range(args.repeats)],
            "majority_vote_em": float(np.mean([r["majority_em"] for r in per_q])),
            "per_question": per_q,
        }
    (outdir / "summary.json").write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(args),
        "summaries": summary, "hy4": client.summary(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== repeat summary ===")
    for pid, s in summary.items():
        print(f"{pid}: wording-unstable {s['wording_unstable_rate']:.2f}, "
              f"scored-unstable {s['scored_unstable_rate']:.2f}, Jaccard {s['mean_pair_jaccard']:.3f}, "
              f"EM/run {s['em_per_run']}, majority {s['majority_vote_em']:.3f}")


if __name__ == "__main__":
    main()
