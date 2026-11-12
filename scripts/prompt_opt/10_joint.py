#!/usr/bin/env python
"""Phase 5 / Phase 3.3 — run a Reader prompt on a frozen graph-retrieval result.

Contexts are reconstructed from the retrieval record and checked token-for-token
against the stored 4096 budget, so changing the Reader prompt never changes the
evidence the model sees.

Outputs:
  joint/{variant}_{pid}.jsonl
  joint/summary.json (merged)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True,
                    help="comma separated retrieval variant dirs, e.g. G_raw,cmpo_comp-p4")
    ap.add_argument("--pids", default="P0")
    ap.add_argument("--split", default="all", choices=["all", "dev", "val"])
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--shard", default="jo")
    ap.add_argument("--records-root", default="comprehension/retrieval",
                    help="directory containing {variant}/records.jsonl")
    args = ap.parse_args()

    import os
    os.environ["HY4_CACHE_SHARD"] = args.shard
    from src.adapters.hy4_client import HY4Client

    conv = load_conv("conv-30")
    docs = session_docs(conv)
    split = json.loads((OUT_ROOT / "splits" / "split_dev40_val65.json").read_text())
    qfilter = {"all": None, "dev": set(split["dev_qids"]), "val": set(split["val_qids"])}[args.split]

    client = HY4Client(tag="joint")
    outdir = OUT_ROOT / "joint"
    outdir.mkdir(parents=True, exist_ok=True)
    summary_all = {}
    for variant in args.variants.split(","):
        rec_path = OUT_ROOT / args.records_root / variant / "records.jsonl"
        recs = [json.loads(l) for l in open(rec_path, encoding="utf-8")]
        if qfilter is not None:
            recs = [r for r in recs if int(r["qid"].split("q")[-1]) in qfilter]
        for pid in args.pids.split(","):
            spec = RP.get(pid)
            rows = []
            def one(rec):
                fz = freeze_context(rec, docs, 4096)
                stored = rec["budget"]["4096"]
                if fz["context_tokens"] != stored["tokens"] or \
                        set(fz["selected_docs"]) != set(stored["selected_ids"]):
                    return {"error": f"context mismatch {rec['qid']}", "qid": rec["qid"],
                            "qidx": int(rec["qid"].split("q")[-1]), "pid": pid}
                system, user = render(spec, fz["question"], fz["context"])
                try:
                    out = client.chat([{"role": "system", "content": system},
                                       {"role": "user", "content": user}],
                                      max_tokens=MAX_TOKENS, temperature=0.0,
                                      tag=f"joint_{variant}_{pid}")
                    return {
                        "pid": pid, "variant": variant, "qid": rec["qid"],
                        "qidx": int(rec["qid"].split("q")[-1]), "question": fz["question"],
                        "category": rec["category"],
                        "context_docs": fz["selected_docs"], "context_tokens": fz["context_tokens"],
                        "system": system, "user": user, "prompt_sha256": sha(system + "\n" + user),
                        "raw": out["content"], "pred_answer": parse_answer(pid, out["content"]),
                        "usage": out.get("usage"), "finish_reason": out.get("finish_reason"),
                        "cached": out.get("cached"), "latency_s": out.get("latency_s"),
                        "error": None, "recall@5": rec["retrieval"]["recall@5"],
                        "evidence_full@4096": rec["budget"]["4096"]["evidence_full"],
                        "diag": rec.get("diag", {}),
                    }
                except Exception as exc:  # noqa: BLE001
                    return {"pid": pid, "variant": variant, "qid": rec["qid"],
                            "qidx": int(rec["qid"].split("q")[-1]), "question": fz["question"],
                            "category": rec["category"], "pred_answer": "", "raw": "",
                            "usage": {}, "finish_reason": "error", "cached": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "recall@5": rec["retrieval"]["recall@5"],
                            "evidence_full@4096": rec["budget"]["4096"]["evidence_full"]}

            t0 = time.time()
            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                rows = list(ex.map(one, recs))
            bad = [r for r in rows if (r.get("error") or "").startswith("context mismatch")]
            if bad:
                raise SystemExit(f"{len(bad)} context mismatches; refusing to write")
            gold = {int(r["qid"].split("q")[-1]): r["gold_answer"] for r in recs}
            cats = {int(r["qid"].split("q")[-1]): int(r["category"]) for r in recs}
            for r in rows:
                if not r.get("error"):
                    r.update(evaluate(r["pred_answer"], gold[r["qidx"]], cats[r["qidx"]]))
            key = f"{variant}|{pid}"
            write_jsonl(outdir / f"{variant.replace('/', '_')}_{pid}.jsonl", rows)
            s = summarize_prompt(rows)
            s["mean_context_tokens"] = mean(r.get("context_tokens") for r in rows)
            s["recall@5"] = mean(r.get("recall@5") for r in rows)
            s["evidence_full@4096"] = mean(r.get("evidence_full@4096") for r in rows)
            s["variant"], s["pid"], s["split"] = variant, pid, args.split
            summary_all[key] = s
            print(f"  {key}: n={len(rows)} EM={s['project_em']:.3f} F1={s['project_f1']:.3f} "
                  f"offF1={s['official_f1']:.3f} refusal={s['refusal_rate_answerable']:.3f} "
                  f"{time.time()-t0:.0f}s")

    p = outdir / "summary.json"
    prev = json.loads(p.read_text()) if p.exists() else {"summaries": {}}
    prev.setdefault("summaries", {}).update(summary_all)
    prev["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    prev["hy4"] = client.summary()
    p.write_text(json.dumps(prev, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
