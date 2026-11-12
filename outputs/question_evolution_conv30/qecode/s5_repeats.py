#!/usr/bin/env python
"""Stage 5 - repeated fresh runs of the final candidates (HY4 nondeterminism).

Every LLM call is made with ``use_cache=False`` so identical prompts really hit
the endpoint.  Each method is run three times on the evaluation split; we report
run-level means, per-question EM stability, and wording changes.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from . import common
from . import skills as S
from . import retrieval_evolve as R
from .s3_joint import load_answer_memory, load_retrieval_library
from .s4_react import revise, verify_answer


def _add_usage(a, b):
    a = dict(a or {})
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        a[k] = int(a.get(k, 0) or 0) + int((b or {}).get(k, 0) or 0)
    return a


def run_one(method: str, qidx: int, embedder, evolved_mem, evolved_ret) -> Dict:
    q = common.qa_map()[qidx]
    skills = []
    if method in ("E2", "E6", "R5"):
        skills = S.select_skills(evolved_mem.active(), q.question, int(q.category),
                                 embedder=embedder,
                                 skill_vecs=S.skill_vectors(evolved_mem.active(), embedder))
    trace = None
    if method in ("E0", "E2"):
        ctx, sel, ntok = common.record_budget_docs(qidx)
        n_calls = 1
        usage = {}
    else:
        trace = R.retrieval_pipeline(qidx, evolved_ret if method in ("E6", "R5") else (),
                                     mode="qdm_suff" if method in ("E6", "R5") else "base",
                                     use_cache=False, embedder=embedder)
        ctx, sel, ntok = common.build_context_from_ranked(trace["final_ranked"], 4096)
        n_calls = 1 + int(trace.get("n_llm_calls", 0))
        usage = dict(trace.get("usage") or {})
    r = common.answer(qidx, skills, stage=f"repeat_{method}", purpose=f"repeat_{method}",
                      use_cache=False, context_override=(ctx, sel, ntok))
    usage = _add_usage(usage, r.get("usage"))
    verify = None
    if method == "R5":
        verify = verify_answer(q.question, ctx, r.get("pred_answer", ""), q.qid, use_cache=False)
        n_calls += 1
        usage = _add_usage(usage, verify.get("usage"))
        if verify["verdict"] == "refuse" or (verify["should_refuse"] and not r.get("refused")):
            r["pred_answer"] = "No answer"
            r["refused"] = True
        elif verify["verdict"] == "revise" and (verify["issues"] or verify["missing_parts"]):
            rev = revise(q.question, ctx, skills,
                         list(verify["issues"]) + [f"missing: {x}" for x in verify["missing_parts"]],
                         q.qid, use_cache=False)
            n_calls += 1
            usage = _add_usage(usage, rev.get("usage"))
            r["pred_answer"] = rev["pred_answer"]
            r["refused"] = common.is_refusal(rev["pred_answer"])
        r.update(common.score_answer(r["pred_answer"], qidx))
    r = dict(r)
    r.update({"method": method, "context_docs": sel, "context_tokens": ntok,
              "skills_used": [s["skill_id"] for s in skills],
              "answer_support": common.answer_support(r.get("pred_answer", ""), ctx),
              "verifier": verify, "n_llm_calls": n_calls, "usage": usage,
              "retrieval_n_llm_calls": (trace or {}).get("n_llm_calls", 0),
              "retrieval": {k: (trace or {}).get(k) for k in
                            ("recall@5", "evidence_full@4096")}})
    return r


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="E0,E2,E6,R5")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    suffix = args.out_suffix
    embedder = R.load_embedder()
    evolved_mem = load_answer_memory()
    evolved_ret = load_retrieval_library()
    qids = common.qidx_list("evaluation")
    reps = 3
    all_rows, summary = [], {}
    for method in methods:
        runs = []
        for k in range(reps):
            rows = [run_one(method, q, embedder, evolved_mem, evolved_ret) for q in qids]
            runs.append(rows)
            all_rows.extend([{**r, "run": k} for r in rows])
            s = common.summarize(rows)
            print(f"{method} run{k}: EM={s['project_em']:.3f} F1={s['project_f1']:.3f} "
                  f"offF1={s['official_f1']:.3f} adv={s['adversarial_refusal']:.3f}", flush=True)
        per_q = []
        for i, q in enumerate(qids):
            answers = [runs[k][i]["pred_answer"] for k in range(reps)]
            ems = [runs[k][i]["em"] for k in range(reps)]
            f1s = [runs[k][i]["f1"] for k in range(reps)]
            norms = [re.sub(r"\s+", " ", str(a).lower().strip()) for a in answers]
            per_q.append({
                "qidx": q, "category": common.category_of(q),
                "answers": answers, "ems": ems, "f1s": f1s,
                "em_stable": len(set(ems)) == 1,
                "wording_stable": len(set(norms)) == 1,
                "f1_std": float(np.std(f1s)),
            })
        run_means = [common.summarize(r) for r in runs]
        summary[method] = {
            "repeats": reps,
            "run_em": [s["project_em"] for s in run_means],
            "run_f1": [s["project_f1"] for s in run_means],
            "run_official_f1": [s["official_f1"] for s in run_means],
            "run_adversarial": [s["adversarial_refusal"] for s in run_means],
            "mean_em": float(np.mean([s["project_em"] for s in run_means])),
            "std_em": float(np.std([s["project_em"] for s in run_means])),
            "mean_f1": float(np.mean([s["project_f1"] for s in run_means])),
            "std_f1": float(np.std([s["project_f1"] for s in run_means])),
            "mean_official_f1": float(np.mean([s["official_f1"] for s in run_means])),
            "std_official_f1": float(np.std([s["official_f1"] for s in run_means])),
            "em_unstable_qids": [r["qidx"] for r in per_q if not r["em_stable"]],
            "wording_unstable_qids": [r["qidx"] for r in per_q if not r["wording_stable"]],
            "em_unstable_rate": float(np.mean([not r["em_stable"] for r in per_q])),
            "wording_unstable_rate": float(np.mean([not r["wording_stable"] for r in per_q])),
            "mean_f1_std_across_runs": float(np.mean([r["f1_std"] for r in per_q])),
        }
        print(method, json.dumps({k: round(v, 3) for k, v in summary[method].items()
                                  if isinstance(v, float)}, ensure_ascii=False), flush=True)
    common.write_jsonl(common.OUT / "artifacts" / f"repeat_predictions{suffix}.jsonl", all_rows)
    common.write_json(common.OUT / "artifacts" / f"repeat_summary{suffix}.json", summary)

    lines = ["# Repeated-run stability (eval set, fresh calls)", "",
             "| Method | runs | EM mean±sd | F1 mean±sd | official F1 mean±sd | adv refusal | "
             "EM-unstable questions | wording-unstable |",
             "|---|---:|---|---|---|---|---:|---:|"]
    for m, s in summary.items():
        lines.append(f"| {m} | {s['repeats']} | {s['mean_em']:.3f}±{s['std_em']:.3f} | "
                     f"{s['mean_f1']:.3f}±{s['std_f1']:.3f} | "
                     f"{s['mean_official_f1']:.3f}±{s['std_official_f1']:.3f} | "
                     f"{np.mean(s['run_adversarial']):.3f} | {len(s['em_unstable_qids'])} | "
                     f"{len(s['wording_unstable_qids'])} |")
    lines += ["", "Per-method unstable question IDs are in `artifacts/repeat_summary.json`.", ""]
    (common.OUT / "repeat_stability.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
