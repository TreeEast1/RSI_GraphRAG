#!/usr/bin/env python
"""Stage 3 - the E0-E6 joint matrix and the equal-budget static control.

E0 no skills / fixed P7
E1 hand-written static Answer Skills
E2 evolved Answer Skill memory
E3 static Retrieval Skills + fixed P7
E4 evolved Retrieval Skills + fixed P7
E5 static Retrieval + evolved Answer
E6 evolved Retrieval + evolved Answer
EB long static Answer prompt (all hand-written rules, same per-question call budget)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import common
from . import skills as S
from . import retrieval_evolve as R

METHODS = {
    "E0": {"ret": "base", "ans": "p7", "label": "no skills + P7"},
    "E1": {"ret": "base", "ans": "static_retrieved", "label": "static Answer Skills"},
    "E2": {"ret": "base", "ans": "evolved", "label": "evolved Answer Skills"},
    "E3": {"ret": "static", "ans": "p7", "label": "static Retrieval + P7"},
    "E4": {"ret": "evolved", "ans": "p7", "label": "evolved Retrieval + P7"},
    "E5": {"ret": "static", "ans": "evolved", "label": "static Retrieval + evolved Answer"},
    "E6": {"ret": "evolved", "ans": "evolved", "label": "evolved Retrieval + evolved Answer"},
    "EB": {"ret": "base", "ans": "static_all", "label": "equal-budget long static prompt"},
}


def selected_answer_step() -> int:
    """Choose the frozen snapshot on DEV only, using official LoCoMo F1 (main metric)."""
    cands = []
    for p in sorted(common.SNAP.glob("answer_dev_eval_step*.json")):
        d = common.read_json(p)
        step = int(d.get("step", int(p.stem.split("step")[-1])))
        cands.append((float(d.get("official_f1", -1)), float(d.get("project_f1", -1)), step))
    if cands:
        return max(cands)[2]
    state = common.read_json(common.OUT / "artifacts" / "answer_evolution_state.json")
    return int(state["best_snapshot_step"])


def load_answer_memory() -> S.AnswerSkillMemory:
    step = selected_answer_step()
    path = common.SNAP / f"answer_memory_step{step:03d}.json"
    m = S.AnswerSkillMemory.load(path)
    m.meta["selected_dev_step"] = step
    return m


def selected_retrieval_step() -> int:
    """Choose the retrieval snapshot on DEV by evidence-full, then R@5."""
    cands = []
    for p in sorted(common.SNAP.glob("retrieval_dev_eval_step*.json")):
        d = common.read_json(p)
        cands.append((float(d.get("evidence_full@4096", -1)),
                      float(d.get("recall@5", -1)),
                      int(d.get("step", int(p.stem.split("step")[-1])))))
    return max(cands)[2] if cands else 0


def load_retrieval_library() -> List[Dict]:
    step = selected_retrieval_step()
    path = common.SNAP / f"retrieval_memory_step{step:03d}.json"
    if not path.exists():
        return []
    data = common.read_json(path)
    return [s for s in data.get("skills", []) if s.get("status") == "active"]


def _answer_skills(kind: str, qidx: int, embedder, evolved_mem: S.AnswerSkillMemory,
                   static_mem: S.AnswerSkillMemory) -> List[Dict]:
    if kind == "p7":
        return []
    if kind == "static_retrieved":
        mem = static_mem
    elif kind == "static_all":
        return list(static_mem.skills)
    else:
        mem = evolved_mem
    return S.select_skills(mem.active(), common.qa_map()[qidx].question,
                           common.category_of(qidx), k=2, embedder=embedder,
                           skill_vecs=S.skill_vectors(mem.active(), embedder))


def run_method(method: str, qids: Sequence[int], embedder, evolved_mem, static_mem,
               static_ret, evolved_ret) -> List[Dict]:
    spec = METHODS[method]
    rows = []
    for qidx in qids:
        q = common.qa_map()[qidx]
        if spec["ret"] == "base":
            if method == "E0":
                r = common.answer_context_cached(qidx, "P7")
                if r is None:
                    r = common.answer(qidx, stage=f"joint_{method}", purpose="E0")
                r = dict(r)
                trace = R.retrieval_pipeline(qidx, (), mode="base", embedder=embedder)
                ctx, sel, ntok = common.record_budget_docs(qidx)
            else:
                trace = R.retrieval_pipeline(qidx, (), mode="base", embedder=embedder)
                ctx, sel, ntok = common.build_context_from_ranked(trace["final_ranked"], 4096)
        else:
            lib = static_ret if spec["ret"] == "static" else evolved_ret
            trace = R.retrieval_pipeline(qidx, lib, mode="qdm_suff", embedder=embedder)
            ctx, sel, ntok = common.build_context_from_ranked(trace["final_ranked"], 4096)

        if method == "E0":
            pass
        else:
            skills = _answer_skills(spec["ans"], qidx, embedder, evolved_mem, static_mem)
            r = common.answer(qidx, skills, stage=f"joint_{method}", purpose=f"{method}_reader",
                              context_override=(ctx, sel, ntok))
            r["selected_skill_details"] = [
                {"skill_id": s["skill_id"], "score": s.get("_select_score"),
                 "reasons": s.get("_select_reasons")} for s in skills]
        r = dict(r)
        r["method"] = method
        r["method_label"] = spec["label"]
        r["context_docs"] = sel
        r["context_tokens"] = ntok
        r["answer_support"] = common.answer_support(r.get("pred_answer", ""), ctx)
        r["retrieval"] = {k: trace.get(k) for k in
                          ("recall@3", "recall@5", "evidence_full@4096",
                           "evidence_partial@4096", "multi_evidence_complete")}
        r["retrieval_queries"] = trace.get("queries", [])
        r["retrieval_skills"] = trace.get("skills_used", [])
        r["retrieval_n_llm_calls"] = trace.get("n_llm_calls", 0)
        r["retrieval_usage"] = trace.get("usage", {})
        r["n_llm_calls"] = int(r.get("n_llm_calls", 1)) + int(trace.get("n_llm_calls", 0))
        u = dict(r.get("usage") or {})
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            u[k] = int(u.get(k, 0) or 0) + int((trace.get("usage") or {}).get(k, 0) or 0)
        r["usage"] = u
        rows.append(r)
    return rows


def summarize_method(rows: List[Dict]) -> Dict:
    s = common.summarize(rows)
    s["answer_support"] = common.mean(r.get("answer_support") for r in rows)
    s["retrieval_recall@5"] = common.mean((r.get("retrieval") or {}).get("recall@5") for r in rows)
    s["retrieval_recall@3"] = common.mean((r.get("retrieval") or {}).get("recall@3") for r in rows)
    s["evidence_full@4096"] = common.mean((r.get("retrieval") or {}).get("evidence_full@4096")
                                          for r in rows)
    s["multi_evidence_complete"] = common.mean((r.get("retrieval") or {}).get("multi_evidence_complete")
                                               for r in rows)
    s["retrieval_query_calls"] = common.mean(r.get("retrieval_n_llm_calls", 0) for r in rows)
    return s


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--splits", default="development,evaluation")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    suffix = args.out_suffix
    embedder = R.load_embedder()
    evolved_mem = load_answer_memory()
    static_mem = S.memory_from_skills(S.static_answer_skills())
    static_ret = S.static_retrieval_skills()
    evolved_ret = load_retrieval_library()
    print("evolved answer skills:", len(evolved_mem.active()),
          "step", evolved_mem.meta.get("selected_dev_step"),
          "| evolved retrieval skills:", len(evolved_ret))

    all_summary = {}
    all_rows = []
    for split_name in [x.strip() for x in args.splits.split(",") if x.strip()]:
        qids = common.qidx_list(split_name)
        for method in methods:
            rows = run_method(method, qids, embedder, evolved_mem, static_mem,
                              static_ret, evolved_ret)
            s = summarize_method(rows)
            s["split"] = split_name
            s["method"] = method
            s["label"] = METHODS[method]["label"]
            all_summary[f"{split_name}|{method}"] = s
            all_rows.extend(rows)
            print(f"{split_name:12s} {method:3s} n={len(rows)} EM={s['project_em']:.3f} "
                  f"F1={s['project_f1']:.3f} offF1={s['official_f1']:.3f} "
                  f"adv={s['adversarial_refusal']:.3f} Ev={s['evidence_full@4096']:.3f}", flush=True)
    common.write_jsonl(common.OUT / "artifacts" / f"joint_predictions{suffix}.jsonl", all_rows)
    common.write_json(common.OUT / "artifacts" / f"joint_summary{suffix}.json",
                      {"methods": {m: METHODS[m] for m in methods}, "summaries": all_summary,
                       "answer_memory_step": evolved_mem.meta.get("selected_dev_step"),
                       "n_evolved_answer_skills": len(evolved_mem.active()),
                       "n_evolved_retrieval_skills": len(evolved_ret)})

    lines = ["# Joint Experiments (eval set unless noted)", "",
             "All methods share the frozen COMP-P4_top5 base ranking and the P7 Reader; "
             "only the retrieval/answer skill layer changes.", "",
             "| Exp | Retrieval | Answer | n | EM | F1 | official F1 | adv refusal | "
             "Ev@4096 | multi-ev complete | R@5 | LLM calls |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in methods:
        s = all_summary[f"evaluation|{method}"]
        lines.append(
            f"| {method} | {METHODS[method]['ret']} | {METHODS[method]['ans']} | {s['n']} | "
            f"{s['project_em']:.3f} | {s['project_f1']:.3f} | {s['official_f1']:.3f} | "
            f"{s['adversarial_refusal']:.3f} | {s['evidence_full@4096']:.3f} | "
            f"{s['multi_evidence_complete']:.3f} | {s['retrieval_recall@5']:.3f} | "
            f"{s['mean_calls']:.1f} |")
    lines += ["", "## By category (eval, F1 / official F1 / n)", "",
              "| Exp | multi-hop | temporal | single-hop | adversarial refusal |",
              "|---|---|---|---|---:|"]
    for method in methods:
        s = all_summary[f"evaluation|{method}"]
        bc = s["by_category"]
        cells = []
        for c in ("1", "2", "4"):
            if c in bc:
                cells.append(f"{bc[c]['f1']:.3f} / {bc[c]['official_f1']:.3f} ({bc[c]['n']})")
            else:
                cells.append("-")
        lines.append(f"| {method} | " + " | ".join(cells) + " | "
                     f"{s['adversarial_refusal']:.3f} |")
    lines += ["", "## Development-set replication", "",
              "| Exp | EM | F1 | official F1 | adv refusal | Ev@4096 |", "|---|---:|---:|---:|---:|---:|"]
    for method in methods:
        s = all_summary.get(f"development|{method}")
        if not s:
            continue
        lines.append(f"| {method} | {s['project_em']:.3f} | {s['project_f1']:.3f} | "
                     f"{s['official_f1']:.3f} | {s['adversarial_refusal']:.3f} | "
                     f"{s['evidence_full@4096']:.3f} |")
    (common.OUT / "joint_experiments.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote joint_experiments.md")


if __name__ == "__main__":
    main()
