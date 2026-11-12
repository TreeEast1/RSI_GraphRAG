#!/usr/bin/env python
"""Stage 4 - lightweight Question-Time ReAct + grounded verifier.

Variants on the evaluation split:
  R0  direct P7
  R1  P7 + verifier (one revision at most)
  R2  P7 + static ReAct (QDM, sufficiency, static retrieval skills, verifier)
  R3  evolved Answer Skills, direct answer
  R4  evolved Answer Skills + verifier
  R5  evolved Answer Skills + evolved Retrieval + verifier (full system)

The verifier sees only question, evidence, and candidate answer.  It never sees
gold answers or gold evidence.  Revision is capped at one extra Reader call.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

from . import common
from . import skills as S
from . import retrieval_evolve as R
from .s3_joint import load_answer_memory, load_retrieval_library

VERIFY_SYSTEM = (
    "You are a grounded answer verifier for a retrieval-augmented QA system. You see only the "
    "question, the retrieved evidence, and a candidate answer. You never use outside knowledge. "
    "Output JSON only."
)
VERIFY_PROMPT = """Question: {question}

Retrieved evidence:
{context}

Candidate answer: {answer}

Check the candidate answer against the evidence:
1. Is every claim supported by the evidence? Quote the shortest supporting span.
2. Does the answer miss a part that the question explicitly asks for?
3. Is there a date, duration, or event-order error?
4. Does it answer about the wrong person, object, or event?
5. Does it over-enumerate irrelevant items?
6. Does the question presuppose something the evidence does not support? If so, the answer
   should refuse.

Return JSON:
{{"supported": true/false,
  "support_quote": "<=25 words or empty",
  "missing_parts": ["..."],
  "date_error": true/false,
  "wrong_target": true/false,
  "over_enumeration": true/false,
  "should_refuse": true/false,
  "issues": ["..."],
  "verdict": "ok|revise|refuse"}}
"""

REVISE_SUFFIX = """

A grounded verifier found these issues with a previous draft:
{issues}

Produce the corrected final answer once. Keep the same output format.
Answer:"""


def verify_answer(question: str, context: str, answer: str, qid: str,
                  use_cache: bool = True) -> Dict:
    out = common.call_json(
        [{"role": "system", "content": VERIFY_SYSTEM},
         {"role": "user", "content": VERIFY_PROMPT.format(context=context, question=question,
                                                          answer=answer)}],
        stage="verify_answer", qid=qid, purpose="grounded_verifier", max_tokens=600,
        use_cache=use_cache)
    d = out.get("json") or {}
    return {"supported": bool(d.get("supported", True)),
            "support_quote": str(d.get("support_quote", ""))[:160],
            "missing_parts": d.get("missing_parts") or [],
            "date_error": bool(d.get("date_error", False)),
            "wrong_target": bool(d.get("wrong_target", False)),
            "over_enumeration": bool(d.get("over_enumeration", False)),
            "should_refuse": bool(d.get("should_refuse", False)),
            "issues": [str(x)[:160] for x in (d.get("issues") or [])][:6],
            "verdict": str(d.get("verdict", "ok")),
            "usage": out.get("usage"), "cached": out.get("cached")}


def revise(question: str, context: str, skills: Sequence[Dict], issues: Sequence[str],
           qid: str, use_cache: bool = True) -> Dict:
    system, user = common.render_prompt(question, context, skills)
    user = user.rstrip()
    if user.endswith("Answer:"):
        user = user[:-len("Answer:")] + REVISE_SUFFIX.format(
            issues="\n".join(f"- {x}" for x in issues))
    else:
        user += REVISE_SUFFIX.format(issues="\n".join(f"- {x}" for x in issues))
    out = common.call_text(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        stage="revise", qid=qid, purpose="one_revision", max_tokens=256, use_cache=use_cache)
    return {"text": out["text"], "pred_answer": common.parse_answer("P7", out["text"]),
            "usage": out.get("usage"), "cached": out.get("cached")}


def run_variant(name: str, qids: Sequence[int], embedder, evolved_mem, static_mem,
                static_ret, evolved_ret, *, ret_mode: str, ret_lib: Sequence[Dict],
                answer_kind: str, use_verifier: bool) -> List[Dict]:
    rows = []
    for qidx in qids:
        q = common.qa_map()[qidx]
        if ret_mode == "base":
            trace = R.retrieval_pipeline(qidx, (), mode="base", embedder=embedder)
            ctx, sel, ntok = common.build_context_from_ranked(trace["final_ranked"], 4096)
        else:
            trace = R.retrieval_pipeline(qidx, ret_lib, mode="qdm_suff", embedder=embedder)
            ctx, sel, ntok = common.build_context_from_ranked(trace["final_ranked"], 4096)
        if answer_kind == "p7":
            skills = []
        elif answer_kind == "evolved":
            skills = S.select_skills(evolved_mem.active(), q.question, int(q.category),
                                     embedder=embedder,
                                     skill_vecs=S.skill_vectors(evolved_mem.active(), embedder))
        else:
            skills = S.select_skills(static_mem.active(), q.question, int(q.category),
                                     embedder=embedder,
                                     skill_vecs=S.skill_vectors(static_mem.active(), embedder))
        r = common.answer(qidx, skills, stage=f"react_{name}", purpose=f"{name}_reader",
                          context_override=(ctx, sel, ntok))
        n_calls = 1 + int(trace.get("n_llm_calls", 0))
        usage = dict(r.get("usage") or {})
        verify = None
        revised = False
        if use_verifier:
            verify = verify_answer(q.question, ctx, r.get("pred_answer", ""), q.qid)
            n_calls += 1
            usage = _add_usage(usage, verify.get("usage"))
            verdict = verify["verdict"]
            if verdict == "refuse" or (verify["should_refuse"] and not r.get("refused")):
                r["pred_answer"] = "No answer"
                r["refused"] = True
                revised = True
            elif verdict == "revise" and (verify["issues"] or verify["missing_parts"]):
                rev = revise(q.question, ctx, skills,
                             list(verify["issues"]) + [f"missing: {x}" for x in verify["missing_parts"]],
                             q.qid)
                n_calls += 1
                usage = _add_usage(usage, rev.get("usage"))
                r["pred_answer"] = rev["pred_answer"]
                r["refused"] = common.is_refusal(rev["pred_answer"])
                revised = True
            r.update(common.score_answer(r["pred_answer"], qidx))
        r = dict(r)
        r["method"] = name
        r["context_docs"] = sel
        r["context_tokens"] = ntok
        r["answer_support"] = common.answer_support(r.get("pred_answer", ""), ctx)
        r["verifier"] = verify
        r["revised"] = revised
        r["skills_used"] = [s["skill_id"] for s in skills]
        r["retrieval"] = {k: trace.get(k) for k in
                          ("recall@3", "recall@5", "evidence_full@4096",
                           "multi_evidence_complete")}
        r["retrieval_n_llm_calls"] = trace.get("n_llm_calls", 0)
        r["n_llm_calls"] = n_calls
        r["usage"] = usage
        rows.append(r)
    return rows


def _add_usage(a: Dict, b: Optional[Dict]) -> Dict:
    a = dict(a or {})
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        a[k] = int(a.get(k, 0) or 0) + int((b or {}).get(k, 0) or 0)
    return a


def summarize(rows: List[Dict]) -> Dict:
    s = common.summarize(rows)
    s["answer_support"] = common.mean(r.get("answer_support") for r in rows)
    s["verifier_supported"] = common.mean((r.get("verifier") or {}).get("supported") for r in rows)
    s["revision_rate"] = common.mean(float(r.get("revised", False)) for r in rows)
    s["evidence_full@4096"] = common.mean((r.get("retrieval") or {}).get("evidence_full@4096")
                                          for r in rows)
    return s


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="R0,R1,R2,R3,R4,R5")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    keep = [x.strip() for x in args.variants.split(",") if x.strip()]
    suffix = args.out_suffix
    embedder = R.load_embedder()
    evolved_mem = load_answer_memory()
    static_mem = S.memory_from_skills(S.static_answer_skills())
    static_ret = S.static_retrieval_skills()
    evolved_ret = load_retrieval_library()
    qids = common.qidx_list("evaluation")
    variants = {
        "R0": dict(ret_mode="base", ret_lib=(), answer_kind="p7", use_verifier=False),
        "R1": dict(ret_mode="base", ret_lib=(), answer_kind="p7", use_verifier=True),
        "R2": dict(ret_mode="qdm_suff", ret_lib=static_ret, answer_kind="p7", use_verifier=True),
        "R3": dict(ret_mode="base", ret_lib=(), answer_kind="evolved", use_verifier=False),
        "R4": dict(ret_mode="base", ret_lib=(), answer_kind="evolved", use_verifier=True),
        "R5": dict(ret_mode="qdm_suff", ret_lib=evolved_ret, answer_kind="evolved",
                   use_verifier=True),
    }
    labels = {"R0": "direct P7", "R1": "P7 + verifier",
              "R2": "P7 + static ReAct + verifier",
              "R3": "evolved Answer Skills", "R4": "evolved Answer Skills + verifier",
              "R5": "evolved Answer + evolved Retrieval + verifier"}
    all_rows, summary = [], {}
    for name, cfg in variants.items():
        if name not in keep:
            continue
        rows = run_variant(name, qids, embedder, evolved_mem, static_mem, static_ret,
                           evolved_ret, **cfg)
        s = summarize(rows)
        s["label"] = labels[name]
        summary[name] = s
        all_rows.extend(rows)
        print(f"{name}: EM={s['project_em']:.3f} F1={s['project_f1']:.3f} "
              f"offF1={s['official_f1']:.3f} adv={s['adversarial_refusal']:.3f} "
              f"support={s['answer_support']:.3f} revise={s['revision_rate']:.3f} "
              f"calls={s['mean_calls']:.1f}", flush=True)
    common.write_jsonl(common.OUT / "artifacts" / f"react_predictions{suffix}.jsonl", all_rows)
    common.write_json(common.OUT / "artifacts" / f"react_summary{suffix}.json", summary)

    lines = ["# Question-Time ReAct + Grounded Verification (eval set)", "",
             "| Variant | Description | EM | F1 | official F1 | adv refusal | "
             "answer support | revision rate | LLM calls |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {labels[name]} | {s['project_em']:.3f} | {s['project_f1']:.3f} | "
                     f"{s['official_f1']:.3f} | {s['adversarial_refusal']:.3f} | "
                     f"{s['answer_support']:.3f} | {s['revision_rate']:.3f} | "
                     f"{s['mean_calls']:.1f} |")
    lines += ["", "The verifier never receives gold answers, gold evidence, or evaluation labels. "
              "Revision is capped at one extra Reader call; the retrieval loop is capped at two "
              "rounds.", ""]
    (common.OUT / "react_and_verification.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote react_and_verification.md")


if __name__ == "__main__":
    main()
