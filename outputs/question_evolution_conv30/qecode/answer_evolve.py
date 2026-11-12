"""Answer-Skill self-evolution.

Loop per adaptation question
----------------------------
  1. retrieve relevant Answer Skills from memory (category + pattern + embedding);
  2. answer with the P7 reader plus the selected skills;
  3. score against the training feedback (gold answer is available only here);
  4. if wrong, reflect on the error and propose one general rule;
  5. verify the candidate by replaying the failed question, similar training
     questions, dissimilar questions, and adversarial questions;
  6. accept / update / reject; log every decision.

Verification never looks at the evaluation split.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import common
from . import skills as S

REFLECT_SYSTEM = (
    "You are a reflective skill-learning agent for a retrieval-augmented QA system. "
    "You analyse one failed training question and propose at most one reusable Answer Skill. "
    "You must generalise: never store the question's people, places, events, dates, numbers, "
    "or gold answer. A skill is a short imperative rule with clear trigger conditions. "
    "Output JSON only."
)

REFLECT_PROMPT = """Training question: {question}
Category: {category_name} ({category})
Gold answer: {gold}
System draft answer: {draft}
Project EM={em:.2f}, F1={f1:.2f}

Gold evidence (the evidence that should have been used):
{gold_evidence}

Retrieved sessions (top 5, truncated):
{sessions}

Are the gold evidence sessions inside the retrieved context? {gold_in_context}

Current relevant skills:
{current_skills}

Diagnose the failure. Choose exactly one error_type:
evidence_not_retrieved, evidence_not_used, temporal_arithmetic, temporal_granularity,
multi_fact_omission, over_verbose, format_mismatch, wrong_refusal, unsupported_inference,
wrong_target_or_premise, other.

Rules:
- If error_type is evidence_not_retrieved, set error_axis="retrieval" and candidate_skill=null.
- If the answer is semantically right but the format differs, propose a format rule.
- Otherwise set error_axis="answer" and propose one general rule that would fix this class
  of errors without knowing this specific question.
- If an existing skill already covers the issue, set update_skill_id and give only the
  missing clause in action_instructions.
- Do NOT mention any person, place, organisation, product, specific date, number, or the
  gold answer in the candidate skill.
- action_instructions must be <= 55 words, imperative, and composable with other skills.
- applicable_question_patterns are 1-3 generic question patterns.
- categories is a subset of [1,2,4,5].

Return JSON:
{{"error_axis": "answer|retrieval",
  "error_type": "...",
  "diagnosis": "<=40 words",
  "generalizable_pattern": "<=25 words",
  "update_skill_id": null or "ans_...",
  "candidate_skill": null or {{
     "title": "...",
     "trigger_conditions": "...",
     "applicable_question_patterns": ["..."],
     "categories": [2],
     "action_instructions": "...",
     "expected_output_format": "...",
     "evidence_requirements": "...",
     "failure_modes": ["..."]
  }}}}
"""

REPAIR_SUFFIX = """

The previous candidate was rejected by verification. Previous candidate:
{previous}

Verification failures:
{failures}

Propose a revised candidate that fixes the failed check while remaining general and short.
Return the same JSON schema.
"""


def load_embedder():
    try:
        from src.config import PATHS
        from src.embedding.nvembed import NVEmbedder
        from src.graph.runtime import EmbeddingProvider, make_base_config
        cfg = make_base_config(PATHS["raw_graphs"] / "_warmup", conv_id="qe-skills")
        provider = EmbeddingProvider.get(cfg)
        return NVEmbedder(provider.model, cache_path=common.ART / "qe_skill_embeddings.pkl")
    except Exception as exc:  # noqa: BLE001
        print("embedder unavailable, falling back to lexical skill retrieval:", exc)
        return None


def _candidate_text(c: Dict) -> str:
    parts = [c.get("title", ""), c.get("trigger_conditions", ""),
             c.get("action_instructions", ""), c.get("expected_output_format", ""),
             c.get("evidence_requirements", "")]
    parts += list(c.get("applicable_question_patterns") or [])
    parts += list(c.get("failure_modes") or [])
    return " ".join(str(p) for p in parts if p)


def sanitize_candidate(raw: Dict, qidx: int) -> Tuple[Optional[Dict], Dict]:
    """Validate schema, length, category scope and the no-memorisation constraint."""
    info = {"ok": False, "reason": None, "leak_hits": []}
    if not isinstance(raw, dict):
        info["reason"] = "not_an_object"
        return None, info
    action = str(raw.get("action_instructions") or "").strip()
    title = str(raw.get("title") or "").strip()
    if len(action) < 10:
        info["reason"] = "action_too_short"
        return None, info
    if len(action) > 700:
        action = action[:700].rsplit(" ", 1)[0]
    cats = []
    for c in raw.get("categories") or []:
        try:
            c = int(c)
        except (TypeError, ValueError):
            continue
        if c in (1, 2, 4, 5):
            cats.append(c)
    pats = [str(x).strip() for x in (raw.get("applicable_question_patterns") or []) if str(x).strip()][:4]
    cand = {
        "title": title[:90] or "Learned answer rule",
        "trigger_conditions": str(raw.get("trigger_conditions") or "").strip()[:240],
        "applicable_question_patterns": pats or ["relevant question"],
        "categories": sorted(set(cats)),
        "action_instructions": action,
        "expected_output_format": str(raw.get("expected_output_format") or "").strip()[:180],
        "evidence_requirements": str(raw.get("evidence_requirements") or "").strip()[:180],
        "failure_modes": [str(x)[:120] for x in (raw.get("failure_modes") or [])][:4],
    }
    ok, hits = S.leak_check(_candidate_text(cand), qidx)
    info["leak_hits"] = hits
    if not ok:
        info["reason"] = "leak_check"
        return None, info
    info["ok"] = True
    return cand, info


def reflect(qidx: int, draft_row: Dict, current_skills: Sequence[Dict], *,
            previous: Optional[Dict] = None, failures: Optional[List[str]] = None,
            use_cache: bool = True) -> Dict:
    q = common.qa_map()[qidx]
    cat = int(q.category)
    if cat == 5:
        gold = ("No answer. The question's premise is unsupported: it attributes a fact to the "
                "wrong person or to an event that the context does not state.")
    else:
        gold = str(q.answer or "")
    context, sel, _ = common.record_budget_docs(qidx)
    gold_in_ctx = set(common.gold_doc_ids(qidx)) <= set(sel)
    cur = "\n".join(f"- [{s['skill_id']}] {s.get('action_instructions', '')}"
                    for s in current_skills[:6]) or "(none)"
    prompt = REFLECT_PROMPT.format(
        question=q.question, category_name=common.CATEGORY_NAMES.get(cat, "?"), category=cat,
        gold=gold, draft=draft_row.get("pred_answer", ""), em=draft_row.get("em", 0.0),
        f1=draft_row.get("f1", 0.0), gold_evidence=common.gold_evidence_text(qidx) or "(none)",
        sessions=common.compact_sessions(sel, k=5, limit=420), gold_in_context=gold_in_ctx,
        current_skills=cur,
    )
    if previous:
        prompt += REPAIR_SUFFIX.format(
            previous=json.dumps(previous, ensure_ascii=False)[:1200],
            failures="\n".join(f"- {f}" for f in (failures or []))[:2000],
        )
    out = common.call_json(
        [{"role": "system", "content": REFLECT_SYSTEM}, {"role": "user", "content": prompt}],
        stage="reflect", qid=q.qid, purpose="reflect_answer", max_tokens=1100,
        use_cache=use_cache,
    )
    data = out.get("json") or {}
    data["_usage"] = out.get("usage")
    data["_cached"] = out.get("cached")
    data["_gold_in_context"] = gold_in_ctx
    data["_raw"] = out.get("raw", "")[:2000]
    return data


def _base_row(qidx: int, history: Dict[int, Dict], memory: S.AnswerSkillMemory,
              embedder, prefer_history: bool = True) -> Dict:
    """Baseline for verification.

    For the source question we want the actual failed draft (current memory).
    For replay questions we want the no-skill P7 baseline, so removing a
    previously useful memory skill is not mistaken for candidate harm.
    """
    if prefer_history and qidx in history:
        return history[qidx]
    r = common.answer_context_cached(qidx, "P7")
    if r is not None:
        return r
    sel = S.select_skills(memory.active(), common.qa_map()[qidx].question,
                          common.category_of(qidx), k=2, embedder=embedder,
                          skill_vecs=S.skill_vectors(memory.active(), embedder))
    return common.answer(qidx, sel, stage="verify", purpose="baseline_unseen")


def _rank_similar(qidx: int, candidates: Sequence[int], embedder) -> List[int]:
    if not candidates:
        return []
    if embedder is None:
        q = common.qa_map()[qidx].question
        return sorted(candidates, key=lambda x: -S._jaccard(q, common.qa_map()[x].question))
    texts = [common.qa_map()[qidx].question] + [common.qa_map()[x].question for x in candidates]
    vecs = embedder.encode_queries(texts)
    sims = vecs[1:] @ vecs[0]
    return [candidates[i] for i in np.argsort(-sims)]


def verify_candidate(candidate: Dict, qidx: int, memory: S.AnswerSkillMemory,
                     history: Dict[int, Dict], order: Sequence[int], step: int,
                     embedder, use_cache: bool = True) -> Dict:
    """Deterministic acceptance test with positive fix, replay and safety checks."""
    cat = common.category_of(qidx)
    base_src = _base_row(qidx, history, memory, embedder)
    cand_skills = [dict(candidate, skill_id="__candidate__", version=1)]
    fix = common.answer(qidx, cand_skills, stage="verify", purpose="candidate_fix",
                        use_cache=use_cache)
    fix_pass = (fix["f1"] >= base_src["f1"] + 0.10) or (fix["em"] == 1.0 and base_src["em"] == 0.0)

    # similar training questions
    seen = [q for q in order if q in history and q != qidx]
    similar = [q for q in _rank_similar(qidx, [q for q in seen if common.category_of(q) == cat],
                                        embedder)[:3]]
    # dissimilar training questions: one from each other answerable category
    dissimilar = []
    for c in (1, 2, 4):
        if c == cat:
            continue
        pool = [q for q in seen if common.category_of(q) == c
                and (common.answer_context_cached(q, "P7") or history[q]).get("em") == 1.0]
        if not pool:
            pool = [q for q in seen if common.category_of(q) == c]
        if pool:
            dissimilar.append(pool[0])
    # adversarial safety questions: seen cat-5 first, else fixed held-in adaptation cat-5
    adv_pool = [q for q in seen if common.category_of(q) == 5]
    if len(adv_pool) < 2:
        adv_pool += [q for q in common.qidx_list("adaptation")
                     if common.category_of(q) == 5 and q != qidx and q not in adv_pool]
    adversarial = adv_pool[:2]

    def replay(qq: int, purpose: str) -> Dict:
        b = _base_row(qq, history, memory, embedder, prefer_history=False)
        c = common.answer(qq, cand_skills, stage="verify", purpose=purpose, use_cache=use_cache)
        return {"qidx": qq, "category": common.category_of(qq),
                "baseline_answer": b.get("pred_answer"), "baseline_em": b.get("em"),
                "baseline_f1": b.get("f1"), "candidate_answer": c.get("pred_answer"),
                "candidate_em": c.get("em"), "candidate_f1": c.get("f1"),
                "baseline_refused": bool(b.get("refused")),
                "candidate_refused": bool(c.get("refused"))}

    similar_rows = [replay(q, "candidate_similar") for q in similar]
    dissimilar_rows = [replay(q, "candidate_dissimilar") for q in dissimilar]
    adv_rows = [replay(q, "candidate_adversarial") for q in adversarial]

    hard_neg = sum(1 for r in similar_rows + dissimilar_rows
                   if (r["baseline_em"] == 1.0 and r["candidate_em"] == 0.0))
    soft_neg = sum(1 for r in similar_rows + dissimilar_rows
                   if (r["baseline_f1"] - r["candidate_f1"]) > 0.25)
    adv_viol = sum(1 for r in adv_rows if r["baseline_refused"] and not r["candidate_refused"])
    mean_sim_delta = float(np.mean([r["candidate_f1"] - r["baseline_f1"] for r in similar_rows])) \
        if similar_rows else 0.0
    checks = {
        "fix_pass": bool(fix_pass),
        "fix_f1_delta": round(float(fix["f1"] - base_src["f1"]), 3),
        "similar_pass": hard_neg == 0,
        "dissimilar_pass": sum(1 for r in dissimilar_rows
                               if r["baseline_em"] == 1.0 and r["candidate_em"] == 0.0) == 0,
        "adversarial_pass": adv_viol == 0,
        "leak_pass": True,
    }
    accept = (checks["fix_pass"] and checks["similar_pass"] and
              checks["dissimilar_pass"] and checks["adversarial_pass"])
    failures = []
    if not checks["fix_pass"]:
        failures.append(f"did not fix the source question (F1 delta {checks['fix_f1_delta']:+.2f})")
    if not checks["similar_pass"]:
        failures.append(f"{hard_neg} similar question(s) flipped from correct to wrong")
    if not checks["dissimilar_pass"]:
        failures.append("a dissimilar correct question flipped to wrong")
    if not checks["adversarial_pass"]:
        failures.append(f"{adv_viol} adversarial question(s) stopped refusing")
    return {
        "candidate": {k: candidate.get(k) for k in ("title", "action_instructions", "categories",
                                                    "applicable_question_patterns")},
        "source": {"qidx": qidx, "baseline_answer": base_src.get("pred_answer"),
                   "baseline_em": base_src.get("em"), "baseline_f1": base_src.get("f1"),
                   "candidate_answer": fix.get("pred_answer"), "candidate_em": fix.get("em"),
                   "candidate_f1": fix.get("f1")},
        "similar": similar_rows, "dissimilar": dissimilar_rows, "adversarial": adv_rows,
        "hard_negative_transfers": hard_neg, "soft_negative_transfers": soft_neg,
        "adversarial_violations": adv_viol, "mean_similar_f1_delta": round(mean_sim_delta, 3),
        "checks": checks, "accepted": bool(accept), "failures": failures,
    }


def dev_evaluate(memory: S.AnswerSkillMemory, step: int, embedder,
                 use_cache: bool = True) -> Dict:
    rows = []
    for qidx in common.qidx_list("development"):
        sel = S.select_skills(memory.active(), common.qa_map()[qidx].question,
                              common.category_of(qidx), k=2, embedder=embedder,
                              skill_vecs=S.skill_vectors(memory.active(), embedder))
        if not sel:
            r = common.answer_context_cached(qidx, "P7")
            if r is None:
                r = common.answer(qidx, stage="dev", purpose="dev_empty")
            r = dict(r)
            r["skills_used"] = []
        else:
            r = common.answer(qidx, sel, stage="dev", purpose="dev_with_skills", use_cache=use_cache)
        context, _, _ = common.record_budget_docs(qidx)
        r["answer_support"] = common.answer_support(r.get("pred_answer", ""), context)
        r["selected_skill_details"] = [
            {"skill_id": s["skill_id"], "score": s.get("_select_score"),
             "reasons": s.get("_select_reasons")} for s in sel]
        rows.append(r)
    summary = common.summarize(rows)
    summary["step"] = step
    summary["n_active_skills"] = len(memory.active())
    common.write_jsonl(common.SNAP / f"answer_dev_predictions_step{step:03d}.jsonl", rows)
    common.write_json(common.SNAP / f"answer_dev_eval_step{step:03d}.json", summary)
    return {"rows": rows, "summary": summary}


def main():
    import os
    embedder = load_embedder()
    memory = S.AnswerSkillMemory(common.OUT / "artifacts" / "answer_skill_memory.json")
    order = common.qidx_list("adaptation_order")
    max_steps = int(os.environ.get("QE_MAX_STEPS", "0") or 0)
    if max_steps:
        order = order[:max_steps]
        print(f"DEBUG: limiting adaptation to {len(order)} questions")
    trace_path = common.OUT / ("skill_evolution_trace.debug.jsonl" if max_steps
                               else "skill_evolution_trace.jsonl")
    history: Dict[int, Dict] = {}
    events: List[Dict] = []
    snapshots: Dict[int, Dict] = {}

    # step 0: empty memory
    memory.snapshot(0, "empty")
    ev0 = dev_evaluate(memory, 0, embedder)
    snapshots[0] = {"step": 0, "dev": ev0["summary"], "n_active": 0}

    step = 0
    best = {"step": 0, "f1": ev0["summary"]["project_f1"],
            "official_f1": ev0["summary"]["official_f1"], "memory": memory.snapshot(0, "empty")}
    for i, qidx in enumerate(order, 1):
        step = i
        q = common.qa_map()[qidx]
        skill_vecs = S.skill_vectors(memory.active(), embedder)
        selected = S.select_skills(memory.active(), q.question, int(q.category),
                                   k=2, embedder=embedder, skill_vecs=skill_vecs)
        if not selected:
            r = common.answer_context_cached(qidx, "P7")
            if r is None:
                r = common.answer(qidx, stage="adapt", purpose="adapt_empty")
            r = dict(r)
            r["skills_used"] = []
        else:
            r = common.answer(qidx, selected, stage="adapt", purpose="adapt_with_skills")
        context, _, _ = common.record_budget_docs(qidx)
        r["answer_support"] = common.answer_support(r.get("pred_answer", ""), context)
        r["selected_skill_details"] = [
            {"skill_id": s["skill_id"], "score": s.get("_select_score"),
             "reasons": s.get("_select_reasons")} for s in selected]
        history[qidx] = r
        # Counterfactual attribution: compare with the cacheable no-skill P7 answer.
        base_r = common.answer_context_cached(qidx, "P7")
        if selected and base_r is not None:
            memory.record_marginal([s["skill_id"] for s in selected],
                                   baseline_correct=bool(base_r.get("em") == 1.0),
                                   with_correct=bool(r["em"] == 1.0), feedback_step=i)

        event = {
            "event": "adapt_answer", "step": step, "qid": q.qid, "qidx": qidx,
            "category": int(q.category), "question": q.question, "answer": r.get("pred_answer"),
            "em": r.get("em"), "f1": r.get("f1"), "official_f1": r.get("official_f1"),
            "refused": r.get("refused"), "skills_used": [s["skill_id"] for s in selected],
            "skill_details": r["selected_skill_details"], "n_active_before": len(memory.active()),
        }

        wrong = (r["em"] == 0.0) or (int(q.category) != 5 and r["f1"] < 0.5)
        if wrong:
            reflection = reflect(qidx, r, selected)
            event["reflection"] = {
                "error_axis": reflection.get("error_axis"),
                "error_type": reflection.get("error_type"),
                "diagnosis": reflection.get("diagnosis"),
                "generalizable_pattern": reflection.get("generalizable_pattern"),
                "update_skill_id": reflection.get("update_skill_id"),
                "gold_in_context": reflection.get("_gold_in_context"),
                "raw_candidate": reflection.get("candidate_skill"),
            }
            gold_in_ctx = bool(reflection.get("_gold_in_context"))
            cand_raw = reflection.get("candidate_skill")
            axis = reflection.get("error_axis")
            if not gold_in_ctx:
                cand_raw = None
                event["reflection"]["forced_axis"] = "retrieval: gold evidence not in context"
            if cand_raw:
                cand, info = sanitize_candidate(cand_raw, qidx)
                event["candidate_sanitize"] = info
                if cand:
                    ver = verify_candidate(cand, qidx, memory, history, order, step, embedder)
                    if not ver["accepted"]:
                        # one self-repair attempt with the verification feedback
                        ref2 = reflect(qidx, r, selected, previous=cand,
                                       failures=ver.get("failures") or [])
                        cand2_raw = ref2.get("candidate_skill")
                        event["repair_reflection"] = {
                            "error_type": ref2.get("error_type"),
                            "diagnosis": ref2.get("diagnosis"),
                            "raw_candidate": cand2_raw,
                        }
                        if cand2_raw:
                            cand2, info2 = sanitize_candidate(cand2_raw, qidx)
                            event["repair_sanitize"] = info2
                            if cand2:
                                ver2 = verify_candidate(cand2, qidx, memory, history, order, step,
                                                        embedder)
                                event["repair_verification"] = ver2
                                cand, ver = cand2, ver2
                    event["verification"] = ver
                    if ver["accepted"]:
                        upd = cand_raw.get("update_skill_id") or reflection.get("update_skill_id")
                        existing = memory.get(upd) if upd else None
                        if existing and existing.get("status") == "active":
                            skill = memory.update(existing["skill_id"], cand, step, qidx,
                                                  mode="append")
                            event["decision"] = "updated"
                            event["skill_id"] = existing["skill_id"]
                        else:
                            skill = memory.add(cand, step, qidx)
                            event["decision"] = "added"
                            event["skill_id"] = skill["skill_id"]
                        event["skill_version"] = skill.get("version") if skill else None
                    else:
                        event["decision"] = "rejected"
                else:
                    event["decision"] = "rejected_sanitize"
            else:
                event["decision"] = "no_candidate"
        else:
            event["decision"] = "correct"

        common.append_jsonl(trace_path, event)
        events.append(event)

        # snapshot every 10 feedback questions and at the end
        if step % 10 == 0 or step == len(order):
            p = memory.snapshot(step, f"after_{step}_feedback")
            ev = dev_evaluate(memory, step, embedder)
            snapshots[step] = {"step": step, "dev": ev["summary"],
                               "n_active": len(memory.active()), "snapshot": str(p)}
            print(f"step {step}: dev F1={ev['summary']['project_f1']:.3f} "
                  f"offF1={ev['summary']['official_f1']:.3f} active={len(memory.active())} "
                  f"best={best['step']}", flush=True)
            common.append_jsonl(trace_path, {"event": "snapshot", "step": step,
                                             "dev": ev["summary"],
                                             "n_active": len(memory.active())})
            if ev["summary"]["project_f1"] < best["f1"] - 0.05:
                # roll back to the best development memory
                old = [s for s in memory.skills if s.get("status") == "active"
                       and s.get("created_at_feedback", 0) > best["step"]]
                prev = common.read_json(Path(best["memory"]))
                memory.restore(prev)
                memory.meta["n_rollbacks"] += 1
                rollback = {"event": "rollback", "step": step, "to_step": best["step"],
                            "dev_f1_before": ev["summary"]["project_f1"],
                            "disabled_new_skills": [s["skill_id"] for s in old],
                            "reason": "development F1 dropped > 0.05"}
                common.append_jsonl(trace_path, rollback)
                events.append(rollback)
                print(f"  rollback to step {best['step']}: {rollback['disabled_new_skills']}",
                      flush=True)
            elif ev["summary"]["project_f1"] > best["f1"]:
                best = {"step": step, "f1": ev["summary"]["project_f1"],
                        "official_f1": ev["summary"]["official_f1"],
                        "memory": p}

    # final consolidation: merge duplicates, save
    merges = memory.merge_duplicates(embedder, threshold=0.90, feedback_step=step)
    if merges:
        common.append_jsonl(trace_path, {"event": "merge", "step": step, "merges": merges})
    memory.save()
    final = memory.snapshot(step, "final_after_consolidation")
    common.write_json(common.OUT / "artifacts" / "answer_evolution_state.json",
                      {"snapshots": {str(k): {kk: vv for kk, vv in v.items() if kk != "snapshot"}
                                     for k, v in snapshots.items()},
                       "best_snapshot_step": best["step"],
                       "best_dev_f1": best["f1"],
                       "final_n_active": len(memory.active()),
                       "n_skills_total": len(memory.skills),
                       "merges": merges,
                       "meta": memory.meta})
    # canonical skill_memory.jsonl
    with open(common.OUT / "skill_memory.jsonl", "w", encoding="utf-8") as f:
        for s in memory.skills:
            f.write(json.dumps({**s, "snapshot_step": step}, ensure_ascii=False) + "\n")
    print("answer evolution done:", len(memory.active()), "active /", len(memory.skills), "total")
    return snapshots, events


if __name__ == "__main__":
    main()
