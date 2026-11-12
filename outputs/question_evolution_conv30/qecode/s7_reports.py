#!/usr/bin/env python
"""Stage 7 - generate the narrative reports from the measured artifacts."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from . import common
from . import skills as S


def _fmt(x, nd=3):
    try:
        if x is None or x != x:
            return "-"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def load_trace() -> List[Dict]:
    rows = common.read_jsonl(common.OUT / "skill_evolution_trace.jsonl")
    seen, out = {}, []
    for r in rows:
        ev = r.get("event")
        if ev in ("adapt_answer", "adapt_retrieval"):
            k = (ev, r.get("step"), r.get("qid"))
        elif ev in ("snapshot", "retrieval_snapshot"):
            k = (ev, r.get("step"))
        else:
            k = None
        if k is None:
            out.append(r)
        elif k in seen:
            out[seen[k]] = r
        else:
            seen[k] = len(out)
            out.append(r)
    return out


def answer_report(trace: List[Dict]) -> None:
    state = common.read_json(common.OUT / "artifacts" / "answer_evolution_state.json")
    cands = []
    for p in sorted(common.SNAP.glob("answer_dev_eval_step*.json")):
        d = common.read_json(p)
        cands.append((float(d.get("official_f1", -1)), float(d.get("project_f1", -1)),
                      int(d.get("step", int(p.stem.split("step")[-1])))))
    best = max(cands)[2] if cands else int(state["best_snapshot_step"])
    mem = common.read_json(common.SNAP / f"answer_memory_step{best:03d}.json")
    skills = mem.get("skills", [])
    events = [e for e in trace if e.get("event") == "adapt_answer"]
    decisions = Counter(e.get("decision", "correct") for e in events)
    wrong = [e for e in events if e.get("decision") != "correct"]
    error_types = Counter((e.get("reflection") or {}).get("error_type") for e in wrong
                          if e.get("reflection"))
    sanitize_fail = [e for e in events if (e.get("candidate_sanitize") or {}).get("ok") is False]
    leak_fail = [e for e in sanitize_fail
                 if (e.get("candidate_sanitize") or {}).get("reason") == "leak_check"]
    ver_accepted = sum(1 for e in events if (e.get("verification") or {}).get("accepted"))
    ver_rejected = sum(1 for e in events
                       if e.get("verification") is not None
                       and not (e.get("verification") or {}).get("accepted"))
    repair_attempts = sum(1 for e in events if "repair_verification" in e)
    repair_accepted = sum(1 for e in events
                          if (e.get("repair_verification") or {}).get("accepted"))

    # usage / reuse from the adaptation trace
    uses = defaultdict(list)
    for e in events:
        for sid in e.get("skills_used") or []:
            uses[sid].append(e)
    reuse_rows = []
    for s in skills:
        sid = s["skill_id"]
        created = int(s.get("created_at_feedback", 0) or 0)
        later = [e for e in uses.get(sid, []) if int(e["step"]) > created]
        success = sum(1 for e in later if e.get("em") == 1.0)
        reuse_rows.append({
            "skill_id": sid, "title": s.get("title"), "created": created,
            "version": s.get("version"), "status": s.get("status"),
            "categories": s.get("categories"), "success_count": s.get("success_count"),
            "failure_count": s.get("failure_count"), "confidence": s.get("confidence"),
            "reuse_count": len(later), "reuse_success": success,
            "first_reuse_step": min((int(e["step"]) for e in later), default=None),
            "source_qids": s.get("source_training_qids"),
            "action": s.get("action_instructions"),
        })

    lines = ["# Answer Skill Evolution", "",
             "Adaptation split: 43 questions.  The Reader is P7 plus at most two retrieved "
             "Answer Skills.  Gold answers are visible only in adaptation/reflection and in the "
             "scorer; the evaluation split is untouched until the final snapshot is frozen.", "",
             "## Adaptation outcome", "",
             f"- Adaptation questions: **{len(events)}**",
             f"- Correct without a new skill: **{decisions.get('correct', 0)}**",
             f"- Skills added: **{decisions.get('added', 0)}**; updated: **{decisions.get('updated', 0)}**",
             f"- Candidates rejected by verification: **{ver_rejected}**; accepted: **{ver_accepted}**; "
             f"verification pass rate: **{ver_accepted}/{ver_accepted + ver_rejected}**",
             f"- Sanitizer rejections: **{len(sanitize_fail)}** (leak-check: {len(leak_fail)})",
             f"- Self-repair attempts / accepted: **{repair_attempts} / {repair_accepted}**",
             f"- Active skills in the selected snapshot ({best}): **{state.get('final_n_active', len([s for s in skills if s.get('status') == 'active']))}**",
             f"- Total skills ever created (including disabled/merged): **{len(skills)}**",
             f"- Merges: **{len(state.get('merges') or [])}**; rollbacks: **{mem.get('meta', {}).get('n_rollbacks', 0)}**",
             "", "### Error types driving reflection", "",
             "| error_type | count |", "|---|---:|"]
    for k, v in error_types.most_common():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Development snapshots (snapshot selection)", "",
              "| feedback questions | active skills | dev EM | dev F1 | dev official F1 | adv refusal |",
              "|---:|---:|---:|---:|---:|---:|"]
    for p in sorted((common.SNAP).glob("answer_dev_eval_step*.json")):
        d = common.read_json(p)
        lines.append(f"| {d.get('step')} | {d.get('n_active_skills')} | {_fmt(d.get('project_em'))} | "
                     f"{_fmt(d.get('project_f1'))} | {_fmt(d.get('official_f1'))} | "
                     f"{_fmt(d.get('adversarial_refusal'))} |")
    trig_path = common.OUT / "artifacts" / "answer_dev_eval_step043_triggers.json"
    trig_line = ""
    if trig_path.exists():
        td = common.read_json(trig_path)
        trig_line = (f"Ablation: rendering each skill's trigger conditions in the Reader prompt "
                     f"(instead of action text only) scored dev official F1 "
                     f"{_fmt(td.get('official_f1'))} / project F1 {_fmt(td.get('project_f1'))}, "
                     f"versus {_fmt(common.read_json(common.SNAP / f'answer_dev_eval_step{best:03d}.json').get('official_f1'))} "
                     f"/ {_fmt(common.read_json(common.SNAP / f'answer_dev_eval_step{best:03d}.json').get('project_f1'))} "
                     f"for the deployed action-only rendering.  It was not deployed because dev "
                     f"selection preferred the action-only format.")
    lines += ["", f"Selected snapshot: **step {best}** (highest development F1; evaluation never "
              "participated in selection).", ""]
    if trig_line:
        lines += [trig_line, ""]
    lines += ["## Skills in the selected memory", "",
              "| skill_id | title | cats | v | status | success/failure | confidence | created | reused later (success) |",
              "|---|---|---|---:|---|---:|---:|---:|---:|"]
    for r in sorted(reuse_rows, key=lambda x: (x["created"], x["skill_id"])):
        lines.append(f"| {r['skill_id']} | {r['title']} | {r['categories']} | {r['version']} | "
                     f"{r['status']} | {r['success_count']}/{r['failure_count']} | "
                     f"{_fmt(r['confidence'], 2)} | {r['created']} | "
                     f"{r['reuse_count']} ({r['reuse_success']}) |")
    lines += ["", "## Reuse across questions (RSI evidence)", ""]
    reusable = [r for r in reuse_rows if r["reuse_count"] > 0]
    if reusable:
        for r in reusable[:12]:
            lines.append(f"- `{r['skill_id']}` was learned at feedback step {r['created']} and "
                         f"selected on {r['reuse_count']} later adaptation questions "
                         f"(first reuse step {r['first_reuse_step']}), {r['reuse_success']} correct.")
    else:
        lines.append("- No skill was reused on a later adaptation question; this is evidence "
                     "against cross-question generalisation in this run.")
    lines += ["", "## Selected skill texts", ""]
    for r in sorted(reuse_rows, key=lambda x: (x["created"], x["skill_id"])):
        if r["status"] != "active":
            continue
        lines.append(f"### {r['skill_id']} (v{r['version']})")
        lines.append("")
        lines.append(f"**{r['title']}** — created at step {r['created']}, source qids "
                     f"{r['source_qids']}.")
        lines.append("")
        lines.append(f"> {r['action']}")
        lines.append("")
    lines += ["## Reading", "",
              "- Acceptance requires the candidate to fix its source question, pass similar-question "
              "replay, pass dissimilar-question replay, and preserve adversarial refusal.",
              "- The sanitizer rejects candidates that mention any distinctive answer token, "
              "proper noun, or explicit year from the source question.",
              "- A skill that is created and never reused is a local fix, not evidence of RSI "
              "generalisation.  The reuse table is the direct test.",
              "- Reuse is frequent (16/14/7 later adaptation questions) but successful reuse is "
              "rare (2 correct answers per skill).  Selection is category-driven, so the skills "
              "are applied to many questions whose trigger they do not really match.",
              "- On the evaluation split, 28/43 questions retrieved at least one skill, but only "
              "1 hit was helpful, 1 harmful and 26 neutral (`error_analysis.md`).  Frequent "
              "retrieval is therefore not evidence of benefit.", ""]
    (common.OUT / "answer_skill_evolution.md").write_text("\n".join(lines), encoding="utf-8")


def retrieval_report(trace: List[Dict]) -> None:
    state_path = common.OUT / "artifacts" / "retrieval_evolution_state.json"
    state = common.read_json(state_path) if state_path.exists() else {}
    mem_path = common.OUT / "retrieval_skill_memory.jsonl"
    skills = common.read_jsonl(mem_path) if mem_path.exists() else []
    cands = []
    for fp in sorted(common.SNAP.glob("retrieval_dev_eval_step*.json")):
        dd = common.read_json(fp)
        cands.append((float(dd.get("evidence_full@4096", -1)),
                      float(dd.get("recall@5", -1)),
                      int(dd.get("step", int(fp.stem.split("step")[-1])))))
    sel_step = max(cands)[2] if cands else 0
    sel_path = common.SNAP / f"retrieval_memory_step{sel_step:03d}.json"
    deployed = []
    if sel_path.exists():
        deployed = [x for x in common.read_json(sel_path).get("skills", [])
                    if x.get("status") == "active"]
    events = [e for e in trace if e.get("event") == "adapt_retrieval"]
    decisions = Counter(e.get("decision", "?") for e in events)
    recovered, lost = [], []
    for e in events:
        q = int(e["qidx"])
        b = common.evidence_metrics(q, common.base_ranked(q))
        bf = b.get("evidence_full@4096")
        af = e.get("evidence_full")
        if bf == 0 and af == 1:
            recovered.append(e)
        elif bf == 1 and af != 1:
            lost.append(e)
    lines = ["# Retrieval Skill Evolution", "",
             "The base ranking is the frozen COMP-P4_top5 HippoRAG2 ranking.  Retrieval skills "
             "only add queries; their dense session rankings are merged by RRF.  QDM and the "
             "evidence-sufficiency check are shared across static/evolved libraries; only the "
             "skill library differs.", "",
             "## Adaptation outcome", "",
             f"- Adaptation questions: **{len(events)}**",
             f"- Evidence already complete after the base + QDM/sufficiency pipeline: "
             f"**{decisions.get('sufficient', 0)}**",
             f"- Retrieval skills added: **{decisions.get('added', 0)}**; updated: "
             f"**{decisions.get('updated', 0)}**; rejected: **{decisions.get('rejected', 0)}**; "
             f"candidate pass rate: **{decisions.get('added', 0)}/"
             f"{decisions.get('added', 0) + decisions.get('rejected', 0)}**",
             f"- Questions whose gold evidence was recovered by the skill layer: **{len(recovered)}**",
             f"- Questions whose gold evidence was lost relative to the base ranking: **{len(lost)}**",
             f"- Active retrieval skills in the raw final memory: **{len([s for s in skills if s.get('status') == 'active'])}**",
             f"- Snapshot chosen on development: **step {sel_step}**, deploying **{len(deployed)}** active skills.",
             "", "## Development snapshots (retrieval-only, no LLM)", "",
             "| feedback | active skills | R@5 | evidence full@4096 | multi-evidence complete |",
             "|---:|---:|---:|---:|---:|"]
    for p in sorted(common.SNAP.glob("retrieval_dev_eval_step*.json")):
        d = common.read_json(p)
        lines.append(f"| {d.get('step')} | {d.get('n_active')} | {_fmt(d.get('recall@5'))} | "
                     f"{_fmt(d.get('evidence_full@4096'))} | {_fmt(d.get('multi_evidence_complete'))} |")
    lines += ["", "## Retrieval skills", "",
              "| skill_id | title | cats | v | status | queries | source qids |",
              "|---|---|---|---:|---|---|---|"]
    for s in skills:
        dep = any(d["skill_id"] == s["skill_id"] for d in deployed)
        status = s.get("status") if dep or s.get("status") != "active" else "rejected_by_dev"
        lines.append(f"| {s['skill_id']} | {s.get('title')} | {s.get('categories')} | "
                     f"{s.get('version')} | {status} | "
                     f"{'; '.join(s.get('query_templates') or [])[:160]} | "
                     f"{s.get('source_training_qids')} |")
    js_path = common.OUT / "artifacts" / "joint_summary.json"
    if js_path.exists():
        js = common.read_json(js_path)["summaries"]
        lines += ["", "## Independent retrieval contribution (eval)", "",
                  "| Exp | R@3 | R@5 | evidence full@4096 | multi-evidence complete |",
                  "|---|---:|---:|---:|---:|"]
        for m in ("E0", "E3", "E4"):
            s = js.get(f"evaluation|{m}")
            if s:
                lines.append(f"| {m} | {s.get('retrieval_recall@3', float('nan')):.3f} | "
                             f"{s.get('retrieval_recall@5', float('nan')):.3f} | "
                             f"{s.get('evidence_full@4096', float('nan')):.3f} | "
                             f"{s.get('multi_evidence_complete', float('nan')):.3f} |")
    ep = common.OUT / "artifacts" / "error_cases.json"
    if ep.exists():
        aa = common.read_json(ep).get("retrieval_always_apply") or {}
        if aa.get("n"):
            lines += ["", "## Auxiliary: raw learned retrieval skill applied without the sufficiency gate", "",
                      "This is the skill *before* development snapshot selection; it is shown only "
                      "as an ablation, not as the deployed E4/E6 memory.", "",
                      f"- Active skills applied: {aa.get('n_skills')}",
                      f"- Base R@5 -> applied R@5: {_fmt(aa.get('base_recall@5'))} -> {_fmt(aa.get('applied_recall@5'))}",
                      f"- Base evidence full -> applied: {_fmt(aa.get('base_evidence_full'))} -> {_fmt(aa.get('applied_evidence_full'))}",
                      f"- Recovered questions: {len(aa.get('recovered') or [])}; lost: {len(aa.get('lost') or [])}",
                      ""]
    lines += ["", "## Reading", "",
              f"- The learned skill passed adaptation verification but lowered development "
              f"evidence coverage (step 0 vs later snapshots).  Snapshot selection therefore chose "
              f"**step {sel_step}**; the skill is retained in the trace as a verified-but-rejected "
              f"negative-transfer case, not deployed in E4/E6.",
              "- The pipeline is capped at two retrieval rounds (QDM round, then one skill/missing-"
              "relation round).",
              "- At evaluation time the sufficiency check sees only the question and retrieved "
              "sessions; gold evidence is never used to decide whether to retrieve again.",
              "- Retrieval-skill verification is local and LLM-free: replay the source, similar "
              "and dissimilar training questions with dense retrieval only.", ""]
    (common.OUT / "retrieval_skill_evolution.md").write_text("\n".join(lines), encoding="utf-8")


def final_report(trace: List[Dict]) -> None:
    js_path = common.OUT / "artifacts" / "joint_summary.json"
    js = common.read_json(js_path)["summaries"] if js_path.exists() else {}
    rs_path = common.OUT / "artifacts" / "react_summary.json"
    rs = common.read_json(rs_path) if rs_path.exists() else {}
    rep_path = common.OUT / "artifacts" / "repeat_summary.json"
    rep = common.read_json(rep_path) if rep_path.exists() else {}
    ans_state_path = common.OUT / "artifacts" / "answer_evolution_state.json"
    ans_state = common.read_json(ans_state_path) if ans_state_path.exists() else {}
    ret_state_path = common.OUT / "artifacts" / "retrieval_evolution_state.json"
    ret_state = common.read_json(ret_state_path) if ret_state_path.exists() else {}
    costs_path = common.OUT / "artifacts" / "cost_summary.json"
    costs = common.read_json(costs_path) if costs_path.exists() else {}
    events = [e for e in trace if e.get("event") == "adapt_answer"]
    decisions = Counter(e.get("decision") for e in events)

    def ev(m):
        return js.get(f"evaluation|{m}", {})

    def dev(m):
        return js.get(f"development|{m}", {})

    def d(a, b, key="project_f1"):
        try:
            return float(ev(b).get(key, 0)) - float(ev(a).get(key, 0))
        except (TypeError, ValueError):
            return float("nan")

    lines = ["# FINAL REPORT — LoCoMo conv-30 Question-Time Self-Evolving Answer & Retrieval Skills", "",
             "Date: 2026-11-10.  Project: RSI GraphRAG.  Base graph/retrieval: frozen "
             "**COMP-P4_top5** HippoRAG2 (no LLM fact reranker), 4096-token context.  "
             "LLM: HY4 (`no_think`), embedding: NV-Embed-v2.", "",
             "## 1. Executive summary", "",
             "A closed loop was implemented and executed: adaptation feedback → error "
             "reflection → candidate Answer/Retrieval Skill → replay/adversarial verification → "
             "versioned Skill Memory → snapshot selection on development → frozen evaluation. "
             "The evaluation split (43 questions) was never used for skill selection.", "",
             "**Headline:** the loop is mechanically functional, but on the 43-question evaluation "
             "split no skill layer beats the fixed P7 Reader.  The evolved Answer Skills beat the "
             "hand-written static rules and the long static prompt, but still lose to P7; the "
             "retrieval layer degrades evidence recall.  This is a negative generalisation result, "
             "not a positive one.", "",
             f"- Answer Skill evolution: {decisions.get('added', 0)} skills added, "
             f"{decisions.get('updated', 0)} updated, "
             f"{sum(1 for e in events if e.get('decision') == 'rejected')} candidates rejected, "
             f"{len([s for s in (common.read_json(common.SNAP / ('answer_memory_step%03d.json' % int(ans_state.get('best_snapshot_step', 0)))).get('skills', [])) if s.get('status') == 'active'])} active at the selected snapshot.",
             f"- Retrieval Skill evolution: {ret_state.get('final_n_active', 0)} learned skill; development "
             f"snapshot selection deployed 0 (the learned skill lowered dev evidence coverage). "
             f"Snapshots exist at 0/10/20/30/43 feedback questions.",
             f"- Evaluation matrix E0–E6 and a ReAct/verifier loop were run on the same frozen "
             f"retrieval records.",
             f"- Total LLM call records: {costs.get('total_calls', '?')} "
             f"(cache hits {costs.get('total_cache_hits', '?')}; endpoint attempts "
             f"{(costs.get('api') or {}).get('api_rows', '?')}, success rate "
             f"{_fmt((costs.get('api') or {}).get('success_rate'))}).", "",
             "## 2. Main results", "",
             "### 2.1 E0–E6 matrix (evaluation split, n=43)", "",
             "| Exp | Answer layer | Retrieval layer | EM | F1 | official F1 | adv refusal | Ev@4096 | multi-ev | calls |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    labels = {"E0": ("P7", "base"), "E1": ("static answer", "base"), "E2": ("evolved answer", "base"),
              "E3": ("P7", "static retrieval"), "E4": ("P7", "evolved retrieval"),
              "E5": ("evolved answer", "static retrieval"), "E6": ("evolved answer", "evolved retrieval"),
              "EB": ("long static prompt", "base")}
    for m in ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "EB"):
        s = ev(m)
        if not s:
            continue
        a, r = labels[m]
        lines.append(f"| {m} | {a} | {r} | {_fmt(s.get('project_em'))} | {_fmt(s.get('project_f1'))} | "
                     f"{_fmt(s.get('official_f1'))} | {_fmt(s.get('adversarial_refusal'))} | "
                     f"{_fmt(s.get('evidence_full@4096'))} | {_fmt(s.get('multi_evidence_complete'))} | "
                     f"{_fmt(s.get('mean_calls'), 2)} |")
    lines += ["", "Note: E4/E6 deploy 0 learned retrieval skills because development snapshot "
              "selection rejected the one learned skill; they therefore measure QDM + "
              "sufficiency without a learned skill.  E3/E5 use the hand-written static retrieval "
              "library.", "", "### 2.2 ReAct / verifier (evaluation)", "",
              "| Variant | EM | F1 | official F1 | adv refusal | answer support | revise rate | calls |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m, s in rs.items():
        lines.append(f"| {m} | {_fmt(s.get('project_em'))} | {_fmt(s.get('project_f1'))} | "
                     f"{_fmt(s.get('official_f1'))} | {_fmt(s.get('adversarial_refusal'))} | "
                     f"{_fmt(s.get('answer_support'))} | {_fmt(s.get('revision_rate'))} | "
                     f"{_fmt(s.get('mean_calls'), 2)} |")
    lines += ["", "## 3. Answers to the required questions", ""]

    e0, e2, e6 = ev("E0"), ev("E2"), ev("E6")
    lines += ["### Q1. Can Answer Skills self-evolve without changing model weights?", "",
              f"Yes, mechanically: the loop generated and versioned {len([s for s in (ans_state.get('snapshots') or {})])} "
              f"snapshots and accepted {decisions.get('added', 0) + decisions.get('updated', 0)} skill "
              f"writes.  The selected memory contains "
              f"{ans_state.get('final_n_active', '?')} active skills.  No weights were modified; "
              f"the full policy is prompt text in `skill_memory.jsonl`.", ""]
    lines += ["### Q2. Does it beat static P7?", "",
              f"E2 − E0: EM {d('E0', 'E2', 'project_em'):+.3f}, F1 {d('E0', 'E2'):+.3f}, "
              f"official F1 {d('E0', 'E2', 'official_f1'):+.3f}, adversarial refusal "
              f"{d('E0', 'E2', 'adversarial_refusal'):+.3f}.  "
              f"E2 vs the long static prompt (EB): F1 {float(e2.get('project_f1', 0)) - float(ev('EB').get('project_f1', 0)):+.3f}, "
              f"official F1 {float(e2.get('official_f1', 0)) - float(ev('EB').get('official_f1', 0)):+.3f}.  "
              f"On 43 evaluation questions a change of one question is about 2.3 points of EM; "
              f"the repeated-run spread in Section 4 bounds the noise.  **E2 therefore does not beat "
              f"static P7 on this evaluation split.**", ""]
    bc0, bc2 = e0.get("by_category", {}), e2.get("by_category", {})
    lines += ["### Q3. Did it fix P7's single-hop / multi-hop regression?", "",
              "| category | E0 F1 | E2 F1 | delta | E0 official F1 | E2 official F1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for c in ("1", "2", "4"):
        a = bc0.get(c, {})
        b = bc2.get(c, {})
        lines.append(f"| {common.CATEGORY_NAMES[int(c)]} | {_fmt(a.get('f1'))} | {_fmt(b.get('f1'))} | "
                     f"{float(b.get('f1', 0)) - float(a.get('f1', 0)):+.3f} | "
                     f"{_fmt(a.get('official_f1'))} | {_fmt(b.get('official_f1'))} |")
    lines += ["", "**Interpretation:** E2 improves single-hop F1 (+0.094) and official F1 (+0.087) "
              "but regresses temporal (-0.260 F1) and multi-hop (-0.186 F1).  The learned "
              "temporal-granularity rule over-applies to temporal questions, and the single-hop "
              "phrase rules do not transfer cleanly to multi-hop.  It therefore does not fix both "
              "P7 regressions; it trades temporal/multi-hop accuracy for single-hop accuracy.", "",
              "Read this together with the fixed 43-question size: multi-hop has only five "
              "evaluation questions, so one flip is ±0.2 EM in that stratum.", ""]
    lines += ["### Q4. Can Retrieval Skills improve multi-evidence recall?", "",
              f"E4 − E0: R@5 {float(ev('E4').get('retrieval_recall@5', 0)) - float(e0.get('retrieval_recall@5', 0)):+.3f}, "
              f"evidence full@4096 {d('E0', 'E4', 'evidence_full@4096'):+.3f}, "
              f"multi-evidence complete {float(ev('E4').get('multi_evidence_complete', 0)) - float(e0.get('multi_evidence_complete', 0)):+.3f}.  "
              f"E4 is evaluated with QDM + sufficiency + at most one extra retrieval round.  The loop learned "
              f"{ret_state.get('final_n_active', 1)} retrieval skill, but development snapshot "
              f"selection chose step {ret_state.get('selected_dev_step', 0)} (0 deployed skills), so "
              f"E4 is QDM without a learned retrieval skill.  **It does not improve multi-evidence "
              f"recall; the QDM dense merge is the main cause of the drop.**", ""]
    lines += ["### Q5. Does Skill Memory accumulate, update and reuse?", "",
              f"Snapshots were written at 0/10/20/30/43 feedback questions.  Answer memory: "
              f"{ans_state.get('final_n_active', '?')} active skills, "
              f"{len((ans_state.get('merges') or []))} merges, "
              f"and reuse counts are listed in `answer_skill_evolution.md`.  Retrieval memory: "
              f"{ret_state.get('final_n_active', 0)} learned skill with "
              f"{len((ret_state.get('merges') or []))} merges, but the development-selected "
              f"snapshot deploys 0 skills.  The number of skills a later "
              f"question actually selected is the operational test of reuse, not the number of "
              f"skills stored.", ""]
    lines += ["### Q6. Is ReAct + verification worth the extra cost?", "",
              f"R1 − R0: F1 {float(rs.get('R1', {}).get('project_f1', 0)) - float(rs.get('R0', {}).get('project_f1', 0)):+.3f} "
              f"at {float(rs.get('R1', {}).get('mean_calls', 0)) - float(rs.get('R0', {}).get('mean_calls', 0)):+.2f} extra calls/question.  "
              f"R5 − R3: F1 {float(rs.get('R5', {}).get('project_f1', 0)) - float(rs.get('R3', {}).get('project_f1', 0)):+.3f} "
              f"at {float(rs.get('R5', {}).get('mean_calls', 0)) - float(rs.get('R3', {}).get('mean_calls', 0)):+.2f} extra calls.  "
              f"(R5 was run with the raw learned retrieval skill before development snapshot "
              f"selection rejected it; it is an upper-bound exploratory variant, not the frozen "
              f"system.)  "
              f"The verifier is a grounded check, not a gold-answer oracle; it can suppress "
              f"hallucinations but cannot create missing evidence.  **On this split the verifier is not "
              f"worth the extra calls: it revises a third to a half of answers without an F1 gain.**", ""]
    lines += ["### Q7. Which skills generalize and which memorize the training questions?", "",
              "The sanitizer rejects any candidate containing a distinctive gold-answer token, a "
              "proper noun from the question, or an explicit year.  Skills that were never selected "
              "after their creation are local fixes; skills with reuse_count > 0 are the "
              "generalisation candidates.  Per-skill reuse and success are in "
              "`answer_skill_evolution.md` and `retrieval_skill_evolution.md`.  On the evaluation "
              "split, 28/43 questions retrieved at least one Answer Skill, but only 1 was helpful, "
              "1 was harmful and 26 were neutral; the harmful hit (a temporal answer changed to a "
              "relative phrase) is a concrete over-generalisation.  The skill-hit table in "
              "`error_analysis.md` lists every case.", ""]
    lines += ["### Q8. Can this be an independent component of a full RSI GraphRAG?", "",
              "Yes as a policy layer: it is independent of graph construction and model weights, "
              "it has persistent versioned memory, a strict adaptation/evaluation split, and a "
              "verification gate.  Its ceiling is set by the frozen retrieval evidence: with "
              "P7 on gold-evidence oracle reaching ~0.60 F1 in the prior study, answer-side skills "
              "can only recover the errors whose evidence is already present.", "",
              "## 4. Repeated-run stability", "",
              "| configuration | EM A | EM B | F1 A | F1 B | adv refusal A/B | EM-unstable |",
              "|---|---:|---:|---:|---:|---|---:|"]
    pairs = _paired_rows()
    for d in pairs:
        lines.append(f"| {d['a']} vs {d['b']} | {_fmt(d['ema'])} | {_fmt(d['emb'])} | "
                     f"{_fmt(d['f1a'])} | {_fmt(d['f1b'])} | {_fmt(d['adva'])} / {_fmt(d['advb'])} | "
                     f"{len(d['unstable'])} |")
    lines += ["", "Two independent runs were available per configuration (the endpoint quota did "
              "not allow three fresh runs).  Per-question flips are in `repeat_stability.md`.", "",
              "## 5. Cost", "",
              f"- LLM call records {costs.get('total_calls', '?')}; prompt tokens "
              f"{costs.get('total_prompt_tokens', '?')}; completion tokens "
              f"{costs.get('total_completion_tokens', '?')}.",
              f"- Endpoint latency P50/P95: {_fmt(costs.get('latency_p50'), 2)}s / "
              f"{_fmt(costs.get('latency_p95'), 2)}s.",
              "- Adaptation reflection/verification is a one-off training cost; evaluation "
              "methods E0–E6 use 1–3 Reader-side calls per question.", "",
              "## 6. Limitations", "",
              "- conv-30 has been studied in prior rounds.  The 43-question evaluation split is "
              "fixed before learning, but it is an internal exploratory test, not an independent "
              "final benchmark.",
              "- The retrieval base is frozen; the study isolates skill effects but does not "
              "rebuild the graph.",
              "- Category-5 questions are entity-swapped adversarial mirrors; the official strict "
              "keyword protocol and the project adapted-refusal protocol are both reported.",
              "- HY4 at temperature=0 is not deterministic; repeated runs are reported for the "
              "final candidates.",
              "- The Answer Skill memory is prompt text.  It improves policy, not knowledge; it "
              "cannot recover evidence the retrieval layer never returned.",
              "- The deployed Reader shows only each skill's action text.  A trigger-condition-"
              "aware rendering was tested on development and was neutral/slightly worse "
              "(official F1 0.683 vs 0.687), so it was not deployed; this is a remaining design "
              "choice, not evidence that trigger rendering would fix over-application.",
              "- Small strata (5 multi-hop evaluation questions) make category-level claims "
              "directional.", "",
              "## 7. Artifact index", "",
              "| file | contents |", "|---|---|",
              "| `DATA_SPLIT.md` | fixed stratified split and seed |",
              "| `baseline_results.md` | P0/P7/P8 baselines |",
              "| `answer_skill_evolution.md` | answer-skill loop, snapshots, reuse |",
              "| `retrieval_skill_evolution.md` | retrieval-skill loop, query templates, evidence effects |",
              "| `joint_experiments.md` | E0–E6 + equal-budget static control |",
              "| `react_and_verification.md` | ReAct variants and grounded verifier |",
              "| `error_analysis.md` | fixed/worsened questions, skill hits, overgeneralisation |",
              "| `cost_latency.md` | calls, tokens, latency by stage |",
              "| `repeat_stability.md` | independent repeated runs of the key configurations |",
              "| `qa_predictions.jsonl` | per-question predictions for all main methods |",
              "| `skill_memory.jsonl` / `retrieval_skill_memory.jsonl` | final memories |",
              "| `skill_evolution_trace.jsonl` | every adaptation/verification decision |",
              "| `skill_snapshots/` | memory snapshots and dev evaluations at 0/10/20/30/43 |", ""]
    (common.OUT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")



def joint_report() -> None:
    js_path = common.OUT / "artifacts" / "joint_summary.json"
    if not js_path.exists():
        return
    js = common.read_json(js_path)["summaries"]
    labels = {"E0": ("P7", "base"), "E1": ("static answer", "base"),
              "E2": ("evolved answer", "base"), "E3": ("P7", "static retrieval"),
              "E4": ("P7", "evolved retrieval"), "E5": ("evolved answer", "static retrieval"),
              "E6": ("evolved answer", "evolved retrieval"),
              "EB": ("long static prompt", "base")}
    lines = ["# Joint Experiments (evaluation split, n=43)", "",
             "The base retrieval is the frozen COMP-P4_top5 HippoRAG2 ranking.  Every method "
             "uses the same P7 Reader unless stated.  E3-E6 use QDM + the evidence-sufficiency "
             "check + at most one extra retrieval round.", "",
             "| Exp | Answer | Retrieval | EM | F1 | official F1 | adv refusal | Ev@4096 | multi-ev | R@5 | LLM calls |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "EB"):
        s = js.get(f"evaluation|{m}")
        if not s:
            continue
        a, r = labels[m]
        lines.append(f"| {m} | {a} | {r} | {_fmt(s.get('project_em'))} | {_fmt(s.get('project_f1'))} | "
                     f"{_fmt(s.get('official_f1'))} | {_fmt(s.get('adversarial_refusal'))} | "
                     f"{_fmt(s.get('evidence_full@4096'))} | {_fmt(s.get('multi_evidence_complete'))} | "
                     f"{_fmt(s.get('retrieval_recall@5'))} | {_fmt(s.get('mean_calls'), 2)} |")
    lines += ["", "## By category (F1 / official F1 / n)", "",
              "| Exp | multi-hop | temporal | single-hop |", "|---|---|---|---|"]
    for m in ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "EB"):
        s = js.get(f"evaluation|{m}")
        if not s:
            continue
        bc = s.get("by_category", {})
        cells = []
        for c in ("1", "2", "4"):
            d = bc.get(c)
            cells.append(f"{_fmt(d.get('f1'))} / {_fmt(d.get('official_f1'))} ({d.get('n')})"
                         if d else "-")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "## Reading", "",
              "- E0 is the strongest method on EM/F1/official F1 and on evidence recall.  Every "
              "skill layer lowers at least one of these on the 43-question evaluation split.",
              "- E2 (evolved answer) beats E1 (retrieved static answer rules) and EB (one long "
              "static prompt), so the evolved rules are less damaging than hand-written rules, "
              "but neither exceeds P7.",
              "- E4/E6 deploy 0 learned retrieval skills after development selection; they are "
              "QDM + sufficiency without a learned skill.  E3/E5 use the static retrieval library.",
              "- E3-E6 show that QDM + dense-query merging degrades the strong graph ranking: "
              "R@5 falls from 0.932 to about 0.80-0.89 and evidence-full from 0.909 to 0.82-0.85.",
              "- Within the retrieval layer, the static 4-rule library (E3, R@5 0.803) is worse "
              "than QDM-only (E4, R@5 0.886), so adding hand-written retrieval queries creates "
              "ranking noise rather than recovering evidence.",
              "- One question is 2.3 EM points on this split; the joint differences are larger "
              "than that for E1/E4/E6, but the overall conclusion is still negative.",
              ""]
    (common.OUT / "joint_experiments.md").write_text("\n".join(lines), encoding="utf-8")


def react_report() -> None:
    p = common.OUT / "artifacts" / "react_summary.json"
    if not p.exists():
        return
    rs = common.read_json(p)
    labels = {"R0": "direct P7", "R1": "P7 + verifier",
              "R2": "P7 + static ReAct + verifier",
              "R3": "evolved Answer Skills", "R4": "evolved Answer + verifier",
              "R5": "evolved Answer + evolved Retrieval + verifier"}
    lines = ["# Question-Time ReAct + Grounded Verification (evaluation split)", "",
             "| Variant | Description | EM | F1 | official F1 | adv refusal | "
             "answer support | revision rate | LLM calls |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in ("R0", "R1", "R2", "R3", "R4", "R5"):
        s = rs.get(m)
        if not s:
            continue
        lines.append(f"| {m} | {labels[m]} | {_fmt(s.get('project_em'))} | {_fmt(s.get('project_f1'))} | "
                     f"{_fmt(s.get('official_f1'))} | {_fmt(s.get('adversarial_refusal'))} | "
                     f"{_fmt(s.get('answer_support'))} | {_fmt(s.get('revision_rate'))} | "
                     f"{_fmt(s.get('mean_calls'), 2)} |")
    lines += ["", "The verifier sees only the question, retrieved evidence and candidate answer.  "
              "It never receives gold answers, gold evidence or evaluation labels.  Revision is "
              "capped at one extra Reader call; retrieval is capped at two rounds.", "",
              "Reading: the verifier revises 33-54% of answers but does not improve F1; it mainly "
              "adds calls.  The full evolved system (R5) is the worst variant because both the "
              "answer and retrieval skill layers over-apply.", ""]
    (common.OUT / "react_and_verification.md").write_text("\n".join(lines), encoding="utf-8")


def _paired_rows():
    import numpy as np
    jp = common.OUT / "artifacts" / "joint_predictions.jsonl"
    rp = common.OUT / "artifacts" / "react_predictions.jsonl"
    if not jp.exists() or not rp.exists():
        return []
    jrows = common.read_jsonl(jp)
    rrows = common.read_jsonl(rp)

    def index(rows, method):
        return {int(r["qidx"]): r for r in rows if r.get("method") == method}

    out = []
    for a, b in (("E0", "R0"), ("E2", "R3")):
        A, B = index(jrows, a), index(rrows, b)
        qs = [q for q in sorted(set(A) & set(B)) if common.category_of(q) != 5]
        ema = float(np.mean([A[q]["em"] for q in qs]))
        emb = float(np.mean([B[q]["em"] for q in qs]))
        f1a = float(np.mean([A[q]["f1"] for q in qs]))
        f1b = float(np.mean([B[q]["f1"] for q in qs]))
        adv_q = [q for q in sorted(set(A) & set(B)) if common.category_of(q) == 5]
        adva = float(np.mean([A[q]["em"] for q in adv_q])) if adv_q else float("nan")
        advb = float(np.mean([B[q]["em"] for q in adv_q])) if adv_q else float("nan")
        unstable = [q for q in qs if A[q]["em"] != B[q]["em"]]
        big = [q for q in qs if abs(A[q]["f1"] - B[q]["f1"]) > 0.25]
        out.append({"a": a, "b": b, "ema": ema, "emb": emb, "f1a": f1a, "f1b": f1b,
                    "adva": adva, "advb": advb, "unstable": unstable, "big": big,
                    "A": A, "B": B})
    return out


def stability_report() -> None:
    pairs = _paired_rows()
    if not pairs:
        return
    lines = ["# Repeated-run stability", "",
             "The endpoint quota allowed two independent runs of the two key configurations "
             "instead of the planned three fresh runs.  E0/E2 are the main cached runs; R0/R3 "
             "are independent fresh calls of the same configuration.", "",
             "| configuration | EM A | EM B | F1 A | F1 B | adv refusal A/B | EM-unstable | F1 change > 0.25 |",
             "|---|---:|---:|---:|---:|---|---:|---:|"]
    for d in pairs:
        lines.append(f"| {d['a']} vs {d['b']} | {_fmt(d['ema'])} | {_fmt(d['emb'])} | "
                     f"{_fmt(d['f1a'])} | {_fmt(d['f1b'])} | {_fmt(d['adva'])} / {_fmt(d['advb'])} | "
                     f"{len(d['unstable'])} | {len(d['big'])} |")
    lines += ["", "## Per-question changes (answerable questions)", ""]
    for d in pairs:
        lines.append(f"### {d['a']} vs {d['b']}")
        lines.append("")
        for q in sorted(set(d["unstable"]) | set(d["big"])):
            lines.append(f"- q{q}: EM A={d['A'][q]['em']} B={d['B'][q]['em']}; "
                         f"F1 A={_fmt(d['A'][q]['f1'])} B={_fmt(d['B'][q]['f1'])}")
        lines.append("")
    lines += ["A difference of two EM points on 33 answerable questions is about 0.7 question.  "
              "The per-question flips show that several apparent skill gains/losses are within "
              "HY4 nondeterminism.", ""]
    (common.OUT / "repeat_stability.md").write_text("\n".join(lines), encoding="utf-8")



def main():
    trace = load_trace()
    answer_report(trace)
    retrieval_report(trace)
    joint_report()
    react_report()
    stability_report()
    final_report(trace)
    print("wrote answer_skill_evolution.md, retrieval_skill_evolution.md, FINAL_REPORT.md")


if __name__ == "__main__":
    main()
