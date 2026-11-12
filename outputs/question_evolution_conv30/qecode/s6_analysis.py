#!/usr/bin/env python
"""Stage 6 - error analysis, cost/latency accounting, consolidated predictions."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from . import common


def _fmt(x, nd=3):
    try:
        if x is None or x != x:
            return "-"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def load_joint_eval() -> Dict[str, Dict[int, Dict]]:
    rows = common.read_jsonl(common.OUT / "artifacts" / "joint_predictions.jsonl")
    ev = set(common.qidx_list("evaluation"))
    out: Dict[str, Dict[int, Dict]] = defaultdict(dict)
    for r in rows:
        if int(r["qidx"]) in ev:
            out[r["method"]][int(r["qidx"])] = r
    return out


def load_react_eval() -> Dict[str, Dict[int, Dict]]:
    rows = common.read_jsonl(common.OUT / "artifacts" / "react_predictions.jsonl")
    ev = set(common.qidx_list("evaluation"))
    out: Dict[str, Dict[int, Dict]] = defaultdict(dict)
    for r in rows:
        if int(r["qidx"]) in ev:
            out[r["method"]][int(r["qidx"])] = r
    return out


def _case(r: Dict) -> Dict:
    qidx = int(r["qidx"])
    q = common.qa_map()[qidx]
    return {
        "qidx": qidx, "qid": q.qid, "category": int(q.category),
        "category_name": common.CATEGORY_NAMES.get(int(q.category), "?"),
        "question": q.question, "gold": q.answer if int(q.category) != 5 else "No answer",
        "pred_answer": r.get("pred_answer"), "em": r.get("em"), "f1": r.get("f1"),
        "refused": r.get("refused"), "skills_used": r.get("skills_used", []),
        "skill_details": r.get("selected_skill_details", []),
        "context_docs": r.get("context_docs"),
        "gold_docs": common.gold_doc_ids(qidx) if int(q.category) != 5 else [],
    }


def compare(method_a: str, method_b: str, A: Dict[int, Dict], B: Dict[int, Dict]) -> Dict:
    fixed, worsened, same_wrong = [], [], []
    for q in sorted(set(A) & set(B)):
        a, b = A[q], B[q]
        if a.get("em") == 0 and b.get("em") == 1:
            fixed.append(_case(b))
        elif a.get("em") == 1 and b.get("em") == 0:
            worsened.append(_case(b))
        elif a.get("em") == 0 and b.get("em") == 0:
            same_wrong.append(_case(b))
    return {"method_a": method_a, "method_b": method_b, "fixed": fixed,
            "worsened": worsened, "same_wrong": same_wrong}


def skill_usage_analysis(method: str, rows: Dict[int, Dict], baseline: Dict[int, Dict]) -> Dict:
    hits = helpful = harmful = neutral = 0
    per_skill = defaultdict(lambda: {"uses": 0, "helpful": 0, "harmful": 0, "neutral": 0})
    unmatched_wrong = []
    for q, r in rows.items():
        used = r.get("skills_used") or []
        if used:
            hits += 1
            b = baseline.get(q, {})
            if b.get("em") == 0 and r.get("em") == 1:
                helpful += 1
                cls = "helpful"
            elif b.get("em") == 1 and r.get("em") == 0:
                harmful += 1
                cls = "harmful"
            else:
                neutral += 1
                cls = "neutral"
            for sid in used:
                per_skill[sid]["uses"] += 1
                per_skill[sid][cls] += 1
        elif r.get("em") == 0:
            unmatched_wrong.append(_case(r))
    return {"questions_with_skill_hit": hits, "helpful": helpful, "harmful": harmful,
            "neutral": neutral, "per_skill": dict(per_skill),
            "wrong_without_skill_hit": unmatched_wrong}


def evidence_present_but_unused(rows: Dict[int, Dict]) -> List[Dict]:
    out = []
    for q, r in rows.items():
        if common.category_of(q) == 5:
            continue
        docs = set(r.get("context_docs") or [])
        gold = set(common.gold_doc_ids(q))
        if gold <= docs and r.get("em") == 0 and r.get("refused"):
            out.append(_case(r))
    return out


def cost_summary() -> Dict:
    rows = [x for p in sorted(common.LOGS.glob("call_costs*.jsonl")) for x in common.read_jsonl(p)]
    api = [x for p in sorted(common.LOGS.glob("hy4_calls*.jsonl")) for x in common.read_jsonl(p)]
    status = Counter(str(r.get("status", "?")) for r in api)
    retries = sum(1 for r in api if int(r.get("attempt", 0) or 0) > 0)
    api_info = {"api_rows": len(api), "status": dict(status), "retried_rows": retries,
                "success_rate": (status.get("ok", 0) / len(api)) if api else None}
    by_stage = defaultdict(lambda: {"calls": 0, "cache_hits": 0, "prompt_tokens": 0,
                                    "completion_tokens": 0, "latencies": []})
    all_lat = []
    for r in rows:
        st = r.get("stage", "?")
        d = by_stage[st]
        d["calls"] += 1
        d["cache_hits"] += int(bool(r.get("cached")))
        u = r.get("usage") or {}
        d["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        d["completion_tokens"] += int(u.get("completion_tokens") or 0)
        lat = r.get("latency_s")
        if lat is not None:
            d["latencies"].append(float(lat))
            all_lat.append(float(lat))
    out = {"total_calls": len(rows),
           "total_cache_hits": sum(int(bool(r.get("cached"))) for r in rows),
           "total_prompt_tokens": sum(int((r.get("usage") or {}).get("prompt_tokens") or 0)
                                      for r in rows),
           "total_completion_tokens": sum(int((r.get("usage") or {}).get("completion_tokens") or 0)
                                          for r in rows),
           "by_stage": {}}
    for st, d in sorted(by_stage.items()):
        lat = d.pop("latencies")
        d["latency_p50"] = float(np.percentile(lat, 50)) if lat else None
        d["latency_p95"] = float(np.percentile(lat, 95)) if lat else None
        out["by_stage"][st] = d
    out["latency_p50"] = float(np.percentile(all_lat, 50)) if all_lat else None
    out["latency_p95"] = float(np.percentile(all_lat, 95)) if all_lat else None
    out["api"] = api_info
    return out



def retrieval_always_apply() -> Dict:
    """Auxiliary: apply every active retrieval skill without the sufficiency gate.

    Local dense retrieval only; no LLM calls.  This separates the value of the
    learned query templates from the gate that decides when to use them.
    """
    from . import retrieval_evolve as R
    from . import skills as S
    path = common.OUT / "retrieval_skill_memory.jsonl"
    if not path.exists():
        return {}
    lib = [json.loads(l) for l in open(path, encoding="utf-8")]
    lib = [s for s in lib if s.get("status") == "active"]
    if not lib:
        return {"n_skills": 0}
    rows = []
    for q in common.qidx_list("evaluation"):
        if common.category_of(q) == 5:
            continue
        base = common.base_ranked(q)
        sel = S.select_skills(lib, common.qa_map()[q].question, common.category_of(q), k=2)
        lists = [base]
        for s in sel:
            for qq in R.instantiate_templates(s, common.qa_map()[q].question)[:2]:
                lists.append(R.dense().retrieve(qq, 10))
        merged = R.merge_rrf(lists, top=20) if len(lists) > 1 else base
        b = common.evidence_metrics(q, base)
        m = common.evidence_metrics(q, merged)
        rows.append({"qidx": q, "skills": [s["skill_id"] for s in sel],
                     "base_full": b.get("evidence_full@4096"),
                     "applied_full": m.get("evidence_full@4096"),
                     "base_r5": b.get("recall@5"), "applied_r5": m.get("recall@5")})
    return {
        "n_skills": len(lib), "n": len(rows),
        "base_recall@5": common.mean(r["base_r5"] for r in rows),
        "applied_recall@5": common.mean(r["applied_r5"] for r in rows),
        "base_evidence_full": common.mean(r["base_full"] for r in rows),
        "applied_evidence_full": common.mean(r["applied_full"] for r in rows),
        "recovered": [r["qidx"] for r in rows if r["base_full"] == 0 and r["applied_full"] == 1],
        "lost": [r["qidx"] for r in rows if r["base_full"] == 1 and r["applied_full"] != 1],
        "rows": rows,
    }



def merge_shards() -> None:
    """Merge per-process summary/prediction shards written by s3/s4/s5."""
    art = common.OUT / "artifacts"
    for base, key in (("joint_summary", "summaries"), ("react_summary", None),
                      ("repeat_summary", None)):
        files = sorted(art.glob(f"{base}*.json"))
        if len(files) <= 1:
            continue
        merged = {}
        methods = {}
        for f in files:
            d = common.read_json(f)
            if key and isinstance(d.get(key), dict):
                merged.update(d[key])
                methods.update(d.get("methods") or {})
            else:
                merged.update({k: v for k, v in d.items()
                               if isinstance(v, dict) and k not in ("methods",)})
        out = {key: merged} if key else merged
        if methods:
            out["methods"] = methods
        common.write_json(art / f"{base}.json", out)
    for base, keys in (("joint_predictions", ("method", "qidx")),
                       ("react_predictions", ("method", "qidx")),
                       ("repeat_predictions", ("method", "qidx", "run"))):
        files = sorted(art.glob(f"{base}*.jsonl"))
        if len(files) <= 1:
            continue
        rows = {}
        for f in files:
            for r in common.read_jsonl(f):
                rows[tuple(r.get(k) for k in keys)] = r
        common.write_jsonl(art / f"{base}.jsonl", list(rows.values()))


def main():
    merge_shards()
    joint = load_joint_eval()
    react = load_react_eval()
    analysis = {}
    for a, b in (("E0", "E1"), ("E0", "E2"), ("E0", "E6"), ("E2", "E6"), ("E0", "EB")):
        if a in joint and b in joint:
            analysis[f"{a}_vs_{b}"] = compare(a, b, joint[a], joint[b])
    if "E2" in joint:
        analysis["E2_skill_usage"] = skill_usage_analysis("E2", joint["E2"], joint["E0"])
        analysis["E6_skill_usage"] = skill_usage_analysis("E6", joint["E6"], joint["E0"])
        analysis["E2_evidence_present_but_unused"] = evidence_present_but_unused(joint["E2"])
        analysis["E6_evidence_present_but_unused"] = evidence_present_but_unused(joint["E6"])
    # adversarial flips
    adv = []
    for q in sorted(joint.get("E0", {})):
        if common.category_of(q) != 5:
            continue
        a = joint["E0"][q].get("refused")
        for m in ("E1", "E2", "E6", "EB"):
            if m in joint and q in joint[m]:
                b = joint[m][q].get("refused")
                if a != b:
                    adv.append({"qidx": q, "method": m, "E0_refused": a, "method_refused": b,
                                "question": common.qa_map()[q].question})
    analysis["adversarial_refusal_flips"] = adv
    analysis["retrieval_always_apply"] = retrieval_always_apply()
    # repeat instability
    rep_path = common.OUT / "artifacts" / "repeat_summary.json"
    if rep_path.exists():
        rep = common.read_json(rep_path)
        analysis["repeat_unstable"] = {
            m: {"em_unstable_qids": s["em_unstable_qids"],
                "wording_unstable_qids": s["wording_unstable_qids"]}
            for m, s in rep.items()}
    common.write_json(common.OUT / "artifacts" / "error_cases.json", analysis)
    costs = cost_summary()
    common.write_json(common.OUT / "artifacts" / "cost_summary.json", costs)

    # consolidated predictions for all main methods on the eval split
    preds = []
    for m, rows in joint.items():
        preds.extend(dict(r, method=m, family="E-matrix") for r in rows.values())
    for m, rows in react.items():
        preds.extend(dict(r, method=m, family="ReAct") for r in rows.values())
    common.write_jsonl(common.OUT / "qa_predictions.jsonl", preds)

    # ---- error_analysis.md
    lines = ["# Error Analysis (evaluation split)", ""]
    for key in ("E0_vs_E1", "E0_vs_E2", "E0_vs_E6", "E2_vs_E6"):
        if key not in analysis:
            continue
        c = analysis[key]
        lines += [f"## {c['method_a']} -> {c['method_b']}", "",
                  f"Fixed: **{len(c['fixed'])}**; worsened: **{len(c['worsened'])}**; "
                  f"still wrong: {len(c['same_wrong'])}.", ""]
        if c["fixed"]:
            lines.append("### Fixed")
            lines.append("")
            for x in c["fixed"]:
                lines.append(f"- **q{x['qidx']}** ({x['category_name']}) {x['question'][:120]}")
                lines.append(f"  - gold: `{str(x['gold'])[:160]}`")
                lines.append(f"  - {c['method_b']}: `{str(x['pred_answer'])[:160]}` "
                             f"(skills: {', '.join(x['skills_used']) or 'none'})")
            lines.append("")
        if c["worsened"]:
            lines.append("### Worsened")
            lines.append("")
            for x in c["worsened"]:
                lines.append(f"- **q{x['qidx']}** ({x['category_name']}) {x['question'][:120]}")
                lines.append(f"  - gold: `{str(x['gold'])[:160]}`")
                lines.append(f"  - {c['method_b']}: `{str(x['pred_answer'])[:160]}` "
                             f"(skills: {', '.join(x['skills_used']) or 'none'})")
            lines.append("")
    if "E2_skill_usage" in analysis:
        u = analysis["E2_skill_usage"]
        lines += ["## Skill hit analysis (E2)", "",
                  f"- Questions where at least one skill was retrieved: **{u['questions_with_skill_hit']}**",
                  f"- Helpful (E0 wrong -> E2 right): **{u['helpful']}**",
                  f"- Harmful (E0 right -> E2 wrong): **{u['harmful']}**",
                  f"- Neutral: **{u['neutral']}**",
                  f"- Wrong without any skill hit: **{len(u['wrong_without_skill_hit'])}**", "",
                  "| skill | uses | helpful | harmful | neutral |", "|---|---:|---:|---:|---:|"]
        for sid, d in sorted(u["per_skill"].items(), key=lambda x: -x[1]["uses"]):
            lines.append(f"| {sid} | {d['uses']} | {d['helpful']} | {d['harmful']} | {d['neutral']} |")
        lines.append("")
    if "E6_skill_usage" in analysis:
        u = analysis["E6_skill_usage"]
        lines += ["## Skill hit analysis (E6, full system)", "",
                  f"Hit/helpful/harmful/neutral/no-hit-wrong: {u['questions_with_skill_hit']} / "
                  f"{u['helpful']} / {u['harmful']} / {u['neutral']} / "
                  f"{len(u['wrong_without_skill_hit'])}", ""]
    lines += ["## Evidence present but the Reader still refused", ""]
    for m in ("E2", "E6"):
        key = f"{m}_evidence_present_but_unused"
        if key not in analysis:
            continue
        lines.append(f"- {m}: **{len(analysis[key])}** questions")
        for x in analysis[key]:
            lines.append(f"  - q{x['qidx']} ({x['category_name']}): {x['question'][:110]} "
                         f"-> `{str(x['pred_answer'])[:80]}`")
    lines.append("")
    if analysis.get("adversarial_refusal_flips"):
        lines += ["## Adversarial refusal flips", "", "| qidx | method | E0 refused | method refused | question |",
                  "|---:|---|---:|---:|---|"]
        for x in analysis["adversarial_refusal_flips"]:
            lines.append(f"| {x['qidx']} | {x['method']} | {x['E0_refused']} | "
                         f"{x['method_refused']} | {x['question'][:100]} |")
        lines.append("")
    if "repeat_unstable" in analysis:
        lines += ["## Repeated-run instability", ""]
        for m, d in analysis["repeat_unstable"].items():
            lines.append(f"- {m}: EM unstable on qids {d['em_unstable_qids']}; "
                         f"wording unstable on {len(d['wording_unstable_qids'])} questions")
        lines.append("")
    lines += ["## Interpretation", "",
              "- A skill hit does not mean the skill caused the change; compare the same question "
              "under E0.  The per-skill table separates helpful from harmful hits.",
              "- Skills that are retrieved but never change an answer are neutral, not evidence of "
              "generalisation.",
              "- Repeated-run instability bounds how much of any single-question flip can be "
              "attributed to the skill layer.", ""]
    (common.OUT / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- cost_latency.md
    lines = ["# Cost and Latency", "", "## Total endpoint usage (this study, cached + uncached)", "",
             f"- Calls logged: **{costs['total_calls']}** (cache hits: {costs['total_cache_hits']})",
             f"- Prompt tokens: **{costs['total_prompt_tokens']:,}**",
             f"- Completion tokens: **{costs['total_completion_tokens']:,}**",
             f"- Call latency P50/P95: **{costs['latency_p50']:.2f}s / {costs['latency_p95']:.2f}s**",
             "", f"- Endpoint success rate (all logged attempts): "
             f"{_fmt((costs.get('api') or {}).get('success_rate'))}; "
             f"rows with at least one retry: {(costs.get('api') or {}).get('retried_rows', '?')}.",
             "## By stage", "",
             "| stage | calls | cache hits | prompt tok | completion tok | latency P50 | latency P95 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for st, d in costs["by_stage"].items():
        lines.append(f"| {st} | {d['calls']} | {d['cache_hits']} | {d['prompt_tokens']:,} | "
                     f"{d['completion_tokens']:,} | {d['latency_p50']:.2f} | {d['latency_p95']:.2f} |")
    js_path = common.OUT / "artifacts" / "joint_summary.json"
    if js_path.exists():
        js = common.read_json(js_path)["summaries"]
        lines += ["", "## Per-question cost on the evaluation split", "",
                  "| Exp | LLM calls | prompt tokens | completion tokens | reader latency s |",
                  "|---|---:|---:|---:|---:|"]
        for m in ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "EB"):
            s = js.get(f"evaluation|{m}")
            if s:
                lines.append(f"| {m} | {s['mean_calls']:.2f} | {s['mean_prompt_tokens']:.0f} | "
                             f"{s['mean_completion_tokens']:.1f} | {s['mean_latency_s']:.2f} |")
    rs_path = common.OUT / "artifacts" / "react_summary.json"
    if rs_path.exists():
        rs = common.read_json(rs_path)
        lines += ["", "## ReAct variants", "",
                  "| Variant | LLM calls | prompt tokens | completion tokens |",
                  "|---|---:|---:|---:|"]
        for m, s in rs.items():
            lines.append(f"| {m} | {s['mean_calls']:.2f} | {s['mean_prompt_tokens']:.0f} | "
                         f"{s['mean_completion_tokens']:.1f} |")
    lines += ["", "Costs are measured from the HY4 response usage fields.  Cached calls are "
              "content-addressed repeats and cost no endpoint quota; adaptation reflection and "
              "verification are one-off training costs, while the E-matrix columns are per "
              "evaluation question.", ""]
    (common.OUT / "cost_latency.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote error_analysis.md, cost_latency.md, qa_predictions.jsonl")


if __name__ == "__main__":
    main()
