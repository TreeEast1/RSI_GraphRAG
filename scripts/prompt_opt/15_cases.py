#!/usr/bin/env python
"""Build the case-study file used by FINAL_REPORT.md section 6."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import PATHS  # noqa: E402
from src.prompt_optimization.runner import evaluate  # noqa: E402

OUT = ROOT / "outputs" / "prompt_optimization_conv30"
BASE = [json.loads(l) for l in open(PATHS["qa"] / "baselines_hipporag_conv-30.jsonl", encoding="utf-8")]


def load_records(name):
    p = OUT / "comprehension" / "retrieval" / name / "records.jsonl"
    return {r["qid"]: r for r in [json.loads(l) for l in open(p, encoding="utf-8")]}


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def load_answers(pid):
    rows = {}
    for sub in ("pilot", "final"):
        p = OUT / "reader" / sub / f"results_{pid}.jsonl"
        if p.exists():
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                rows[r["qidx"]] = r.get("pred_answer")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="cmpo_compp4")
    ap.add_argument("--pid", default="P7")
    ap.add_argument("--qids", default="16,17,24,29,31,33,54,4,7,13,19,30,38,39,71,72,78,85,63,67")
    args = ap.parse_args()

    c2 = load_records("C2")
    graw = load_records("G_raw")
    var = load_records(args.variant)
    p0 = load_answers("P0") or {i: r["pred_answer"] for i, r in enumerate(BASE)}
    p7 = load_answers(args.pid)
    joint = {}
    for tag in ("P0", "P7"):
        jp = OUT / "joint" / f"{args.variant}_{tag}.jsonl"
        if jp.exists():
            joint[tag] = {r["qidx"]: r.get("pred_answer") for r in load_rows(jp)}
    import re
    m = re.search(r"compp(\d+)", args.variant)
    ext = OUT / "comprehension" / "extractions" / f"COMP-P{m.group(1)}.json" if m else None
    rels = json.loads(ext.read_text())["relations"] if ext and ext.exists() else []

    out = []
    for q in [int(x) for x in args.qids.split(",")]:
        r = BASE[q]
        qid = r["qid"]
        def pack(rec):
            if qid not in rec:
                return None
            x = rec[qid]
            return {"top5": [d.split("::")[1] for d in x["ranked_docs"][:5]],
                    "gold_rank": [x["ranked_docs"].index(g) + 1 for g in x["gold_doc_ids"]
                                  if g in x["ranked_docs"]],
                    "evidence_full@4096": x["budget"]["4096"]["evidence_full"],
                    "selected": [d.split("::")[1] for d in x["budget"]["4096"]["selected_ids"]],
                    "fallback_dense": x.get("diag", {}).get("fallback_dense"),
                    "n_facts_after": x.get("diag", {}).get("n_facts_after"),
                    "top_facts": x.get("diag", {}).get("top_facts", [])[:3]}
        gold_sessions = r["gold_sessions"]
        touching = [x for x in rels if set(x["source_sessions"]) & set(gold_sessions)]
        item = {
            "qid": qid, "qidx": q, "category": r["category"], "question": r["question"],
            "gold_answer": r["gold_answer"], "gold_sessions": gold_sessions, "gold_turns": r["gold_turns"],
            "G_raw": pack(graw), "C2": pack(c2), args.variant: pack(var),
            "P0_answer": p0.get(q), "P7_answer": p7.get(q),
            "E2_answer": joint.get("P0", {}).get(q),
            "E3_answer": joint.get("P7", {}).get(q),
            "P0_eval": evaluate(p0.get(q), r["gold_answer"], int(r["category"])),
            "P7_eval": evaluate(p7.get(q), r["gold_answer"], int(r["category"])),
            "relations_touching_gold_sessions": [
                {"subject": x["subject"], "predicate": x["predicate"], "object": x["object"],
                 "type": x["relation_type"], "sessions": x["source_sessions"],
                 "confidence": x["confidence"]} for x in touching],
        }
        out.append(item)

    (OUT / "cases").mkdir(exist_ok=True)
    (OUT / "cases" / "cases.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# Case studies", ""]
    for it in out:
        L += [f"## q{it['qidx']} (cat{it['category']})", "",
              f"**Q**: {it['question']}  ",
              f"**Gold**: {it['gold_answer']}  ",
              f"**Gold sessions**: {it['gold_sessions']} turns {it['gold_turns']}", ""]
        L += ["| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |",
              "|---|---|---|---:|---|---:|"]
        for g in ("G_raw", "C2", args.variant):
            x = it[g]
            if x:
                L.append(f"| {g} | {', '.join(x['top5'])} | {x['gold_rank']} | "
                         f"{x['evidence_full@4096']} | {x['fallback_dense']} | {x['n_facts_after']} |")
        L += ["", f"**P0 on G_raw (E0)**: {it['P0_answer']!r} (EM {it['P0_eval']['em']:.0f}, F1 {it['P0_eval']['f1']:.2f})  ",
              f"**P7 on G_raw (E1)**: {it['P7_answer']!r} (EM {it['P7_eval']['em']:.0f}, F1 {it['P7_eval']['f1']:.2f})  ",
              f"**P0 on this graph (E2)**: {it.get('E2_answer')!r}  ",
              f"**P7 on this graph (E3)**: {it.get('E3_answer')!r}", ""]
        if it["relations_touching_gold_sessions"]:
            L.append("Relations touching gold sessions:")
            for x in it["relations_touching_gold_sessions"]:
                L.append(f"- `{x['subject']} --{x['predicate']}--> {x['object']}` "
                         f"[{x['type']}] sessions={x['sessions']} conf={x['confidence']}")
            L.append("")
    (OUT / "cases" / "cases.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} cases -> {OUT/'cases'/'cases.md'}")


if __name__ == "__main__":
    main()
