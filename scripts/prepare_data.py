#!/usr/bin/env python
"""Parse LoCoMo into the project's canonical form and write an audit report.

Outputs
-------
data/processed/{sample_id}/documents.jsonl   session documents (indexing order)
data/processed/{sample_id}/turns.jsonl       turn-level records for evidence mapping
data/processed/{sample_id}/qa.jsonl          QA items with evidence mapped to sessions/turns
data/splits/split.json                       dev/test split (fixed seed)
outputs/reports/DATASET_AUDIT.md             dataset statistics and quality notes
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.locomo import CATEGORY_NAMES, load_locomo, make_splits  # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw" / "locomo" / "locomo10.json"))
    ap.add_argument("--seed", type=int, default=20240501)
    ap.add_argument("--n-dev", type=int, default=3)
    args = ap.parse_args()

    convs = load_locomo(args.raw)
    split = make_splits(convs, n_dev=args.n_dev, seed=args.seed)

    proc = ROOT / "data" / "processed"
    for conv in convs:
        out = proc / conv.sample_id
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "documents.jsonl", "w", encoding="utf-8") as f:
            for d in conv.documents():
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        with open(out / "turns.jsonl", "w", encoding="utf-8") as f:
            for s in conv.sessions:
                for t in s.turns:
                    f.write(json.dumps({
                        "document_id": s.document_id,
                        "session_id": s.session_id,
                        "dia_id": t.dia_id,
                        "speaker": t.speaker,
                        "text": t.text,
                    }, ensure_ascii=False) + "\n")
        with open(out / "qa.jsonl", "w", encoding="utf-8") as f:
            for q in conv.qa:
                f.write(json.dumps({
                    "qid": q.qid,
                    "conversation_id": q.conversation_id,
                    "question": q.question,
                    "answer": q.answer,
                    "category": q.category,
                    "category_name": CATEGORY_NAMES.get(q.category, str(q.category)),
                    "evidence": q.evidence,
                    "evidence_sessions": q.evidence_sessions,
                    "evidence_turns": q.evidence_turns,
                    "unmapped_evidence": q.unmapped_evidence,
                    "adversarial_answer": q.adversarial_answer,
                }, ensure_ascii=False) + "\n")

    splits_dir = ROOT / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- audit
    lines = ["# LoCoMo Dataset Audit", ""]
    lines.append(f"Source: `{args.raw}`")
    lines.append(f"Split seed: {split['seed']} — dev={split['dev']}, test={split['test']}")
    lines.append("")

    total_sessions = total_turns = total_qa = 0
    tok_by_conv = {}
    cat_all = Counter()
    cat_dev = Counter()
    cat_test = Counter()
    ev_counts = Counter()
    missing_ev = 0
    unmapped = 0
    multi_sess_qa = 0
    total_ev = 0

    rows = []
    for conv in convs:
        toks = [s.n_tokens for s in conv.sessions]
        tok_by_conv[conv.sample_id] = toks
        total_sessions += len(conv.sessions)
        total_turns += sum(len(s.turns) for s in conv.sessions)
        total_qa += len(conv.qa)
        where = "dev" if conv.sample_id in split["dev"] else "test"
        for q in conv.qa:
            cat_all[q.category] += 1
            (cat_dev if where == "dev" else cat_test)[q.category] += 1
            n_ev = len(q.evidence)
            ev_counts[n_ev] += 1
            total_ev += n_ev
            if n_ev == 0:
                missing_ev += 1
            unmapped += len(q.unmapped_evidence)
            if len(q.evidence_sessions) > 1:
                multi_sess_qa += 1
        rows.append((conv.sample_id, where, len(conv.sessions), sum(len(s.turns) for s in conv.sessions),
                     len(conv.qa), round(statistics.mean(toks), 1), max(toks), min(toks)))

    lines += ["## Conversation summary", "",
              "| conversation | split | sessions | turns | QA | avg session tokens | max | min |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(*r))

    all_toks = [t for v in tok_by_conv.values() for t in v]
    lines += ["", "## Totals", "",
              f"- Conversations: {len(convs)}",
              f"- Sessions (documents): {total_sessions}",
              f"- Turns: {total_turns}",
              f"- QA items: {total_qa}",
              f"- Session tokens: mean={statistics.mean(all_toks):.1f}, median={statistics.median(all_toks):.1f}, "
              f"max={max(all_toks)}, total={sum(all_toks)} (cl100k proxy)", ""]

    lines += ["## QA categories", "",
              "| category | name | all | dev | test |", "|---|---|---|---|---|"]
    for cat in sorted(cat_all):
        lines.append(f"| {cat} | {CATEGORY_NAMES.get(cat, '?')} | {cat_all[cat]} | {cat_dev[cat]} | {cat_test[cat]} |")

    lines += ["", "## Evidence", "",
              f"- QA rows with evidence: {total_qa - missing_ev} / {total_qa}",
              f"- QA rows without evidence: {missing_ev}",
              f"- Unmapped evidence dia_ids: {unmapped}",
              f"- QA spanning >1 session: {multi_sess_qa} ({multi_sess_qa/total_qa:.1%})",
              f"- Mean evidence items per QA: {total_ev/total_qa:.2f}", ""]
    lines += ["| #evidence | QA count |", "|---|---|"]
    for k in sorted(ev_counts):
        lines.append(f"| {k} | {ev_counts[k]} |")

    lines += ["", "## Notes", "",
              "- Category 5 (adversarial) rows have no gold `answer`; they carry `adversarial_answer` "
              "and must be scored as refusal/robustness, not EM/F1.",
              "- `session_summary`, `observation` and `event_summary` fields are withheld from all "
              "indexing paths to avoid leakage; they may only be used for analysis.",
              "- Token counts use cl100k_base as a budget proxy; actual HY4 tokenisation may differ.", ""]

    out = ROOT / "outputs" / "reports" / "DATASET_AUDIT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print(f"sessions={total_sessions} turns={total_turns} qa={total_qa} unmapped_ev={unmapped} no_ev={missing_ev}")


if __name__ == "__main__":
    main()
