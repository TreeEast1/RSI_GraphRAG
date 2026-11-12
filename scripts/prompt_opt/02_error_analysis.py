#!/usr/bin/env python
"""Phase 0.3 — per-question error attribution for conv-30 G_raw (Task B baseline).

Reads only frozen artifacts (no LLM, no retrieval). For every one of the 105
questions it records the actual Reader context, evidence availability, the
answer, both scoring regimes, and a rule-based error label with a rationale.

Labels
------
A retrieval error        : a required gold session is absent from the reader context
B incomplete evidence    : only part of the gold evidence is in the context
C temporal reasoning     : evidence present, date/duration/order still wrong
D multi-fact integration : evidence present, only part of a multi-part answer given
E answer expression      : semantically close but too short / wrong surface form
F over-refusal           : evidence sufficient but model answers "No answer"
G unsupported answer     : model asserts something the context cannot support
H evaluator artefact     : answer is semantically right but the scorer disagrees

Outputs:
  audit/error_analysis_conv30.jsonl   (one row per question)
  audit/error_analysis_conv30.csv
  audit/graw_contexts/qNNN.txt        (exact reader context)
  audit/error_analysis_summary.md
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import PATHS  # noqa: E402
from src.dataset.locomo import load_locomo, num_tokens  # noqa: E402
from src.evaluation.metrics import select_by_budget  # noqa: E402
from src.prompt_optimization.official_metrics import (f1_score as off_f1,  # noqa: E402
                                                      normalize_answer,
                                                      official_qa_score)
from src.retrieval.reader import READER_TEMPLATE, format_context  # noqa: E402

from nltk.stem import PorterStemmer  # noqa: E402

_PS = PorterStemmer()

OUT = ROOT / "outputs" / "prompt_optimization_conv30" / "audit"
REFUSAL = ("no answer", "cannot", "not mentioned", "unknown", "unanswerable", "not stated")
STOP = {"the", "and", "for", "with", "from", "that", "this", "his", "her", "their", "was",
        "were", "has", "have", "had", "did", "does", "you", "your", "she", "he", "they",
        "are", "but", "not", "all", "own", "about", "into", "when", "what", "which", "how",
        "why", "who", "where", "does", "s"}


def is_refusal(ans) -> bool:
    low = str(ans or "").lower()
    return any(k in low for k in REFUSAL)


def content_tokens(text) -> set:
    toks = normalize_answer(text).split()
    return {t for t in toks if len(t) > 1 and t not in STOP}


def gold_item_coverage(pred, gold):
    items = [x.strip() for x in str(gold or "").split(",") if x.strip()]
    if not items:
        return 1.0, []
    pred_parts = [x.strip() for x in str(pred or "").split(",") if x.strip()] or [str(pred or "")]
    hits = []
    for it in items:
        best = max(off_f1(p, it) for p in pred_parts)
        hits.append((it, round(best, 3)))
    covered = sum(1 for _, s in hits if s >= 0.5)
    return covered / len(items), hits


def date_in_context(gold, ctx_norm, ctx_tokens):
    raw = str(gold or "")
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s*(\d{4})", raw)
    if m:
        day, month, year = m.group(1), m.group(2).lower(), m.group(3)
        return (month in ctx_tokens) and (day in ctx_tokens or day.lstrip("0") in ctx_tokens)
    m = re.search(r"([A-Za-z]+),?\s*(\d{4})", raw)
    if m:
        return m.group(1).lower() in ctx_tokens
    return normalize_answer(raw) in ctx_norm


def classify(r, feat):
    """Return (main, secondary list, rationale)."""
    cat = int(r["category"])
    if cat == 5:
        if is_refusal(r.get("pred_answer")):
            return None, [], "correct refusal on adversarial question"
        return "G", [], "answered an adversarial question instead of refusing"
    if r.get("em") == 1.0:
        return None, [], "correct"
    sec = []
    if feat["semantically_equivalent"] and not r.get("em") == 1.0:
        sec.append("H")
    elif feat["official_f1"] - (r.get("f1") or 0.0) >= 0.25:
        sec.append("H")

    if is_refusal(r.get("pred_answer")):
        if feat["evidence_full"] == 1.0:
            sec2 = list(sec)
            if cat == 2:
                sec2.append("C")  # date questions need date arithmetic, not just lookup
            return "F", sec2, "gold evidence in context but model refused"
        if feat["evidence_partial"] > 0:
            return "B", sec + ["F"], "only part of gold evidence in context; model refused"
        return "A", sec, "no gold evidence in context; model refused"

    if feat["evidence_full"] == 0.0:
        # the gold session is missing; if the answer content is still present, note it
        if feat["gold_recall_in_context"] >= 0.75:
            return "A", sec + ["B"], "gold session missing but answer content largely present"
        return "A", sec + (["B"] if feat["evidence_partial"] > 0 else []), "gold evidence missing from context"

    # evidence session present
    if cat == 2:
        if not feat["date_in_context"]:
            return "A", sec + ["C"], "gold session present but the exact date is not in context"
        return "C", sec, "date evidence present but temporal reasoning failed"
    if cat == 1:
        if feat["item_coverage"] < 1.0:
            return "D", sec, "multi-hop answer missing part(s) of the gold facts"
        if (r.get("f1") or 0) >= 0.5 or feat["stem_overlap"] >= 1:
            return "E", sec, "semantically related but surface form / completeness differs"
        return "G", sec, "unsupported or wrong multi-hop answer"
    # categories 3/4
    if feat["item_coverage"] < 1.0 and len(feat["items"]) > 1:
        return "D", sec, "list answer incomplete"
    if (r.get("f1") or 0) >= 0.5 or feat["official_f1"] >= 0.4 or feat["stem_overlap"] >= 1:
        return "E", sec, "semantically close but too short / different wording"
    if feat["gold_recall_in_context"] < 0.5:
        return "A", sec, "gold evidence session present but answer content largely absent"
    return "G", sec, "answer not supported by the provided context"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx_dir = OUT / "graw_contexts"
    ctx_dir.mkdir(exist_ok=True)
    conv = {c.sample_id: c for c in load_locomo(ROOT / "data" / "raw" / "locomo" / "locomo10.json")}["conv-30"]
    docs = {s.document_id: {"document_id": s.document_id, "session_id": s.session_id,
                            "header": f"Session {s.session_id} | {s.date_time_raw}", "content": s.text}
            for s in conv.sessions}
    rows = [json.loads(l) for l in open(PATHS["qa"] / "baselines_hipporag_conv-30.jsonl", encoding="utf-8")]

    out_rows = []
    for r in rows:
        ctx_docs = select_by_budget([docs[d] for d in r["ranked_docs"] if d in docs], 4096, num_tokens)
        context = format_context(ctx_docs)
        ctx_norm = normalize_answer(context)
        ctx_tokens = set(ctx_norm.split())
        qidx = int(r["qid"].split("q")[-1])
        (ctx_dir / f"q{qidx:03d}.txt").write_text(context, encoding="utf-8")

        gold = r["gold_answer"]
        cov, item_hits = gold_item_coverage(r.get("pred_answer"), gold)
        g_tokens = content_tokens(gold)
        p_tokens = content_tokens(r.get("pred_answer"))
        ps = _PS
        stem_overlap = len({ps.stem(t) for t in g_tokens} & {ps.stem(t) for t in p_tokens})
        g_stems = {ps.stem(t) for t in g_tokens}
        p_stems = {ps.stem(t) for t in p_tokens}
        equivalent = bool(g_stems and p_stems and (
            g_stems <= p_stems or p_stems <= g_stems))
        gold_recall = (len(g_tokens & ctx_tokens) / len(g_tokens)) if g_tokens else 0.0
        off = official_qa_score(r.get("pred_answer"), gold, int(r["category"]))
        feat = {
            "gold_sessions": r["gold_sessions"], "gold_turns": r["gold_turns"],
            "required_sessions": len(r["gold_sessions"]),
            "selected_docs": [d["document_id"] for d in ctx_docs],
            "missing_gold_sessions": sorted(set(r["gold_doc_ids"]) - {d["document_id"] for d in ctx_docs}),
            "evidence_full": r["budget"]["4096"]["evidence_full"],
            "evidence_partial": r["budget"]["4096"]["evidence_partial"],
            "context_tokens": sum(num_tokens(d["content"]) for d in ctx_docs),
            "gold_recall_in_context": round(gold_recall, 3),
            "stem_overlap": stem_overlap,
            "semantically_equivalent": equivalent,
            "gold_content_tokens": sorted(g_tokens),
            "item_coverage": round(cov, 3), "item_hits": item_hits,
            "items": [x.strip() for x in str(gold or "").split(",") if x.strip()],
            "date_in_context": date_in_context(gold, ctx_norm, ctx_tokens),
            "refused": is_refusal(r.get("pred_answer")),
            "answer_words": len(str(r.get("pred_answer") or "").split()),
            "completion_tokens": (r.get("reader_usage") or {}).get("completion_tokens"),
            "truncated_proxy": bool(((r.get("reader_usage") or {}).get("completion_tokens") or 0) >= 250),
            "official_f1": off["official_f1"], "official_em": off["official_em"],
            "official_refused_strict": off.get("official_refused_strict"),
            "official_refused_adapted": off.get("official_refused_adapted"),
        }
        main_label, sec, why = classify(r, feat)
        rec = {
            "qid": r["qid"], "qidx": qidx, "category": r["category"],
            "category_name": r["category_name"], "question": r["question"],
            "gold_answer": gold, "gold_sessions": r["gold_sessions"], "gold_turns": r["gold_turns"],
            "top5_sessions": [d.split("::")[1] for d in r["ranked_docs"][:5]],
            "ranked_docs": r["ranked_docs"], "context_docs": feat["selected_docs"],
            "context_file": f"graw_contexts/q{qidx:03d}.txt",
            "pred_answer": r.get("pred_answer"), "project_em": r["em"], "project_f1": r["f1"],
            "official_f1": feat["official_f1"], "official_em": feat["official_em"],
            "answer_correct": r["em"] == 1.0,
            "correct_refusal": (int(r["category"]) == 5 and feat["refused"]),
            "truncated": feat["truncated_proxy"],
            "evidence_missing": feat["evidence_full"] == 0.0,
            "main_error": main_label, "secondary_errors": sorted(set(sec)),
            "fixable_by_qa_prompt": main_label in {"C", "D", "E", "F", "H"},
            "rationale": why,
            "semantically_equivalent": feat["semantically_equivalent"],
            **{k: feat[k] for k in ("missing_gold_sessions", "evidence_full", "evidence_partial",
                                    "context_tokens", "gold_recall_in_context", "item_coverage",
                                    "date_in_context", "refused", "answer_words", "completion_tokens")},
        }
        out_rows.append(rec)

    with open(OUT / "error_analysis_conv30.jsonl", "w", encoding="utf-8") as f:
        for rec in out_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fields = ["qidx", "qid", "category", "category_name", "question", "gold_answer", "top5_sessions",
              "context_docs", "pred_answer", "project_em", "project_f1", "official_f1", "correct_refusal",
              "truncated", "evidence_missing", "evidence_full", "evidence_partial", "gold_recall_in_context",
              "item_coverage", "date_in_context", "main_error", "secondary_errors", "fixable_by_qa_prompt",
              "rationale"]
    with open(OUT / "error_analysis_conv30.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    ans = [r for r in out_rows if r["category"] != 5]
    adv = [r for r in out_rows if r["category"] == 5]
    main_counter = Counter(r["main_error"] for r in ans if r["main_error"])
    sec_counter = Counter(s for r in ans for s in r["secondary_errors"])
    lines = ["# conv-30 G_raw 错误归因（105 题，规则化 + 人工核对）", "",
             f"- 可回答 {len(ans)} 题：正确 {sum(r['answer_correct'] for r in ans)} 题，错误 {sum(not r['answer_correct'] for r in ans)} 题。",
             f"- 对抗 {len(adv)} 题：正确拒答 {sum(r['correct_refusal'] for r in adv)} 题，幻觉 {sum(not r['correct_refusal'] for r in adv)} 题。",
             f"- 证据完整进入 4096 context：{sum(r['evidence_full'] == 1.0 for r in ans)}/{len(ans)}；证据完全缺失：{sum(r['evidence_full'] == 0.0 for r in ans)}/{len(ans)}。",
             f"- 答案截断（completion>=250）：{sum(r['truncated'] for r in out_rows)} 题。",
             f"- 语义等价但被 EM/项目 F1 判错（H 次要标签）：{sum('H' in r['secondary_errors'] for r in ans)} 题；"
             f"其中严格子集/超集关系 {sum(r['semantically_equivalent'] and not r['answer_correct'] for r in ans)} 题。",
             "", "## 主错误分布（仅可回答且错误）", "",
             "| 标签 | 含义 | 题数 | 占比(错误题) |", "|---|---|---:|---:|"]
    names = {"A": "检索错误", "B": "证据不完整", "C": "时间推理错误", "D": "多事实整合错误",
             "E": "答案表达问题", "F": "过度拒答", "G": "错误推断/幻觉", "H": "评测器问题"}
    n_err = sum(not r["answer_correct"] for r in ans)
    for k, v in main_counter.most_common():
        lines.append(f"| {k} | {names[k]} | {v} | {v/n_err:.1%} |")
    lines += ["", "## 次要标签分布", "", "| 标签 | 题数 |", "|---|---:|"]
    for k, v in sec_counter.most_common():
        lines.append(f"| {k} {names[k]} | {v} |")
    lines += ["", "## 主错误 × 证据是否在 context 内", "",
              "| 主错误 | 证据完整 | 部分证据 | 无证据 |", "|---|---:|---:|---:|"]
    for lab in "ABCDEFGH":
        sub = [r for r in ans if r["main_error"] == lab]
        if not sub:
            continue
        lines.append(f"| {lab} | {sum(r['evidence_full']==1.0 for r in sub)} | "
                     f"{sum(0<r['evidence_partial']<1 for r in sub)} | {sum(r['evidence_full']==0.0 for r in sub)} |")
    lines += ["", "## 按类别", "", "| 类别 | n | 正确 | 主错误分布 |", "|---|---:|---:|---|"]
    for c in (1, 2, 3, 4):
        sub = [r for r in ans if r["category"] == c]
        if not sub:
            continue
        cc = Counter(r["main_error"] for r in sub if r["main_error"])
        lines.append(f"| cat{c} {sub[0]['category_name']} | {len(sub)} | {sum(r['answer_correct'] for r in sub)} | "
                     + ", ".join(f"{k}:{v}" for k, v in cc.most_common()) + " |")
    (OUT / "error_analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
