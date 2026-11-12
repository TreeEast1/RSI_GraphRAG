#!/usr/bin/env python
"""Parse QASPER validation split, map gold evidence to paragraphs, write the audit.

Outputs (all under outputs/qasper_readtime/):
  dataset/papers/{paper_id}/{paragraphs,qa}.jsonl + meta.json
  dataset/summary.json                  machine-readable statistics
  dataset/sample.json                   frozen 100-paper sample (seed in file)
  reports/dataset_audit.md              narrative audit
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.qasper.dataset import load_qasper, stable_paper_order, write_paper_jsonl  # noqa: E402

RAW = ROOT / "data" / "raw" / "qasper"
OUT = ROOT / "outputs" / "qasper_readtime"
SEED = 20261110
N_SAMPLE = 100


def pct(vals, q):
    if not vals:
        return float("nan")
    return float(statistics.quantiles(vals, n=100, method="inclusive")[q - 1])


def main() -> None:
    papers = load_qasper(RAW / "qasper-dev-v0.3.json")
    print(f"papers={len(papers)}")
    proc = OUT / "dataset" / "papers"
    for pid, paper in papers.items():
        write_paper_jsonl(paper, proc / pid)

    # ----------------------------------------------------------- paper stats
    paper_rows = []
    for pid, p in papers.items():
        paper_rows.append({
            "paper_id": pid,
            "title": p.title,
            "n_paragraphs": len(p.paragraphs),
            "n_tokens": p.n_tokens,
            "n_sections": len({x.section_index for x in p.paragraphs}),
            "n_figures_tables": len(p.figures_tables),
            "n_questions": len(p.qas),
            "paragraph_tokens_mean": round(statistics.mean([x.n_tokens for x in p.paragraphs]), 1) if p.paragraphs else 0,
            "paragraph_tokens_max": max([x.n_tokens for x in p.paragraphs], default=0),
        })
    pool_sizes = [r["n_paragraphs"] for r in paper_rows]
    paper_tokens = [r["n_tokens"] for r in paper_rows]

    # ----------------------------------------------------------- qa stats
    qa_all = [q for p in papers.values() for q in p.qas]
    match_counts = Counter()
    unmatched_records = []
    for p in papers.values():
        for q in p.qas:
            for a in q.annotations:
                for m in a.evidence_match:
                    match_counts[m["method"]] += 1
                    if m["method"] == "unmatched":
                        unmatched_records.append({
                            "qid": q.qid, "paper_id": p.paper_id,
                            "evidence": m["evidence"],
                        })
    n_evidence_items = sum(match_counts.values())
    non_text = sum(match_counts.get(k, 0) for k in ("float", "heading", "empty"))
    mapped = n_evidence_items - match_counts.get("unmatched", 0) - non_text
    match_rate = mapped / max(1, n_evidence_items - non_text)

    answerable = [q for q in qa_all if q.answerable]
    with_gold = [q for q in answerable if q.annotator_sets]
    # primary per-question evidence-set view: annotator alternatives
    single = [q for q in with_gold if q.n_gold_max == 1]
    multi = [q for q in with_gold if q.n_gold_max > 1]
    cross = [q for q in with_gold if q.cross_section]
    unans = [q for q in qa_all if not q.answerable]
    tf_only = [q for q in qa_all if q.table_figure_only]
    unmapped_q = [q for q in qa_all if q.has_unmapped_evidence]
    n_annotations = Counter(len(q.annotations) for q in qa_all)

    def dist(name, vals):
        return {"name": name, "n": len(vals), "mean": round(statistics.mean(vals), 2) if vals else None,
                "p10": round(pct(vals, 10), 1) if vals else None,
                "p50": round(pct(vals, 50), 1) if vals else None,
                "p90": round(pct(vals, 90), 1) if vals else None,
                "max": max(vals) if vals else None}

    pool_dist = dist("paragraphs_per_paper", pool_sizes)
    token_dist = dist("tokens_per_paper", paper_tokens)
    para_token_dist = dist("tokens_per_paragraph",
                           [x.n_tokens for p in papers.values() for x in p.paragraphs])
    gold_n_dist = dist("gold_paragraphs_per_question_per_annotator",
                       [len(s) for q in with_gold for s in q.annotator_sets])

    # ----------------------------------------------------------- frozen sample
    eligible = [pid for pid, p in papers.items() if any(q.answerable and q.annotator_sets for q in p.qas)]
    ordered = stable_paper_order(eligible, SEED)
    sample = sorted(ordered[:N_SAMPLE])
    sample_qas = [q for q in qa_all if q.paper_id in set(sample)]
    sample_answerable = [q for q in sample_qas if q.answerable]
    sample_with_gold = [q for q in sample_answerable if q.annotator_sets]
    split = {
        "seed": SEED,
        "rule": ("uniform random sample without replacement over validation papers that have at least "
                 "one answerable question with mapped gold evidence; ordering by sha1(f'{seed}:{paper_id}'), "
                 "take the first 100. Rule frozen before any retrieval/QA evaluation."),
        "n_eligible": len(eligible),
        "n_sample": len(sample),
        "sample": sample,
        "ordered_full": ordered,
    }
    (OUT / "dataset").mkdir(parents=True, exist_ok=True)
    (OUT / "dataset" / "sample.json").write_text(json.dumps(split, indent=1), encoding="utf-8")

    # ----------------------------------------------------------- subsets
    multiev_subset = [q.qid for q in sample_with_gold if q.n_gold_max > 1 or len(q.gold_passages_union) > 1]
    cross_subset = [q.qid for q in sample_with_gold if q.cross_section]
    (OUT / "dataset" / "subsets.json").write_text(json.dumps({
        "multi_evidence": multiev_subset,
        "cross_section": cross_subset,
        "all_sample_answerable_with_gold": [q.qid for q in sample_with_gold],
    }, indent=1), encoding="utf-8")

    # evidence length and paper-length stratification
    pmap = {p.paper_id: p.paragraph_map() for p in papers.values()}
    gold_para_tokens = []
    for q in with_gold:
        for s in q.annotator_sets:
            for pid in s:
                p = pmap[q.paper_id].get(pid)
                if p:
                    gold_para_tokens.append(p.n_tokens)
    paper_tok_sorted = sorted(paper_rows, key=lambda r: r["n_tokens"])
    quint = max(1, len(paper_tok_sorted) // 5)
    by_len = []
    for qi in range(5):
        chunk = paper_tok_sorted[qi * quint:(qi + 1) * quint] if qi < 4 else paper_tok_sorted[4 * quint:]
        ids = {r["paper_id"] for r in chunk}
        qs = [q for q in with_gold if q.paper_id in ids]
        by_len.append({
            "quintile": qi + 1,
            "paper_tokens_range": [chunk[0]["n_tokens"], chunk[-1]["n_tokens"]] if chunk else [],
            "n_papers": len(chunk), "n_questions": len(qs),
            "single_share": round(sum(1 for q in qs if q.n_gold_max == 1) / max(1, len(qs)), 4),
            "multi_share": round(sum(1 for q in qs if q.n_gold_max > 1) / max(1, len(qs)), 4),
            "cross_share": round(sum(1 for q in qs if q.cross_section) / max(1, len(qs)), 4),
            "mean_gold": round(statistics.mean([q.n_gold_max for q in qs]), 2) if qs else None,
        })

    gold_n_counts = Counter(len(s) for q in with_gold for s in q.annotator_sets)
    summary = {
        "split": "validation (qasper-dev-v0.3.json)",
        "n_papers": len(papers),
        "n_questions": len(qa_all),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unans),
        "n_with_mapped_gold": len(with_gold),
        "n_annotations_per_question": dict(sorted(n_annotations.items())),
        "evidence_match": dict(match_counts),
        "evidence_match_rate": round(match_rate, 4),
        "n_text_evidence_items": n_evidence_items - non_text,
        "n_unmatched_evidence_items": len(unmatched_records),
        "n_questions_with_unmapped_evidence": len(unmapped_q),
        "gold_paragraphs": gold_n_dist,
        "gold_paragraph_count_distribution": dict(sorted(gold_n_counts.items())),
        "single_gold_question_share": round(len(single) / max(1, len(with_gold)), 4),
        "multi_gold_question_share": round(len(multi) / max(1, len(with_gold)), 4),
        "cross_section_question_share": round(len(cross) / max(1, len(with_gold)), 4),
        "table_figure_only_share": round(len(tf_only) / max(1, len(qa_all)), 4),
        "pool": pool_dist,
        "paper_tokens": token_dist,
        "paragraph_tokens": para_token_dist,
        "gold_paragraph_tokens": dist("gold_paragraph_tokens", gold_para_tokens),
        "by_paper_length_quintile": by_len,
        "n_float_evidence_items": match_counts.get("float", 0),
        "n_heading_evidence_items": match_counts.get("heading", 0),
        "topk_share_of_pool": {
            str(k): {"mean": round(statistics.mean([min(k, n) / n for n in pool_sizes]), 4),
                     "p50": round(pct([k / n for n in pool_sizes], 50), 4)}
            for k in (3, 5, 10)
        },
        "sample": {
            "n_papers": len(sample), "n_questions": len(sample_qas),
            "n_answerable_with_gold": len(sample_with_gold),
            "n_multi_evidence": len(multiev_subset),
            "n_cross_section": len(cross_subset),
            "pool": dist("pool_size_sampled", [len(papers[p].paragraphs) for p in sample]),
        },
    }
    (OUT / "dataset" / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (OUT / "dataset" / "papers.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in paper_rows) + "\n", encoding="utf-8")
    (OUT / "dataset" / "unmatched_evidence.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in unmatched_records) + "\n", encoding="utf-8")

    # ----------------------------------------------------------- markdown
    L = []
    L.append("# QASPER Dataset Audit\n")
    L.append("Source: `qasper-dev-v0.3.json` (official validation split, QASPER v0.3).\n")
    L.append("Passage unit: one **paragraph** (`paper_id::s{section:02d}::p{para:03d}`); retrieval is always "
             "restricted to the question's own paper.\n")
    L.append("## Corpus\n")
    L.append(f"- papers: **{len(papers)}**, questions: **{len(qa_all)}**, answerable: **{len(answerable)}**, "
             f"unanswerable: **{len(unans)}**")
    L.append(f"- evidence-bearing answerable questions: **{len(with_gold)}**")
    L.append(f"- gold paragraphs per question/annotator: mean {gold_n_dist['mean']}, "
             f"P10 {gold_n_dist['p10']}, P50 {gold_n_dist['p50']}, P90 {gold_n_dist['p90']}, max {gold_n_dist['max']}")
    L.append(f"- gold-paragraph count distribution: "
             + ", ".join(f"{k}→{v}" for k, v in sorted(gold_n_counts.items())))
    L.append(f"- single-paragraph gold: **{summary['single_gold_question_share']:.1%}**; "
             f"multi-paragraph: **{summary['multi_gold_question_share']:.1%}**; "
             f"cross-section: **{summary['cross_section_question_share']:.1%}**")
    L.append(f"- questions whose evidence is figures/tables only: **{summary['table_figure_only_share']:.1%}**")
    L.append(f"- annotations per question: {dict(sorted(n_annotations.items()))}\n")
    L.append("## Candidate pool (paragraphs per paper)\n")
    L.append(f"mean **{pool_dist['mean']}**, P10 **{pool_dist['p10']}**, P50 **{pool_dist['p50']}**, "
             f"P90 **{pool_dist['p90']}**, max **{pool_dist['max']}**")
    L.append(f"Paper length: mean {token_dist['mean']} tokens, P50 {token_dist['p50']}, P90 {token_dist['p90']}.")
    L.append(f"Paragraph length: mean {para_token_dist['mean']} tokens, P50 {para_token_dist['p50']}, "
             f"P90 {para_token_dist['p90']}, max {para_token_dist['max']}.")
    gt = summary["gold_paragraph_tokens"]
    L.append(f"Gold evidence paragraph length: mean {gt['mean']} tokens, P50 {gt['p50']}, P90 {gt['p90']}, "
             f"max {gt['max']}.")
    L.append("\n### Questions by paper-length quintile\n")
    L.append("| quintile | paper tokens | papers | questions | single | multi | cross-section | mean gold |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in by_len:
        L.append(f"| {r['quintile']} | {r['paper_tokens_range']} | {r['n_papers']} | {r['n_questions']} | "
                 f"{100*r['single_share']:.1f}% | {100*r['multi_share']:.1f}% | "
                 f"{100*r['cross_share']:.1f}% | {r['mean_gold']} |")
    L.append("")
    L.append(f"Top-3/5/10 as a fraction of the pool: "
             + ", ".join(f"top-{k} {v['mean']:.1%}" for k, v in summary["topk_share_of_pool"].items()) + ".\n")
    L.append("## Evidence mapping\n")
    L.append(f"- evidence items by method: {dict(match_counts)}")
    L.append(f"- mapped rate: **{summary['evidence_match_rate']:.2%}**; unmatched items: "
             f"{len(unmatched_records)} across {len(unmapped_q)} questions")
    L.append(f"- **multiple annotators are treated as alternative evidence sets.** All-Evidence@k is computed "
             f"per annotator set and averaged over sets, never against the union. Any-Evidence@k uses the union. "
             f"Identical annotator sets are de-duplicated.\n")
    L.append("## Frozen first-round sample\n")
    L.append(f"- seed **{SEED}**, rule: {split['rule']}")
    L.append(f"- {len(sample)} papers, {len(sample_qas)} questions, {len(sample_with_gold)} answerable with gold")
    L.append(f"- multi-evidence subset: {len(multiev_subset)} questions; cross-section subset: {len(cross_subset)}")
    L.append(f"- sample pool: mean {summary['sample']['pool']['mean']}, P10 {summary['sample']['pool']['p10']}, "
             f"P90 {summary['sample']['pool']['p90']}\n")
    L.append("## Decision-gate reading\n")
    small_pool = pool_dist["p90"] and pool_dist["p90"] < 80
    L.append(f"The paragraph pool is small relative to a standard RAG corpus "
             f"(P90 = {pool_dist['p90']} paragraphs/paper; top-5 covers {summary['topk_share_of_pool']['5']['mean']:.1%} "
             f"of the pool on average). {'This is flagged as a saturation risk.' if small_pool else ''} "
             f"Difficulty must therefore be judged on **All-Evidence@k** and on the multi-paragraph / "
             f"cross-section subsets rather than on Any-Evidence@5 alone.\n")
    L.append("Unmatched evidence records are in `dataset/unmatched_evidence.jsonl`; figure/table evidence "
             "(`FLOAT SELECTED` paragraphs) is never mapped to a text paragraph and is excluded from text-evidence "
             "metrics, but reported separately.\n")
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / "dataset_audit.md").write_text("\n".join(L), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("pool", "paper_tokens", "paragraph_tokens")}, indent=1))
    print("wrote", OUT / "reports" / "dataset_audit.md")


if __name__ == "__main__":
    main()
