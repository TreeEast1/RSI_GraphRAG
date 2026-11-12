"""Shared helpers for QASPER experiment scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "qasper_readtime"
RAW = ROOT / "data" / "raw" / "qasper"

from .dataset import Paper, QAItem, load_qasper  # noqa: E402


def load_papers(scope: str = "all") -> Dict[str, Paper]:
    papers = load_qasper(RAW / "qasper-dev-v0.3.json")
    if scope == "all":
        return papers
    sample = json.loads((OUT / "dataset" / "sample.json").read_text())
    if scope == "smoke20":
        keep = sample["ordered_full"][:20]
    elif scope == "sample100":
        keep = sample["sample"]
    else:
        raise ValueError(scope)
    return {p: papers[p] for p in keep}


def paper_list(scope: str) -> List[str]:
    if scope == "all":
        return [p.paper_id for p in load_papers("all").values()]
    sample = json.loads((OUT / "dataset" / "sample.json").read_text())
    return sample["ordered_full"][:20] if scope == "smoke20" else sample["sample"]


def subsets() -> Dict[str, List[str]]:
    return json.loads((OUT / "dataset" / "subsets.json").read_text())


def gold_sets(q: QAItem) -> List[List[str]]:
    return [s for s in q.annotator_sets if s]


def evaluate_paper_questions(paper: Paper, ranker, methods: Sequence[str],
                             top_k: int = 20, budgets=(2048, 4096, 8192)) -> List[Dict]:
    """Run one paper's questions through a ``ranker(question, method, k) -> [ids]``.

    Returns one row per (question, method) with metrics for every method.
    """
    from .retrieval_metrics import coverage_at_budget, question_metrics

    tok = {p.passage_id: p.n_tokens for p in paper.paragraphs}
    rows = []
    for q in paper.qas:
        if not q.answerable:
            continue  # retrieval metrics need gold evidence
        sets = gold_sets(q)
        if not sets:
            continue
        for method in methods:
            ranked = [x if isinstance(x, str) else x["passage_id"] for x in ranker(q.question, method, top_k)]
            m = question_metrics(ranked, sets)
            rec = {
                "method": method,
                "paper_id": paper.paper_id,
                "qid": q.qid,
                "question": q.question,
                "n_gold": max(len(s) for s in sets),
                "n_annotator_sets": len(sets),
                "cross_section": q.cross_section,
                "multi_evidence": max(len(s) for s in sets) > 1 or len(q.gold_passages_union) > 1,
                "ranked": ranked,
                "metrics": m,
                "budget": {str(b): coverage_at_budget(ranked, sets, tok, b) for b in budgets},
            }
            rows.append(rec)
    return rows


def summarize(rows: List[Dict], subset_qids: Optional[Iterable[str]] = None) -> Dict:
    from .retrieval_metrics import aggregate_questions, bootstrap_ci

    sub = set(subset_qids) if subset_qids is not None else None
    if sub is not None:
        rows = [r for r in rows if r["qid"] in sub]
    agg = aggregate_questions(rows)
    for b in ("2048", "4096", "8192"):
        vals = [r["budget"][b]["full"] for r in rows if r.get("budget", {}).get(b)]
        ci = bootstrap_ci(vals)
        agg[f"cov_full@{b}"] = ci["mean"]
        agg[f"cov_full@{b}_ci"] = f"[{ci['lo']:.3f},{ci['hi']:.3f}]"
    agg["n_questions"] = len({r["qid"] for r in rows})
    return agg


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
