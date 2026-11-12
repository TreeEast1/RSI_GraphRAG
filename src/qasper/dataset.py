"""QASPER loading, paragraph-level passage modelling and gold-evidence mapping.

Design
------
* One paper == one independent corpus / knowledge graph. Retrieval never crosses
  papers (the official QASPER task is per-paper QA).
* One paragraph == one retrieval passage, with a stable id
  ``{paper_id}::s{section_idx:02d}::p{para_idx:03d}``.
* Multiple annotators are kept as **alternative** answer/evidence sets. Nothing
  in this module merges them into a single required set; the evaluation layer
  decides how to score alternatives.
* Nothing here is visible to the read-time models except paragraph text: gold
  answers and evidence live only on the annotation objects used by evaluation.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..dataset.locomo import num_tokens

_WS = re.compile(r"\s+")


def norm_text(s: str) -> str:
    """Normalise for evidence matching: whitespace + lowercase, punctuation kept."""
    return _WS.sub(" ", str(s or "")).strip().lower()


def stable_paper_order(paper_ids: Iterable[str], seed: int) -> List[str]:
    """Deterministic order independent of JSON insertion order."""
    return sorted(paper_ids, key=lambda p: hashlib.sha1(f"{seed}:{p}".encode()).hexdigest())


@dataclass
class Paragraph:
    passage_id: str
    paper_id: str
    section_index: int
    paragraph_index: int
    global_index: int
    section_name: str
    text: str
    n_tokens: int = 0

    def __post_init__(self):
        if not self.n_tokens:
            self.n_tokens = num_tokens(self.text)

    def to_dict(self) -> Dict:
        return {
            "passage_id": self.passage_id,
            "paper_id": self.paper_id,
            "section_index": self.section_index,
            "paragraph_index": self.paragraph_index,
            "global_index": self.global_index,
            "section_name": self.section_name,
            "text": self.text,
            "n_tokens": self.n_tokens,
        }


@dataclass
class FigureTable:
    caption_id: str
    paper_id: str
    file: str
    caption: str
    n_tokens: int = 0

    def __post_init__(self):
        if not self.n_tokens:
            self.n_tokens = num_tokens(self.caption)


@dataclass
class Annotation:
    worker_id: str
    annotation_id: str
    unanswerable: bool
    answer_type: str
    answer: str
    evidence: List[str] = field(default_factory=list)
    highlighted_evidence: List[str] = field(default_factory=list)
    evidence_passage_ids: List[str] = field(default_factory=list)
    evidence_match: List[Dict] = field(default_factory=list)  # per evidence item
    has_float_evidence: bool = False
    has_text_evidence: bool = False

    def to_dict(self) -> Dict:
        return {
            "worker_id": self.worker_id,
            "annotation_id": self.annotation_id,
            "unanswerable": self.unanswerable,
            "answer_type": self.answer_type,
            "answer": self.answer,
            "evidence": self.evidence,
            "highlighted_evidence": self.highlighted_evidence,
            "evidence_passage_ids": self.evidence_passage_ids,
            "evidence_match": self.evidence_match,
            "has_float_evidence": self.has_float_evidence,
            "has_text_evidence": self.has_text_evidence,
        }


@dataclass
class QAItem:
    qid: str
    paper_id: str
    question: str
    annotations: List[Annotation] = field(default_factory=list)
    nlp_background: str = ""
    topic_background: str = ""
    # Derived gold views (evaluation only).
    answerable: bool = True
    gold_passages_union: List[str] = field(default_factory=list)
    annotator_sets: List[List[str]] = field(default_factory=list)
    n_gold_min: int = 0
    n_gold_max: int = 0
    cross_section: bool = False
    has_unmapped_evidence: bool = False
    table_figure_only: bool = False

    def to_dict(self) -> Dict:
        return {
            "qid": self.qid,
            "paper_id": self.paper_id,
            "question": self.question,
            "annotations": [a.to_dict() for a in self.annotations],
            "answerable": self.answerable,
            "gold_passages_union": self.gold_passages_union,
            "annotator_sets": self.annotator_sets,
            "n_gold_min": self.n_gold_min,
            "n_gold_max": self.n_gold_max,
            "cross_section": self.cross_section,
            "has_unmapped_evidence": self.has_unmapped_evidence,
            "table_figure_only": self.table_figure_only,
        }


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    paragraphs: List[Paragraph]
    figures_tables: List[FigureTable]
    qas: List[QAItem]
    sections: List[str]
    paragraph_aliases: Dict[str, str] = field(default_factory=dict)  # dropped duplicate -> canonical

    @property
    def full_text(self) -> str:
        return "\n\n".join(
            f"[{p.section_name} | {p.passage_id}]\n{p.text}" for p in self.paragraphs
        )

    @property
    def n_tokens(self) -> int:
        return sum(p.n_tokens for p in self.paragraphs)

    def paragraph_map(self) -> Dict[str, Paragraph]:
        return {p.passage_id: p for p in self.paragraphs}


# --------------------------------------------------------------------------- #
# Evidence mapping
# --------------------------------------------------------------------------- #
def _build_paragraph_index(paragraphs: Sequence[Paragraph]) -> Dict[str, str]:
    return {norm_text(p.text): p.passage_id for p in paragraphs}


def map_evidence_item(
    evidence: str,
    paragraphs: Sequence[Paragraph],
    exact_index: Dict[str, str],
    section_names: Optional[Set[str]] = None,
) -> Tuple[Optional[str], str]:
    """Map one gold evidence string to a paragraph id.

    Returns (passage_id, method). method is one of
    exact | concat | substring | fuzzy | float | heading | empty | unmatched.
    """
    raw = str(evidence or "")
    if "FLOAT SELECTED" in raw:
        return None, "float"
    n = norm_text(raw)
    if not n:
        return None, "empty"
    if section_names and n in section_names:
        return None, "heading"
    if n in exact_index:
        return exact_index[n], "exact"
    # multi-paragraph concatenation (consecutive paragraphs)
    for i in range(len(paragraphs)):
        joined = ""
        for j in range(i, min(i + 4, len(paragraphs))):
            joined = (joined + " " + norm_text(paragraphs[j].text)).strip()
            if joined == n:
                return paragraphs[i].passage_id, "concat"
            if len(joined) > len(n) + 50:
                break
    # substring either direction
    for p in paragraphs:
        pn = norm_text(p.text)
        if n and (n in pn or (len(pn) > 80 and pn in n)):
            return p.passage_id, "substring"
    # token-overlap fuzzy match, only for long evidence paragraphs
    best, best_j = None, 0.0
    et = set(n.split())
    for p in paragraphs:
        pt = set(norm_text(p.text).split())
        if not et or not pt:
            continue
        j = len(et & pt) / len(et | pt)
        if j > best_j:
            best, best_j = p.passage_id, j
    if best is not None and best_j >= 0.85:
        return best, "fuzzy"
    return None, "unmatched"


def _answer_view(a: Dict) -> Tuple[bool, str, str]:
    unans = bool(a.get("unanswerable"))
    if unans:
        return True, "none", "Unanswerable"
    if a.get("extractive_spans"):
        return False, "extractive", ", ".join(str(x) for x in a["extractive_spans"])
    if a.get("free_form_answer"):
        return False, "abstractive", str(a["free_form_answer"])
    if a.get("yes_no") is not None:
        return False, "boolean", "Yes" if a["yes_no"] else "No"
    return False, "unknown", ""


def load_qasper(path: str | Path) -> Dict[str, Paper]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    papers: Dict[str, Paper] = {}
    for paper_id, data in raw.items():
        sections = [s.get("section_name", "") for s in data.get("full_text", [])]
        paragraphs: List[Paragraph] = []
        aliases: Dict[str, str] = {}
        seen_text: Dict[str, str] = {}
        gidx = 0
        for si, sec in enumerate(data.get("full_text", [])):
            sname = sec.get("section_name", "")
            for pi, text in enumerate(sec.get("paragraphs", [])):
                text = str(text or "").strip()
                if not text:
                    continue
                pid = f"{paper_id}::s{si:02d}::p{pi:03d}"
                key = norm_text(text)
                if key in seen_text:
                    # Identical text in two places: collapse to one graph/retrieval
                    # unit so chunk hashing and gold matching stay unambiguous.
                    aliases[pid] = seen_text[key]
                    continue
                seen_text[key] = pid
                paragraphs.append(Paragraph(pid, paper_id, si, pi, gidx, sname, text))
                gidx += 1
        figs = [
            FigureTable(f"{paper_id}::fig{i:03d}", paper_id,
                        str(ft.get("file", "")), str(ft.get("caption", "")).strip())
            for i, ft in enumerate(data.get("figures_and_tables", []))
        ]
        exact_index = _build_paragraph_index(paragraphs)
        section_names = {norm_text(s) for s in sections if s}
        qas: List[QAItem] = []
        for q in data.get("qas", []):
            annotations: List[Annotation] = []
            for ann in q.get("answers", []):
                ainfo = ann.get("answer", {}) or {}
                unans, atype, ans = _answer_view(ainfo)
                ev = [str(x) for x in (ainfo.get("evidence") or [])]
                hl = [str(x) for x in (ainfo.get("highlighted_evidence") or [])]
                mapped, matches = [], []
                for item in ev:
                    pid, method = map_evidence_item(item, paragraphs, exact_index, section_names)
                    if pid and pid in aliases:
                        pid = aliases[pid]
                    matches.append({"evidence": item[:200], "passage_id": pid, "method": method})
                    if pid:
                        mapped.append(pid)
                has_float = any("FLOAT SELECTED" in x for x in ev)
                has_text = any("FLOAT SELECTED" not in x and norm_text(x) for x in ev)
                annotations.append(Annotation(
                    worker_id=str(ann.get("worker_id", "")),
                    annotation_id=str(ann.get("annotation_id", "")),
                    unanswerable=unans,
                    answer_type=atype,
                    answer=ans,
                    evidence=ev,
                    highlighted_evidence=hl,
                    evidence_passage_ids=sorted(set(mapped)),
                    evidence_match=matches,
                    has_float_evidence=has_float,
                    has_text_evidence=has_text,
                ))
            answerable = any(not a.unanswerable for a in annotations)
            sets = [sorted(set(a.evidence_passage_ids)) for a in annotations if a.evidence_passage_ids]
            # Alternative annotator sets: dedupe identical sets.
            uniq_sets = []
            for s in sets:
                if s not in uniq_sets:
                    uniq_sets.append(s)
            union = sorted({p for s in uniq_sets for p in s})
            n_min = min((len(s) for s in uniq_sets), default=0)
            n_max = max((len(s) for s in uniq_sets), default=0)
            sec_of = {p.passage_id: p.section_index for p in paragraphs}
            cross = any(len({sec_of[x] for x in s if x in sec_of}) > 1 for s in uniq_sets)
            unmapped = any(m["method"] == "unmatched" for a in annotations for m in a.evidence_match)
            tf_only = bool(uniq_sets) is False and any(a.has_text_evidence is False and a.has_float_evidence for a in annotations)
            qas.append(QAItem(
                qid=str(q.get("question_id", "")),
                paper_id=paper_id,
                question=str(q.get("question", "")),
                annotations=annotations,
                nlp_background=str(q.get("nlp_background", "")),
                topic_background=str(q.get("topic_background", "")),
                answerable=answerable,
                gold_passages_union=union,
                annotator_sets=uniq_sets,
                n_gold_min=n_min,
                n_gold_max=n_max,
                cross_section=cross,
                has_unmapped_evidence=unmapped,
                table_figure_only=tf_only,
            ))
        papers[paper_id] = Paper(
            paper_id=paper_id,
            title=str(data.get("title", "")),
            abstract=str(data.get("abstract", "")),
            paragraphs=paragraphs,
            figures_tables=figs,
            qas=qas,
            sections=sections,
            paragraph_aliases=aliases,
        )
    return papers


def write_paper_jsonl(paper: Paper, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "paragraphs.jsonl", "w", encoding="utf-8") as f:
        for p in paper.paragraphs:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    with open(out_dir / "qa.jsonl", "w", encoding="utf-8") as f:
        for q in paper.qas:
            f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "paper_id": paper.paper_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "sections": paper.sections,
            "n_paragraphs": len(paper.paragraphs),
            "n_tokens": paper.n_tokens,
            "figures_and_tables": [ft.__dict__ for ft in paper.figures_tables],
        }, f, ensure_ascii=False, indent=1)
