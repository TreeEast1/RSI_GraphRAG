"""DetectiveQA loading, prefix-safe chunking, gold clue mapping and frozen splits.

DetectiveQA asks every question at an annotated story position.  A method may
only read paragraphs strictly before ``answer_position``.  Chunks are therefore
contiguous paragraph spans and never cross a question cutoff.  Cutoffs are the
question timestamps, not answer content, and every method sees the same chunks.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "detectiveqa"
OUT = ROOT / "outputs" / "detectiveqa"
_PARA_RE = re.compile(r"^\[(\d+)\]\s*(.*)$", re.M)
_BAD_OPTION_PREFIX = re.compile(r"^\s*200\s*:\s*")


@dataclass
class Paragraph:
    pid: int
    text: str


@dataclass
class Novel:
    novel_id: str
    title: str
    author: str
    paragraphs: List[Paragraph]

    def paragraph_map(self) -> Dict[int, str]:
        return {p.pid: p.text for p in self.paragraphs}


@dataclass
class Question:
    qid: str
    novel_id: str
    source: str
    question_index: int
    question: str
    options: Dict[str, str]
    answer: str
    answer_position: int
    cutoff: int
    clue_paragraphs: List[int]
    clue_texts: List[str] = field(default_factory=list)
    annotator_reasoning: str = ""
    distraction: Dict[str, str] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    novel_id: str
    index: int
    start_pid: int
    end_pid: int
    paragraph_ids: List[int]
    text: str
    n_words: int

    @property
    def available_at(self) -> int:
        return self.end_pid + 1


def _novel_meta(path: Path) -> Tuple[str, str, str]:
    parts = path.stem.split("-", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return parts[0], path.stem, ""


def load_novels(lang: str = "en", raw: Path = RAW) -> Dict[str, Novel]:
    novels = {}
    for path in sorted((raw / f"novel_data_{lang}").glob("*.txt")):
        novel_id, title, author = _novel_meta(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        paragraphs = [Paragraph(int(pid), body.strip()) for pid, body in _PARA_RE.findall(text)]
        novels[novel_id] = Novel(novel_id, title, author, paragraphs)
    return novels


def clean_option(text: str) -> str:
    return _BAD_OPTION_PREFIX.sub("", str(text).strip())


def load_questions(lang: str = "en", raw: Path = RAW) -> List[Question]:
    """Load questions with unique ids.

    Two annotation files can share a ``novel_id`` (DetectiveQA ships a second
    English annotation of novel 117 as ``149.json``).  The first file in sorted
    order keeps the canonical ``source:novel:index`` id; later files get their
    file stem as an extra discriminator so no question is silently overwritten
    by ``load_prepared``'s qid-keyed dict.  This is the minimal fix for a raw
    data collision, not a change to the dataset or to any method.
    """
    items: List[Tuple[str, Path, Dict]] = []
    for source in ("human_anno", "AIsup_anno"):
        for path in sorted((raw / f"anno_data_{lang}" / source).glob("*.json")):
            for item in json.loads(path.read_text(encoding="utf-8")):
                items.append((source, path, item))

    paths_by_key: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for source, path, item in items:
        paths_by_key[(source, str(item["novel_id"]))].append(path)
    canonical_paths = {key: sorted(set(paths)) for key, paths in paths_by_key.items()}

    questions = []
    for source, path, item in items:
        novel_id = str(item["novel_id"])
        siblings = canonical_paths[(source, novel_id)]
        suffix = "" if siblings.index(path) == 0 else f":{path.stem}"
        for index, raw_q in enumerate(item["questions"]):
            answer_position = int(raw_q["answer_position"])
            pairs = list(zip(raw_q.get("reasoning", []), raw_q.get("clue_position", [])))
            clues = sorted({int(pos) for _, pos in pairs
                            if isinstance(pos, int) and 0 < pos < answer_position})
            clue_texts = [str(text).strip() for text, pos in pairs
                          if isinstance(pos, int) and 0 < pos < answer_position]
            final_reasoning = next((str(text).strip() for text, pos in pairs if pos == -1), "")
            questions.append(Question(
                qid=f"{source}:{novel_id}{suffix}:{index:02d}",
                novel_id=novel_id,
                source=source,
                question_index=index,
                question=str(raw_q["question"]).strip(),
                options={k: clean_option(v) for k, v in sorted(raw_q["options"].items())},
                answer=str(raw_q["answer"]).strip(),
                answer_position=answer_position,
                cutoff=answer_position,
                clue_paragraphs=clues,
                clue_texts=clue_texts,
                annotator_reasoning=final_reasoning,
                distraction={k: str(v).strip() for k, v in raw_q.get("distraction", {}).items()},
            ))
    return questions


def _chunk_text(paragraphs: Sequence[Paragraph]) -> str:
    return "\n".join(p.text for p in paragraphs if p.text)


def build_chunks(novel: Novel, cutoffs: Iterable[int], target_words: int = 200,
                 max_words: int = 320) -> List[Chunk]:
    """Greedy contiguous chunks with hard boundaries before every cutoff."""
    boundaries = set(int(x) for x in cutoffs)
    groups: List[List[Paragraph]] = []
    current: List[Paragraph] = []
    words = 0
    for paragraph in novel.paragraphs:
        n = len(paragraph.text.split())
        must_split = paragraph.pid in boundaries
        too_long = current and words >= target_words and words + n > max_words
        if current and (must_split or too_long):
            groups.append(current)
            current, words = [], 0
        current.append(paragraph)
        words += n
    if current:
        groups.append(current)

    chunks = []
    seen = Counter()
    for index, group in enumerate(groups):
        text = _chunk_text(group)
        seen[text] += 1
        if seen[text] > 1:
            text = f"{text}\n[paragraphs {group[0].pid}-{group[-1].pid}]"
        chunks.append(Chunk(
            chunk_id=f"dq-{novel.novel_id}-{index:05d}",
            novel_id=novel.novel_id,
            index=index,
            start_pid=group[0].pid,
            end_pid=group[-1].pid,
            paragraph_ids=[p.pid for p in group],
            text=text,
            n_words=len(text.split()),
        ))
    return chunks


def paragraph_to_chunk(chunks: Sequence[Chunk]) -> Dict[int, str]:
    return {pid: chunk.chunk_id for chunk in chunks for pid in chunk.paragraph_ids}


def visible_chunk_ids(chunks: Sequence[Chunk], cutoff: int) -> List[str]:
    return [chunk.chunk_id for chunk in chunks if chunk.end_pid < cutoff]


def gold_chunk_ids(question: Question, chunks: Sequence[Chunk]) -> List[str]:
    mapping = paragraph_to_chunk(chunks)
    return list(dict.fromkeys(mapping[pid] for pid in question.clue_paragraphs if pid in mapping))


def make_splits(questions: Sequence[Question], seed: int = 20261111,
                cross_dev_novels: int = 10) -> Dict[str, List[str]]:
    """Freeze novel-disjoint cross-novel splits and within-novel streams.

    * eval_human: every human-annotated question, never used for updates.
    * within_practice: AI-assisted questions on the same human-annotated novels.
      A memory may affect an evaluation question only when the memory became
      available before that question's cutoff.
    * cross_practice/cross_dev: AI-only novels, used to learn transferable skills.
    """
    human_novels = sorted({q.novel_id for q in questions if q.source == "human_anno"}, key=int)
    ai_only = sorted({q.novel_id for q in questions if q.novel_id not in human_novels}, key=int)
    rng = random.Random(seed)
    shuffled = list(ai_only)
    rng.shuffle(shuffled)
    dev_novels = sorted(shuffled[:cross_dev_novels], key=int)
    practice_novels = sorted(shuffled[cross_dev_novels:], key=int)

    by_novel = defaultdict(list)
    for q in questions:
        by_novel[q.novel_id].append(q)
    ordered = lambda novels, source=None: [  # noqa: E731
        q.qid for novel in novels for q in sorted(by_novel[novel], key=lambda x: (x.cutoff, x.qid))
        if source is None or q.source == source
    ]
    practice_order = ordered(practice_novels)
    rng.shuffle(practice_order)
    return {
        "eval_human": ordered(human_novels, "human_anno"),
        "within_practice": ordered(human_novels, "AIsup_anno"),
        "cross_practice": ordered(practice_novels),
        "cross_practice_order": practice_order,
        "cross_dev": ordered(dev_novels),
        "human_novels": human_novels,
        "cross_practice_novels": practice_novels,
        "cross_dev_novels": dev_novels,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    return values[int((len(values) - 1) * q)]


def audit(novels: Dict[str, Novel], questions: Sequence[Question],
          chunks_by_novel: Dict[str, List[Chunk]], splits: Dict[str, List[str]]) -> Dict:
    by_qid = {q.qid: q for q in questions}
    spans = [max(q.clue_paragraphs) - min(q.clue_paragraphs)
             for q in questions if len(q.clue_paragraphs) > 1]
    clue_counts = [len(q.clue_paragraphs) for q in questions]
    visible = []
    gold_chunks = []
    for q in questions:
        chunks = chunks_by_novel[q.novel_id]
        visible.append(len(visible_chunk_ids(chunks, q.cutoff)))
        gold_chunks.append(len(gold_chunk_ids(q, chunks)))
    return {
        "novels": len(novels),
        "questions": len(questions),
        "unique_qids": len({q.qid for q in questions}),
        "duplicate_qids": len(questions) - len({q.qid for q in questions}),
        "questions_by_source": dict(Counter(q.source for q in questions)),
        "chunks": sum(len(x) for x in chunks_by_novel.values()),
        "chunk_words_median": _percentile([c.n_words for x in chunks_by_novel.values() for c in x], .5),
        "clues_median": _percentile(clue_counts, .5),
        "clue_span_paragraphs_median": _percentile(spans, .5),
        "clue_span_paragraphs_p90": _percentile(spans, .9),
        "visible_chunks_median": _percentile(visible, .5),
        "gold_chunks_median": _percentile(gold_chunks, .5),
        "splits": {k: len(v) for k, v in splits.items()},
        "eval_human_with_earlier_practice": sum(
            any(by_qid[p].novel_id == by_qid[e].novel_id and by_qid[p].cutoff < by_qid[e].cutoff
                for p in splits["within_practice"])
            for e in splits["eval_human"]
        ),
    }


def write_jsonl(path: Path, rows: Iterable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row) if hasattr(row, "__dataclass_fields__") else row,
                               ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_prepared(out: Path = OUT) -> Tuple[Dict[str, Question], Dict[str, List[Chunk]], Dict]:
    questions = {row["qid"]: Question(**row) for row in read_jsonl(out / "questions.jsonl")}
    chunks = defaultdict(list)
    for row in read_jsonl(out / "chunks.jsonl"):
        chunks[row["novel_id"]].append(Chunk(**row))
    splits = json.loads((out / "splits.json").read_text())
    return questions, dict(chunks), splits
