"""LoCoMo dataset loading, canonical session/turn modelling and evidence mapping.

Design notes
------------
* One Conversation == one independent corpus / knowledge graph.
* One Session == one document (passage) for the first version.
* Every document keeps a turn-level back-pointer so that evidence recall can be
  measured both at document level (session) and at turn level (dia_id).

Nothing in this module looks at gold answers/evidence; the mapping helpers are
only used by the evaluation code.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tiktoken

_SESSION_RE = re.compile(r"^session_(\d+)$")
_DIA_RE = re.compile(r"^D(\d+):(\d+)$")
_DATE_FORMATS = (
    "%I:%M %p on %d %B, %Y",
    "%I:%M %p on %d %b, %Y",
    "%H:%M on %d %B, %Y",
    "%H:%M on %d %b, %Y",
)

_ENCODER = None


def num_tokens(text: str) -> int:
    """Token count proxy (cl100k_base). Used for budgeting, not billing."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return len(_ENCODER.encode(text))


def parse_date(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


@dataclass
class Turn:
    dia_id: str
    speaker: str
    text: str
    session_id: int
    turn_index: int  # 1-based index inside the session (matches D<s>:<i>)

    def render(self) -> str:
        return f"{self.speaker}: {self.text}"


@dataclass
class Session:
    conversation_id: str
    session_id: int
    date_time_raw: str
    document_id: str
    turns: List[Turn] = field(default_factory=list)
    date_time: Optional[datetime] = None

    @property
    def text(self) -> str:
        header = f"[Session {self.session_id} | {self.date_time_raw}]"
        return header + "\n" + "\n".join(t.render() for t in self.turns)

    @property
    def n_tokens(self) -> int:
        return num_tokens(self.text)

    @property
    def dia_ids(self) -> List[str]:
        return [t.dia_id for t in self.turns]

    def turn_map(self) -> Dict[str, Turn]:
        return {t.dia_id: t for t in self.turns}


@dataclass
class QAItem:
    qid: str
    conversation_id: str
    question: str
    answer: Optional[str]
    category: int
    evidence: List[str]
    adversarial_answer: Optional[str] = None
    evidence_sessions: List[int] = field(default_factory=list)
    evidence_turns: List[str] = field(default_factory=list)
    unmapped_evidence: List[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return self.category != 5 and bool(self.answer)


@dataclass
class Conversation:
    sample_id: str
    speakers: Tuple[str, str]
    sessions: List[Session]
    qa: List[QAItem]

    @property
    def conversation_id(self) -> str:
        return self.sample_id

    def session_map(self) -> Dict[int, Session]:
        return {s.session_id: s for s in self.sessions}

    def documents(self) -> List[Dict]:
        """Document records in the exact order used for indexing."""
        return [
            {
                "document_id": s.document_id,
                "session_id": s.session_id,
                "date_time_raw": s.date_time_raw,
                "content": s.text,
                "n_tokens": s.n_tokens,
                "dia_ids": s.dia_ids,
            }
            for s in self.sessions
        ]


CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def _parse_evidence(ev: List[str], conv: Conversation) -> Tuple[List[int], List[str], List[str]]:
    """Map dia_ids to (session ids, valid dia_ids, unmapped)."""
    by_dia = {t.dia_id: t for s in conv.sessions for t in s.turns}
    sessions, turns, unmapped = set(), [], []
    for item in ev or []:
        if not isinstance(item, str):
            unmapped.append(str(item))
            continue
        m = _DIA_RE.match(item.strip())
        if not m:
            unmapped.append(item)
            continue
        dia = f"D{int(m.group(1))}:{int(m.group(2))}"
        if dia in by_dia:
            turns.append(dia)
            sessions.add(by_dia[dia].session_id)
        else:
            unmapped.append(item)
    return sorted(sessions), turns, unmapped


def load_conversation(raw: Dict) -> Conversation:
    sample_id = raw["sample_id"]
    cv = raw["conversation"]
    speakers = (cv["speaker_a"], cv["speaker_b"])

    session_ids = sorted(
        int(m.group(1)) for k in cv if (m := _SESSION_RE.match(k))
    )
    sessions: List[Session] = []
    for sid in session_ids:
        doc_id = f"{sample_id}::s{sid:02d}"
        sess = Session(
            conversation_id=sample_id,
            session_id=sid,
            date_time_raw=cv.get(f"session_{sid}_date_time", ""),
            document_id=doc_id,
            date_time=parse_date(cv.get(f"session_{sid}_date_time", "")),
        )
        for i, turn in enumerate(cv[f"session_{sid}"], start=1):
            sess.turns.append(
                Turn(
                    dia_id=turn.get("dia_id", f"D{sid}:{i}"),
                    speaker=turn.get("speaker", ""),
                    text=turn.get("text", ""),
                    session_id=sid,
                    turn_index=i,
                )
            )
        sessions.append(sess)

    conv = Conversation(sample_id=sample_id, speakers=speakers, sessions=sessions, qa=[])

    qa_items: List[QAItem] = []
    for i, q in enumerate(raw.get("qa", [])):
        item = QAItem(
            qid=f"{sample_id}::q{i}",
            conversation_id=sample_id,
            question=q["question"],
            answer=q.get("answer"),
            category=int(q.get("category", -1)),
            evidence=list(q.get("evidence") or []),
            adversarial_answer=q.get("adversarial_answer"),
        )
        item.evidence_sessions, item.evidence_turns, item.unmapped_evidence = _parse_evidence(
            item.evidence, conv
        )
        qa_items.append(item)
    conv.qa = qa_items
    return conv


def load_locomo(path: str | Path) -> List[Conversation]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    convs = [load_conversation(raw) for raw in data]
    convs.sort(key=lambda c: c.sample_id)
    return convs


def make_splits(convs: List[Conversation], n_dev: int = 3, seed: int = 20240501) -> Dict[str, List[str]]:
    import random

    ids = [c.sample_id for c in convs]
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    dev = sorted(shuffled[:n_dev])
    test = sorted(shuffled[n_dev:])
    return {"seed": seed, "dev": dev, "test": test}
