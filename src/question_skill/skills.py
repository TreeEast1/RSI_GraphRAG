"""Persistent, versioned retrieval-skill library (experiment E4)."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..config import PATHS


@dataclass
class Skill:
    skill_id: str
    trigger_condition: str          # human readable
    failure_pattern: str
    retrieval_strategy: str         # template with {entity} / {question} slots
    applicable_question_type: str   # temporal | multi-hop | single-hop | open-domain | any
    example: str = ""
    observed_benefit: float = 0.0
    usage_count: int = 0
    version: int = 1
    enabled: bool = True
    metadata: Dict = field(default_factory=dict)

    def triggers(self, question: str, category: Optional[int] = None) -> bool:
        if not self.enabled:
            return False
        if self.applicable_question_type not in ("any", None):
            expected = {"temporal": 2, "multi-hop": 1, "open-domain": 3, "single-hop": 4}
            if category is not None and expected.get(self.applicable_question_type) != category:
                return False
        return True


class SkillLibrary:
    def __init__(self, path: str | Path = PATHS["skills"] / "retrieval_skills.json"):
        self.path = Path(path)
        self.skills: List[Skill] = []
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.skills = [Skill(**s) for s in data.get("skills", [])]

    def add(self, skill: Skill) -> None:
        self.skills.append(skill)

    def active(self) -> List[Skill]:
        return [s for s in self.skills if s.enabled]

    def get(self, skill_id: str) -> Optional[Skill]:
        return next((s for s in self.skills if s.skill_id == skill_id), None)

    def save(self, meta: Optional[Dict] = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "meta": meta or {},
            "skills": [asdict(s) for s in self.skills],
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def record_usage(self, skill_ids: List[str], benefit: float) -> None:
        for sid in skill_ids:
            s = self.get(sid)
            if s:
                s.usage_count += 1
                s.observed_benefit = 0.8 * s.observed_benefit + 0.2 * benefit
        self.save()


DEFAULT_SKILLS = [
    Skill(
        skill_id="sk_temporal_anchor",
        trigger_condition="question asks when/when did/how long/before/after",
        failure_pattern="retrieval returns the event but not the dated session, so the reader cannot answer a time question",
        retrieval_strategy='{event} date time month year',
        applicable_question_type="temporal",
        example="Q: When did X happen? -> retrieve 'X happened <date>'",
    ),
    Skill(
        skill_id="sk_bridge_person",
        trigger_condition="question mentions an event or object but the relation to a person is missing",
        failure_pattern="only one hop of a multi-hop chain is retrieved",
        retrieval_strategy='{subject} {relation} {object}',
        applicable_question_type="multi-hop",
    ),
    Skill(
        skill_id="sk_state_change",
        trigger_condition="question involves before/after/change/now/used to",
        failure_pattern="earlier and later states of the same entity are retrieved from only one period",
        retrieval_strategy='{entity} moved job changed started new',
        applicable_question_type="any",
    ),
]
