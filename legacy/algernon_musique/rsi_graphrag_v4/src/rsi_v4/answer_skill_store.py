"""Answer skills selected at read time for the QA reader."""

from .skill_store import SkillStore


class AnswerSkillStore(SkillStore):
    def __init__(self) -> None:
        super().__init__("answer")
