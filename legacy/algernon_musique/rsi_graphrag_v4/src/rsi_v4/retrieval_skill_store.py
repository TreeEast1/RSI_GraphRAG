"""Retrieval skills selected at query time for decomposition."""

from .skill_store import SkillStore


class RetrievalSkillStore(SkillStore):
    def __init__(self) -> None:
        super().__init__("retrieval")
