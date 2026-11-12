"""Shared skill records. Concrete stores live in the retrieval and answer modules."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Dict, List

from .hashing import sha256_text
from .validation import leaks_gold_answer, token_jaccard


class SkillStore:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.skills: Dict[str, dict] = {}
        self.version = 0

    def snapshot(self) -> dict:
        return {"skills": copy.deepcopy(self.skills), "version": self.version, "kind": self.kind}

    def restore(self, snapshot: dict) -> None:
        self.skills = copy.deepcopy(snapshot["skills"])
        self.version = snapshot["version"]

    def active(self) -> List[dict]:
        return [skill for skill in self.skills.values() if skill["status"] == "active"]

    def select(self, question: str, k: int = 2) -> List[dict]:
        scored = []
        for skill in self.active():
            score = token_jaccard(
                question,
                skill["trigger_condition"] + " " + skill["strategy_description"],
            )
            if score > 0:
                scored.append((score, skill["skill_id"], skill))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored[:k]]

    def propose(self, candidate: dict, *, question: str, answers: List[str], source_case_id: str) -> tuple:
        trigger = (candidate.get("trigger_condition") or "").strip()
        strategy = (candidate.get("strategy_description") or "").strip()
        guidance = (candidate.get("executable_guidance") or "").strip()
        blob = " ".join([trigger, strategy, guidance])
        if len(guidance) < 20:
            return None, "guidance too short"
        if question and question.strip() and question.strip() in blob:
            return None, "skill copies the source question"
        if leaks_gold_answer(blob, answers):
            return None, "skill contains a gold answer"
        skill_id = self.kind[:3] + "-" + sha256_text(trigger + "|" + guidance)[:16]
        skill = {
            "skill_id": skill_id,
            "kind": self.kind,
            "trigger_condition": trigger,
            "strategy_description": strategy,
            "executable_guidance": guidance,
            "source_case_ids": [source_case_id],
            "validation_cases": [],
            "effectiveness_stats": {"accepted": 0, "rejected": 0},
            "version": 1,
            "status": "candidate",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return skill, ""

    def stage(self, skill: dict) -> None:
        existing = self.skills.get(skill["skill_id"])
        if existing:
            if skill["source_case_ids"][0] not in existing["source_case_ids"]:
                existing["source_case_ids"].append(skill["source_case_ids"][0])
            existing["version"] += 1
            existing["status"] = "candidate"
        else:
            self.skills[skill["skill_id"]] = skill
        self.version += 1

    def commit(self, skill_id: str, case_id: str, helpful: bool) -> None:
        skill = self.skills[skill_id]
        skill["status"] = "active"
        skill["validation_cases"].append({"case_id": case_id, "helpful": helpful})
        key = "accepted" if helpful else "rejected"
        skill["effectiveness_stats"][key] = skill["effectiveness_stats"].get(key, 0) + 1
        self.version += 1

    def reject(self, skill_id: str) -> None:
        if skill_id in self.skills:
            self.skills[skill_id]["status"] = "rejected"
            self.version += 1
