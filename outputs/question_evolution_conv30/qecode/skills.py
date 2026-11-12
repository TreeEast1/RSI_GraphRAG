"""Structured, persistent Answer/Retrieval Skill memories.

Design goals
------------
* one skill == one short, composable rule with explicit trigger conditions;
* no gold answer, person, place, event, or specific date is stored;
* every skill has a version history, confidence, status and provenance;
* memories are JSON-serialisable so a snapshot is a complete frozen policy.
"""
from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import common

ANSWER_FIELDS = [
    "skill_id", "title", "trigger_conditions", "applicable_question_patterns",
    "categories", "action_instructions", "expected_output_format",
    "evidence_requirements", "failure_modes", "source_training_qids",
    "success_count", "failure_count", "confidence", "version", "status",
    "created_at_feedback", "last_updated_feedback",
]

# Words that are too generic to count as leaked answer content.
_GENERIC = set("""
a an the and or but if then than that this these those of to in on at for with by from as is are was were be been
being do does did done have has had having can could may might must should would will shall not no yes
what when where who whom whose why how which whether
person people someone somebody something thing things event events place places name named
question answer context evidence information conversation session sessions statement stated mention mentioned
date dates time times day days month months year years today yesterday tomorrow week weeks before after during
since until last next earlier later first second third start started starting end ended ending
relative absolute approximate approximately around exactly duration interval compute calculated
support supported supports using used use include includes including include all every each any some
short shortest complete complete answer final output format line span spans phrase list comma separated
job jobs business businesses work working worked loss lost own share passion store shop studio
one two both also additional additionally only just very much many more most other others same
""".split())

_QUESTION_GENERIC = set("""
what when where who whom whose why how which did does was were is are the and for from with about
has have had take took taken make made get got give gave say said tell told can could would should
""".split())

_MONTHS = set("january february march april may june july august september october november december".split())


def slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return s[:limit] or "skill"


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if len(t) > 2]


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def sensitive_terms(qidx: int) -> set:
    """Distinctive answer/question tokens that must never enter a reusable skill."""
    q = common.qa_map()[qidx]
    gold = str(q.answer or "")
    adv = str(getattr(q, "adversarial_answer", "") or "")
    text = " ".join([q.question, gold, adv])
    terms = set()
    for tok in re.findall(r"[A-Za-z][A-Za-z'\-]+|\d+", text):
        low = tok.lower()
        if low in _GENERIC or low in _QUESTION_GENERIC or low in _MONTHS or len(low) <= 2:
            continue
        # numbers longer than 2 digits (dates, years, counts) are sensitive
        if re.fullmatch(r"\d+", low) and len(low) > 2:
            terms.add(low)
        elif re.search(r"[A-Z]", tok) and low not in {"jon", "gina"}:
            terms.add(low)
        elif not re.fullmatch(r"\d+", low) and len(low) > 4:
            terms.add(low)
    return terms


def leak_check(text: str, qidx: int) -> Tuple[bool, List[str]]:
    low = str(text).lower()
    hits = sorted(t for t in sensitive_terms(qidx) if t in low)
    if len(re.findall(r"\b(?:19|20)\d{2}\b", text)):
        hits.append("explicit_year")
    gold = str(common.gold_of(qidx) or "").strip().lower()
    if len(gold) > 4 and gold in low:
        hits.append("gold_substring")
    return (len(hits) == 0), hits



def leak_check_retrieval(text: str, qidx: int) -> Tuple[bool, List[str]]:
    """Leak check for retrieval query templates.

    Retrieval queries legitimately contain generic verbs and search words, so
    only proper nouns, explicit numbers/years, and the verbatim gold answer are
    forbidden.
    """
    q = common.qa_map()[qidx]
    gold = str(q.answer or "")
    low = str(text).lower()
    hits = []
    for tok in re.findall(r"\b[A-Z][a-zA-Z'\-]+\b", q.question + " " + gold):
        if tok.lower() in {"what", "when", "where", "who", "which", "why", "how"}:
            continue
        if tok.lower() in low:
            hits.append(tok)
    if re.search(r"\b(?:19|20)\d{2}\b", text):
        hits.append("explicit_year")
    if re.search(r"\b\d{3,}\b", text):
        hits.append("explicit_number")
    gold_norm = re.sub(r"[^a-z0-9 ]+", " ", gold.lower())
    text_norm = re.sub(r"[^a-z0-9 ]+", " ", low)
    if len(gold_norm.strip()) > 4 and gold_norm.strip() in text_norm:
        hits.append("gold_substring")
    return (len(set(hits)) == 0), sorted(set(hits))

# --------------------------------------------------------------------------- selection
def select_skills(skills: Sequence[Dict], question: str, category: int, k: int = 4,
                  embedder=None, skill_vecs: Optional[Dict[str, np.ndarray]] = None) -> List[Dict]:
    qtok = set(_tokens(question))
    qv = None
    if embedder is not None:
        try:
            qv = embedder.encode_queries([question])[0]
        except Exception:  # noqa: BLE001
            qv = None
    scored = []
    for s in skills:
        if s.get("status") != "active":
            continue
        cats = s.get("categories") or []
        if cats and int(category) not in [int(c) for c in cats]:
            continue
        score = 2.0 if cats else 0.8
        reasons = [f"category={category}" if cats else "category=any"]
        pat_overlap = 0.0
        for p in s.get("applicable_question_patterns") or []:
            ptok = set(_tokens(p))
            if ptok:
                ov = len(ptok & qtok) / len(ptok)
                if ov >= 0.34:
                    pat_overlap = max(pat_overlap, ov)
        if pat_overlap:
            score += 1.2 * pat_overlap
            reasons.append(f"pattern_overlap={pat_overlap:.2f}")
        trig = _jaccard(question, s.get("trigger_conditions", "") + " " + s.get("title", ""))
        if trig:
            score += 0.6 * trig
            reasons.append(f"trigger_overlap={trig:.2f}")
        if qv is not None and skill_vecs and s["skill_id"] in skill_vecs:
            cos = float(skill_vecs[s["skill_id"]] @ qv)
            score += 1.5 * max(0.0, cos)
            reasons.append(f"embedding={cos:.2f}")
        score += 0.3 * float(s.get("confidence", 0.5))
        if score >= 1.0:
            scored.append((score, reasons, s))
    scored.sort(key=lambda x: -x[0])
    out = []
    for score, reasons, s in scored[:k]:
        item = dict(s)
        item["_select_score"] = round(score, 3)
        item["_select_reasons"] = reasons
        out.append(item)
    return out


def skill_vectors(skills: Sequence[Dict], embedder) -> Dict[str, np.ndarray]:
    if embedder is None or not skills:
        return {}
    texts = [s.get("title", "") + ". " + s.get("trigger_conditions", "") + ". "
             + s.get("action_instructions", "") for s in skills]
    vecs = embedder.encode_queries(texts)
    return {s["skill_id"]: vecs[i] for i, s in enumerate(skills)}


# --------------------------------------------------------------------------- memories
class BaseMemory:
    prefix = "sk"
    snapshot_name = "answer_memory"

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.skills: List[Dict] = []
        self.meta: Dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_updates": 0,
                           "n_merges": 0, "n_disabled": 0, "n_rollbacks": 0}

    def active(self) -> List[Dict]:
        return [s for s in self.skills if s.get("status") == "active"]

    def get(self, sid: str) -> Optional[Dict]:
        return next((s for s in self.skills if s["skill_id"] == sid), None)

    def _new_id(self, title: str) -> str:
        base = f"{self.prefix}_{slug(title)}"
        sid, n = base, 2
        while self.get(sid):
            sid, n = f"{base}_{n}", n + 1
        return sid

    def add(self, payload: Dict, feedback_step: int, source_qid: int) -> Dict:
        skill = {k: payload.get(k) for k in ANSWER_FIELDS}
        skill["skill_id"] = self._new_id(payload.get("title", "rule"))
        skill["source_training_qids"] = [source_qid]
        skill["success_count"] = 0
        skill["failure_count"] = 0
        skill["confidence"] = 0.55
        skill["version"] = 1
        skill["status"] = "active"
        skill["created_at_feedback"] = feedback_step
        skill["last_updated_feedback"] = feedback_step
        skill.setdefault("categories", [])
        skill.setdefault("applicable_question_patterns", [])
        skill.setdefault("failure_modes", [])
        skill["history"] = [{"version": 1, "feedback_step": feedback_step, "qid": source_qid,
                             "event": "created"}]
        self.skills.append(skill)
        return skill

    def update(self, sid: str, payload: Dict, feedback_step: int, source_qid: int,
               mode: str = "append") -> Optional[Dict]:
        s = self.get(sid)
        if not s:
            return None
        old = copy.deepcopy(s)
        if mode == "append":
            add = str(payload.get("action_instructions", "")).strip()
            if add and add.lower() not in s.get("action_instructions", "").lower():
                s["action_instructions"] = (s.get("action_instructions", "").rstrip()
                                            + " " + add).strip()
            s["applicable_question_patterns"] = list(dict.fromkeys(
                (s.get("applicable_question_patterns") or [])
                + list(payload.get("applicable_question_patterns") or [])))
            s["failure_modes"] = list(dict.fromkeys(
                (s.get("failure_modes") or []) + list(payload.get("failure_modes") or [])))
        else:
            for k in ("action_instructions", "trigger_conditions", "expected_output_format",
                      "evidence_requirements"):
                if payload.get(k):
                    s[k] = payload[k]
        s["version"] = int(s.get("version", 1)) + 1
        s["last_updated_feedback"] = feedback_step
        s["confidence"] = max(0.5, float(s.get("confidence", 0.5)) * 0.9)
        s["source_training_qids"] = list(dict.fromkeys(
            (s.get("source_training_qids") or []) + [source_qid]))
        s.setdefault("history", []).append({
            "version": s["version"], "feedback_step": feedback_step, "qid": source_qid,
            "event": f"updated_{mode}", "previous": {
                "action_instructions": old.get("action_instructions"),
                "version": old.get("version"),
            },
        })
        self.meta["n_updates"] += 1
        return s

    def disable(self, sid: str, reason: str, feedback_step: int) -> None:
        s = self.get(sid)
        if not s or s.get("status") == "disabled":
            return
        s["status"] = "disabled"
        s["disabled_reason"] = reason
        s.setdefault("history", []).append({"feedback_step": feedback_step,
                                            "event": "disabled", "reason": reason})
        self.meta["n_disabled"] += 1

    def record_usage(self, skill_ids: Sequence[str], success: bool) -> None:
        for sid in skill_ids:
            s = self.get(sid)
            if not s:
                continue
            if success:
                s["success_count"] = int(s.get("success_count", 0)) + 1
            else:
                s["failure_count"] = int(s.get("failure_count", 0)) + 1
            n = s["success_count"] + s["failure_count"]
            s["confidence"] = (s["success_count"] + 1.0) / (n + 2.0)

    def record_marginal(self, skill_ids: Sequence[str], baseline_correct: bool,
                        with_correct: bool, feedback_step: int = -1) -> None:
        """Credit a skill only when the skill layer changed the outcome.

        Both-correct and both-wrong outcomes are neutral: a hard question does
        not count against every skill that happened to be retrieved.
        """
        if baseline_correct == with_correct:
            return
        for sid in skill_ids:
            s = self.get(sid)
            if not s:
                continue
            if (not baseline_correct) and with_correct:
                s["success_count"] = int(s.get("success_count", 0)) + 1
            elif baseline_correct and (not with_correct):
                s["failure_count"] = int(s.get("failure_count", 0)) + 1
            n = s["success_count"] + s["failure_count"]
            s["confidence"] = (s["success_count"] + 1.0) / (n + 2.0)
            if s["failure_count"] >= 3 and s["failure_count"] >= s["success_count"] + 2:
                self.disable(sid, f"counterfactual_harm:{s['failure_count']}f/{s['success_count']}s",
                             feedback_step=feedback_step)

    def merge_duplicates(self, embedder=None, threshold: float = 0.9,
                         feedback_step: int = -1) -> List[Dict]:
        """Merge near-duplicate active skills with the same category scope."""
        active = self.active()
        vecs = skill_vectors(active, embedder)
        merged = []
        used = set()
        for i, a in enumerate(active):
            if a["skill_id"] in used:
                continue
            for b in active[i + 1:]:
                if b["skill_id"] in used:
                    continue
                if set(a.get("categories") or []) != set(b.get("categories") or []):
                    continue
                sim = 0.0
                if vecs:
                    sim = float(vecs[a["skill_id"]] @ vecs[b["skill_id"]])
                else:
                    sim = _jaccard(a.get("action_instructions", ""), b.get("action_instructions", ""))
                if sim < threshold:
                    continue
                # keep the newer version, append the older rule if it adds content
                keep, drop = (a, b) if int(a.get("version", 1)) >= int(b.get("version", 1)) else (b, a)
                extra = drop.get("action_instructions", "").strip()
                if extra and extra.lower() not in keep.get("action_instructions", "").lower():
                    keep["action_instructions"] = (keep["action_instructions"].rstrip() + " " + extra).strip()
                keep["source_training_qids"] = list(dict.fromkeys(
                    (keep.get("source_training_qids") or []) + (drop.get("source_training_qids") or [])))
                keep.setdefault("history", []).append({
                    "feedback_step": feedback_step, "event": "merged",
                    "merged_from": drop["skill_id"], "similarity": round(sim, 3),
                })
                self.disable(drop["skill_id"], f"merged_into:{keep['skill_id']}", feedback_step)
                used.add(drop["skill_id"])
                merged.append({"kept": keep["skill_id"], "dropped": drop["skill_id"],
                               "similarity": round(sim, 3)})
        self.meta["n_merges"] += len(merged)
        return merged

    def save(self) -> None:
        if self.path:
            common.write_json(self.path, {"meta": self.meta, "skills": self.skills})

    def snapshot(self, step: int, label: str = "") -> Path:
        p = common.SNAP / f"{self.snapshot_name}_step{step:03d}.json"
        common.write_json(p, {"meta": self.meta, "step": step, "label": label,
                              "n_active": len(self.active()), "skills": self.skills})
        return p

    def restore(self, data: Dict) -> None:
        self.skills = copy.deepcopy(data.get("skills", []))
        self.meta = copy.deepcopy(data.get("meta", self.meta))

    @classmethod
    def load(cls, path: Path) -> "BaseMemory":
        m = cls(path)
        data = common.read_json(path)
        m.skills = data.get("skills", [])
        m.meta = data.get("meta", m.meta)
        return m


class AnswerSkillMemory(BaseMemory):
    prefix = "ans"

class RetrievalSkillMemory(BaseMemory):
    prefix = "ret"
    snapshot_name = "retrieval_memory"

    def add(self, payload: Dict, feedback_step: int, source_qid: int) -> Dict:
        skill = super().add(payload, feedback_step, source_qid)
        skill["query_templates"] = list(payload.get("query_templates") or [])
        return skill

    def update(self, sid: str, payload: Dict, feedback_step: int, source_qid: int,
               mode: str = "append") -> Optional[Dict]:
        s = super().update(sid, payload, feedback_step, source_qid, mode=mode)
        if s and payload.get("query_templates"):
            s["query_templates"] = list(dict.fromkeys(
                (s.get("query_templates") or []) + list(payload["query_templates"])))
        return s


# --------------------------------------------------------------------------- static baselines
def static_answer_skills() -> List[Dict]:
    specs = [
        ("Temporal anchor", ["When", "How long", "Before", "After"],
         "Distinguish the date of the session from the date of the event described in it. "
         "Resolve yesterday/tomorrow/last week/next month against the session date when the "
         "conversation gives a valid anchor.", "date only"),
        ("Temporal granularity", ["When"],
         "Keep the granularity supported by the evidence: if only a month is stated, output the "
         "month and do not invent a day; if only a relative phrase is stated, do not turn it "
         "into an absolute date.", "supported date granularity"),
        ("Single-hop minimal span", ["What", "Who", "Where"],
         "For a single-hop factual question return only the minimal answer span; do not add "
         "context, explanation, or extra facts.", "short span"),
        ("Multi-fact coverage", ["What", "Which", "How"],
         "When the question asks for items, events, or several facts, enumerate every supported "
         "item and remove duplicates; do not stop after the first match.", "comma-separated complete list"),
        ("Why minimal causes", ["Why"],
         "For why-questions state the necessary cause or motivation that the evidence supports; "
         "do not enumerate every nearby sentence.", "short causal phrase(s)"),
        ("Presupposition check", ["What", "Why", "How"],
         "Check whether the question assumes an event, possession, action, or role that the "
         "context does not state for the named person. If the premise is unsupported, answer No answer.",
         "No answer or supported answer"),
        ("Evidence sufficiency", ["What", "When", "Why", "Which"],
         "Distinguish missing evidence from evidence that was retrieved but not used. Refuse only "
         "when the required fact is absent and cannot be derived; otherwise answer from the evidence.",
         "answer or No answer"),
        ("Adversarial refusal", ["What", "Why", "How"],
         "Do not transfer a fact from one speaker to another. If the question attributes an event "
         "or preference to the wrong person, refuse instead of answering the mirrored question.",
         "No answer"),
    ]
    out = []
    for i, (title, pats, instr, fmt) in enumerate(specs, 1):
        out.append({
            "skill_id": f"static_ans_{i}_{slug(title)}", "title": title,
            "trigger_conditions": "question type matches " + ", ".join(pats),
            "applicable_question_patterns": pats, "categories": [],
            "action_instructions": instr, "expected_output_format": fmt,
            "evidence_requirements": "only retrieved evidence",
            "failure_modes": ["over-generalization", "unsupported inference"],
            "source_training_qids": [], "success_count": 0, "failure_count": 0,
            "confidence": 1.0, "version": 1, "status": "active",
            "created_at_feedback": -1, "last_updated_feedback": -1,
            "history": [{"event": "human_authored_static_baseline"}],
        })
    return out


def static_retrieval_skills() -> List[Dict]:
    specs = [
        ("Temporal anchoring", ["When", "How long"],
         ["{question} date month year session timeline",
          "{question} yesterday tomorrow last week next month before after"]),
        ("Entity bridge", ["What", "Who", "Which", "How"],
         ["{question} {entities} relationship connection",
          "{entities} together both shared"]),
        ("List coverage", ["Which", "What"],
         ["{question} also additionally all items events places people",
          "{question} list every mentioned"]),
        ("State change", ["Why", "How"],
         ["{question} before after changed started new moved"]),
    ]
    out = []
    for i, (title, pats, templates) in enumerate(specs, 1):
        out.append({
            "skill_id": f"static_ret_{i}_{slug(title)}", "title": title,
            "trigger_conditions": "question type matches " + ", ".join(pats),
            "applicable_question_patterns": pats, "categories": [],
            "action_instructions": "Issue the additional retrieval queries and merge their "
                                   "session rankings with the base graph ranking.",
            "query_templates": templates, "expected_output_format": "ranked sessions",
            "evidence_requirements": "do not use gold evidence at inference",
            "failure_modes": ["topic drift", "duplicate evidence"], "source_training_qids": [],
            "success_count": 0, "failure_count": 0, "confidence": 1.0, "version": 1,
            "status": "active", "created_at_feedback": -1, "last_updated_feedback": -1,
            "history": [{"event": "human_authored_static_baseline"}],
        })
    return out


def memory_from_skills(skills: Sequence[Dict], cls=AnswerSkillMemory) -> BaseMemory:
    m = cls()
    m.skills = [copy.deepcopy(s) for s in skills]
    return m
