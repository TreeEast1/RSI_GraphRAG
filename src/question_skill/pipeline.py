"""Question-time evolution: QDM, retrieval skill, answer skill."""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..adapters.hy4_client import HY4Client
from ..retrieval.reader import READER_TEMPLATE, format_context

QDM_SYSTEM = (
    "You decompose questions about a long conversation into independent retrieval sub-questions. "
    "Only decompose when the question genuinely requires chaining several facts. Output JSON."
)

QDM_PROMPT = """Question: {question}

Decide whether this question needs decomposition before retrieval.
- Single-hop factual questions ("What did X do?", "When did Y happen?") do NOT need decomposition.
- Multi-hop questions that chain two or more facts DO need it.
- Temporal questions about a sequence of events may need it.

Return JSON:
{{"needs_decomposition": true/false,
  "sub_questions": ["..."],
  "reason": "<=20 words"}}
"""

RETRIEVAL_SKILL_SYSTEM = (
    "You diagnose whether the retrieval results for a question have a missing link. You never "
    "invent facts. Output JSON."
)

RETRIEVAL_SKILL_PROMPT = """Question: {question}
Sub-questions answered so far: {subqs}

Retrieved session summaries (best first):
{evidence}

Task: identify at most {max_new} relation chains that are still missing to answer the question.
A missing chain means: the retrieved sessions mention some entities of the question but not the
relation that connects them (e.g. a person is found but not how they met another person).
If the evidence already covers the question, return an empty list.

Return JSON:
{{"missing": [{{"query": "<short retrieval query>", "why": "<=15 words"}}],
  "sufficient": true/false}}
"""

ANSWER_SKILL_SYSTEM = (
    "You verify and finalise answers using only the retrieved context. "
    "You must refuse when the context does not support an answer. Output JSON."
)

ANSWER_SKILL_PROMPT = """Context:
{context}

Question: {question}
Draft answer: {draft}

Check the draft against the context:
1. Is every claim in the draft supported by the context?
2. Is the answer in the best-supported form (exact name/date/number from the text)?
3. If the context has no support, the answer must be "No answer".
4. Prefer the shortest complete answer.

Return JSON:
{{"answer": "...", "answerable": true/false, "support": "<=25 word quote or empty"}}
"""


def decompose(client: HY4Client, question: str, tag: str = "qdm") -> Dict:
    try:
        out = client.complete_json(QDM_PROMPT.format(question=question),
                                   system=QDM_SYSTEM, max_tokens=512, tag=tag)
    except Exception:
        return {"needs_decomposition": False, "sub_questions": [], "reason": "parse_error"}
    subqs = [str(s).strip() for s in (out.get("sub_questions") or []) if str(s).strip()]
    needs = bool(out.get("needs_decomposition"))
    if not needs:
        subqs = []
    return {"needs_decomposition": needs, "sub_questions": subqs[:4],
            "reason": str(out.get("reason", ""))[:120]}


def judge_missing(client: HY4Client, question: str, subqs: Sequence[str],
                  ranked_docs: Sequence[Dict], max_new: int = 2,
                  tag: str = "ret_skill") -> Dict:
    def _short(doc: Dict, limit: int = 280) -> str:
        return re.sub(r"\s+", " ", doc["content"])[:limit]

    evidence = "\n".join(
        f"[{i+1}] {d['document_id']}: {_short(d)}"
        for i, d in enumerate(ranked_docs[:5])
    )
    prompt = RETRIEVAL_SKILL_PROMPT.format(
        question=question,
        subqs="; ".join(subqs) if subqs else "(none)",
        evidence=evidence or "(nothing retrieved)",
        max_new=max_new,
    )
    try:
        out = client.complete_json(prompt, system=RETRIEVAL_SKILL_SYSTEM,
                                   max_tokens=512, tag=tag)
    except Exception:
        return {"missing": [], "sufficient": True}
    missing = []
    for m in out.get("missing") or []:
        if isinstance(m, dict) and m.get("query"):
            missing.append({"query": str(m["query"]).strip(), "why": str(m.get("why", ""))[:80]})
    return {"missing": missing[:max_new], "sufficient": bool(out.get("sufficient", not missing))}


def rrf_merge(ranked_lists: List[List[Dict]], k: int = 60, top: int = 20) -> List[Dict]:
    """Reciprocal rank fusion over several ranked document lists."""
    scores: Dict[str, float] = {}
    by_id: Dict[str, Dict] = {}
    for lst in ranked_lists:
        for rank, doc in enumerate(lst):
            did = doc["document_id"]
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
            by_id.setdefault(did, doc)
    ordered = sorted(scores.items(), key=lambda x: -x[1])[:top]
    return [{**by_id[d], "score": s, "rrf_score": s} for d, s in ordered]


def answer_skill(client: HY4Client, question: str, docs: List[Dict], draft: str,
                 tag: str = "ans_skill") -> Dict:
    context = format_context(docs)
    prompt = ANSWER_SKILL_PROMPT.format(context=context, question=question, draft=draft)
    try:
        out = client.complete_json(prompt, system=ANSWER_SKILL_SYSTEM,
                                   max_tokens=512, tag=tag)
        ans = str(out.get("answer", "")).strip() or draft
        return {"answer": ans, "answerable": bool(out.get("answerable", True)),
                "support": str(out.get("support", ""))[:200], "revised": ans != draft}
    except Exception:
        return {"answer": draft, "answerable": True, "support": "", "revised": False}
