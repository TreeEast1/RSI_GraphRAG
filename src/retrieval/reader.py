"""Unified QA reader.

Every method (dense RAG, HippoRAG2, comprehension graphs, full context) is read
by this exact function with the same prompt, so QA differences cannot be
explained by reader prompt drift.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..adapters.hy4_client import HY4Client, parse_json_loose

READER_SYSTEM = (
    "You are a careful reading-comprehension assistant. You answer questions about a long "
    "conversation between two speakers using only the retrieved context. "
    "Use the dates and session numbers to reason about time when needed; resolve relative "
    "expressions such as 'yesterday' or 'last week' against the session date. "
    "If the context truly does not contain the answer, reply exactly: No answer"
)

READER_TEMPLATE = """Retrieved context:
{context}

Question: {question}

Format rules for the final line:
- It must start with 'Answer: ' and contain only the answer span itself.
- No explanation, no full sentence, no quotes, no trailing punctuation.
- For a date, give it exactly as written in the context.
- If the answer is not in the context, write 'Answer: No answer'.

Thought: think briefly about where the answer is in the context.
Answer:"""

SHORT_ANSWER_SUFFIX = ""


def format_context(docs: List[Dict], header_key: str = "document_id") -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        header = doc.get("header") or doc.get(header_key, f"doc{i}")
        parts.append(f"[{header}]\n{doc.get('content', '')}")
    return "\n\n".join(parts)


def _parse_answer(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"Answer:\s*(.*)", text, flags=re.S | re.I)
    ans = m.group(1) if m else text
    ans = ans.strip().split("\n")[0].strip()
    return ans


class Reader:
    def __init__(self, client: HY4Client, tag: str = "reader"):
        self.client = client
        self.tag = tag

    def answer(self, question: str, docs: List[Dict], max_tokens: int = 256) -> Dict:
        context = format_context(docs)
        prompt = READER_TEMPLATE.format(context=context, question=question)
        out = self.client.chat(
            [
                {"role": "system", "content": READER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            tag=self.tag,
        )
        raw = out["content"]
        return {
            "answer": _parse_answer(raw),
            "raw": raw,
            "usage": out["usage"],
            "finish_reason": out["finish_reason"],
            "cached": out["cached"],
        }

    def answer_full_context(self, question: str, context_text: str, max_tokens: int = 256) -> Dict:
        prompt = READER_TEMPLATE.format(context=context_text, question=question)
        out = self.client.chat(
            [
                {"role": "system", "content": READER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            tag="reader_fullctx",
        )
        raw = out["content"]
        return {
            "answer": _parse_answer(raw),
            "raw": raw,
            "usage": out["usage"],
            "finish_reason": out["finish_reason"],
            "cached": out["cached"],
        }

    # ------------------------------------------------------------ batching
    def answer_batch(self, items: List[tuple], max_workers: int = 16, max_tokens: int = 256) -> List[Dict]:
        """items: list of (question, docs). Returns results in order."""
        from concurrent.futures import ThreadPoolExecutor

        def _one(it):
            q, docs = it
            return self.answer(q, docs, max_tokens=max_tokens)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(_one, items))

    def answer_full_context_batch(self, items: List[tuple], max_workers: int = 16,
                                  max_tokens: int = 256) -> List[Dict]:
        from concurrent.futures import ThreadPoolExecutor

        def _one(it):
            q, ctx = it
            return self.answer_full_context(q, ctx, max_tokens=max_tokens)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(_one, items))
