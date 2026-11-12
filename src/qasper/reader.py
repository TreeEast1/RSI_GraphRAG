"""Unified QASPER reader used by every method, plus context assembly.

The reader prompt is fixed across methods; only the retrieved context differs.
Context assembly always respects one token budget, including comprehension
units shown to the reader (they are never free extra text).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..dataset.locomo import num_tokens

READER_SYSTEM = (
    "You answer questions about one scientific paper using only the provided excerpts. "
    "You never use outside knowledge and never invent numbers or entities. "
    "If the excerpts do not contain the answer, you say so."
)

READER_TEMPLATE = """Excerpts from the paper:
{context}

Question: {question}

Rules for the final line:
- Start the final line with 'Answer: '.
- Give a short answer span or short phrase; for yes/no questions answer Yes or No.
- Use exact numbers and units from the excerpts.
- If the excerpts do not contain the answer, write 'Answer: Unanswerable'.
- No explanation after the answer.

Answer:"""


@dataclass
class ContextItem:
    kind: str          # paragraph | unit | summary
    id: str
    text: str
    n_tokens: int


def _item(kind: str, id_: str, text: str) -> ContextItem:
    return ContextItem(kind, id_, text, num_tokens(text))


def build_context(paragraphs: Dict[str, Dict], ordered_passage_ids: Sequence[str],
                  budget: int = 4096,
                  units: Optional[Sequence[Dict]] = None,
                  unit_budget: int = 0,
                  unit_label: str = "Insight") -> Dict:
    """Greedy paragraph selection under ``budget``; optional insight block first.

    ``units`` must be ranked with fields concise_text / source_passage_ids.
    Total tokens (units + paragraphs) never exceed ``budget``.
    """
    blocks: List[str] = []
    used = 0
    selected_units = []
    if units and unit_budget > 0:
        u_used = 0
        for u in units:
            text = f"[{unit_label} | {u['unit_id']}] {u['concise_text']}"
            n = num_tokens(text)
            if u_used + n > unit_budget:
                continue
            blocks.append(text)
            u_used += n
            selected_units.append(u["unit_id"])
        used += u_used
    selected = []
    for pid in ordered_passage_ids:
        p = paragraphs.get(pid)
        if not p:
            continue
        text = f"[{pid} | {p['section_name']}]\n{p['text']}"
        n = num_tokens(text)
        if used + n > budget:
            continue
        blocks.append(text)
        used += n
        selected.append(pid)
    return {"context": "\n\n".join(blocks), "tokens": used, "passages": selected,
            "units": selected_units}


class QasperReader:
    def __init__(self, client, tag: str = "qasper_reader"):
        self.client = client
        self.tag = tag

    def answer(self, question: str, context: str, max_tokens: int = 256) -> Dict:
        prompt = READER_TEMPLATE.format(context=context, question=question)
        out = self.client.chat(
            [{"role": "system", "content": READER_SYSTEM},
             {"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.0, tag=self.tag,
        )
        raw = out["content"] or ""
        return {"answer": parse_answer(raw), "raw": raw, "usage": out["usage"],
                "finish_reason": out["finish_reason"], "cached": out["cached"]}


def parse_answer(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"Answer:\s*(.*)", text, flags=re.S | re.I)
    ans = m.group(1) if m else text
    return ans.strip().split("\n")[0].strip()


def batch_answer(reader: QasperReader, items: Sequence[tuple], max_workers: int = 8,
                 max_tokens: int = 256) -> List[Dict]:
    from concurrent.futures import ThreadPoolExecutor

    def _one(it):
        q, ctx = it
        try:
            return reader.answer(q, ctx, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001
            return {"answer": "", "raw": "", "usage": {}, "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_one, items))
