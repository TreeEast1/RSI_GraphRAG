"""Candidate Reader / QA prompts (Task B).

Every prompt keeps the same hard interface as the baseline:
  * the document block and the question are inserted by ``Reader``-style rendering
  * the final line must start with ``Answer:`` so the shared parser is unchanged

P0 is the project baseline verbatim (imported from ``src.retrieval.reader``).
P1-P5 are the five designs from the experiment brief.  P6 is an
official-LoC0Mo-style reference (short phrase, exact words from context, generic
date hint instead of the dataset's category label).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..retrieval.reader import READER_SYSTEM, READER_TEMPLATE


@dataclass(frozen=True)
class ReaderPrompt:
    pid: str
    name: str
    system: str
    template: str
    notes: str = ""


P0 = ReaderPrompt(
    pid="P0",
    name="Baseline (current project prompt)",
    system=READER_SYSTEM,
    template=READER_TEMPLATE,
    notes="exact copy of src/retrieval/reader.py; saved baseline outputs are reused, no re-call",
)

P1 = ReaderPrompt(
    pid="P1",
    name="Concise but complete",
    system=(
        "You answer questions about a long conversation between two people. "
        "Use only the provided conversation evidence."
    ),
    template="""Conversation evidence:
{context}

Question: {question}

Answer the question accurately and completely.

Rules:
1. Identify exactly what the question asks.
2. Include all necessary facts supported by the evidence.
3. Be concise, but never omit required information just to make the answer shorter.
4. For list questions, include every supported item needed to answer.
5. For why questions, include the relevant causes or motivations.
6. For date questions, give the precise date when it is supported.
7. Preserve names, dates, places, and other factual details.
8. Do not invent or assume missing information.
9. If the evidence is insufficient, answer "No answer".

Write exactly one line of the form:
Answer: <answer>
""",
    notes="length constraint is softened: complete before concise",
)

P2 = ReaderPrompt(
    pid="P2",
    name="Integrate evidence before answering",
    system=(
        "You are a careful reading-comprehension assistant. You answer from retrieved "
        "conversation evidence and never invent facts."
    ),
    template="""Retrieved conversation sessions:
{context}

Question: {question}

Before answering, silently do the following:
1. Identify the people, events, and facts relevant to the question.
2. Collect evidence across all relevant sessions, not only the first one that looks related.
3. Resolve references to the same person, event, or object across sessions.
4. Combine multiple facts when the question requires more than one.
5. Check that the evidence supports the complete answer.

Answer requirements:
- Include every necessary part of the answer.
- Do not omit supporting causes, events, or entities.
- Prefer exact information from the conversation.
- Do not use unsupported assumptions.
- If the information is insufficient, answer "No answer".

Do not output your reasoning. Write exactly one line:
Answer: <answer>
""",
    notes="forces cross-session aggregation internally, no visible chain of thought",
)

P3 = ReaderPrompt(
    pid="P3",
    name="Temporal and event-state sensitive",
    system=(
        "You answer questions about conversations that take place across multiple dates "
        "and sessions. Each session header gives the date of that conversation. Use only "
        "the provided evidence and never fabricate a date or duration."
    ),
    template="""Conversation evidence (each session starts with its date):
{context}

Question: {question}

Rules:
1. Distinguish the date of a conversation from the date of an event mentioned in it.
2. Resolve relative expressions such as yesterday, last week, next month, or six months
   later only when a valid temporal anchor is available.
3. For duration questions, identify the start and end events before computing the interval.
4. For event-order questions, distinguish planned, ongoing, completed, and cancelled events.
5. If several dates appear, choose the one that corresponds to the event asked about.
6. Never fabricate a date or duration.
7. For non-temporal questions, answer normally and completely.

Write exactly one line:
Answer: <answer>
""",
    notes="explicit session-date anchoring; duration arithmetic",
)

P4 = ReaderPrompt(
    pid="P4",
    name="Evidence-sufficiency and refusal calibration",
    system=(
        "You answer questions using retrieved conversation evidence. You may combine "
        "explicitly supported facts across sessions, including event order and simple "
        "date arithmetic, but you never invent facts."
    ),
    template="""Conversation evidence:
{context}

Question: {question}

First decide silently whether the available evidence is sufficient.

- You may combine explicitly supported facts across sessions, including event order and
  simple date calculations.
- Do not invent facts, dates, motivations, or events that the evidence does not support.
- If the necessary evidence is present, answer the question instead of refusing.
- If a required fact is missing and cannot be reliably derived, answer "No answer".

Keep the answer concise but complete. Write exactly one line:
Answer: <answer>
""",
    notes="lowers over-refusal; adds explicit permission for supported cross-session derivation",
)

P5 = ReaderPrompt(
    pid="P5",
    name="Adapt answer form to the question",
    system=(
        "You answer questions based on retrieved long-term conversation records. "
        "You use only the provided evidence."
    ),
    template="""Conversation evidence:
{context}

Question: {question}

Decide what kind of information the question requires, then give the shortest complete answer.

- Who: the correct person or people.
- When: the exact supported date or time.
- How long: the duration computed from supported dates.
- Where: the relevant location or locations.
- Which events/items: all relevant items, enumerated.
- Why: all necessary causes or motivations.
- How: the relevant outcome or process.
- Other: the requested fact.

Use evidence from multiple sessions when necessary.
Preserve distinctions between different people, dates, and events.
Do not omit required facts just to shorten the answer.
Do not add unnecessary explanation.
If the information is insufficient, answer "No answer".

Write exactly one line:
Answer: <answer>
""",
    notes="question-type-conditioned completeness; question text only, no gold category",
)

P6 = ReaderPrompt(
    pid="P6",
    name="Official-LoCoMo-style short phrase",
    system=(
        "You answer questions based on the provided conversation context. "
        "Answer with exact words from the context whenever possible."
    ),
    template="""{context}

Based on the above context, write an answer in the form of a short phrase for the following question.

Question: {question}

If the question asks for a date, use the date of the conversation to answer with an approximate date.
If the context does not contain the answer, write "Not mentioned in the conversation".

Short answer:""",
    notes="mirrors official QA_PROMPT text; generic date hint instead of gold category",
)


ALL_P0_P6 = [P0, P1, P2, P3, P4, P5, P6]
NEW_PROMPTS = [P1, P2, P3, P4, P5, P6]

# P7 is a dev-informed synthesis (added after P0-P6 dev results): keep the
# baseline's short-span/refusal behaviour, add the temporal rules that fixed
# temporal over-refusal in P3, and make list questions enumerate every item.
P7 = ReaderPrompt(
    pid="P7",
    name="Short-span complete (synthesis of P0/P3/P6)",
    system=(
        "You are a careful reading-comprehension assistant answering questions about a long "
        "conversation between two speakers. Use only the retrieved context and answer with "
        "exact words from the context whenever possible. Each session header gives the date of "
        "that conversation. If the context truly does not contain the answer, reply exactly: "
        "No answer"
    ),
    template="""Retrieved context:
{context}

Question: {question}

Rules:
1. Give only the answer span or a comma-separated list of spans, using exact words from the
   context. Do not write a sentence and do not explain your reasoning.
2. If the question asks which items, events, people, or places, list every supported item
   separated by commas.
3. If the question asks why, give all supported reasons as a short phrase or as
   comma-separated short phrases.
4. If the question asks when, output only the date, with no surrounding words such as
   "around", "approximately", or "in the conversation". The date may be the session date when
   the event is described as "yesterday", "last week", or "next month" relative to it.
5. If the question asks how long, output only the duration computed from the supported start
   and end dates.
6. Combine facts from different sessions when needed. Do not refuse when the required
   evidence is present; only write "No answer" if it is genuinely absent from the context
   and cannot be simply derived from it.
7. Never invent facts or dates.

Answer:""",
    notes="dev-informed synthesis; P7 was added after seeing P0-P6 on the 40 dev questions",
)


# P8 = P7 + an explicit presupposition check (dev showed P7 losing one
# adversarial question because it answered a question whose premise the context
# does not support).
P8 = ReaderPrompt(
    pid="P8",
    name="Short-span complete + presupposition check",
    system=P7.system,
    template=P7.template.replace(
        """6. Combine facts from different sessions when needed. Do not refuse when the required
   evidence is present; only write "No answer" if it is genuinely absent from the context
   and cannot be simply derived from it.
7. Never invent facts or dates.""",
        """6. Combine facts from different sessions when needed. Do not refuse when the required
   evidence is present; only write "No answer" if it is genuinely absent from the context
   and cannot be simply derived from it.
7. Check the premise of the question. If the question assumes that a person did, attended,
   participated in, or possesses something that the context does not state, write "No answer"
   instead of answering a different question.
8. Never invent facts or dates."""),
    notes="P7 plus a generic presupposition check for adversarial questions",
)

ALL_PROMPTS = [P0, P1, P2, P3, P4, P5, P6, P7, P8]


def get(pid: str) -> ReaderPrompt:
    for p in ALL_PROMPTS:
        if p.pid == pid:
            return p
    raise KeyError(pid)

