"""Comprehension prompt candidates (COMP-P0..P5) for Task A.

P0 is the existing prompt in ``src.comprehension.prompts.COMPREHENSION_INSTRUCTIONS``
and is not re-generated here: the original C3 extraction is reused verbatim.

All new prompts keep the same JSON schema as the existing pipeline so that
``src.comprehension.extract`` and ``src.graph.variants`` can consume them.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..comprehension.prompts import COMPREHENSION_INSTRUCTIONS, COMPREHENSION_SYSTEM

SCHEMA = """Return JSON of exactly this shape:
{"relations": [
  {"subject": "Jon", "predicate": "attended_fair_to_promote", "object": "Jon's dance studio",
   "relation_type": "enabling_link", "source_dia_ids": ["D10:1", "D15:2"],
   "supporting_fact_ids": ["f12", "f40"],
   "evidence": ["Yesterday, I went to a fair to show off my studio",
                "Still working on opening a dance studio."],
   "derivation": "Composed", "confidence": 0.9}
]}
"""

COMMON_SYSTEM = (
    "You are a careful analyst building a cross-document knowledge graph from a long "
    "conversation. You only state relations that the cited dialogue turns actually support. "
    "You output only valid JSON."
)


def _base_block(instructions: str, constraints: str) -> str:
    return f"{instructions}\n\n{constraints}\n\n{SCHEMA}"


COMP_P1 = _base_block(
    """You are reading multiple conversation sessions to construct new, evidence-grounded
knowledge relations. Your task is NOT to repeat individual facts.

Identify relations that emerge only when evidence from at least two different sessions is
combined. A relation is valuable only if a reader of a single session could not state it
without the other session.

Prioritise:
- goal -> action -> outcome connections
- causal relations that the dialogue explicitly supports
- changes of state (before state vs after state)
- enabling links between two documented events
- identity resolution of one specific event or object mentioned twice
- event dependencies that are not already implied by the session dates

For every proposed relation:
1. Identify the supporting facts (use the f-ids from the atomic fact list when possible).
2. Cite dialogue ids from at least two different sessions.
3. Make sure the relation is not already expressed by one atomic fact.
4. Reject unsupported or redundant relations.

Do not invent motivations, causality, dates, or identity links. If no useful new relation
exists, return an empty list.""",
    """Hard requirements:
1. At least two source_dia_ids from DIFFERENT sessions.
2. evidence: one short verbatim quote (max 25 words) per cited session.
3. subject and object are short canonical noun phrases (<= 6 words): a person, place,
   organisation, event, object or concrete state. Do NOT put a whole clause in the subject
   or object.
4. predicate is a short verb or verb phrase (1-4 words, e.g. "led_to", "enabled",
   "opened_after", "became"). Do NOT use long compound predicates and do NOT encode
   "before"/"after" into the predicate unless you are explicitly reporting event order.
5. relation_type is one of: causal, enabling_link, state_change, goal_outcome,
   event_dependency, identity_unification, temporal_ordering. Use temporal_ordering only
   when the two events are separately described and their order is NOT already obvious
   from the session dates; it is the least valuable relation type here.
6. derivation is "Composed" when combining explicit facts needs no world knowledge and
   "Inferred" when common-sense inference is needed.
7. confidence is your calibrated probability (0-1).
8. Do NOT restate, paraphrase, or synonymise an existing atomic fact.
9. If nothing qualifies, return {"relations": []}.""")


COMP_P2 = _base_block(
    """Read the given conversation sessions as a connected history, not as independent
documents.

First identify the important entities, events, and states across the sessions. Then look for
relations that connect information distributed across DIFFERENT sessions. A relation is
worth adding only when the connection itself is not written in any single session.

Prefer bridges of these kinds:
- a plan stated in one session and its outcome in a later session
- an event in one session and its documented consequence in another
- the state of a business, project, or relationship before and after a major change
- two mentions that clearly refer to the same specific event or object
- evidence in one session that explains why a later action happened
- a sequence of separate facts that only together describe what happened

Create a relation only when the source evidence supports the connection itself. A shared
person name is not a relation. Do not repeat existing facts. Use specific, unambiguous
names for people, events, and objects. Return no relation when the evidence is insufficient.""",
    """Hard requirements:
1. Cite dialogue ids from at least two different sessions; every cited id must appear in
   the excerpts.
2. evidence: short verbatim quotes from each cited session (<= 25 words each).
3. subject/object: named people, organisations, places, events or concrete states (<= 6
   words). The predicate is a short verb phrase (1-4 words).
4. relation_type is one of: plan_outcome, event_consequence, state_change, identity_unification,
   causal, enabling_link, event_sequence. Do NOT use temporal_ordering as a fallback when you
   cannot find a real bridge.
5. Do NOT generate a relation merely because two statements mention the same person.
6. derivation is "Composed" or "Inferred"; confidence is 0-1.
7. Return {"relations": []} if no cross-session bridge is supported.""")


COMP_P3 = _base_block(
    """Analyse the conversation as a timeline of entities, events, goals, and changing states.

First identify the important people, businesses, projects, events, and goals. Then track
across sessions:
- initial goals
- planned actions
- ongoing activities
- completed outcomes
- changes in status
- changes in relationships
- changes in plans

For each important state transition:
1. Identify the earlier state.
2. Identify the later state.
3. Find the exact supporting sessions.
4. Decide whether a meaningful new relation can be derived from the change.
5. Reject relations that rely on unsupported assumptions about cause or intention.

Generate concise, evidence-grounded relations that explicitly describe the change. Prefer a
state transition over a vague ordering. Preserve event dates when they are stated.""",
    """Hard requirements:
1. Cite dialogue ids from at least two different sessions.
2. evidence: verbatim quotes from the supporting sessions (<= 25 words each).
3. subject/object are short canonical noun phrases (<= 6 words) naming the entity, project,
   event, or state. The predicate is a short verb phrase (1-4 words) describing the
   transition (e.g. "expanded_to", "changed_from", "became", "led_to", "recovered_from").
4. relation_type is one of: state_transition, goal_change, plan_outcome, status_change,
   relationship_change, causal. Prefer these over temporal_ordering.
5. If two events happened in a certain order but there is no evidence that one caused or
   enabled the other, report only the documented state change, not a causal claim.
6. Do not restate existing facts. confidence is 0-1.
7. Return {"relations": []} if there is no supported state change.""")


COMP_P4 = _base_block(
    """You are constructing a reusable comprehension graph for future questions that are not
known in advance. Read the provided conversation sessions and identify important relations
that are missing from independent fact extraction.

A useful relation should:
1. represent a distinct, evidence-grounded fact or cross-session inference;
2. connect specific entities, events, or states;
3. preserve enough semantic detail to be found by a future natural-language query;
4. point back to the original supporting sessions;
5. avoid vague predicates and generic links;
6. avoid repetition of existing facts.

Think about how a future question would be phrased. The subject and object should be names
that a question could plausibly mention (a person, business, event, place, project, or
state), and the predicate should state a specific relation between them. Do NOT hide the
whole answer inside a long predicate.

Do not invent relations to increase graph size. Do not use any question-answer pairs or gold
labels (none are provided). If the context does not support a valuable new relation, return
an empty list.""",
    """Hard requirements:
1. At least two source_dia_ids from DIFFERENT sessions.
2. subject/object: short, specific, human-readable noun phrases (<= 6 words). Prefer names
   that already occur in the atomic fact list, so the relation attaches to existing graph
   entities.
3. predicate: a short, specific verb phrase (1-4 words), lower-case with underscores
   (e.g. "promoted", "enabled_opening_of", "expanded_into", "resulted_in"). Avoid
   "related_to", "associated_with", and any predicate longer than 4 words.
4. evidence: short verbatim quote (<= 25 words) per cited session.
5. relation_type is one of: causal, enabling_link, state_change, goal_outcome,
   event_dependency, identity_unification, temporal_ordering. Use temporal_ordering only
   when order is the actual new information and is not implied by session dates.
6. derivation is "Composed" or "Inferred"; confidence is 0-1.
7. Do NOT output a relation whose text is already an atomic fact, and do not output two
   relations that say the same thing with different words.
8. Return {"relations": []} if nothing qualifies.""")


# P5 generator: P4 with an explicit request for a slightly larger candidate pool.
COMP_P5_GEN = _base_block(
    """You are constructing a reusable comprehension graph for future questions that are not
known in advance. Read the provided conversation sessions and propose candidate relations
that are missing from independent fact extraction.

Generate up to 12 candidates per reading group. Candidates are cheap; a separate verification
step will reject weak ones. Still, every candidate must be grounded in the supplied sessions.

A useful candidate:
1. combines evidence from at least two different sessions;
2. connects specific entities, events, projects, or states;
3. uses a short subject/object noun phrase and a short specific verb predicate;
4. is not already stated by an atomic fact.

Prefer causal links, enabling links, goal -> outcome chains, state changes, and identity
unification. Use temporal_ordering only when the order is the actual new information.

Do not invent facts. Do not use question-answer pairs or gold labels.""",
    """Hard requirements:
1. At least two source_dia_ids from DIFFERENT sessions.
2. subject/object: short specific noun phrases (<= 6 words), preferably names already in the
   atomic fact list.
3. predicate: short specific verb phrase (1-4 words), lower-case with underscores.
4. evidence: short verbatim quote per cited session.
5. relation_type: causal, enabling_link, state_change, goal_outcome, event_dependency,
   identity_unification, or temporal_ordering.
6. derivation is "Composed" or "Inferred"; confidence is 0-1.
7. Return {"relations": []} if nothing qualifies.""")


VERIFY_SYSTEM = (
    "You verify candidate comprehension relations for a knowledge graph. Reject only "
    "relations that are unsupported, redundant, ambiguous, or likely to mislead retrieval. "
    "Output only valid JSON."
)

VERIFY_INSTRUCTIONS = """Evaluate every proposed comprehension relation using only the source
conversation evidence and the existing atomic facts.

For each candidate assess:
1. Evidence support: do the cited dialogue ids contain the quoted evidence?
2. Cross-session necessity: is the relation actually supported by combining two sessions,
   or does one session already state it?
3. Factual correctness: does the relation misstate who/what/when?
4. Novelty: is it a paraphrase of an existing atomic fact?
5. Specificity: are subject, predicate, and object short and unambiguous?
6. Source attribution: are the cited sessions the right ones?
7. Retrieval risk: could this relation create a misleading graph connection?

Accept a relation when it is clearly supported by explicit facts, even if it is simple.
Reject when it is unsupported, redundant, ambiguous, or misleading. Do not reject merely
because the wording is not ideal.

Return JSON:
{"verdicts": [{"id": "c0", "keep": true, "reason": "<short reason>",
               "evidence_ok": true, "cross_session": true, "novel": true,
               "relation_type": "<type>"}]}
"""


@dataclass(frozen=True)
class CompPrompt:
    pid: str
    name: str
    instructions: str
    system: str = COMMON_SYSTEM


COMP_P0 = CompPrompt("COMP-P0", "Original comprehension prompt", COMPREHENSION_INSTRUCTIONS,
                     COMPREHENSION_SYSTEM)
COMP_P1_SPEC = CompPrompt("COMP-P1", "Evidence-grounded composition", COMP_P1)
COMP_P2_SPEC = CompPrompt("COMP-P2", "Cross-session bridge discovery", COMP_P2)
COMP_P3_SPEC = CompPrompt("COMP-P3", "Event and state tracking", COMP_P3)
COMP_P4_SPEC = CompPrompt("COMP-P4", "Retrieval-oriented comprehension", COMP_P4)
COMP_P5_SPEC = CompPrompt("COMP-P5", "Generate then verify", COMP_P5_GEN)

ALL = [COMP_P0, COMP_P1_SPEC, COMP_P2_SPEC, COMP_P3_SPEC, COMP_P4_SPEC, COMP_P5_SPEC]


def get(pid: str) -> CompPrompt:
    for p in ALL:
        if p.pid == pid:
            return p
    raise KeyError(pid)


def render(instructions: str, sessions_block: str, facts_block: str) -> str:
    return (f"{instructions}\n\nConversation excerpts:\n{sessions_block}\n\n"
            f"Atomic facts already extracted:\n{facts_block}\n\nReturn the JSON now.")
