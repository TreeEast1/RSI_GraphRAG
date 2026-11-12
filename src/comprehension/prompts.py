"""Prompts for read-time extraction and comprehension.

Fairness rules baked into these prompts
--------------------------------------
* C1 (session-level) and C2 (cluster-level) share the *same* schema and the same
  instructions. Only the number of sessions in the input block differs.
* C5 (full conversation) reuses the C2 prompt verbatim, with the whole
  conversation as one block.
* C3 adds a second, clearly separated comprehension step. It never changes the
  extraction prompt, so C3 - C2 isolates the comprehension layer.
"""
from __future__ import annotations

EXTRACTION_SYSTEM = (
    "You are a precise information-extraction engine building a knowledge graph from a long "
    "conversation between two speakers. You output only valid JSON."
)

EXTRACTION_INSTRUCTIONS = """Extract every verifiable fact stated in the conversation excerpts below.

Rules:
1. Each triple is [subject, predicate, object]. Subject and object must be short, canonical noun
   phrases (a person, place, organisation, event, object or concrete state).
2. Resolve pronouns ("I", "you", "she", "he") to the actual speaker name, using the speaker labels.
3. Normalise aliases: use one canonical name per person/place (e.g. always "Caroline", not "Caroline" and "Mel").
4. Predicates must be short verbs or verb phrases ("works_at", "met", "moved_to", "attended", "bought").
5. Include temporal facts explicitly: use the session date as the object when a date is stated or implied.
6. source_dia_ids must list the dialogue ids (e.g. "D3:12") that directly support the triple.
   Every triple must cite at least one dialogue id that appears in the excerpts.
7. evidence must be a short verbatim quote (max 25 words) from those turns.
8. Do not invent facts. Do not infer beyond what is stated. Do not output duplicate triples.
9. Do not summarise; output individual atomic facts.

Return JSON of exactly this shape:
{"triples": [
  {"subject": "Caroline", "predicate": "attended", "object": "LGBTQ support group",
   "source_dia_ids": ["D1:3"], "evidence": "I went to the LGBTQ support group yesterday",
   "derivation": "Explicit"}
]}
"""

COMPREHENSION_SYSTEM = (
    "You are a careful analyst building a cross-document knowledge graph from a long conversation. "
    "You only state relations that the cited dialogue turns actually support. You output only valid JSON."
)

COMPREHENSION_INSTRUCTIONS = """You are given several conversation sessions that were grouped together, and the atomic
facts already extracted from them. Your job is to find relations that the atomic facts do NOT
explicitly state, but which are jointly supported by facts from *different* sessions.

What counts as a useful new relation:
* Two facts about the same entity from different sessions that jointly establish a link
  (e.g. Alice starts a new job at Company X in S5, and meets Bob at Company X in S10
  -> "Alice" "colleague_of" "Bob" IS NOT allowed unless the text states they are colleagues).
* A change of state over time (moved_to city, then changed_job, then lives_in ...).
* Temporal ordering of events described in different sessions.
* Identity unification: two mentions across sessions that clearly refer to the same event or object.
* Causal or enabling links asserted or clearly implied by the dialogue text.

Hard requirements:
1. Every relation MUST cite at least two source_dia_ids from DIFFERENT sessions, or at least one
   dialogue id plus at least one supporting fact id from another session.
2. evidence must contain verbatim quotes (max 25 words each) from each cited session.
3. derivation must be "Composed" when combining explicit facts needs no world knowledge, and
   "Inferred" when common-sense or semantic inference is needed.
4. confidence is your calibrated probability (0-1) that the relation is true given the evidence.
5. Do NOT restate facts that are already in the atomic fact list. Do NOT produce synonyms or
   paraphrases of existing triples. Do NOT produce vague relations such as "related_to".
6. If you cannot find any cross-session relation that satisfies rule 1, return {"relations": []}.

Return JSON of exactly this shape:
{"relations": [
  {"subject": "Alice", "predicate": "moved_to", "object": "New York",
   "relation_type": "state_change", "source_dia_ids": ["D1:4", "D5:2"],
   "supporting_fact_ids": ["f12"],
   "evidence": ["I'm moving to New York next month", "Since I moved here I started at Acme"],
   "derivation": "Composed", "confidence": 0.9}
]}
"""

VERIFY_SYSTEM = (
    "You verify candidate relations for a knowledge graph. Be strict: reject anything that is not "
    "directly supported by the quoted dialogue turns. Output only valid JSON."
)

VERIFY_INSTRUCTIONS = """For each candidate relation, decide whether it is supported by the quoted dialogue turns.

Checks:
1. Do the cited dialogue ids exist and contain the quoted evidence?
2. Does the relation genuinely require more than one session, or is it restating a single fact?
3. Is it a synonym/paraphrase of an existing atomic fact (then reject)?
4. Does it contradict any existing fact?
5. Would knowing this relation plausibly help answer a question about this conversation?

Return JSON:
{"verdicts": [{"id": "<candidate id>", "keep": true, "reason": "<short reason>", "confidence": 0.0}]}
"""


def render_extraction_prompt(sessions_block: str) -> str:
    return (
        f"{EXTRACTION_INSTRUCTIONS}\n\n"
        f"Conversation excerpts:\n{sessions_block}\n\n"
        f"Return the JSON now."
    )


def render_comprehension_prompt(sessions_block: str, facts_block: str) -> str:
    return (
        f"{COMPREHENSION_INSTRUCTIONS}\n\n"
        f"Conversation excerpts:\n{sessions_block}\n\n"
        f"Atomic facts already extracted:\n{facts_block}\n\n"
        f"Return the JSON now."
    )


def render_verification_prompt(candidates_block: str) -> str:
    return f"{VERIFY_INSTRUCTIONS}\n\nCandidates:\n{candidates_block}\n\nReturn the JSON now."
