"""Versioned prompts for DetectiveQA read-time and question-time experiments.

Every prompt asks for evidence ids that are visible in the supplied context.  No
prompt receives annotator reasoning or clue positions.  The question-time memory
writer may see the outcome of an attempt, but its output must be supported by
retrieved story text and is later verified without the question or answer.
"""
from __future__ import annotations

PROMPT_VERSION = "detectiveqa-v2"

EVIDENCE_SYSTEM = (
    "You read excerpts from a detective novel. Use only the excerpts. Treat dialogue as a claim by "
    "its speaker, not automatically as a fact. Cite chunk ids exactly. Output valid JSON only."
)

JOINT_OPENIE = """Extract atomic facts from this packet of story excerpts.

Rules:
1. Each item must be explicitly stated by one cited chunk.
2. Resolve pronouns only when the excerpt makes the referent clear.
3. Keep claims, lies, suspicions and observations attributed to the speaker or observer.
4. Use short canonical noun phrases and short verb predicates.
5. Do not add cross-chunk conclusions.
6. Return at most 40 facts. Prefer distinct events and entity relations that matter for later
   detective questions; do not list every sentence.

Return:
{{"facts": [{{"subject": "...", "predicate": "...", "object": "...",
  "source_chunk_ids": ["dq-..."], "evidence": "short quote"}}]}}

Packet:
{packet}
"""

COMPREHENSION = """Find comprehension relations in this packet of detective-novel excerpts.

A comprehension relation is not an atomic fact. It is a short relation that becomes justified only
after combining evidence from at least two cited chunks. Useful relation types:
- alias_or_identity: two names, descriptions or objects refer to the same thing
- alibi_or_opportunity: evidence fixes who could or could not be present
- motive_or_interest: evidence connects a person to a concrete gain, fear, conflict or secret
- method_or_capability: evidence connects a person or object to how an act could be done
- contradiction_or_deception: one claim conflicts with another observation or claim
- timeline_or_causal: one event constrains, enables or explains another event
- state_change: a person, object, relationship or belief changes across the story

Requirements:
1. Cite at least two different chunk ids. Every cited chunk must be necessary.
2. Give one short quote from every cited chunk.
3. Do not output a fact that one chunk already states.
4. Do not name a culprit, motive or method unless the cited text makes that conclusion strong.
5. Use "suspected_", "claims_" or "contradicts_" predicates for uncertain or disputed claims.
6. subject and object must be short names or noun phrases; predicate must be 1-4 words.
7. If no relation passes all checks, return an empty list.

Atomic facts already in this packet:
{facts}

Return:
{{"relations": [{{"subject": "...", "predicate": "...", "object": "...",
  "relation_type": "alias_or_identity|alibi_or_opportunity|motive_or_interest|method_or_capability|contradiction_or_deception|timeline_or_causal|state_change",
  "source_chunk_ids": ["dq-...", "dq-..."],
  "evidence": [{{"chunk_id": "dq-...", "quote": "..."}}],
  "why_cross_chunk": "<=25 words", "status": "supported|claimed|suspected", "confidence": 0.0}}]}}

Packet:
{packet}
"""

VERIFY_RELATIONS = """Verify candidate comprehension relations against the cited excerpts only.

Keep a candidate only if all checks pass:
1. Every quote appears in its cited chunk.
2. The relation follows from the cited chunks, with uncertainty labelled correctly.
3. At least two cited chunks are necessary.
4. It is not a paraphrase of one atomic fact.
5. Subject, predicate and object are specific enough to be useful later.

Return:
{{"verdicts": [{{"id": "c0", "keep": true, "fixed_status": "supported|claimed|suspected",
  "reason": "<=20 words"}}]}}

Candidates:
{candidates}

Excerpts:
{packet}
"""

READER_SYSTEM = (
    "You answer a multiple-choice question about the part of a detective novel shown below. "
    "Use only the excerpts, not memory of the book. Distinguish facts from claims and suspicions. "
    "Output valid JSON only."
)

# Diagnostic upper bound on parameter memory.  The strict READER_SYSTEM forbids
# using prior knowledge; this variant explicitly permits it when no excerpts are
# supplied.  It is never used as a retrieval/QA method.
MEMORY_SYSTEM = (
    "You answer a multiple-choice question about a detective novel. No story excerpts are provided. "
    "If you recognize the novel, use your prior knowledge; otherwise choose the most plausible option. "
    "Do not invent quotations or chunk ids. Output valid JSON only."
)

READER = """Story excerpts, ordered by story position:
{context}

Question: {question}
Options:
{options}

First identify the decisive evidence ids. Then choose the option best supported by the excerpts.
If evidence is incomplete, choose the most supported option and set insufficient_evidence=true.

Return:
{{"evidence_ids": ["dq-..."], "reasoning": "<=60 words", "answer": "A|B|C|D",
  "insufficient_evidence": false}}
"""

COMPREHENSION_V3 = """Find retrieval-useful comprehension relations in these detective-novel excerpts.

Goal: write relations that would help a later reader find and connect evidence across chunks.
Each relation must combine at least two different cited chunks. Do not merely restate one atomic fact.

Allowed relation_type values:
alias_or_identity, alibi_or_opportunity, motive_or_interest, method_or_capability,
contradiction_or_deception, timeline_or_causal, state_change.

Rules:
1. subject and object are concrete names, objects, events, places or claims.
2. predicate is a specific 1-5 word verb phrase. Never use a relation_type name as predicate.
   Bad: "Mary | motive_or_interest | money". Good: "Mary | stands to inherit | uncle's estate".
3. Cite every necessary chunk and give one short exact quote from each cited chunk.
4. status: supported = narrated or independently established; claimed = a character says it;
   suspected = plausible link or name variant that is not confirmed.
5. Name variants may be alias_or_identity with status suspected when context strongly matches.
6. Prefer relations involving people, objects, times, locations, lies, secrets or causal links.
7. Do not identify a culprit unless the cited text explicitly makes that conclusion.
8. Return at most 12 relations; return an empty list if none are useful.

Atomic facts already extracted from the same chunks:
{facts}

Return:
{{"relations": [{{"subject": "...", "predicate": "...", "object": "...",
  "relation_type": "...", "source_chunk_ids": ["dq-...", "dq-..."],
  "evidence": [{{"chunk_id": "dq-...", "quote": "..."}}],
  "why_cross_chunk": "<=25 words", "status": "supported|claimed|suspected",
  "retrieval_cues": ["2-6 short phrases"], "confidence": 0.0}}]}}

Packet:
{packet}
"""

VERIFY_RELATIONS_V3 = """Audit candidate comprehension relations.

The quotes have already been mechanically verified as present in their cited chunks. Do not reject
a candidate because of punctuation, smart quotes or a small paraphrase in your memory of the text.

Judge only:
1. semantic_support: does the relation follow from the cited excerpts?
2. cross_chunk_necessary: are at least two cited chunks needed?
3. specificity: is the triple specific enough to retrieve useful evidence later?
4. status: choose supported, claimed or suspected. If the relation is true but status is wrong,
   keep it and correct fixed_status.
5. alias_or_identity: keep plausible translation variants as suspected; keep a speaker's statement
   as claimed; reject if names clearly refer to different entities.

Return one verdict for every candidate id:
{{"verdicts": [{{"id": "c0", "keep": true, "fixed_status": "supported|claimed|suspected",
  "semantic_support": "strong|partial|none", "cross_chunk_necessary": true,
  "specificity": "high|medium|low", "reason": "<=20 words"}}]}}

Candidates:
{candidates}

Excerpts:
{packet}
"""

CASE_JUDGE_SYSTEM = (
    "You are an evaluation auditor for detective-novel QA. You may see the gold answer and annotated "
    "clues. Use them only to diagnose failures. Output valid JSON only."
)

CASE_JUDGE = """Diagnose this question across several context conditions.

Question: {question}
Options:
{options}
Gold answer: {gold_answer}
Cutoff position ratio: {cutoff_ratio:.3f}

Annotated clue excerpts:
{gold_context}

Recent prefix excerpts:
{recent_context}

Condition outputs:
{conditions}

Classify the item and each failed or surprising condition. Labels:
- reveal_before_cutoff: recent text explicitly gives or nearly gives the answer
- gold_incomplete: annotated clues omit decisive evidence
- gold_ambiguous_or_wrong: gold answer or option is ambiguous/questionable
- retrieval_miss: needed evidence absent from that condition
- partial_evidence: some but not enough evidence present
- reader_reasoning_error: enough evidence present but reader chose wrong
- entity_alias_error: names/translation variants confuse the reader
- option_bias_or_memory: answer likely comes from option priors or memory
- correct_by_shortcut: correct, but not from intended evidence
- correct_supported: correct with adequate evidence

Return:
{{"item_labels": ["..."], "answer_evidence_location": "recent|retrieved|gold|elsewhere|unclear",
  "explicit_reveal_before_cutoff": true, "gold_sufficient": true,
  "conditions": {{"name": {{"label": "...", "reason": "<=35 words"}}}},
  "summary": "<=60 words"}}
"""

DIAGNOSE_SYSTEM = (
    "You audit one failed retrieval-augmented reasoning trace. Diagnose from the trace and "
    "retrieved story text. Never turn the answer option into memory. Output valid JSON only."
)

DIAGNOSE = """Question: {question}
Options:
{options}
System answer: {system_answer}
Feedback: {feedback}

Retrieval queries:
{queries}

Retrieved excerpts:
{context}

Reader trace:
{trace}

Choose one failure type:
- knowledge_gap: the needed connection is spread across visible text and is missing from memory
- retrieval_gap: relevant evidence probably exists but the queries or graph seeds missed it
- reasoning_gap: the decisive evidence is already present but was combined incorrectly
- unclear: the trace cannot support a confident diagnosis

Propose at most one update:
- comprehension_memory: a cross-chunk relation that cites only retrieved chunk ids
- retrieval_skill: a reusable query-planning rule with no novel-specific names or answer content
- reasoning_skill: a reusable verification rule with no novel-specific names or answer content
- none

The update must still be true if the question is deleted.

Return:
{{"failure_type": "knowledge_gap|retrieval_gap|reasoning_gap|unclear",
  "diagnosis": "<=40 words", "update_type": "comprehension_memory|retrieval_skill|reasoning_skill|none",
  "memory": null or {{"subject": "...", "predicate": "...", "object": "...", "relation_type": "...",
    "source_chunk_ids": ["dq-...", "dq-..."], "evidence": [{{"chunk_id": "dq-...", "quote": "..."}}],
    "status": "supported|claimed|suspected"}},
  "skill": null or {{"title": "...", "trigger": "...", "procedure": "<=60 words",
    "checks": ["..."]}}}}
"""

REWRITE_QUERY = """Write up to three short retrieval queries for the question below.
Use the skill only if its trigger matches. Do not answer the question.

Skill:
{skill}

Question: {question}
Options:
{options}

Return: {{"queries": ["..."]}}
"""
