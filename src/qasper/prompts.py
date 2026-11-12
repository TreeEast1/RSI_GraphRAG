"""Prompts for QASPER read-time extraction, comprehension, verification, summary.

Leakage rules
-------------
Inputs are the paper title, paragraph text and atomic facts only. Questions,
gold answers and gold evidence never enter any prompt. Examples are schematic
placeholders, never questions or facts from the evaluation set.
"""
from __future__ import annotations

ATOMIC_SYSTEM = (
    "You extract atomic facts from scientific paper paragraphs. You never infer beyond the text. "
    "You output only valid JSON."
)

ATOMIC_INSTRUCTIONS = """You are given numbered paragraphs from one scientific paper, plus its title.

Extract the atomic facts explicitly stated in the paragraphs.

Definitions and rules:
1. An atomic fact is one verifiable statement {"subject","predicate","object"}. Subject and object are short
   canonical noun phrases taken from the paper (method, component, dataset, metric, result, assumption,
   limitation). Predicate is a short verb phrase.
2. Copy numbers, units, dataset names and metric names exactly as written.
3. Each fact must cite exactly ONE paragraph id: the paragraph in which it is explicitly stated.
   Never combine information from two paragraphs.
4. Do not infer, evaluate, or summarise. Do not add external knowledge.
5. Skip vague statements, pure citations, section headings and acknowledgements.
6. No duplicate facts.
7. Output at most 4 facts per paragraph and at most 24 facts total. Prefer specific, quantitative facts.

Return JSON of exactly this shape:
{"facts":[{"fact_id":"f1","subject":"<entity>","predicate":"<verb phrase>","object":"<entity or value>",
"source_passage_ids":["<passage id>"],"evidence":"<verbatim quote from that paragraph, max 30 words>"}]}
"""

COMPREHENSION_SYSTEM = (
    "You build a cross-paragraph understanding layer over one scientific paper. You only state insights "
    "that the cited paragraphs jointly support, and you always ground them in passage ids. "
    "You output only valid JSON."
)

COMPREHENSION_INSTRUCTIONS = """You are given the title of one scientific paper, several numbered paragraphs, and the atomic
facts already extracted from those paragraphs.

Find insights that require combining information from AT LEAST TWO DIFFERENT paragraphs. An insight is
new: it is not stated by any single paragraph and it is not a paraphrase of one atomic fact.

What qualifies:
* Method -> experiment -> result chains, where the parts are in different paragraphs.
* A claim in one paragraph and the evidence for it in another paragraph.
* An assumption and the limitation it creates, stated in different places.
* An experimental setting and the outcome observed under it.
* A problem and the proposed solution when they are described separately.
* Coreference or terminology unification across sections (the same method, dataset or component under
  different names), when needed to connect two passages.
* A conclusion that follows only from two or more facts stated in different paragraphs.

Hard requirements:
1. source_passage_ids must contain at least TWO ids belonging to different paragraphs, and every id must
   be one of the input ids.
2. supporting_fact_ids must reference input atomic fact ids.
3. concise_text is ONE self-contained sentence of at most 40 words, understandable without reading the
   paper. Use exact entity, dataset and metric names and exact numbers from the input. Do not add any
   entity, number, dataset, or conclusion that is absent from the cited paragraphs.
4. triple expresses the same insight as short subject / predicate / object. Use a short canonical
   predicate such as improves_on, evaluated_on, outperforms, requires, causes, limits, extends.
5. relation_type is one of: method_result, claim_evidence, assumption_limitation, setting_outcome,
   problem_solution, entity_resolution, combined_fact.
6. derivation is "Composed" when combining explicit statements is enough, "Inferred" when ordinary
   domain inference is required.
7. confidence is a calibrated probability between 0 and 1.
8. Do not turn co-occurrence or correlation into causation unless the text states the causal link.
9. Do not restate atomic facts. If no insight satisfies these rules, return {"insights":[]}.
10. Return at most 10 insights, ordered by importance.

Return JSON of exactly this shape:
{"insights":[{"insight_id":"i1","concise_text":"<one sentence>",
"triple":{"subject":"<entity>","predicate":"<verb phrase>","object":"<entity or value>"},
"relation_type":"<type>","source_passage_ids":["<id>","<id>"],"supporting_fact_ids":["<fact id>"],
"derivation":"Composed","confidence":0.8}]}
"""

VERIFY_SYSTEM = (
    "You verify candidate insights against their cited source paragraphs of one scientific paper. "
    "You are strict about unsupported content but do not reject a correct insight because of wording. "
    "You output only valid JSON."
)

VERIFY_INSTRUCTIONS = """For each candidate insight, decide whether it is supported by the cited paragraphs.

Checks:
1. Do all cited passage ids exist among the provided sources?
2. Do the cited paragraphs jointly support the insight exactly as written?
3. Is the insight merely a restatement or paraphrase of a fact contained in a single paragraph? If yes, reject.
4. Does it assert a causal relation that the text only supports by co-occurrence or correlation? If yes, reject.
5. Does it introduce numbers, entities, databases, datasets or conclusions absent from the sources? If yes, reject.
6. Is the triple consistent with concise_text?

Return JSON of exactly this shape:
{"verdicts":[{"insight_id":"<id>","keep":true,"support":0.9,"novelty":0.7,"causality_error":false,
"reason":"<short reason>"}]}
"""

SUMMARY_SYSTEM = (
    "You write faithful paragraph summaries for scientific papers. You never add information. "
    "You output only valid JSON."
)

SUMMARY_INSTRUCTIONS = """For each numbered paragraph, write one self-contained summary sentence of at most 30 words.
Preserve entity, dataset and metric names and exact numbers. Do not combine paragraphs and do not add
information that is not in that paragraph.

Return JSON of exactly this shape:
{"summaries":[{"passage_id":"<id>","summary":"<one sentence>"}]}
"""


def render_paragraph_block(paper, passage_ids, include_facts: bool = False,
                           facts_by_passage=None) -> str:
    by_id = paper.paragraph_map()
    parts = [f"Paper title: {paper.title}"]
    for pid in passage_ids:
        p = by_id[pid]
        parts.append(f"[PASSAGE {pid}] (section: {p.section_name})\n{p.text}")
    return "\n\n".join(parts)


def render_atomic_prompt(paper, passage_ids) -> str:
    return (f"{ATOMIC_INSTRUCTIONS}\n\nParagraphs:\n{render_paragraph_block(paper, passage_ids)}\n\n"
            "Return the JSON now.")


def render_comprehension_prompt(paper, passage_ids, facts) -> str:
    lines = []
    for f in facts:
        lines.append(json_line(f))
    facts_block = "\n".join(lines) if lines else "(none)"
    return (f"{COMPREHENSION_INSTRUCTIONS}\n\nParagraphs:\n{render_paragraph_block(paper, passage_ids)}\n\n"
            f"Atomic facts:\n{facts_block}\n\nReturn the JSON now.")


def json_line(f: dict) -> str:
    import json
    return json.dumps({k: f.get(k) for k in ("fact_id", "subject", "predicate", "object",
                                              "source_passage_ids")}, ensure_ascii=False)


def render_verify_prompt(paper, insights, sources_by_id) -> str:
    import json
    parts = []
    for ins in insights:
        src = "\n".join(f"[PASSAGE {pid}]\n{sources_by_id[pid]}" for pid in ins["source_passage_ids"]
                        if pid in sources_by_id)
        parts.append(
            f"CANDIDATE {ins['insight_id']}\n"
            f"concise_text: {ins['concise_text']}\n"
            f"triple: {json.dumps(ins['triple'], ensure_ascii=False)}\n"
            f"relation_type: {ins.get('relation_type')}\n"
            f"derivation: {ins.get('derivation')}\n"
            f"cited sources:\n{src}"
        )
    return (f"Paper title: {paper.title}\n\n"
            f"{VERIFY_INSTRUCTIONS}\n\nCandidates:\n" + "\n\n".join(parts) + "\n\nReturn the JSON now.")


def render_summary_prompt(paper, passage_ids) -> str:
    return (f"{SUMMARY_INSTRUCTIONS}\n\nParagraphs:\n{render_paragraph_block(paper, passage_ids)}\n\n"
            "Return the JSON now.")


def split_blocks(passage_ids, paper, max_paragraphs: int = 6, max_tokens: int = 2500):
    """Split a reading group into small atomic-extraction blocks."""
    by_id = paper.paragraph_map()
    blocks, cur, cur_tok = [], [], 0
    for pid in passage_ids:
        n = by_id[pid].n_tokens
        if cur and (len(cur) >= max_paragraphs or cur_tok + n > max_tokens):
            blocks.append(cur)
            cur, cur_tok = [], 0
        cur.append(pid)
        cur_tok += n
    if cur:
        blocks.append(cur)
    return blocks
