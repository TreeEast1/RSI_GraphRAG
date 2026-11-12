"""Layered read-time verifier.

V1 locates quoted spans in the full passage and expands them to the sentences
that contain them. V2 checks that the claim's named entities occur in those
sentences. V3 applies deterministic date checks and blocks causal or succession
upgrades. V4 compares the claim with the OpenIE triples of the cited passages,
which is the global G_raw view of those passages, not the triples that happened
to survive a cluster cut. V5 emits a decision and a failure stage.

The generator's self-label is ignored.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

V4_SRC = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/experiments/rsi_graphrag_v4/src")
HIPPO_SRC = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")
for item in (V4_SRC, HIPPO_SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from hipporag.utils.misc_utils import text_processing  # noqa: E402

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
STOP = {
    "a", "an", "the", "of", "to", "for", "in", "on", "was", "were", "is", "are", "be",
    "been", "by", "and", "or", "who", "which", "with", "as", "from", "his", "her",
    "their", "its", "at", "that", "this", "into", "over", "than", "respectively",
    "during", "while", "part", "became", "become", "made", "make", "had", "have",
    "has", "but", "also", "their", "them", "he", "she", "it", "they",
}
CAUSAL_CLAIM = re.compile(r"\b(because|motivated|motivation|in order to|due to|caused|so that|led to)\b", re.I)
CAUSAL_EVIDENCE = re.compile(r"\b(because|motivated|in order to|due to|caused|so that|led to|to live with|following)\b", re.I)
SUCCESSION_CLAIM = re.compile(r"\b(succeeded|succession|replaced|successor|took over)\b", re.I)
SUCCESSION_EVIDENCE = re.compile(
    r"\b(succeeded|successor|replac(?:e|ed|es|ing)|appointed the new|the new commander|debut(?:ed)? as the host|became host|new commander)\b",
    re.I,
)
TEMPORAL_CLAIM = re.compile(r"\b(before|earlier|prior to|after)\b", re.I)
TEMPORAL_EVIDENCE = re.compile(r"\b(before|earlier|prior|after|following|until|from|debuted|appointed)\b", re.I)


def _norm_ws(text: str) -> str:
    text = (text or "").replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("``", '"').replace("''", '"')
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> List[str]:
    processed = text_processing(text or "")
    return [tok for tok in processed.split() if tok and tok not in STOP and not tok.isdigit()]


def _content_tokens(text: str) -> List[str]:
    processed = text_processing(text or "")
    return [tok for tok in processed.split() if tok]


def locate_span(passage: str, span: str) -> Optional[dict]:
    tokens = re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", span or "")
    tokens = [tok for tok in tokens if tok.strip()]
    if len("".join(tokens)) < 8:
        return None
    pattern = r"\s*".join(re.escape(tok) for tok in tokens)
    match = re.search(pattern, passage or "", flags=re.IGNORECASE)
    if match is None:
        return None
    start, end = match.start(), match.end()
    left = 0
    for mark in (".", "?", "!"):
        pos = (passage.rfind(mark, 0, start))
        if pos > left:
            left = pos + 1
    right = len(passage)
    for mark in (".", "?", "!"):
        pos = passage.find(mark, end)
        if pos != -1:
            right = min(right, pos + 1)
    sentence = passage[left:right].strip()
    return {"start": start, "end": end, "matched": passage[start:end], "sentence": sentence}


def _parse_dates(text: str) -> List[tuple]:
    found = []
    pattern_a = re.compile(
        r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
        re.I,
    )
    pattern_b = re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s*(\d{4})\b",
        re.I,
    )
    pattern_c = re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
        re.I,
    )
    for day, month, year in pattern_a.findall(text or ""):
        found.append((int(year), MONTHS[month.lower()], int(day)))
    for month, day, year in pattern_b.findall(text or ""):
        found.append((int(year), MONTHS[month.lower()], int(day)))
    if not found:
        for month, year in pattern_c.findall(text or ""):
            found.append((int(year), MONTHS[month.lower()], 1))
    return found


def _triple_blob(triples: Sequence[Sequence[str]]) -> str:
    parts = []
    for triple in triples:
        parts.append(" ".join(str(part) for part in triple))
    return " ".join(parts)


_ALIASES = {"st": {"st", "saint"}, "saint": {"saint", "st"}}


def _names_in_claim(claim: str) -> List[str]:
    raw = re.findall(r"\b[A-Z][A-Za-z0-9.'’-]*(?:\s+[A-Z][A-Za-z0-9.'’-]*)+", claim or "")
    return raw


def _name_tokens(name: str) -> List[str]:
    return [tok for tok in text_processing(name).split() if tok not in {"of", "the", "and"}]


def _token_present(token: str, available: set) -> bool:
    return bool(_ALIASES.get(token, {token}) & available)


def _title_matches(name: str, titles: set) -> bool:
    norm = text_processing(name)
    if not norm:
        return False
    for title in titles:
        if norm == title or title.startswith(norm + " ") or norm.startswith(title):
            return True
    return False


def _name_grounded(name: str, evidence_norm: str) -> bool:
    tokens = _name_tokens(name)
    if not tokens:
        return True
    available = set(evidence_norm.split())
    return all(_token_present(tok, available) for tok in tokens)


def classify_type(candidate: dict, evidence_text: str, passages_used: int, novel: bool, logic_ok: bool, n_sentences: int = 1) -> str:
    if not logic_ok:
        return "UNSUPPORTED_HYPOTHESIS"
    if not novel:
        return "REDUNDANT"
    relation = (candidate.get("new_semantic_relation") or "").lower()
    text = candidate.get("comprehension_text") or ""
    if passages_used >= 2:
        return "CROSS_PASSAGE_REASONING"
    if re.search(r"\b(first|earliest|history|all-women|shortcut)\b", text, re.I) and len(_tokens(text)) > 12:
        # abstraction is decided by residue later; keep compositional if multiple spans
        pass
    spans = candidate.get("evidence_spans") or []
    if (len(spans) >= 2 or len(candidate.get("chain") or []) >= 2) and n_sentences >= 2:
        if passages_used >= 2:
            return "CROSS_PASSAGE_REASONING"
        return "COMPOSITIONAL_REASONING"
    # A single explicit sentence that the triples missed.
    if evidence_text and _tokens(text):
        overlap = len(set(_tokens(text)) & set(_tokens(evidence_text))) / max(1, len(set(_tokens(text))))
        if overlap >= 0.55:
            return "EXTRACTION_REPAIR"
    if relation in {"earlier first appearance than", "earlier_than"}:
        return "CROSS_PASSAGE_REASONING" if passages_used >= 2 else "COMPOSITIONAL_REASONING"
    return "SINGLE_PASSAGE_CONSOLIDATION"


def verify_candidate(candidate: dict, passages: Dict[str, str], passage_triples: Dict[str, List[Sequence[str]]]) -> dict:
    spans = list(candidate.get("evidence_spans") or [])
    claim = candidate.get("comprehension_text") or ""
    stages = []
    located = []
    missing = []
    for span in spans:
        pid = span.get("passage_id")
        quote = span.get("span") or ""
        passage = passages.get(pid or "")
        if not pid or passage is None:
            missing.append({"passage_id": pid, "span": quote, "problem": "unknown_passage"})
            continue
        hit = locate_span(passage, quote)
        if hit is None:
            missing.append({"passage_id": pid, "span": quote, "problem": "span_not_in_passage"})
            continue
        located.append({"passage_id": pid, **hit})
    if not spans or missing or not located:
        return {
            "candidate_id": candidate.get("candidate_id"),
            "decision": "EVIDENCE_MISMATCH",
            "failure_stage": "V1",
            "faithfulness": "unlocated",
            "novelty": "unclear",
            "audit_type": "UNLOCATED",
            "reason": "A cited span is not a verbatim substring of the named passage.",
            "missing_spans": missing,
            "located": [{k: v for k, v in row.items() if k != "sentence"} | {"sentence": row.get("sentence", "")[:400]} for row in located],
        }

    evidence_text = " ".join(row["sentence"] for row in located)
    passage_ids = sorted({row["passage_id"] for row in located})
    names = _names_in_claim(claim)
    evidence_norm = text_processing(evidence_text)
    evidence_tokens = set(evidence_norm.split())
    full_norm = text_processing(" ".join(passages.get(pid, "") for pid in passage_ids))
    title_names = set()
    for pid in passage_ids:
        lines = [line.strip() for line in (passages.get(pid) or "").splitlines() if line.strip()]
        if lines:
            title_names.add(text_processing(lines[0]))
    pronoun = bool(re.search(r"(?m)(?:^|[.]\s+)(She|He|They)\b", evidence_text))
    title_anaphor = bool(re.search(r"\b(the show|the series|the station|the character|the film|the pilot)\b", evidence_text, re.I))
    anchored = []
    missing_names = []
    for name in names:
        if any(tok in MONTHS for tok in text_processing(name).split()):
            continue
        if _name_grounded(name, evidence_norm):
            continue
        if (pronoun or title_anaphor) and _title_matches(name, title_names):
            anchored.append(name + " [title coreference]")
            continue
        tokens = _name_tokens(name)
        in_passage = _name_grounded(name, full_norm)
        in_sentence = [tok for tok in tokens if _token_present(tok, evidence_tokens)]
        if in_passage and in_sentence:
            anchored.append(name)
            continue
        missing_names.append(name)
    stages.append({
        "stage": "V2",
        "anchored_partial_names": anchored,
        "missing_names": missing_names,
        "evidence_sentences": [row["sentence"][:500] for row in located],
    })
    if missing_names:
        recoverable = [name for name in missing_names if _name_grounded(name, full_norm)]
        kind = "INCOMPLETE_CITATION" if recoverable and len(recoverable) == len(missing_names) else "ENTITY_BINDING_ERROR"
        return {
            "candidate_id": candidate.get("candidate_id"),
            "decision": "REJECT",
            "failure_stage": "V2",
            "faithfulness": "incomplete_citation" if kind == "INCOMPLETE_CITATION" else "entity_unbound",
            "novelty": "unclear",
            "audit_type": kind,
            "reason": (
                "The cited sentences do not contain "
                + ", ".join(missing_names)
                + (
                    ". Those names occur elsewhere in the cited passage, but no token of the name is in the cited sentence."
                    if kind == "INCOMPLETE_CITATION"
                    else ". Those names are not in the cited passage."
                )
            ),
            "missing_names": missing_names,
            "recoverable_in_passage": recoverable,
            "evidence_sentences": [row["sentence"][:500] for row in located],
        }

    logic_failures = []
    relation = candidate.get("new_semantic_relation") or ""
    combined = claim + " " + relation
    if CAUSAL_CLAIM.search(combined) and not CAUSAL_EVIDENCE.search(evidence_text):
        logic_failures.append("causal_or_motive_not_in_evidence")
    if SUCCESSION_CLAIM.search(combined) and not SUCCESSION_EVIDENCE.search(evidence_text):
        logic_failures.append("succession_not_in_evidence")
    if re.search(r"\bbefore\b", claim, re.I) and not re.search(r"\b(before|prior|earlier|until|following|after)\b", evidence_text, re.I):
        # Coordinated "and" does not license "before".
        if re.search(r"\band\b", evidence_text, re.I):
            logic_failures.append("before_not_entailed_by_coordination")
    dates = _parse_dates(claim)
    if len(dates) >= 2 and re.search(r"\b(before|earlier)\b", claim, re.I):
        if dates[0] > dates[1] and re.search(r"before|earlier", claim, re.I):
            # The first mentioned date should be the earlier one for "A before B, on DATE1 and DATE2".
            # Check the textual order of the two dates against the claim's before-relation.
            evidence_dates = _parse_dates(evidence_text)
            if len(evidence_dates) >= 2 and evidence_dates != sorted(evidence_dates) and dates[0] > dates[1]:
                logic_failures.append("date_order_contradicts_claim")
        if dates[0] > dates[1]:
            logic_failures.append("claimed_earlier_date_is_later")
    stages.append({"stage": "V3", "logic_failures": logic_failures, "dates": dates})
    if logic_failures:
        return {
            "candidate_id": candidate.get("candidate_id"),
            "decision": "REJECT" if "before_not_entailed_by_coordination" in logic_failures or "causal_or_motive_not_in_evidence" in logic_failures or "succession_not_in_evidence" in logic_failures or "claimed_earlier_date_is_later" in logic_failures else "UNCERTAIN",
            "failure_stage": "V3",
            "faithfulness": "unsupported",
            "novelty": "unclear",
            "audit_type": "UNSUPPORTED_HYPOTHESIS",
            "reason": "Logical check failed: " + ", ".join(logic_failures),
            "logic_failures": logic_failures,
            "evidence_sentences": [row["sentence"][:500] for row in located],
        }

    triples = []
    for pid in passage_ids:
        triples.extend(passage_triples.get(pid) or [])
    blob_tokens = set(_content_tokens(_triple_blob(triples)))
    claim_tokens = [tok for tok in _content_tokens(claim) if tok not in STOP]
    residue = [tok for tok in claim_tokens if tok not in blob_tokens]
    # Numeric dates are content. Keep year/day tokens that text_processing left as digits.
    digit_tokens = [tok for tok in text_processing(claim).split() if tok.isdigit() and tok not in blob_tokens]
    residue.extend(digit_tokens)
    novel = len(residue) > 0
    n_sentences = len({row["sentence"] for row in located})
    audit_type = classify_type(candidate, evidence_text, len(passage_ids), novel, True, n_sentences)
    if not novel:
        audit_type = "REDUNDANT"
    warning = ""
    if re.search(r"motiv", relation, re.I) and not CAUSAL_EVIDENCE.search(evidence_text):
        warning = "relation_label_adds_motive"
    if anchored:
        warning = (warning + ";" if warning else "") + "partial_name_anchored:" + ",".join(anchored)
    stages.append({"stage": "V4", "residue": residue, "n_passage_triples": len(triples), "novel": novel})
    if not novel:
        return {
            "candidate_id": candidate.get("candidate_id"),
            "decision": "REJECT",
            "failure_stage": "V4",
            "faithfulness": "supported",
            "novelty": "duplicate",
            "audit_type": "REDUNDANT",
            "reason": "The claim is supported by the cited sentences, and its content words are already covered by OpenIE triples of those passages.",
            "residue": residue,
            "evidence_sentences": [row["sentence"][:500] for row in located],
            "warning": warning,
        }
    return {
        "candidate_id": candidate.get("candidate_id"),
        "decision": "ACCEPT",
        "failure_stage": "",
        "faithfulness": "supported",
        "novelty": "new",
        "audit_type": audit_type,
        "reason": "The expanded source sentences support the claim, and these content words are absent from the passages' OpenIE triples: " + ", ".join(residue[:12]),
        "residue": residue,
        "evidence_sentences": [row["sentence"][:500] for row in located],
        "warning": warning,
        "stages": stages,
    }


def verify_many(candidates: Sequence[dict], passages: Dict[str, str], passage_triples: Dict[str, List[Sequence[str]]]) -> List[dict]:
    rows = []
    for candidate in candidates:
        if (candidate.get("operation") or "ADD").upper() != "ADD":
            continue
        if not (candidate.get("comprehension_text") or "").strip():
            continue
        rows.append(verify_candidate(candidate, passages, passage_triples))
    return rows
