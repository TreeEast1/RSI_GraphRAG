"""Semantic entailment against the cited passages' full OpenIE triples.

Token overlap is not enough. A candidate is redundant only when the triples
already state the same proposition under a closed set of morphological and
frame equivalences. A low-value but true derivation is TRIVIAL_DERIVATION,
not a factual error.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Sequence

HIPPO_SRC = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")
if str(HIPPO_SRC) not in sys.path:
    sys.path.insert(0, str(HIPPO_SRC))

from hipporag.utils.misc_utils import text_processing  # noqa: E402

from v5_mech.verifier import STOP, _parse_dates  # noqa: E402

# General morphology. These are lemmas, not pair-specific paraphrases.
MORPH = {
    "appeared": "appear",
    "appears": "appear",
    "appearance": "appear",
    "appearing": "appear",
    "renewed": "renew",
    "renewal": "renew",
    "renewing": "renew",
    "premiered": "premiere",
    "premieres": "premiere",
    "broadcasts": "broadcast",
    "broadcasting": "broadcast",
    "broadcasted": "broadcast",
    "aired": "air",
    "airing": "air",
    "succeeded": "succeed",
    "succeeds": "succeed",
    "successor": "succeed",
    "succession": "succeed",
    "replaced": "replace",
    "replacing": "replace",
    "replaces": "replace",
    "replacement": "replace",
    "appointed": "appoint",
    "appointing": "appoint",
    "appointment": "appoint",
    "approved": "approve",
    "approval": "approve",
    "approves": "approve",
    "commanded": "command",
    "commander": "command",
    "commanding": "command",
}
ORDINAL = {
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "1st": "1",
    "2nd": "2",
    "3rd": "3",
}
# Words that only frame an appearance/broadcast-on-date event.
APPEAR_FRAME = {"episode", "screen", "during", "broadcast", "air", "shown"}
# Pickup synonyms. Used only when a triple already says picked up / not picked up.
AIR_SYNONYM = {"air", "broadcast"}
ORDER_WORDS = {"earlier", "before", "prior", "after"}
# Relation words that restate an appointment or replacement the triples already store.
PACK_WORDS = {"new", "succeed", "replace", "appoint", "command"}


def stems(text: str) -> List[str]:
    raw = text_processing(text or "").replace("-", " ").split()
    out = []
    for tok in raw:
        tok = MORPH.get(tok, tok)
        tok = ORDINAL.get(tok, tok)
        if tok in STOP or len(tok) <= 1:
            continue
        out.append(tok)
    return out


def _stem_set(text: str) -> set:
    return set(stems(text))


def _triple_text(triples: Sequence[Sequence[str]]) -> str:
    return " ".join(" ".join(str(part) for part in triple) for triple in triples)


def _interval_sentence(evidence: str, dates: list) -> bool:
    """True only when one sentence contains both dates and an interval marker."""
    if len(dates) < 2:
        return False
    parts = re.split(r"(?<=[.!?])\s+", evidence or "")
    for part in parts:
        if not re.search(r"\b(from|until)\b", part, re.I):
            continue
        found = _parse_dates(part)
        if all(date in found for date in dates[:2]):
            return True
    return False


def classify_novelty(claim: str, triples: Sequence[Sequence[str]], evidence: str, n_passages: int) -> dict:
    """Return the information-value label for a claim that is already factually supported."""
    claim_stems = _stem_set(claim)
    triple_stems = _stem_set(_triple_text(triples))
    residue = set(claim_stems - triple_stems)
    claim_dates = _parse_dates(claim)
    reasons = []

    # Drop digits that are part of a date already shared with some triple.
    triple_dates = []
    for triple in triples:
        triple_dates.extend(_parse_dates(" ".join(str(part) for part in triple)))
    if claim_dates and all(date in triple_dates for date in claim_dates):
        for year, month, day in claim_dates:
            residue.discard(str(year))
            residue.discard(str(day))
            residue.discard(str(month))

    # Appearance on a date entails "episode broadcast on that date".
    appear_covered = any(
        "appear" in _stem_set(" ".join(map(str, triple))) and _parse_dates(" ".join(map(str, triple)))
        and set(_parse_dates(" ".join(map(str, triple)))) & set(claim_dates)
        for triple in triples
    ) if claim_dates else False
    if appear_covered:
        dropped = residue & APPEAR_FRAME
        if dropped:
            residue -= APPEAR_FRAME
            reasons.append("appearance_frame")

    # "the show" is an anaphor when the title stems are already in the triples.
    content_names = claim_stems - PACK_WORDS - ORDER_WORDS - APPEAR_FRAME - {"show", "series", "program", "not", "picked", "up"}
    if content_names <= triple_stems and residue & {"show", "series", "program"}:
        residue -= {"show", "series", "program"}
        reasons.append("title_anaphor")

    # "picked up for broadcast" is the same proposition as "picked up for air".
    if residue <= AIR_SYNONYM and any("picked" in _stem_set(" ".join(map(str, triple))) for triple in triples):
        if any(AIR_SYNONYM & _stem_set(" ".join(map(str, triple))) for triple in triples):
            residue -= AIR_SYNONYM
            reasons.append("air_broadcast_synonym")

    # Succession wording is entailed only when the names are already in the
    # triples and a triple already appoints, replaces, or names the commander.
    has_role = any(
        _stem_set(" ".join(map(str, triple))) & {"appoint", "replace", "succeed", "command"}
        for triple in triples
    )
    named = (claim_stems - PACK_WORDS - ORDER_WORDS - APPEAR_FRAME) <= triple_stems
    if has_role and named and residue and residue <= (PACK_WORDS | {"after", "new"}):
        residue -= PACK_WORDS | {"after", "new"}
        reasons.append("succession_wording_entailed")

    order_only = residue <= ORDER_WORDS and bool(residue)
    if order_only and _interval_sentence(evidence, claim_dates):
        return {
            "value_type": "TRIVIAL_DERIVATION",
            "novel": False,
            "faithfulness": "supported",
            "reason": "Both dates are the endpoints of one from/until interval already in the cited sentence.",
            "residue": sorted(residue),
        }
    if not residue:
        return {
            "value_type": "REDUNDANT",
            "novel": False,
            "faithfulness": "supported",
            "reason": "The cited passages' OpenIE triples already entail this proposition. "
            + (", ".join(reasons) if reasons else "Content stems are covered."),
            "residue": [],
        }
    if order_only and n_passages >= 2:
        return {
            "value_type": "COMPOSITIONAL_UNDERSTANDING",
            "novel": True,
            "faithfulness": "supported",
            "reason": "The only addition is an order between dates that come from different passages.",
            "residue": sorted(residue),
        }
    if order_only and n_passages == 1:
        return {
            "value_type": "COMPOSITIONAL_UNDERSTANDING",
            "novel": True,
            "faithfulness": "supported",
            "reason": "The only addition is an order between two dated events in one passage, and they are not endpoints of one interval.",
            "residue": sorted(residue),
        }
    evidence_stems = _stem_set(evidence)
    if residue <= evidence_stems:
        return {
            "value_type": "EXTRACTION_REPAIR",
            "novel": True,
            "faithfulness": "supported",
            "reason": "The cited sentence states these stems, and no OpenIE triple entails them: " + ", ".join(sorted(residue)[:12]),
            "residue": sorted(residue),
        }
    return {
        "value_type": "SEMANTIC_ABSTRACTION",
        "novel": True,
        "faithfulness": "supported",
        "reason": "The claim adds stems that are not entailed by the triples and are not copied from the cited sentence: "
        + ", ".join(sorted(residue)[:12]),
        "residue": sorted(residue),
    }
