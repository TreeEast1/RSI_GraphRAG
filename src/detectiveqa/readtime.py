"""Low-cost read-time packets and validation for DetectiveQA comprehension relations.

Packets are built from the official HippoRAG NER output, never from questions or
annotated clues.  Each packet follows one recurring, non-hub entity across a
novel and samples temporally spread mentions.  The same packet is given to the
joint-OpenIE control and to comprehension extraction, so their difference isolates
the prompt objective rather than the context.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .data import Chunk

_WORD = re.compile(r"[a-z0-9]+")
RELATION_TYPES = {"alias_or_identity", "alibi_or_opportunity", "motive_or_interest",
                  "method_or_capability", "contradiction_or_deception", "timeline_or_causal",
                  "state_change"}
_GENERIC_PREDICATES = RELATION_TYPES | {"related to", "connected to", "associated with", "involves",
                                        "relation", "relationship", "contradicts", "claims", "suspected"}
_GENERIC = {"he", "she", "they", "i", "you", "we", "man", "woman", "people", "room", "house",
            "time", "day", "night", "police", "thing", "something", "someone",
            # story-generic single words that otherwise dominate packet ranking
            "god", "lord", "heaven", "devil", "sir", "madam", "mister", "missus",
            "doctor", "inspector", "sergeant", "constable", "captain", "colonel",
            "father", "mother", "brother", "sister", "uncle", "aunt", "friend",
            "servant", "girl", "boy", "lady", "gentleman", "men", "women"}


@dataclass
class Packet:
    packet_id: str
    novel_id: str
    anchor: str
    chunk_ids: List[str]
    available_at: int
    n_words: int


def norm(text: str) -> str:
    return " ".join(_WORD.findall(str(text).lower()))


def load_openie_entities(state_path: Path, chunks: Sequence[Chunk]) -> Dict[str, List[str]]:
    from ..graph.variants import chunk_id_for

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    by_hash = {row["idx"]: row for row in payload.get("docs", [])}
    out = {}
    for chunk in chunks:
        row = by_hash.get(chunk_id_for(chunk.text), {})
        out[chunk.chunk_id] = list(dict.fromkeys(row.get("extracted_entities", [])))
    return out


def build_packets(chunks: Sequence[Chunk], entities_by_chunk: Dict[str, Sequence[str]], *,
                  max_packets: int = 40, chunks_per_packet: int = 6, max_words: int = 1800,
                  min_mentions: int = 4, max_df: float = 0.20) -> List[Packet]:
    """Choose recurring entities and sample mentions across early/middle/late story time."""
    order = {c.chunk_id: i for i, c in enumerate(chunks)}
    by_id = {c.chunk_id: c for c in chunks}
    mentions: Dict[str, List[str]] = defaultdict(list)
    surface: Dict[str, str] = {}
    for chunk_id, ents in entities_by_chunk.items():
        for ent in ents:
            key = norm(ent)
            if len(key) < 3 or key in _GENERIC:
                continue
            mentions[key].append(chunk_id)
            surface.setdefault(key, ent)
    n = max(1, len(chunks))
    candidates = []
    for key, ids in mentions.items():
        ids = sorted(set(ids), key=order.get)
        if len(ids) < min_mentions or len(ids) / n > max_df:
            continue
        span = order[ids[-1]] - order[ids[0]]
        score = math.log(1 + len(ids)) * math.log(2 + span) * math.log(n / len(ids))
        candidates.append((score, key, ids))
    packets = []
    for score, key, ids in sorted(candidates, reverse=True)[:max_packets]:
        picks = []
        for j in range(min(chunks_per_packet, len(ids))):
            pos = round(j * (len(ids) - 1) / max(1, min(chunks_per_packet, len(ids)) - 1))
            picks.append(ids[pos])
        picks = sorted(dict.fromkeys(picks), key=order.get)
        kept, words = [], 0
        for chunk_id in picks:
            if kept and words + by_id[chunk_id].n_words > max_words:
                continue
            kept.append(chunk_id)
            words += by_id[chunk_id].n_words
        if len(kept) >= 2:
            packets.append(Packet(f"{chunks[0].novel_id}:{len(packets):03d}", chunks[0].novel_id,
                                  surface[key], kept, max(by_id[c].available_at for c in kept), words))
    return packets


def render_packet(packet: Packet, chunks: Sequence[Chunk]) -> str:
    by_id = {c.chunk_id: c for c in chunks}
    return "\n\n".join(f"[{cid} | paragraphs {by_id[cid].start_pid}-{by_id[cid].end_pid}]\n{by_id[cid].text}"
                       for cid in packet.chunk_ids)


def _quote_in_text(quote: str, text: str) -> bool:
    q = norm(quote)
    return bool(q) and q in norm(text)


def validate_relation(item: Dict, packet: Packet, chunks: Sequence[Chunk],
                      existing_facts: Optional[Iterable[Tuple[str, str, str]]] = None) -> Tuple[Optional[Dict], str]:
    """Schema, provenance and novelty checks; returns (relation, reject_reason)."""
    by_id = {c.chunk_id: c for c in chunks}
    s, p, o = (str(item.get(k, "")).strip() for k in ("subject", "predicate", "object"))
    if not (s and p and o):
        return None, "missing_triple"
    if len(s.split()) > 8 or len(o.split()) > 10 or len(p.split()) > 5:
        return None, "too_long"
    if norm(p) in {norm(x) for x in _GENERIC_PREDICATES} or norm(p).replace(" ", "_") in RELATION_TYPES:
        return None, "generic_predicate"
    cited = [str(x) for x in item.get("source_chunk_ids") or [] if str(x) in packet.chunk_ids]
    cited = list(dict.fromkeys(cited))
    if len(cited) < 2:
        return None, "needs_two_chunks"
    quotes = {}
    for ev in item.get("evidence") or []:
        if isinstance(ev, dict) and str(ev.get("chunk_id")) in cited:
            quotes[str(ev["chunk_id"])] = str(ev.get("quote", ""))
    if any(cid not in quotes or not _quote_in_text(quotes[cid], by_id[cid].text) for cid in cited):
        return None, "quote_not_found"
    key = (norm(s), norm(p), norm(o))
    if existing_facts and key in {tuple(norm(x) for x in f) for f in existing_facts}:
        return None, "duplicate_fact"
    status = str(item.get("status") or "supported")
    if status not in {"supported", "claimed", "suspected"}:
        status = "suspected"
    relation_type = str(item.get("relation_type") or "unknown")
    if relation_type not in RELATION_TYPES:
        return None, "bad_relation_type"
    latest = max(cited, key=lambda cid: by_id[cid].available_at)
    try:
        confidence = min(1.0, max(0.0, float(item.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "subject": s, "predicate": p, "object": o,
        "relation_type": relation_type,
        "source_chunk_ids": cited, "attach_chunk_id": latest,
        "available_at": by_id[latest].available_at,
        "evidence": [{"chunk_id": cid, "quote": quotes[cid]} for cid in cited],
        "why_cross_chunk": str(item.get("why_cross_chunk") or "")[:300],
        "retrieval_cues": [str(x)[:80] for x in (item.get("retrieval_cues") or [])[:6]],
        "status": status, "confidence": confidence, "packet_id": packet.packet_id,
    }, ""


def relation_to_triple(relation: Dict) -> Tuple[str, str, str]:
    predicate = relation["predicate"]
    status = relation.get("fixed_status") or relation.get("status")
    if status in {"claimed", "suspected"} and not predicate.startswith(status):
        predicate = f"{status} {predicate}"
    return relation["subject"], predicate, relation["object"]


def extra_by_latest_chunk(relations: Iterable[Dict]) -> Dict[str, List[Tuple[str, str, str]]]:
    """Attach every relation only where all of its evidence has been read."""
    out: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for relation in relations:
        out[relation["attach_chunk_id"]].append(relation_to_triple(relation))
    return dict(out)


def relation_stats(relations: Sequence[Dict], chunks: Sequence[Chunk]) -> Dict:
    order = {c.chunk_id: i for i, c in enumerate(chunks)}
    gaps = [max(order[c] for c in r["source_chunk_ids"]) - min(order[c] for c in r["source_chunk_ids"])
            for r in relations]
    return {
        "n": len(relations),
        "types": dict(Counter(r["relation_type"] for r in relations)),
        "status": dict(Counter(r.get("fixed_status") or r["status"] for r in relations)),
        "median_chunk_gap": sorted(gaps)[len(gaps) // 2] if gaps else None,
    }


def build_question_agnostic_packets(chunks: Sequence[Chunk], entities_by_chunk: Dict[str, Sequence[str]], *,
                                    max_packets: int = 30, chunks_per_packet: int = 5,
                                    max_words: int = 1500, min_mentions: int = 3,
                                    max_df: float = 0.25) -> List[Packet]:
    """Entity packets with lower overlap and broader coverage than the v2 smoke test."""
    raw = build_packets(chunks, entities_by_chunk, max_packets=max_packets * 4,
                        chunks_per_packet=chunks_per_packet, max_words=max_words,
                        min_mentions=min_mentions, max_df=max_df)
    chosen: List[Packet] = []
    covered: Counter = Counter()
    for packet in raw:
        overlap = sum(covered[c] > 0 for c in packet.chunk_ids) / max(1, len(packet.chunk_ids))
        if chosen and overlap > 0.5:
            continue
        chosen.append(Packet(f"{packet.novel_id}:{len(chosen):03d}", packet.novel_id, packet.anchor,
                             packet.chunk_ids, packet.available_at, packet.n_words))
        covered.update(packet.chunk_ids)
        if len(chosen) >= max_packets:
            break
    return chosen
