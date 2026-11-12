"""Read-time extraction and comprehension over LoCoMo sessions."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..adapters.hy4_client import HY4Client, parse_json_loose
from ..dataset.locomo import Conversation, Session, num_tokens
from .prompts import (COMPREHENSION_SYSTEM, EXTRACTION_SYSTEM, VERIFY_SYSTEM,
                      render_comprehension_prompt, render_extraction_prompt,
                      render_verification_prompt)

ROOT = Path(__file__).resolve().parents[2]


def _call_json(client, prompt: str, system: str, max_tokens: int, tag: str, attempts: int = 3):
    """JSON call that survives output truncation by retrying with a bigger budget."""
    last = None
    for attempt in range(attempts):
        budget = min(32768, max_tokens * (2 ** attempt))
        out = client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=budget, temperature=0.0, json_mode=True, tag=tag,
        )
        parsed = parse_json_loose(out["content"])
        last = out
        if parsed is not None:
            return parsed, out, attempt
        if out.get("finish_reason") != "length":
            break
    return None, last, attempts



def sessions_block(conv: Conversation, session_ids: Sequence[int], with_turn_ids: bool = True) -> str:
    by_id = conv.session_map()
    parts = []
    for sid in session_ids:
        s = by_id[sid]
        head = f"=== Session {sid} | {s.date_time_raw} ==="
        lines = [head]
        for t in s.turns:
            prefix = f"[{t.dia_id}] " if with_turn_ids else ""
            lines.append(f"{prefix}{t.speaker}: {t.text}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _norm(s: str) -> str:
    return " ".join(str(s).strip().strip('"').strip("'").lower().split())


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    source_dia_ids: List[str]
    source_sessions: List[int]
    evidence: str
    derivation: str = "Explicit"
    confidence: float = 1.0
    relation_type: str = "explicit"
    supporting_fact_ids: List[str] = field(default_factory=list)

    def key(self) -> Tuple[str, str, str]:
        return (_norm(self.subject), _norm(self.predicate), _norm(self.object))

    def as_fact(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> Dict:
        return {
            "subject": self.subject, "predicate": self.predicate, "object": self.object,
            "source_dia_ids": self.source_dia_ids, "source_sessions": self.source_sessions,
            "evidence": self.evidence, "derivation": self.derivation,
            "confidence": self.confidence, "relation_type": self.relation_type,
            "supporting_fact_ids": self.supporting_fact_ids,
        }


def _coerce_triples(raw: List, conv: Conversation, allowed_dia: Set[str],
                    allowed_sessions: Set[int], record_rejects: Optional[List] = None) -> List[Triple]:
    dia_to_session = {t.dia_id: t.session_id for s in conv.sessions for t in s.turns}
    out: List[Triple] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        s = str(item.get("subject", "")).strip()
        p = str(item.get("predicate", "")).strip()
        o = str(item.get("object", "")).strip()
        if not s or not p or not o:
            continue
        cited = item.get("source_dia_ids") or []
        if not isinstance(cited, list):
            cited = [cited]
        cited = [str(c).strip() for c in cited if str(c).strip()]
        valid = [d for d in cited if d in allowed_dia]
        if not valid:
            if record_rejects is not None:
                record_rejects.append({"reason": "no_valid_dia", "triple": [s, p, o], "cited": cited})
            continue
        sessions = sorted({dia_to_session[d] for d in valid if d in dia_to_session})
        sessions = [x for x in sessions if x in allowed_sessions]
        if not sessions:
            continue
        out.append(Triple(
            subject=s, predicate=p, object=o,
            source_dia_ids=valid, source_sessions=sessions,
            evidence=str(item.get("evidence", ""))[:500],
            derivation=str(item.get("derivation", "Explicit")),
            confidence=float(item.get("confidence", 1.0) or 1.0),
        ))
    return out


def dedupe(triples: Iterable[Triple]) -> List[Triple]:
    seen: Dict[Tuple[str, str, str], Triple] = {}
    for t in triples:
        k = t.key()
        if k in seen:
            prev = seen[k]
            prev.source_dia_ids = sorted(set(prev.source_dia_ids) | set(t.source_dia_ids))
            prev.source_sessions = sorted(set(prev.source_sessions) | set(t.source_sessions))
        else:
            seen[k] = t
    return list(seen.values())


def extract_group(client: HY4Client, conv: Conversation, session_ids: Sequence[int],
                  max_tokens: int = 8192, tag: str = "extract") -> Dict:
    block = sessions_block(conv, session_ids)
    allowed_dia = {t.dia_id for sid in session_ids for t in conv.session_map()[sid].turns}
    prompt = render_extraction_prompt(block)
    try:
        raw, meta, attempts = _call_json(client, prompt, EXTRACTION_SYSTEM, max_tokens, tag)
        if raw is None:
            return {"session_ids": list(session_ids),
                    "error": "unparseable_or_truncated",
                    "finish_reason": meta.get("finish_reason") if meta else None,
                    "output_tokens": (meta or {}).get("usage", {}).get("completion_tokens"),
                    "input_tokens": num_tokens(prompt), "triples": [], "rejected": []}
        triples_raw = raw.get("triples", []) if isinstance(raw, dict) else []
    except Exception as exc:  # noqa: BLE001
        return {"session_ids": list(session_ids), "error": f"{type(exc).__name__}: {exc}",
                "input_tokens": num_tokens(prompt), "triples": [], "rejected": []}
    rejects: List[Dict] = []
    triples = _coerce_triples(triples_raw, conv, allowed_dia, set(session_ids), rejects)
    triples = dedupe(triples)
    return {
        "session_ids": list(session_ids),
        "input_tokens": num_tokens(prompt),
        "n_raw": len(triples_raw),
        "finish_reason": meta.get("finish_reason"),
        "output_tokens": meta.get("usage", {}).get("completion_tokens"),
        "truncated": bool(meta.get("finish_reason") == "length"),
        "retries": attempts,
        "triples": [t.to_dict() for t in triples],
        "rejected": rejects,
    }


def comprehend_group(client: HY4Client, conv: Conversation, session_ids: Sequence[int],
                     triples: List[Triple], max_tokens: int = 8192,
                     tag: str = "comprehend") -> Dict:
    block = sessions_block(conv, session_ids)
    allowed_dia = {t.dia_id for sid in session_ids for t in conv.session_map()[sid].turns}
    facts_block = "\n".join(
        f"f{i}: {t.subject} | {t.predicate} | {t.object}" for i, t in enumerate(triples)
    )
    prompt = render_comprehension_prompt(block, facts_block)
    try:
        raw, meta, attempts = _call_json(client, prompt, COMPREHENSION_SYSTEM, max_tokens, tag)
        if raw is None:
            return {"session_ids": list(session_ids), "error": "unparseable_or_truncated",
                    "input_tokens": num_tokens(prompt), "relations": []}
        rels_raw = raw.get("relations", []) if isinstance(raw, dict) else []
    except Exception as exc:  # noqa: BLE001
        return {"session_ids": list(session_ids), "error": f"{type(exc).__name__}: {exc}",
                "input_tokens": num_tokens(prompt), "relations": []}

    dia_to_session = {t.dia_id: t.session_id for s in conv.sessions for t in s.turns}
    existing = {t.key() for t in triples}
    rels: List[Triple] = []
    rejects = []
    for item in rels_raw or []:
        if not isinstance(item, dict):
            continue
        s = str(item.get("subject", "")).strip()
        p = str(item.get("predicate", "")).strip()
        o = str(item.get("object", "")).strip()
        if not s or not p or not o:
            continue
        cited = item.get("source_dia_ids") or []
        if not isinstance(cited, list):
            cited = [cited]
        cited = [str(c).strip() for c in cited]
        valid = [d for d in cited if d in allowed_dia]
        sessions = sorted({dia_to_session[d] for d in valid})
        if len(sessions) < 2:
            rejects.append({"reason": "needs_two_sessions", "triple": [s, p, o], "sessions": sessions})
            continue
        key = (_norm(s), _norm(p), _norm(o))
        if key in existing:
            rejects.append({"reason": "duplicate_of_explicit", "triple": [s, p, o]})
            continue
        ev = item.get("evidence", "")
        if isinstance(ev, list):
            ev = " || ".join(str(e) for e in ev)
        rels.append(Triple(
            subject=s, predicate=p, object=o,
            source_dia_ids=valid, source_sessions=sessions, evidence=str(ev)[:800],
            derivation=str(item.get("derivation", "Composed")),
            confidence=float(item.get("confidence", 0.5) or 0.5),
            relation_type=str(item.get("relation_type", "cross_session")),
            supporting_fact_ids=[str(x) for x in (item.get("supporting_fact_ids") or [])],
        ))
    rels = dedupe(rels)
    return {
        "session_ids": list(session_ids),
        "input_tokens": num_tokens(prompt),
        "n_raw": len(rels_raw),
        "relations": [r.to_dict() for r in rels],
        "rejected": rejects,
    }


def verify_candidates(client: HY4Client, candidates: List[Dict], batch_size: int = 20,
                      tag: str = "verify") -> List[Dict]:
    """Cheap LLM verification of candidate relations. Returns kept candidates."""
    kept: List[Dict] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        block = json.dumps([
            {"id": f"c{i}", "subject": c["subject"], "predicate": c["predicate"],
             "object": c["object"], "derivation": c.get("derivation"),
             "evidence": c.get("evidence"), "source_dia_ids": c.get("source_dia_ids")}
            for i, c in enumerate(batch)
        ], ensure_ascii=False, indent=1)
        try:
            out = client.complete_json(render_verification_prompt(block),
                                       system=VERIFY_SYSTEM, max_tokens=4096, tag=tag)
            verdicts = {v.get("id"): v for v in out.get("verdicts", []) if isinstance(v, dict)}
        except Exception:
            verdicts = {}
        for i, c in enumerate(batch):
            v = verdicts.get(f"c{i}")
            if v is None or v.get("keep") is True:
                kept.append(c)
    return kept


def run_scope(client: HY4Client, conv: Conversation, groups: List[List[str]],
              *, do_comprehension: bool = False, do_verification: bool = False,
              max_tokens: int = 8192, tag_prefix: str = "rt") -> Dict:
    """Extract (and optionally comprehend) over a list of reading groups."""
    by_doc = conv.session_map()
    doc_to_sid = {s.document_id: s.session_id for s in conv.sessions}
    out_groups = []
    all_triples: List[Triple] = []
    all_relations: List[Triple] = []
    for gi, doc_ids in enumerate(groups):
        sids = [doc_to_sid[d] for d in doc_ids]
        res = extract_group(client, conv, sids, max_tokens=max_tokens,
                            tag=f"{tag_prefix}_extract")
        triples = [Triple(**{k: t[k] for k in
                             ("subject", "predicate", "object", "source_dia_ids", "source_sessions",
                              "evidence", "derivation", "confidence", "relation_type", "supporting_fact_ids")})
                   for t in res["triples"]]
        all_triples.extend(triples)
        entry = {"group_id": gi, "session_ids": sids, "extraction": res}
        if do_comprehension:
            cres = comprehend_group(client, conv, sids, triples, max_tokens=max_tokens,
                                    tag=f"{tag_prefix}_comprehend")
            if do_verification and cres.get("relations"):
                kept = verify_candidates(client, cres["relations"], tag=f"{tag_prefix}_verify")
                cres["relations_verified"] = kept
                cres["n_verified"] = len(kept)
            entry["comprehension"] = cres
            all_relations.extend(Triple(**{k: r[k] for k in
                                          ("subject", "predicate", "object", "source_dia_ids",
                                           "source_sessions", "evidence", "derivation", "confidence",
                                           "relation_type", "supporting_fact_ids")})
                                 for r in cres["relations"])
        out_groups.append(entry)

    uniq_triples = dedupe(all_triples)
    uniq_relations = dedupe(all_relations)
    return {
        "conv_id": conv.conversation_id,
        "scope": {"do_comprehension": do_comprehension, "do_verification": do_verification},
        "groups": out_groups,
        "n_triples": len(uniq_triples),
        "n_relations": len(uniq_relations),
        "triples": [t.to_dict() for t in uniq_triples],
        "relations": [r.to_dict() for r in uniq_relations],
    }
