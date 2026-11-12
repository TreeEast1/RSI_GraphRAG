"""Read-time atomic-fact extraction, cross-paragraph comprehension, verification, summaries.

Only paper text and previously extracted facts enter the prompts. The functions
return fully validated records with source passage ids; nothing downstream is
allowed to use an insight without provenance.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..adapters.hy4_client import parse_json_loose
from .grouping import ReadingGroup
from .prompts import (ATOMIC_SYSTEM, COMPREHENSION_SYSTEM, SUMMARY_SYSTEM, VERIFY_SYSTEM,
                      render_atomic_prompt, render_comprehension_prompt,
                      render_summary_prompt, render_verify_prompt)

OUT = Path(__file__).resolve().parents[2] / "outputs" / "qasper_readtime"
RELATION_TYPES = {"method_result", "claim_evidence", "assumption_limitation", "setting_outcome",
                  "problem_solution", "entity_resolution", "combined_fact"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


_BAD_ENTITIES = {"we", "our", "us", "it", "they", "them", "this paper", "this work", "the paper",
                 "the authors", "authors", "paper", "this study", "the study", "our work",
                 "the proposed method", "the model", "results", "experiments"}
_BAD_PREFIXES = ("section ", "table ", "figure ", "fig. ", "eq. ", "equation ", "appendix ",
                 "bibref", "tabref", "figref", "we ", "our ")


def valid_entity(text: str) -> bool:
    t = str(text or "").strip()
    n = _norm(t)
    if len(n) < 2 or len(n) > 120:
        return False
    if n in _BAD_ENTITIES or any(n.startswith(p) for p in _BAD_PREFIXES):
        return False
    # reject pure math/number/punctuation strings
    if not re.search(r"[A-Za-z]{2}", t):
        return False
    if n.startswith("$") and n.endswith("$"):
        return False
    return True


def _salvage_objects(text: str, key: str) -> List[Dict]:
    """Recover complete objects from a truncated JSON array (e.g. ``"facts": [...]``)."""
    if not text:
        return []
    i = text.find(f'"{key}"')
    if i < 0:
        i = text.find(key)
    j = text.find("[", i)
    if j < 0:
        return []
    depth, in_str, esc, start, items = 0, False, False, None, []
    for k in range(j + 1, len(text)):
        ch = text[k]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = k
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    items.append(json.loads(text[start:k + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
        elif ch == "]" and depth == 0:
            break
    return items


def _call_json(client, prompt: str, system: str, max_tokens: int, tag: str, attempts: int = 2,
               salvage_key: Optional[str] = None):
    """JSON call that retries with a larger budget and salvages truncated arrays."""
    last = None
    for attempt in range(attempts):
        budget = min(16384, max_tokens * (2 ** attempt))
        out = client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=budget, temperature=0.0, json_mode=True, tag=tag,
        )
        parsed = parse_json_loose(out["content"])
        last = out
        if parsed is not None:
            return parsed, out, attempt
        if salvage_key:
            items = _salvage_objects(out["content"], salvage_key)
            if items:
                out["salvaged"] = True
                return {salvage_key: items, "salvaged": True}, out, attempt
        if out.get("finish_reason") != "length":
            break
    return None, last, attempts


def _usage(out) -> Dict:
    return (out or {}).get("usage") or {}


# --------------------------------------------------------------------------- #
# Atomic facts
# --------------------------------------------------------------------------- #
def parse_facts(raw, group: ReadingGroup, allowed: Sequence[str],
                id_prefix: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
    allowed = set(allowed)
    prefix = id_prefix or group.group_id
    facts, rejects, seen = [], [], set()
    for i, item in enumerate(raw or []):
        if not isinstance(item, dict):
            continue
        s, p, o = (str(item.get(k, "")).strip() for k in ("subject", "predicate", "object"))
        src = [str(x) for x in (item.get("source_passage_ids") or [])]
        src = [x for x in src if x in allowed]
        if not s or not p or not o or len(set(src)) != 1:
            rejects.append({"reason": "invalid_fact", "item": item})
            continue
        if not (valid_entity(s) and valid_entity(o)):
            rejects.append({"reason": "invalid_entity", "item": [s, p, o]})
            continue
        key = (_norm(s), _norm(p), _norm(o))
        if key in seen:
            rejects.append({"reason": "duplicate", "item": [s, p, o]})
            continue
        seen.add(key)
        facts.append({
            "fact_id": f"{prefix}:f{len(facts):02d}",
            "subject": s, "predicate": p, "object": o,
            "source_passage_ids": [src[0]],
            "source_section": None,
            "evidence": str(item.get("evidence", ""))[:500],
            "group_id": group.group_id,
        })
    return facts, rejects


def run_atomic_group(client, paper, group: ReadingGroup, max_tokens: int = 3072,
                     block_workers: int = 10) -> Dict:
    """Extract atomic facts in small blocks (blocks run concurrently)."""
    from concurrent.futures import ThreadPoolExecutor

    from .prompts import split_blocks

    blocks = split_blocks(group.passage_ids, paper)

    def _one(pair):
        bi, block = pair
        prompt = render_atomic_prompt(paper, block)
        parsed, out, attempts = _call_json(client, prompt, ATOMIC_SYSTEM, max_tokens,
                                           tag=f"qasper_atomic:{paper.paper_id}",
                                           salvage_key="facts")
        raw = (parsed or {}).get("facts", []) if isinstance(parsed, dict) else []
        facts, rej = parse_facts(raw, group, block, id_prefix=f"{group.group_id}:b{bi}")
        call = {"block": bi, "n_passages": len(block), "n_raw": len(raw),
                "error": None if parsed is not None else "unparseable_or_truncated",
                "salvaged": bool((out or {}).get("salvaged")),
                "finish_reason": (out or {}).get("finish_reason"),
                "usage": _usage(out), "retries": attempts,
                "input_tokens": sum(paper.paragraph_map()[p].n_tokens for p in block)}
        return facts, rej, call

    with ThreadPoolExecutor(max_workers=max(1, min(block_workers, len(blocks)))) as ex:
        results = list(ex.map(_one, list(enumerate(blocks))))
    all_facts, rejects, calls = [], [], []
    for facts, rej, call in results:
        all_facts.extend(facts)
        rejects.extend(rej)
        calls.append(call)
    return {"group_id": group.group_id, "facts": all_facts, "rejects": rejects,
            "n_raw": sum(c["n_raw"] for c in calls),
            "error": next((c["error"] for c in calls if c["error"]), None),
            "calls": calls}


# --------------------------------------------------------------------------- #
# Comprehension insights (triples + units from one call)
# --------------------------------------------------------------------------- #
def parse_insights(raw, group: ReadingGroup, allowed: Sequence[str],
                   fact_ids: Sequence[str]) -> Tuple[List[Dict], List[Dict]]:
    allowed = set(allowed)
    fact_ids = set(fact_ids)
    insights, rejects, seen = [], [], set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("concise_text", "")).strip()
        tri = item.get("triple") or {}
        s, p, o = (str(tri.get(k, "")).strip() for k in ("subject", "predicate", "object"))
        src = [str(x) for x in (item.get("source_passage_ids") or [])]
        src = [x for x in src if x in allowed]
        src = list(dict.fromkeys(src))
        reasons = []
        if not text:
            reasons.append("empty_text")
        if not (s and p and o):
            reasons.append("invalid_triple")
        if len(src) < 2:
            reasons.append("needs_two_passages")
        if not (valid_entity(s) and valid_entity(o)):
            reasons.append("invalid_entity")
        if reasons:
            rejects.append({"reason": "+".join(reasons), "item": {"text": text[:200], "triple": [s, p, o],
                                                                   "source_passage_ids": src}})
            continue
        key = (_norm(s), _norm(p), _norm(o))
        if key in seen:
            rejects.append({"reason": "duplicate_triple", "item": [s, p, o]})
            continue
        seen.add(key)
        rt = str(item.get("relation_type", "combined_fact"))
        insights.append({
            "insight_id": f"{group.group_id}:i{len(insights):02d}",
            "paper_id": group.paper_id,
            "group_id": group.group_id,
            "concise_text": text,
            "triple": {"subject": s, "predicate": p, "object": o},
            "relation_type": rt if rt in RELATION_TYPES else "combined_fact",
            "source_passage_ids": src,
            "supporting_fact_ids": [str(x) for x in (item.get("supporting_fact_ids") or [])
                                    if str(x) in fact_ids],
            "derivation": str(item.get("derivation", "Composed")),
            "confidence": max(0.0, min(1.0, float(item.get("confidence", 0.5) or 0.5))),
            "verification_status": "unverified",
            "verification": None,
        })
    return insights, rejects


def run_comprehension_group(client, paper, group: ReadingGroup, facts: List[Dict],
                            max_tokens: int = 4096) -> Dict:
    prompt = render_comprehension_prompt(paper, group.passage_ids, facts)
    parsed, out, attempts = _call_json(client, prompt, COMPREHENSION_SYSTEM, max_tokens,
                                       tag=f"qasper_compr:{paper.paper_id}",
                                       salvage_key="insights")
    raw = (parsed or {}).get("insights", []) if isinstance(parsed, dict) else []
    insights, rejects = parse_insights(raw, group, group.passage_ids,
                                       [f["fact_id"] for f in facts])
    return {"group_id": group.group_id, "insights": insights, "rejects": rejects,
            "n_raw": len(raw), "error": None if parsed is not None else "unparseable_or_truncated",
            "salvaged": bool((out or {}).get("salvaged")),
            "finish_reason": (out or {}).get("finish_reason"),
            "usage": _usage(out), "retries": attempts}


def verify_paper(client, paper, insights: List[Dict], batch_size: int = 15) -> Tuple[List[Dict], List[Dict]]:
    """One or more verifier calls per paper. Returns (kept_or_flagged, verdict_records)."""
    by_id = paper.paragraph_map()
    sources = {p.passage_id: p.text for p in paper.paragraphs}
    verdicts: Dict[str, Dict] = {}
    calls = []
    for start in range(0, len(insights), batch_size):
        batch = insights[start:start + batch_size]
        prompt = render_verify_prompt(paper, batch, sources)
        try:
            parsed, out, _ = _call_json(client, prompt, VERIFY_SYSTEM, 2048,
                                        tag=f"qasper_verify:{paper.paper_id}",
                                        salvage_key="verdicts")
            err = None
        except Exception as exc:  # noqa: BLE001 - endpoint failure, keep the paper
            parsed, out, err = None, None, f"{type(exc).__name__}: {exc}"[:300]
        v = (parsed or {}).get("verdicts", []) if isinstance(parsed, dict) else []
        for item in v:
            if isinstance(item, dict) and item.get("insight_id"):
                verdicts[str(item["insight_id"])] = item
        calls.append({"n": len(batch), "parsed": parsed is not None, "error": err,
                      "usage": _usage(out)})
    kept, records = [], []
    for ins in insights:
        v = verdicts.get(ins["insight_id"])
        rec = {"insight_id": ins["insight_id"], "verdict": v}
        records.append(rec)
        if v is None:
            ins["verification_status"] = "missing"
            kept.append(ins)
            continue
        keep = bool(v.get("keep", True))
        if keep and v.get("causality_error"):
            keep = False
        ins["verification_status"] = "kept" if keep else "rejected"
        ins["verification"] = {k: v.get(k) for k in ("support", "novelty", "causality_error", "reason")}
        if keep:
            kept.append(ins)
        else:
            records[-1]["reason"] = v.get("reason")
    return kept, {"verdicts": records, "calls": calls}


# --------------------------------------------------------------------------- #
# Single-paragraph summaries (M6 control)
# --------------------------------------------------------------------------- #
def run_summary_group(client, paper, group: ReadingGroup, max_tokens: int = 8192) -> Dict:
    prompt = render_summary_prompt(paper, group.passage_ids)
    parsed, out, attempts = _call_json(client, prompt, SUMMARY_SYSTEM, max_tokens,
                                       tag=f"qasper_summary:{paper.paper_id}",
                                       salvage_key="summaries")
    raw = (parsed or {}).get("summaries", []) if isinstance(parsed, dict) else []
    allowed = set(group.passage_ids)
    summaries, seen = [], set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("passage_id", ""))
        text = str(item.get("summary", "")).strip()
        if pid in allowed and pid not in seen and text:
            seen.add(pid)
            summaries.append({"passage_id": pid, "summary": text, "group_id": group.group_id})
    return {"group_id": group.group_id, "summaries": summaries,
            "n_raw": len(raw), "error": None if parsed is not None else "unparseable_or_truncated",
            "salvaged": bool((out or {}).get("salvaged")),
            "finish_reason": (out or {}).get("finish_reason"),
            "usage": _usage(out), "retries": attempts}


def merge_insights(all_insights: List[Dict]) -> List[Dict]:
    """Deduplicate insights repeated across overlapping groups by triple key."""
    merged: Dict[tuple, Dict] = {}
    order = []
    for ins in all_insights:
        t = ins["triple"]
        key = (_norm(t["subject"]), _norm(t["predicate"]), _norm(t["object"]))
        if key in merged:
            prev = merged[key]
            prev["source_passage_ids"] = sorted(set(prev["source_passage_ids"]) | set(ins["source_passage_ids"]))
            prev["supporting_fact_ids"] = sorted(set(prev["supporting_fact_ids"]) | set(ins["supporting_fact_ids"]))
            prev["confidence"] = max(prev["confidence"], ins["confidence"])
        else:
            merged[key] = ins
            order.append(key)
    out = [merged[k] for k in order]
    for i, ins in enumerate(out):
        ins["unit_id"] = f"{ins['paper_id']}::u{i:03d}"
    return out
