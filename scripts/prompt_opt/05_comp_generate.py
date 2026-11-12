#!/usr/bin/env python
"""Phase 3.1 — generate comprehension relations with COMP-P1..P5 on the frozen C2
reading groups, coerce them to the existing schema, and (for P5) verify them.

COMP-P0 is NOT generated: the original C3 extraction is reused verbatim, so the
baseline graph and its results stay exactly as recorded.

Outputs:
  comprehension/prompts/{pid}_group{g}.txt
  comprehension/extractions/{pid}.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import PATHS  # noqa: E402
from src.dataset.locomo import load_locomo, num_tokens  # noqa: E402
from src.prompt_optimization import comprehension_prompts as CP  # noqa: E402

OUT = ROOT / "outputs" / "prompt_optimization_conv30" / "comprehension"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm(s: str) -> str:
    return " ".join(str(s).strip().strip('"').strip("'").lower().split())


def sessions_block(conv, session_ids):
    by_id = conv.session_map()
    parts = []
    for sid in session_ids:
        s = by_id[sid]
        lines = [f"=== Session {sid} | {s.date_time_raw} ==="]
        lines += [f"[{t.dia_id}] {t.speaker}: {t.text}" for t in s.turns]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def call_json(client, prompt: str, system: str, max_tokens: int, tag: str, attempts: int = 3):
    """Same truncation-aware retry as src.comprehension.extract._call_json."""
    from src.adapters.hy4_client import parse_json_loose
    last = None
    for attempt in range(attempts):
        budget = min(32768, max_tokens * (2 ** attempt))
        out = client.chat([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                          max_tokens=budget, temperature=0.0, json_mode=True, tag=tag)
        parsed = parse_json_loose(out["content"])
        last = out
        if parsed is not None:
            return parsed, out, attempt
        if out.get("finish_reason") != "length":
            break
    return None, last, attempts


def coerce_relations(raw_rels, conv, allowed_dia, allowed_sessions, explicit_keys, group_id):
    dia_to_session = {t.dia_id: t.session_id for s in conv.sessions for t in s.turns}
    rels, rejects = [], []
    seen = set()
    for item in raw_rels or []:
        if not isinstance(item, dict):
            continue
        s = str(item.get("subject", "")).strip()
        p = str(item.get("predicate", "")).strip()
        o = str(item.get("object", "")).strip()
        if not s or not p or not o:
            rejects.append({"reason": "empty_field", "triple": [s, p, o]})
            continue
        cited = item.get("source_dia_ids") or []
        if not isinstance(cited, list):
            cited = [cited]
        cited = [str(c).strip() for c in cited]
        valid = [d for d in cited if d in allowed_dia]
        invalid = [d for d in cited if d not in allowed_dia]
        sessions = sorted({dia_to_session[d] for d in valid})
        if len(sessions) < 2:
            rejects.append({"reason": "needs_two_sessions", "triple": [s, p, o],
                            "sessions": sessions, "invalid": invalid})
            continue
        key = (_norm(s), _norm(p), _norm(o))
        if key in explicit_keys:
            rejects.append({"reason": "duplicate_of_explicit", "triple": [s, p, o]})
            continue
        if key in seen:
            rejects.append({"reason": "duplicate_within_prompt", "triple": [s, p, o]})
            continue
        seen.add(key)
        ev = item.get("evidence", "")
        if isinstance(ev, list):
            ev = " || ".join(str(e) for e in ev)
        rels.append({
            "subject": s, "predicate": p, "object": o,
            "source_dia_ids": valid, "source_sessions": sessions,
            "evidence": str(ev)[:800],
            "derivation": str(item.get("derivation", "Composed")),
            "confidence": float(item.get("confidence", 0.5) or 0.5),
            "relation_type": str(item.get("relation_type", "cross_session")),
            "supporting_fact_ids": [str(x) for x in (item.get("supporting_fact_ids") or [])],
            "group_id": group_id,
            "schema_checks": {
                "subject_words": len(s.split()), "object_words": len(o.split()),
                "predicate_words": len(p.replace("_", " ").split()),
                "n_sessions": len(sessions), "invalid_dia_ids": invalid,
            },
        })
    return rels, rejects


def verify_with_evidence(client, candidates: List[Dict], conv, tag: str) -> (List[Dict], Dict):
    """Ask the verifier to keep/reject each candidate, showing cited turn text."""
    if not candidates:
        return [], {}
    by_dia = {t.dia_id: t for s in conv.sessions for t in s.turns}
    items = []
    for i, c in enumerate(candidates):
        cited = []
        for d in c["source_dia_ids"]:
            t = by_dia.get(d)
            if t:
                cited.append({"dia_id": d, "speaker": t.speaker, "text": t.text})
        items.append({
            "id": f"c{i}", "subject": c["subject"], "predicate": c["predicate"],
            "object": c["object"], "relation_type": c["relation_type"],
            "derivation": c["derivation"], "evidence": c["evidence"],
            "source_dia_ids": c["source_dia_ids"], "cited_turns": cited,
        })
    block = json.dumps(items, ensure_ascii=False, indent=1)
    prompt = f"{CP.VERIFY_INSTRUCTIONS}\n\nCandidates:\n{block}\n\nReturn the JSON now."
    try:
        out = client.complete_json(prompt, system=CP.VERIFY_SYSTEM, max_tokens=8192, tag=tag)
        verdicts = {v.get("id"): v for v in out.get("verdicts", []) if isinstance(v, dict)}
    except Exception as exc:  # noqa: BLE001
        return candidates, {"error": f"{type(exc).__name__}: {exc}", "n": len(candidates)}
    kept, decisions = [], {}
    for i, c in enumerate(candidates):
        v = verdicts.get(f"c{i}")
        decisions[f"c{i}"] = v
        if v is None or v.get("keep") is True:
            c = dict(c)
            c["verification"] = v
            kept.append(c)
    return kept, {"decisions": decisions, "n_candidates": len(candidates), "n_kept": len(kept)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", default="COMP-P1,COMP-P2,COMP-P3,COMP-P4,COMP-P5")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--shard", default="co")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import os
    os.environ["HY4_CACHE_SHARD"] = args.shard
    from src.adapters.hy4_client import HY4Client

    conv = {c.sample_id: c for c in load_locomo(ROOT / "data" / "raw" / "locomo" / "locomo10.json")}["conv-30"]
    c2 = json.loads((PATHS["comprehension"] / "conv-30" / "cluster_hybrid_0.5_b8192_triple.json").read_text())
    explicit_keys = {(_norm(t["subject"]), _norm(t["predicate"]), _norm(t["object"]))
                     for t in c2.get("triples", [])}

    client = HY4Client(tag="comp_gen")
    for pid in args.pids.split(","):
        spec = CP.get(pid)
        out_path = OUT / "extractions" / f"{pid}.json"
        if out_path.exists() and not args.overwrite:
            print(f"skip existing {out_path.name}")
            continue
        print(f"\n=== {pid} {spec.name} ===")
        groups_out = []
        all_rels = []
        for g in c2["groups"]:
            sids = g["session_ids"]
            block = sessions_block(conv, sids)
            facts = g["extraction"]["triples"]
            facts_block = "\n".join(f"f{i}: {t['subject']} | {t['predicate']} | {t['object']}"
                                    for i, t in enumerate(facts))
            prompt = CP.render(spec.instructions, block, facts_block)
            allowed_dia = {t.dia_id for sid in sids for t in conv.session_map()[sid].turns}
            OUT.joinpath("prompts").mkdir(parents=True, exist_ok=True)
            (OUT / "prompts" / f"{pid}_group{g['group_id']}.txt").write_text(
                f"# {pid} {spec.name}\n# system_sha256={sha(spec.system)}\n"
                f"# instructions_sha256={sha(spec.instructions)}\n# prompt_sha256={sha(prompt)}\n"
                f"# prompt_tokens={num_tokens(prompt)}\n\n--- SYSTEM ---\n{spec.system}\n\n"
                f"--- USER ---\n{prompt}\n", encoding="utf-8")
            t0 = time.time()
            raw, meta, attempts = call_json(client, prompt, spec.system, args.budget,
                                            tag=f"{pid}_g{g['group_id']}")
            if raw is None:
                groups_out.append({"group_id": g["group_id"], "session_ids": sids,
                                   "error": "unparseable_or_truncated",
                                   "finish_reason": (meta or {}).get("finish_reason"),
                                   "relations": [], "rejects": []})
                print(f"  group {g['group_id']}: FAILED")
                continue
            rels_raw = raw.get("relations", []) if isinstance(raw, dict) else []
            rels, rejects = coerce_relations(rels_raw, conv, allowed_dia, set(sids),
                                             explicit_keys, g["group_id"])
            groups_out.append({
                "group_id": g["group_id"], "session_ids": sids, "n_facts": len(facts),
                "prompt_tokens": num_tokens(prompt), "prompt_sha256": sha(prompt),
                "n_raw": len(rels_raw), "relations": rels, "rejects": rejects,
                "finish_reason": (meta or {}).get("finish_reason"),
                "usage": (meta or {}).get("usage"), "retries": attempts,
                "elapsed_s": round(time.time() - t0, 1),
            })
            all_rels.extend(rels)
            print(f"  group {g['group_id']}: raw={len(rels_raw)} kept={len(rels)} "
                  f"rejects={len(rejects)} {time.time()-t0:.0f}s")

        verification = None
        if pid == "COMP-P5" and all_rels:
            kept, vlog = verify_with_evidence(client, all_rels, conv, tag="COMP-P5_verify")
            verification = vlog
            # mark which candidates survived verification
            kept_keys = {(_norm(r["subject"]), _norm(r["predicate"]), _norm(r["object"])) for r in kept}
            for r in all_rels:
                r["verified_keep"] = (_norm(r["subject"]), _norm(r["predicate"]),
                                      _norm(r["object"])) in kept_keys
            all_rels = kept
            print(f"  verification kept {len(kept)}/{vlog.get('n_candidates')}")

        dedup = {}
        for r in all_rels:
            k = (_norm(r["subject"]), _norm(r["predicate"]), _norm(r["object"]))
            dedup.setdefault(k, r)
        final = list(dedup.values())
        res = {
            "conv_id": "conv-30", "pid": pid, "name": spec.name,
            "system": spec.system, "system_sha256": sha(spec.system),
            "instructions_sha256": sha(spec.instructions),
            "budget": args.budget, "groups": groups_out,
            "relations": final, "n_relations": len(final),
            "verification": verification,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        OUT.joinpath("extractions").mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  wrote {out_path} n_relations={len(final)}")

    print("HY4:", client.summary())


if __name__ == "__main__":
    main()
