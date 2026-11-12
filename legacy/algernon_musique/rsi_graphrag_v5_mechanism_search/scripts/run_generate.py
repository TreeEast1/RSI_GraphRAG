#!/usr/bin/env python3
"""Generate comprehension candidates for frozen units.

Each method starts from the same unit text and an empty G_comp. G_raw is not
written. A and B are skipped when the formatted unit hits the historical cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V4 = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/experiments/rsi_graphrag_v4")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V4 / "src"))

from rsi_v4.hipporag_adapter import extract_json_object  # noqa: E402

from v5_mech.client import Budget, BudgetExceeded, LunaClient  # noqa: E402
from v5_mech.paths import ART, OPENIE, PROMPTS, V4_PROMPTS  # noqa: E402
from v5_mech.verifier import verify_many  # noqa: E402

METHOD_PROMPTS = {
    "A": V4_PROMPTS / "comprehension_protocol_a.txt",
    "B": V4_PROMPTS / "comprehension_protocol_b.txt",
    "C": PROMPTS / "method_c_evidence_chain.txt",
    "Ccompact": PROMPTS / "method_c_compact.txt",
    "S": PROMPTS / "method_c2_structure_gap.txt",
    "C2": PROMPTS / "method_c2_crossfact_first.txt",
    "D": PROMPTS / "method_d_graph_gap.txt",
    "E": PROMPTS / "method_e_retrieval.txt",
}


def load_units(prefix: str, directory: Path | None = None) -> list:
    directory = directory or (ART / "units")
    rows = []
    for path in sorted(directory.glob(f"{prefix}_*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no frozen units for {prefix}")
    return rows


def passage_views(unit: dict, cited: list) -> tuple:
    passages = {}
    triples = {}
    for pid in cited:
        path = OPENIE / f"{pid}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        passages[pid] = record.get("content") or unit["passages"].get(pid, "")
        triples[pid] = record.get("triples") or []
    for pid, content in unit["passages"].items():
        passages.setdefault(pid, content)
    return passages, triples


def cited_ids(updates: list) -> list:
    found = []
    for update in updates:
        for span in update.get("evidence_spans") or []:
            pid = span.get("passage_id")
            if pid and pid not in found:
                found.append(pid)
        for pid in update.get("source_passage_ids") or []:
            if pid not in found:
                found.append(pid)
        for item in update.get("chain") or []:
            pid = item.get("passage_id")
            if pid and pid not in found:
                found.append(pid)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="hist or new")
    parser.add_argument("--methods", default="C,D")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--force-baselines", action="store_true")
    parser.add_argument("--roles", default="", help="Comma-separated unit roles to keep")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--budget", default="")
    parser.add_argument("--unit-dir", default="")
    parser.add_argument("--input-field", default="unit_text")
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    units = load_units(args.prefix, Path(args.unit_dir) if args.unit_dir else None)
    if args.roles:
        keep = {item.strip() for item in args.roles.split(",") if item.strip()}
        units = [unit for unit in units if unit.get("role") in keep]
        if not units:
            raise SystemExit(f"no units matched {sorted(keep)}")
    out = ART / "runs" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(Path(args.budget) if args.budget else ART / "budget.json")
    client = LunaClient(budget)
    manifest_before = (ART / "units" / "g_raw_triple_manifest.sha256").read_text(encoding="utf-8").strip()
    results = []
    for unit in units:
        hits = unit.get("cache_key_hits") or {}
        for method in methods:
            if method in {"A", "B"} and not args.force_baselines and hits.get(f"{'small' if 'small' in unit['role'] else method}"):
                pass
            if method in {"A", "B"} and not args.force_baselines:
                role_hits = set(hits)
                if method in {key.split("_")[-1] for key in role_hits}:
                    print("skip cached baseline", unit["role"], method, flush=True)
                    results.append({"role": unit["role"], "method": method, "status": "historical_cache", "cache_key_hits": hits})
                    continue
            prompt = METHOD_PROMPTS[method].read_text(encoding="utf-8")
            user_text = unit.get(args.input_field) or unit["unit_text"]
            stage = f"{args.run_id}_{unit['role']}_{method}"
            print("call", stage, "chars", len(user_text), flush=True)
            try:
                response = client.complete(
                    stage=stage,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_text},
                    ],
                    max_completion_tokens=3500,
                    reasoning_effort=args.effort,
                )
            except BudgetExceeded as exc:
                results.append({"role": unit["role"], "method": method, "status": "stopped_budget", "error": str(exc)})
                print("budget stop", exc, flush=True)
                (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                return
            except Exception as exc:  # noqa: BLE001
                results.append({"role": unit["role"], "method": method, "status": "error", "error": str(exc)[:500]})
                print("error", stage, exc, flush=True)
                continue
            raw_path = out / f"{unit['role']}__{method}_raw.txt"
            raw_path.write_text(response["text"], encoding="utf-8")
            try:
                parsed = extract_json_object(response["text"])
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "role": unit["role"],
                    "method": method,
                    "status": "parse_error",
                    "error": str(exc)[:300],
                    "cache_hit": response["cache_hit"],
                    "usage": response["metadata"],
                })
                continue
            if isinstance(parsed, list):
                parsed = {"updates": parsed}
            if not isinstance(parsed, dict):
                results.append({
                    "role": unit["role"],
                    "method": method,
                    "status": "parse_error",
                    "error": f"unexpected json type {type(parsed).__name__}",
                    "cache_hit": response["cache_hit"],
                    "usage": response["metadata"],
                })
                continue
            updates = parsed.get("updates") or parsed.get("operations") or []
            if isinstance(updates, dict):
                updates = [updates]
            passages, triples = passage_views(unit, cited_ids(updates))
            judgments = verify_many(updates, passages, triples)
            record = {
                "role": unit["role"],
                "cluster_id": unit["cluster_id"],
                "unit_id": unit["unit"]["unit_id"],
                "method": method,
                "status": "completed",
                "cache_hit": response["cache_hit"],
                "cache_source": response["cache_source"],
                "usage": response["metadata"],
                "n_updates": len(updates),
                "n_add": sum(1 for item in updates if (item.get("operation") or "").upper() == "ADD"),
                "n_accept": sum(1 for item in judgments if item["decision"] == "ACCEPT"),
                "n_reject": sum(1 for item in judgments if item["decision"] == "REJECT"),
                "n_mismatch": sum(1 for item in judgments if item["decision"] == "EVIDENCE_MISMATCH"),
                "n_uncertain": sum(1 for item in judgments if item["decision"] == "UNCERTAIN"),
                "types": {},
                "gaps": parsed.get("gaps") or [],
                "updates": updates,
                "judgments": judgments,
            }
            for item in judgments:
                record["types"][item.get("audit_type") or "NA"] = record["types"].get(item.get("audit_type") or "NA", 0) + 1
            (out / f"{unit['role']}__{method}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append({key: record[key] for key in ("role", "method", "status", "cache_hit", "n_add", "n_accept", "n_reject", "n_mismatch", "types", "usage")})
            print(" done", unit["role"], method, "add", record["n_add"], "accept", record["n_accept"], "hit", response["cache_hit"], flush=True)
    manifest_after = (ART / "units" / "g_raw_triple_manifest.sha256").read_text(encoding="utf-8").strip()
    summary = {
        "run_id": args.run_id,
        "methods": methods,
        "g_raw_manifest_unchanged": manifest_before == manifest_after,
        "budget": budget.data,
        "results": results,
    }
    (out / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("run_id", "g_raw_manifest_unchanged")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
