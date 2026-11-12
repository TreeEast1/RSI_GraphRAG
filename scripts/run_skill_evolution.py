#!/usr/bin/env python
"""E4 Skill Evolution: learn retrieval-expansion "skills" on dev, freeze, apply to test.

A skill is a (category, query_suffix, observed_benefit) triple.  On dev we
evaluate every candidate suffix for every question category with retrieval-only
metrics, keep the best positive suffix per category, freeze the library to
`outputs/skills/frozen_skill_library.json`, and then apply that frozen library
to the test split.  Gold evidence is never visible to the selection procedure
except through the aggregate R@5 metric on dev, which is the intended dev signal.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PATHS  # noqa: E402
from src.dataset.locomo import load_locomo, num_tokens  # noqa: E402
from src.evaluation.metrics import select_by_budget  # noqa: E402
from src.question_skill.pipeline import rrf_merge  # noqa: E402

sys.path.insert(0, str(PATHS["third_party"] / "src"))

SUFFIXES = {
    "none": "",
    "temporal": "date time year month",
    "person": "who person name met",
    "place": "where place location",
    "reason": "why reason because",
    "sequence": "before after then later",
    "number": "how many number",
}
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def docs_of(conv):
    return [{"document_id": s.document_id, "session_id": s.session_id,
             "header": f"Session {s.session_id} | {s.date_time_raw}", "content": s.text}
            for s in conv.sessions]


def ranked_from_res(res, docs):
    by_content = {d["content"]: d for d in docs}
    out = []
    for d, m, sc in zip(res.docs, res.doc_metadata, res.doc_scores):
        base = by_content.get(d)
        if isinstance(m, dict) and m.get("document_id") and base:
            base = {**base, "document_id": m["document_id"], "session_id": m.get("session_id"),
                    "header": f"Session {m.get('session_id')} | {m.get('date_time','')}"}
        if base:
            out.append({**base, "score": float(sc)})
    return out


def eval_q(qa, ranked):
    gold = {f"{qa.conversation_id}::s{s:02d}" for s in qa.evidence_sessions}
    if not gold:
        return None
    top5 = {d["document_id"] for d in ranked[:5]}
    sel = select_by_budget(ranked, 4096, num_tokens)
    sel_ids = {d["document_id"] for d in sel}
    return {
        "r5": len(top5 & gold) / len(gold),
        "ev": len(sel_ids & gold) / len(gold),
    }


def run_split(conv_ids, convs, hippo_factory, suffixes, library=None, tag=""):
    """If library is None -> evaluate all suffixes; else apply the frozen library."""
    results = defaultdict(list)  # key -> list of metrics
    for conv_id in conv_ids:
        conv = convs[conv_id]
        docs = docs_of(conv)
        with hippo_factory(conv_id) as hippo:
            for qa in conv.qa:
                base = ranked_from_res(hippo.retrieve([qa.question], num_to_retrieve=20)[0], docs)
                base_m = eval_q(qa, base)
                if base_m is None:
                    continue
                results["__base__"].append(base_m)
                if library is not None:
                    results[f"{qa.category}|base"].append(base_m)
                cat = qa.category
                if library is None:
                    for name, suf in suffixes.items():
                        if name == "none":
                            continue
                        q2 = f"{qa.question} {suf}" if suf else qa.question
                        r2 = ranked_from_res(hippo.retrieve([q2], num_to_retrieve=20)[0], docs)
                        merged = rrf_merge([base, r2], top=20)
                        m = eval_q(qa, merged)
                        results[f"{cat}|{name}"].append(m)
                else:
                    name = library.get(str(cat), "none")
                    if name != "none":
                        q2 = f"{qa.question} {suffixes[name]}"
                        r2 = ranked_from_res(hippo.retrieve([q2], num_to_retrieve=20)[0], docs)
                        merged = rrf_merge([base, r2], top=20)
                        m = eval_q(qa, merged)
                    else:
                        m = base_m
                    results[str(cat)].append(m)
                if len(results["__base__"]) % 100 == 0:
                    print(f"  {tag} {conv_id}: {len(results['__base__'])} questions", flush=True)
    return results


def agg(ms):
    n = len(ms)
    return {"n": n, "r5": sum(m["r5"] for m in ms) / n, "ev": sum(m["ev"] for m in ms) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="conv-30,conv-44,conv-48")
    ap.add_argument("--test", default="conv-26,conv-41,conv-42,conv-43,conv-47,conv-49,conv-50")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--suffixes", default="temporal,person,place")
    ap.add_argument("--apply-only", action="store_true")
    ap.add_argument("--library", default="")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    from src.adapters.hy4_client import HY4Client
    from src.graph.hipporag_adapter import HY4CacheOpenAI
    from src.graph.runtime import EmbeddingProvider, open_hipporag, make_base_config
    from src.graph.variants import raw_save_dir

    global SUFFIXES
    keep = {"none": ""}
    for name in args.suffixes.split(","):
        if name in SUFFIXES:
            keep[name] = SUFFIXES[name]
    SUFFIXES = keep
    convs = {c.sample_id: c for c in load_locomo(ROOT / "data" / "raw" / "locomo" / "locomo10.json")}
    if args.max_questions:
        for c in convs.values():
            c.qa = c.qa[: args.max_questions]
    cfg0 = make_base_config(PATHS["raw_graphs"] / "_warmup")
    provider = EmbeddingProvider.get(cfg0)
    llm = HY4CacheOpenAI.from_experiment_config(cfg0)

    def factory(conv_id):
        return open_hipporag(str(raw_save_dir(conv_id)), conv_id=conv_id,
                             embedding_provider=provider, llm=llm)

    t0 = time.time()
    if args.apply_only:
        lib = json.loads(Path(args.library).read_text())
        libmap = lib["library"]
        SUFFIXES = lib["suffixes"]
        print("=== apply-only: frozen library", libmap, "to", args.test)
        test_ids = args.test.split(",")
        test = run_split(test_ids, convs, factory, SUFFIXES, library=libmap, tag="apply")
        applied_ms = []
        for cat in (1, 2, 3, 4):
            applied_ms.extend(test.get(str(cat), []))
        base_ms = []
        for cat in (1, 2, 3, 4):
            base_ms.extend(test.get(f"{cat}|base", []))
        out = {"library": libmap,
               "test_base": agg(base_ms) if base_ms else {"n": 0},
               "test_applied": agg(applied_ms) if applied_ms else {"n": 0},
               "elapsed_s": round(time.time() - t0, 1)}
        (PATHS["reports"] / f"skill_evolution_apply{args.out_suffix}.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return
    dev_ids = args.dev.split(",")
    print(f"=== dev evaluation ({len(dev_ids)} conversations, {len(SUFFIXES)-1} suffixes) ===")
    dev = run_split(dev_ids, convs, factory, SUFFIXES, library=None, tag="dev")
    base = agg(dev["__base__"])
    print("dev base:", base)

    # select best positive suffix per category on dev
    library, report = {}, {"base": base, "candidates": {}}
    for cat in (1, 2, 3, 4):
        best_name, best_r5 = "none", base["r5"]
        per_cat = {}
        for name in SUFFIXES:
            if name == "none":
                continue
            ms = dev.get(f"{cat}|{name}")
            if not ms:
                continue
            a = agg(ms)
            per_cat[name] = a
            if a["r5"] > best_r5 + 0.002:
                best_name, best_r5 = name, a["r5"]
        library[str(cat)] = best_name
        report["candidates"][str(cat)] = per_cat
        print(f"  category {cat} ({CATEGORY_NAMES[cat]}): chose {best_name} (r5 {best_r5:.3f})")

    lib_path = PATHS["skills"] / "frozen_skill_library.json"
    lib_path.parent.mkdir(parents=True, exist_ok=True)
    lib_path.write_text(json.dumps({
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dev_convs": dev_ids,
        "suffixes": SUFFIXES,
        "library": library,
        "dev_report": {k: v for k, v in report.items() if k != "candidates"},
        "dev_candidates": {c: {n: a for n, a in d.items()} for c, d in report["candidates"].items()},
    }, indent=2))
    print("frozen library ->", lib_path, library)

    print("=== test application (frozen) ===")
    test_ids = args.test.split(",")
    test = run_split(test_ids, convs, factory, SUFFIXES, library=library, tag="test")
    test_base = agg(test["__base__"])
    applied = test["__base__"]  # placeholder
    # aggregate applied rows across categories
    applied_ms = []
    for cat in (1, 2, 3, 4):
        applied_ms.extend(test.get(str(cat), []))
    applied = agg(applied_ms) if applied_ms else {"n": 0}
    out = {"library": library, "dev_base": base, "test_base": test_base, "test_applied": applied,
           "elapsed_s": round(time.time() - t0, 1)}
    (PATHS["reports"] / "skill_evolution_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
