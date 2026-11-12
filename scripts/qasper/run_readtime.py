#!/usr/bin/env python
"""Read-time stage for QASPER: atomic facts, cross-paragraph insights, verification, summaries.

No question, answer or gold evidence is ever passed to the model. Every insight
keeps its source passage ids; verified insights are written both as graph
triples (M3) and as retrievable comprehension units (M4).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.adapters.hy4_client import HY4Client, HY4Config, load_env  # noqa: E402
from src.qasper.evalutil import OUT, load_papers  # noqa: E402
from src.qasper.grouping import section_groups, semantic_groups  # noqa: E402
from src.qasper.readtime import (merge_insights, run_atomic_group, run_comprehension_group,
                                 run_summary_group, verify_paper)  # noqa: E402

load_env()


def make_client() -> HY4Client:
    import os

    cfg = HY4Config.from_env(max_concurrency=int(os.environ.get("HY4_MAX_CONCURRENCY", "6")))
    tag = os.environ.get("QASPER_RT_CACHE_TAG", "")
    return HY4Client(cfg, cache_path=OUT / "cache" / f"hy4_readtime{tag}.sqlite",
                     log_path=OUT / "logs" / f"hy4_calls_readtime{tag}.jsonl", tag="qasper_readtime")


def triples_by_passage(records, get_ids, get_triple):
    out: dict = {}
    for r in records:
        t = get_triple(r)
        for pid in get_ids(r):
            out.setdefault(pid, []).append(list(t))
    return out


def _safe(fn, fallback: dict, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - keep the batch alive on endpoint failures
        return {**fallback, "error": f"{type(exc).__name__}: {exc}"[:300]}


def process_paper(client, paper, grouping: str, embedder=None, workers: int = 4, force: bool = False) -> dict:
    out_dir = OUT / "comprehension" / grouping / paper.paper_id
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not force:
        prev = json.loads(meta_path.read_text())
        if prev.get("complete", True):
            return prev
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if grouping == "section":
        groups = section_groups(paper)
    else:
        groups = semantic_groups(paper, embedder)
    (out_dir / "groups.json").write_text(json.dumps([g.to_dict() for g in groups], indent=1))

    # 1) atomic facts per group (concurrent)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        atomic = list(ex.map(
            lambda g: _safe(run_atomic_group, {"group_id": g.group_id, "facts": [], "rejects": [],
                                               "n_raw": 0, "calls": []}, client, paper, g),
            groups))
    facts = [f for r in atomic for f in r["facts"]]
    for f in facts:
        f["source_section"] = paper.paragraph_map()[f["source_passage_ids"][0]].section_name
    (out_dir / "atomic_facts.json").write_text(json.dumps(
        {"facts": facts, "calls": atomic}, indent=1, ensure_ascii=False))

    # 2) comprehension per group, using that group's facts only
    facts_by_group = {}
    for r in atomic:
        facts_by_group.setdefault(r["group_id"], []).extend(r["facts"])
    groups2 = [g for g in groups if len(g.passage_ids) >= 2]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        compr = list(ex.map(
            lambda g: _safe(run_comprehension_group,
                            {"group_id": g.group_id, "insights": [], "rejects": [], "n_raw": 0},
                            client, paper, g, facts_by_group.get(g.group_id, [])),
            groups2))
    raw_insights = [i for r in compr for i in r["insights"]]
    merged = merge_insights(raw_insights)
    (out_dir / "insights_raw.json").write_text(json.dumps(
        {"insights": merged, "calls": compr, "skipped_single_paragraph_groups":
         [g.group_id for g in groups if len(g.passage_ids) < 2]}, indent=1, ensure_ascii=False))

    # 3) grounded verification
    kept, ver = verify_paper(client, paper, merged)
    rejected = [i for i in merged if i["verification_status"] == "rejected"]
    (out_dir / "insights_verified.json").write_text(json.dumps(
        {"kept": kept, "rejected": rejected, "verification": ver}, indent=1, ensure_ascii=False))

    # 4) per-paragraph summaries (M6 control)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        summ = list(ex.map(
            lambda g: _safe(run_summary_group, {"group_id": g.group_id, "summaries": [], "n_raw": 0},
                            client, paper, g),
            groups))
    summaries = [s for r in summ for s in r["summaries"]]
    (out_dir / "summaries.json").write_text(json.dumps(
        {"summaries": summaries, "calls": summ}, indent=1, ensure_ascii=False))

    # 5) knowledge files for graph injection / unit retrieval
    atom_triples = triples_by_passage(facts, lambda f: f["source_passage_ids"],
                                      lambda f: (f["subject"], f["predicate"], f["object"]))
    comp_triples = triples_by_passage(kept, lambda i: i["source_passage_ids"],
                                      lambda i: (i["triple"]["subject"], i["triple"]["predicate"],
                                                 i["triple"]["object"]))
    (out_dir / "triples_atomic_by_passage.json").write_text(json.dumps(atom_triples, indent=1, ensure_ascii=False))
    (out_dir / "triples_comp_by_passage.json").write_text(json.dumps(comp_triples, indent=1, ensure_ascii=False))
    with open(out_dir / "units.jsonl", "w", encoding="utf-8") as f:
        for i, ins in enumerate(kept):
            row = {"unit_id": f"{paper.paper_id}::u{i:03d}", **ins}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def tok(records):
        calls = []
        for r in records:
            calls.extend(r["calls"] if "calls" in r else [r])
        return {k: sum((c.get("usage") or {}).get(k, 0) or 0 for c in calls)
                for k in ("prompt_tokens", "completion_tokens")}

    reasons = {}
    for r in compr:
        for x in r["rejects"]:
            reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
    errors = [r.get("error") for r in atomic + compr + summ if r.get("error")]
    errors += [c.get("error") for c in ver.get("calls", []) if c.get("error")]
    endpoint_fail = any("HY4 call failed" in str(e) or "RateLimit" in str(e) for e in errors)
    meta = {
        "paper_id": paper.paper_id, "grouping": grouping,
        "n_paragraphs": len(paper.paragraphs),
        "n_groups": len(groups),
        "group_tokens": [g.n_tokens for g in groups],
        "n_atomic_facts": len(facts),
        "n_insights_raw": len(merged),
        "n_insights_verified": len(kept),
        "n_insights_rejected": len(rejected),
        "insight_rejects": reasons,
        "n_units": len(kept),
        "n_summaries": len(summaries),
        "atomic_usage": tok(atomic), "comprehension_usage": tok(compr),
        "summary_usage": tok(summ),
        "errors": [e for e in errors if e],
        "complete": not endpoint_fail,
        "runtime_s": round(time.time() - t0, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="smoke20")
    ap.add_argument("--grouping", default="section", choices=["section", "semantic"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--papers-file", default=None)
    args = ap.parse_args()
    papers = load_papers(args.scope)
    if args.papers_file:
        keep = {l.strip() for l in open(args.papers_file) if l.strip()}
        papers = {pid: p for pid, p in papers.items() if pid in keep}
    if args.num_shards > 1:
        papers = {pid: p for i, (pid, p) in enumerate(papers.items())
                  if i % args.num_shards == args.shard}
    client = make_client()
    embedder = None
    if args.grouping == "semantic":
        from src.embedding.nvembed import NVEmbedder
        from src.graph.hipporag_adapter import make_base_config
        from src.graph.runtime import EmbeddingProvider
        prov = EmbeddingProvider.get(make_base_config(OUT / "cache" / "embed_cfg"))
        embedder = NVEmbedder(prov.model, cache_path=OUT / "cache" / "para_embeddings.pkl")
    tot = {"papers": 0, "facts": 0, "verified": 0, "rejected": 0, "units": 0}
    for i, (pid, paper) in enumerate(papers.items(), 1):
        if args.limit and i > args.limit:
            break
        print(f"=== [{i}/{len(papers)}] {pid}: {len(paper.paragraphs)} paragraphs", flush=True)
        m = process_paper(client, paper, args.grouping, embedder, args.workers, args.force)
        tot["papers"] += 1
        tot["facts"] += m.get("n_atomic_facts", 0)
        tot["verified"] += m.get("n_insights_verified", 0)
        tot["rejected"] += m.get("n_insights_rejected", 0)
        tot["units"] += m.get("n_units", 0)
        print(f"    groups={m.get('n_groups')} facts={m.get('n_atomic_facts')} "
              f"insights={m.get('n_insights_verified')}/{m.get('n_insights_raw')} "
              f"summaries={m.get('n_summaries')} {m.get('runtime_s')}s", flush=True)
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
