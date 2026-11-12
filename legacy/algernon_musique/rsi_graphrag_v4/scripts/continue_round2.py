#!/usr/bin/env python3
"""Extend the mechanism pilot by two more update questions.

The first frozen mechanism graph is left untouched. New OpenIE reuses the Luna
cache. The new snapshot is written beside it.
"""

from __future__ import annotations

import json
import random
import sys
import traceback
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")

from rsi_v4.answer_skill_store import AnswerSkillStore  # noqa: E402
from rsi_v4.comprehension_evolver import evolve_unit  # noqa: E402
from rsi_v4.comprehension_store import ComprehensionStore  # noqa: E402
from rsi_v4.corpus_manager import load_corpus, load_eval_questions  # noqa: E402
from rsi_v4.embeddings import NVEmbedEncoder  # noqa: E402
from rsi_v4.evolution_analyzer import diagnose  # noqa: E402
from rsi_v4.experiment_runner import (  # noqa: E402
    ROOT as EXP_ROOT,
    _safe,
    _write_json,
    evaluate_questions,
    resolve_luna_key,
    run_openie,
)
from rsi_v4.graph_analyzer import analyze_openie_records, materialize_cluster  # noqa: E402
from rsi_v4.graph_partitioner import audit_partition, partition_cluster  # noqa: E402
from rsi_v4.graph_patch_manager import GraphPatchManager  # noqa: E402
from rsi_v4.hipporag_adapter import (  # noqa: E402
    build_raw_graph,
    load_openie_dir,
    prompt_manager,
    save_raw_graph,
)
from rsi_v4.llm_client import BudgetExceeded, LLMClient, UsageLedger  # noqa: E402
from rsi_v4.rag import rag  # noqa: E402
from rsi_v4.rag_verifier import verify_candidates  # noqa: E402
from rsi_v4.reporting import write_reports  # noqa: E402
from rsi_v4.retrieval import GraphRetriever  # noqa: E402
from rsi_v4.retrieval_skill_store import RetrievalSkillStore  # noqa: E402

FRESH_CAP = 110
FIRST_QUESTION = "2hop__77134_88165"


def main() -> None:
    config = json.loads((EXP_ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    art = EXP_ROOT / "artifacts"
    out = art / "round2"
    out.mkdir(parents=True, exist_ok=True)
    secret, auth = resolve_luna_key(config, art)
    if not secret:
        raise SystemExit(f"Luna auth failed: {auth}")
    questions = load_eval_questions(str(EXP_ROOT / "data" / "questions_eval.jsonl"))
    corpus = load_corpus(str(EXP_ROOT / "data" / "corpus.jsonl"))
    by_id = {row["passage_id"]: row for row in corpus}
    rng = random.Random(config["seed"])
    update_ids = sorted(row["question_id"] for row in questions if row.get("split_role") == "update")
    rng.shuffle(update_ids)
    if update_ids[0] != FIRST_QUESTION:
        raise SystemExit(f"seeded first update question changed: {update_ids[0]}")
    extra_ids = update_ids[1:3]
    selected_ids = [FIRST_QUESTION] + extra_ids
    selected = [row for row in questions if row["question_id"] in set(selected_ids)]
    if any(row.get("split_role") == "heldout" for row in selected):
        raise SystemExit("held-out question entered round 2")
    passage_ids = []
    for question in selected:
        for paragraph in question["paragraphs"]:
            pid = paragraph["passage_id"]
            if pid in by_id and pid not in passage_ids:
                passage_ids.append(pid)
    print("round2 questions", selected_ids, "passages", len(passage_ids))
    ledger = UsageLedger(max_fresh_calls=FRESH_CAP)
    client = LLMClient(
        cache_path=str(art / "llm_cache.sqlite"),
        deployment=config["llm"]["deployment"],
        endpoint=config["llm"]["endpoint"],
        api_version=config["llm"]["api_version"],
        api_key=secret,
        ledger=ledger,
        seed=config["llm"]["seed"],
    )
    failures = art / "failures.jsonl"
    openie_dir = art / "openie" / "luna"
    openie_status = run_openie(
        [by_id[pid] for pid in passage_ids],
        client,
        openie_dir,
        prompt_manager(),
        config["llm"]["openie_reasoning_effort"],
        config["budget"]["openie_workers"],
        failures,
        secret,
    )
    print("openie", openie_status)
    records = {row["passage_id"]: row for row in load_openie_dir(str(openie_dir))}
    ready = [records[pid] for pid in passage_ids if pid in records]
    manifest = json.loads((EXP_ROOT / "data" / "corpus_manifest.json").read_text(encoding="utf-8"))
    graph = build_raw_graph(
        ready,
        status="mechanism_round2_not_full_corpus",
        model_name=config["llm"]["deployment"],
        corpus_hash=manifest["corpus_hash"],
    )
    graph_path = art / "graphs" / "mechanism_round2_g_raw.json"
    if graph_path.exists() and json.loads(graph_path.read_text(encoding="utf-8")).get("graph_hash") != graph["graph_hash"]:
        graph_path = art / "graphs" / "mechanism_round2b_g_raw.json"
    save_raw_graph(graph, str(graph_path))
    first_graph = art / "graphs" / "mechanism_g_raw.json"
    first_hash = json.loads(first_graph.read_text(encoding="utf-8"))["graph_hash"]
    if json.loads(first_graph.read_text(encoding="utf-8"))["graph_hash"] != first_hash:
        raise SystemExit("first G_raw changed")
    encoder = tiktoken.get_encoding(config["reading"]["tokenizer"])
    token_by_passage = {row["passage_id"]: len(encoder.encode(row["content"])) for row in graph["passages"].values()}
    bundle = analyze_openie_records(
        ready,
        token_by_passage=token_by_passage,
        input_token_budget=config["reading"]["input_token_budget"],
        source_label="luna_round2",
    )
    store = ComprehensionStore()
    manager = GraphPatchManager(graph, store)
    raw_hash = graph["graph_hash"]
    overhead = len(encoder.encode((EXP_ROOT / "prompts" / "comprehension_system.txt").read_text(encoding="utf-8")))
    comp_results = []
    units_seen = 0
    for component in sorted(bundle["relation_components"], key=lambda row: -row["n_nodes"]):
        if units_seen >= 3 or ledger.fresh_calls >= FRESH_CAP - 12:
            break
        cluster = materialize_cluster(bundle, component)
        cluster["graph_id"] = raw_hash[:12]
        units = partition_cluster(
            cluster,
            input_token_budget=config["reading"]["input_token_budget"],
            overlap_hops=1,
            count_tokens=lambda text: len(encoder.encode(text)),
            overhead_tokens=overhead,
        )
        audit = audit_partition(cluster, units, config["reading"]["input_token_budget"])
        for unit in units:
            if units_seen >= 3:
                break
            try:
                outcome = evolve_unit(
                    client,
                    cluster,
                    unit,
                    manager,
                    reasoning_effort=config["llm"]["comprehension_reasoning_effort"],
                )
            except BudgetExceeded:
                outcome = {"unit_id": unit["unit_id"], "status": "stopped_budget"}
            except Exception as exc:  # noqa: BLE001
                outcome = {"unit_id": unit["unit_id"], "status": "error", "error": _safe(exc, secret)}
            comp_results.append(
                {
                    "unit_id": unit["unit_id"],
                    "core_complete": audit["core_coverage_complete"],
                    "n_units": audit["n_units"],
                    "boundary_retention": audit["boundary_retention_rate"],
                    "accepted": len((outcome.get("accepted") or [])),
                    "rejected": len((outcome.get("rejected") or [])),
                    "status": outcome.get("status"),
                    "reject_reasons": [item.get("reason") for item in (outcome.get("rejected") or [])],
                }
            )
            units_seen += 1
            if outcome.get("status") == "stopped_budget":
                break
    assert graph["graph_hash"] == raw_hash
    _write_json(out / "comprehension_store.json", store.to_dict())
    indexed = set(graph["passages"])
    covered = [
        question
        for question in selected
        if question["gold_passage_ids"] and set(question["gold_passage_ids"]) <= indexed
    ]
    retrieval_payload = {
        "covered": [question["question_id"] for question in covered],
        "missing_gold": [
            question["question_id"]
            for question in selected
            if not (question["gold_passage_ids"] and set(question["gold_passage_ids"]) <= indexed)
        ],
    }
    embedder = NVEmbedEncoder(config["embedding_model_path"])
    try:
        embedder.load()
        retrieval_payload["embedding_device"] = embedder.device
        empty = ComprehensionStore()
        retrieval_skills = RetrievalSkillStore()
        answer_skills = AnswerSkillStore()
        r0 = evaluate_questions(
            covered, GraphRetriever(graph, empty, embedder, config["retrieval"]), client,
            retrieval_skills, answer_skills, config["retrieval"]["qa_top_k"],
        )
        r1 = evaluate_questions(
            covered, GraphRetriever(graph, store, embedder, config["retrieval"]), client,
            retrieval_skills, answer_skills, config["retrieval"]["qa_top_k"],
        )
        retrieval_payload["R0"] = r0
        retrieval_payload["R1"] = r1
        retrieval_payload["r1_comp_in_top10"] = sum(1 for row in r1 if row.get("comprehension_ids_in_top10"))
        q_results = []
        by_q = {question["question_id"]: question for question in covered}
        for row in r1:
            question = by_q[row["question_id"]]
            needs_update = row["em"] < 1 or row.get("Recall@5", 0) < 1
            if not needs_update:
                q_results.append({"question_id": row["question_id"], "status": "rollout_already_correct", "r1": row})
                continue
            if ledger.fresh_calls >= FRESH_CAP - 4:
                q_results.append({"question_id": row["question_id"], "status": "stopped_budget_before_question_time"})
                continue
            rollout = rag(
                question["question"], GraphRetriever(graph, store, embedder, config["retrieval"]),
                client, retrieval_skills, answer_skills, qa_top_k=config["retrieval"]["qa_top_k"],
            )
            diagnosis = diagnose(
                client, question=question, rollout=rollout, raw_graph=graph,
                reasoning_effort=config["llm"]["analyzer_reasoning_effort"],
            )
            verification = verify_candidates(
                question=question,
                before_rollout=rollout,
                diagnosis=diagnosis,
                raw_graph=graph,
                store=store,
                retrieval_skills=retrieval_skills,
                answer_skills=answer_skills,
                client=client,
                embedder=embedder,
                retrieval_config=config["retrieval"],
                regression_questions=[item for item in covered if item["question_id"] != question["question_id"]][:1],
                qa_top_k=config["retrieval"]["qa_top_k"],
                raw_hash=raw_hash,
            )
            q_results.append(
                {
                    "question_id": question["question_id"],
                    "status": "verified",
                    "diagnosis_flags": {
                        "graph_error": diagnosis.get("graph_error"),
                        "retrieval_error": diagnosis.get("retrieval_error"),
                        "answer_error": diagnosis.get("answer_error"),
                    },
                    "verification": verification,
                }
            )
        retrieval_payload["question_time"] = q_results
        retrieval_payload["skills"] = {
            "retrieval": retrieval_skills.snapshot(),
            "answer": answer_skills.snapshot(),
        }
    except Exception as exc:  # noqa: BLE001
        retrieval_payload["embedding_error"] = _safe(exc, secret)
        retrieval_payload["trace"] = traceback.format_exc()[-2000:]
        print("embedding/eval failed", retrieval_payload["embedding_error"])
    previous = json.loads((art / "ledger.json").read_text(encoding="utf-8"))
    summary = {
        "questions": selected_ids,
        "extra_questions": extra_ids,
        "n_passages_requested": len(passage_ids),
        "n_passages_in_graph": len(graph["passages"]),
        "n_entities": len(graph["entities"]),
        "n_triples": len(graph["triples"]),
        "graph_hash": graph["graph_hash"],
        "first_graph_hash_unchanged": json.loads(first_graph.read_text(encoding="utf-8"))["graph_hash"] == first_hash,
        "synonymy_status": graph["synonymy_status"],
        "openie": openie_status,
        "comprehension_nodes": len(store.active_nodes()),
        "comprehension_edges": len(store.active_edges()),
        "comprehension_results": comp_results,
        "raw_hash_unchanged": graph["graph_hash"] == raw_hash,
        "retrieval": retrieval_payload,
        "round2_ledger": ledger.snapshot(),
        "previous_fresh_calls": previous.get("fresh_calls"),
        "cumulative_fresh_calls_lower_bound": previous.get("fresh_calls", 0) + ledger.fresh_calls,
        "fresh_cap": FRESH_CAP,
        "response_model": auth.get("response_model"),
        "heldout_used": False,
    }
    _write_json(out / "summary.json", summary)
    _write_json(out / "ledger.json", ledger.snapshot())
    write_reports(EXP_ROOT)
    print("round2 done", summary["comprehension_nodes"], "nodes", "fresh", ledger.fresh_calls)


if __name__ == "__main__":
    main()
