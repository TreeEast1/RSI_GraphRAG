"""Run the bounded RSI GraphRAG v4 pipeline and write only measured artifacts."""

from __future__ import annotations

import argparse
import json
import random
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import tiktoken

from .comprehension_evolver import evolve_unit
from .comprehension_store import ComprehensionStore
from .corpus_manager import (
    assert_read_time_inputs_have_no_questions,
    freeze_musique,
    load_corpus,
    load_eval_questions,
)
from .embeddings import NVEmbedEncoder
from .evolution_analyzer import diagnose
from .graph_analyzer import (
    analyze_openie_records,
    materialize_cluster,
    select_representative_clusters,
)
from .graph_partitioner import audit_partition, partition_cluster
from .graph_patch_manager import GraphPatchManager
from .hipporag_adapter import (
    build_raw_graph,
    hipporag_commit,
    historical_records,
    load_openie_dir,
    official_prompt_hash,
    openie_one,
    prompt_manager,
    save_raw_graph,
    write_openie_record,
)
from .llm_client import BudgetExceeded, LLMClient, UsageLedger, load_api_key
from .metrics import aggregate_questions
from .rag import rag
from .rag_verifier import verify_candidates
from .reporting import write_reports
from .retrieval import GraphRetriever
from .answer_skill_store import AnswerSkillStore
from .evolution_analyzer import score_row
from .retrieval_skill_store import RetrievalSkillStore

ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe(exc: Exception, secret: str) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if secret and secret in text:
        text = text.replace(secret, "[REDACTED]")
    return text[:500]


def _append_failure(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _token_counter(encoder):
    def count(text: str) -> int:
        return len(encoder.encode(text or ""))

    return count


def run_openie(passages: List[dict], client, out_dir: Path, manager, effort: str, workers: int, failures: Path, secret: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [passage for passage in passages if not (out_dir / f"{passage['passage_id']}.json").exists()]

    def job(passage: dict) -> str:
        record = openie_one(client, passage, manager, reasoning_effort=effort)
        write_openie_record(str(out_dir), record)
        return passage["passage_id"]

    written = 0
    stopped = False
    errors = []
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(job, passage) for passage in pending]
            for future in as_completed(futures):
                try:
                    future.result()
                    written += 1
                    if written % 5 == 0:
                        print("openie written", written, flush=True)
                except BudgetExceeded:
                    stopped = True
                except Exception as exc:  # noqa: BLE001
                    errors.append(_safe(exc, secret))
                    _append_failure(failures, {"stage": "openie", "error": _safe(exc, secret)})
    return {
        "requested_new": len(pending),
        "written_new": written,
        "cached_files": len(list(out_dir.glob("*.json"))),
        "stopped_budget": stopped,
        "errors": errors[:10],
    }


def evaluate_questions(questions: List[dict], retriever, client, retrieval_skills, answer_skills, qa_top_k: int) -> List[dict]:
    rows = []
    for question in questions:
        rollout = rag(
            question["question"],
            retriever,
            client,
            retrieval_skills,
            answer_skills,
            qa_top_k=qa_top_k,
        )
        row = score_row(question, rollout)
        row["answer"] = rollout["answer"]
        row["comprehension_ids_in_top10"] = rollout["comprehension_ids_in_top10"]
        row["reader_doc_ids"] = rollout["reader_doc_ids"]
        row["split_role"] = question.get("split_role")
        rows.append(row)
    return rows


def resolve_luna_key(config: dict, art: Path):
    """Pick a key that can call the Luna deployment. The key is never written to disk."""
    import ast

    import httpx
    from openai import AzureOpenAI

    def from_launcher() -> str:
        text = Path(config["hipporag_root"], "main_azure.py").read_text(encoding="utf-8")
        module = ast.parse(text)
        for node in module.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if getattr(target, "id", None) == "API_KEY":
                        return ast.literal_eval(node.value)
        raise RuntimeError("HippoRAG/main_azure.py has no API_KEY assignment")

    def ping(key: str) -> str:
        client = AzureOpenAI(
            api_version=config["llm"]["api_version"],
            azure_endpoint=config["llm"]["endpoint"],
            api_key=key,
            max_retries=0,
            http_client=httpx.Client(timeout=httpx.Timeout(60, read=60)),
        )
        response = client.chat.completions.create(
            model=config["llm"]["deployment"],
            messages=[{"role": "user", "content": "Reply with the single word pong."}],
            max_completion_tokens=20,
            reasoning_effort="low",
        )
        return response.model

    env_key = load_api_key(config["llm"]["api_key_env"]) or ""
    launcher_key = from_launcher()
    errors = []
    for source, key in (
        (config["llm"]["api_key_env"], env_key),
        ("HippoRAG/main_azure.py:API_KEY", launcher_key),
    ):
        if not key:
            continue
        try:
            model_id = ping(key)
        except Exception as exc:  # noqa: BLE001
            errors.append({"source": source, "status": getattr(exc, "status_code", None), "error": type(exc).__name__})
            continue
        meta = {
            "source": source,
            "endpoint": config["llm"]["endpoint"],
            "api_version": config["llm"]["api_version"],
            "deployment": config["llm"]["deployment"],
            "response_model": model_id,
            "rejected_sources": errors,
        }
        _write_json(art / "llm_auth.json", meta)
        return key, meta
    _write_json(art / "llm_auth.json", {"response_model": None, "rejected_sources": errors})
    return "", {"response_model": None, "rejected_sources": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data_dir = ROOT / "data"
    art = ROOT / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    failures = art / "failures.jsonl"
    phases = {"all"} if args.phase == "all" else {args.phase}
    allow_full_read = (art / "APPROVED_FULL_READ_TIME").exists()
    allow_question_time = (art / "APPROVED_QUESTION_TIME").exists()
    if not allow_question_time:
        print("question-time is paused; no analyzer or skill update will run")
    if not allow_full_read:
        print("full read-time is paused; comprehension stays inside an explicit pilot script")
    secret, auth_meta = resolve_luna_key(config, art)
    print("luna auth", auth_meta.get("source"), auth_meta.get("response_model"), "rejected", auth_meta.get("rejected_sources"))

    if phases & {"all", "corpus"}:
        manifest = freeze_musique(
            config["musique_questions"],
            config["musique_corpus"],
            str(data_dir),
            seed=config["seed"],
        )
        corpus_rows = load_corpus(str(data_dir / "corpus.jsonl"))
        assert_read_time_inputs_have_no_questions(corpus_rows)
        _write_json(art / "corpus_stage.json", manifest)
        print("corpus frozen", manifest["n_unique_passages"], manifest["role_counts"])

    if phases & {"all", "historical"} and not (art / "partition_pilot.json").exists():
        print("historical relation-graph audit started")
        encoder = tiktoken.get_encoding(config["reading"]["tokenizer"])
        records = list(historical_records(config["historical_openie"]))
        from hipporag.utils.misc_utils import compute_mdhash_id

        token_by_passage = {
            compute_mdhash_id(record["content"], prefix="chunk-"): len(encoder.encode(record["content"]))
            for record in records
        }
        bundle = analyze_openie_records(
            records,
            token_by_passage=token_by_passage,
            input_token_budget=config["reading"]["input_token_budget"],
            leiden_resolution=config["reading"]["leiden_resolution"],
            source_label="historical_gpt-4o-mini_openie_not_luna",
        )
        summary = bundle["summary"]
        _write_json(art / "historical_graph_stats.json", summary)
        components = [
            row
            for row in bundle["relation_components"]
            if 4 <= row["n_nodes"] <= 250
        ]
        chosen = select_representative_clusters(components, config["reading"]["representative_clusters"])
        overhead = len(encoder.encode((ROOT / "prompts" / "comprehension_system.txt").read_text(encoding="utf-8")))
        count = _token_counter(encoder)
        pilot_rows = []
        for component in chosen:
            cluster = materialize_cluster(bundle, component)
            cluster["graph_id"] = "historical-gpt-4o-mini"
            row = {
                "cluster_id": component["cluster_id"],
                "n_nodes": component["n_nodes"],
                "n_passages": component["n_passages"],
                "estimated_passage_tokens": component["estimated_tokens"],
                "hops": {},
            }
            for budget_name, budget in (
                ("budget_16000", config["reading"]["input_token_budget"]),
                ("stress_800", 800),
            ):
                row["hops"][budget_name] = {}
                for hops in (0, 1, 2):
                    units = partition_cluster(
                        cluster,
                        input_token_budget=budget,
                        overlap_hops=hops,
                        count_tokens=count,
                        overhead_tokens=overhead,
                    )
                    audit = audit_partition(cluster, units, budget)
                    audit["n_flagged_over_budget"] = len(audit["flagged_over_budget_units"])
                    row["hops"][budget_name][str(hops)] = audit
            pilot_rows.append(row)
            print("partitioned", component["cluster_id"], component["n_nodes"])
        def _mean(budget_name: str, hops: str, key: str) -> float:
            values = [row["hops"][budget_name][hops][key] for row in pilot_rows]
            return sum(values) / len(values) if values else 0.0

        def _budget_block(budget_name: str) -> dict:
            return {
                "core_coverage_all": all(
                    row["hops"][budget_name][hop]["core_coverage_complete"]
                    for row in pilot_rows
                    for hop in ("0", "1", "2")
                ),
                "mean_boundary_retention": {hop: _mean(budget_name, hop, "boundary_retention_rate") for hop in ("0", "1", "2")},
                "mean_units": {hop: _mean(budget_name, hop, "n_units") for hop in ("0", "1", "2")},
                "mean_overlap_mention_ratio": {hop: _mean(budget_name, hop, "overlap_mention_ratio") for hop in ("0", "1", "2")},
                "max_input_tokens": {
                    hop: max((row["hops"][budget_name][hop]["max_input_tokens"] for row in pilot_rows), default=0)
                    for hop in ("0", "1", "2")
                },
            }

        pilot = {
            "source_label": summary["source_label"],
            "warning": (
                "This partition pilot uses the historical gpt-4o-mini OpenIE graph. It is not the Luna G_raw. "
                "stress_800 forces splits so overlap can be measured; budget_16000 is the actual reading budget."
            ),
            "n_components_total": summary["relation_component_count"],
            "n_components_expanded": len(pilot_rows),
            "n_components_between_4_and_250": len(components),
            "n_components_above_250_not_expanded": sum(
                1 for row in bundle["relation_components"] if row["n_nodes"] > 250
            ),
            "budget_16000": _budget_block("budget_16000"),
            "stress_800": _budget_block("stress_800"),
            "clusters": pilot_rows,
        }
        _write_json(art / "partition_pilot.json", pilot)
        _write_json(art / "historical_graph_stats.json", summary)
        del bundle
        print("historical audit written")

    if not (phases & {"all", "openie", "comprehend", "eval", "question", "reports"}):
        return

    corpus_manifest = _read_json(data_dir / "corpus_manifest.json", {})
    questions = load_eval_questions(str(data_dir / "questions_eval.jsonl")) if (data_dir / "questions_eval.jsonl").exists() else []
    corpus = load_corpus(str(data_dir / "corpus.jsonl")) if (data_dir / "corpus.jsonl").exists() else []
    by_id = {row["passage_id"]: row for row in corpus}
    ledger = UsageLedger(max_fresh_calls=config["budget"]["max_fresh_llm_calls"])
    client = None
    if phases & {"all", "openie", "comprehend", "eval", "question"}:
        client = LLMClient(
            cache_path=str(art / "llm_cache.sqlite"),
            deployment=config["llm"]["deployment"],
            endpoint=config["llm"]["endpoint"],
            api_version=config["llm"]["api_version"],
            api_key=secret or None,
            ledger=ledger,
            seed=config["llm"]["seed"],
        )
    manager = prompt_manager()
    openie_dir = art / "openie" / "luna"
    formal_ids = [row["passage_id"] for row in sorted(corpus, key=lambda item: item["corpus_index"])][
        : config["budget"]["formal_passage_cap"]
    ]
    rng = random.Random(config["seed"])
    update_ids = sorted(question["question_id"] for question in questions if question.get("split_role") == "update")
    rng.shuffle(update_ids)
    mechanism_questions = [
        question
        for question in questions
        if question["question_id"] in set(update_ids[: config["budget"]["mechanism_question_cap"]])
    ]
    mechanism_ids = []
    missing_mechanism = []
    for question in mechanism_questions:
        for paragraph in question["paragraphs"]:
            pid = paragraph["passage_id"]
            if pid in by_id and pid not in mechanism_ids:
                mechanism_ids.append(pid)
            elif pid not in by_id:
                missing_mechanism.append(pid)

    if client is not None and phases & {"all", "openie"}:
        try:
            probe_passage = by_id[formal_ids[0]]
            probe_status = run_openie(
                [probe_passage],
                client,
                openie_dir,
                manager,
                config["llm"]["openie_reasoning_effort"],
                1,
                failures,
                secret,
            )
            probe_file = openie_dir / f"{probe_passage['passage_id']}.json"
            probe_record = json.loads(probe_file.read_text(encoding="utf-8")) if probe_file.exists() else None
            _write_json(
                art / "luna_probe.json",
                {
                    "requested_deployment": config["llm"]["deployment"],
                    "endpoint": config["llm"]["endpoint"],
                    "api_version": config["llm"]["api_version"],
                    "response_model": None if probe_record is None else probe_record.get("response_model"),
                    "passage_id": probe_passage["passage_id"],
                    "n_entities": 0 if probe_record is None else len(probe_record.get("entities") or []),
                    "n_triples": 0 if probe_record is None else len(probe_record.get("triples") or []),
                    "status": probe_status,
                    "ledger": ledger.snapshot(),
                },
            )
            fresh = ledger.snapshot()
            per_call_prompt = 0
            per_call_completion = 0
            fresh_calls = max(1, fresh["fresh_calls"])
            for row in fresh["by_stage"].values():
                per_call_prompt += row.get("prompt_tokens", 0)
                per_call_completion += row.get("completion_tokens", 0)
            per_call_prompt /= fresh_calls
            per_call_completion /= fresh_calls
            _write_json(
                art / "cost_estimate.json",
                {
                    "basis": "mean tokens of fresh calls so far, extrapolated to 11656 passages x 2 OpenIE calls",
                    "fresh_calls_observed": fresh["fresh_calls"],
                    "mean_prompt_tokens_per_fresh_call": per_call_prompt,
                    "mean_completion_tokens_per_fresh_call": per_call_completion,
                    "projected_full_openie_calls": 11656 * 2,
                    "projected_prompt_tokens": per_call_prompt * 11656 * 2,
                    "projected_completion_tokens": per_call_completion * 11656 * 2,
                    "usd": None,
                    "usd_reason": "No Luna price was configured. Tokens are reported without a dollar conversion.",
                    "automatic_call_cap": config["budget"]["max_fresh_llm_calls"],
                },
            )
            formal_status = run_openie(
                [by_id[pid] for pid in formal_ids],
                client,
                openie_dir,
                manager,
                config["llm"]["openie_reasoning_effort"],
                config["budget"]["openie_workers"],
                failures,
                secret,
            )
            formal_status["target_passages"] = len(formal_ids)
            formal_status["full_corpus_passages"] = len(corpus)
            formal_status["note"] = "Corpus-order prefix of the frozen corpus. Not selected from question gold."
            _write_json(art / "formal_openie_status.json", formal_status)
            mechanism_status = run_openie(
                [by_id[pid] for pid in mechanism_ids],
                client,
                openie_dir,
                manager,
                config["llm"]["openie_reasoning_effort"],
                config["budget"]["openie_workers"],
                failures,
                secret,
            )
            mechanism_status["question_ids"] = [question["question_id"] for question in mechanism_questions]
            mechanism_status["n_passages_requested"] = len(mechanism_ids)
            mechanism_status["n_paragraphs_missing_from_frozen_corpus"] = len(missing_mechanism)
            mechanism_status["warning"] = (
                "Mechanism pool is the paragraph union of a seeded update-split sample, including distractors. "
                "It is not the formal full-corpus G_raw. Held-out questions were not used to choose it."
            )
            _write_json(art / "mechanism_status.json", mechanism_status)
        except Exception as exc:  # noqa: BLE001
            _append_failure(failures, {"stage": "openie_driver", "error": _safe(exc, secret)})
            _write_json(art / "openie_driver_error.json", {"error": _safe(exc, secret), "trace": traceback.format_exc()[-2000:]})
            print("openie driver error", _safe(exc, secret))

    records = {row["passage_id"]: row for row in load_openie_dir(str(openie_dir))} if openie_dir.exists() else {}
    if records and phases & {"all", "openie", "comprehend", "eval", "question"}:
        formal_records = [records[pid] for pid in formal_ids if pid in records]
        if formal_records:
            formal_graph = build_raw_graph(
                formal_records,
                status="partial_prefix" if len(formal_records) < len(corpus) else "complete",
                model_name=config["llm"]["deployment"],
                corpus_hash=corpus_manifest.get("corpus_hash", ""),
            )
            save_raw_graph(formal_graph, str(art / "graphs" / "formal_prefix_g_raw.json"))
        mechanism_records = [records[pid] for pid in mechanism_ids if pid in records]
        if mechanism_records:
            mechanism_graph = build_raw_graph(
                mechanism_records,
                status="mechanism_pilot_not_full_corpus",
                model_name=config["llm"]["deployment"],
                corpus_hash=corpus_manifest.get("corpus_hash", ""),
            )
            save_raw_graph(mechanism_graph, str(art / "graphs" / "mechanism_g_raw.json"))
            _write_json(
                art / "mechanism_graph_meta.json",
                {
                    "graph_hash": mechanism_graph["graph_hash"],
                    "n_passages": len(mechanism_graph["passages"]),
                    "n_entities": len(mechanism_graph["entities"]),
                    "n_triples": len(mechanism_graph["triples"]),
                    "n_context_edges": len(mechanism_graph["context_edges"]),
                    "n_synonymy_edges": len(mechanism_graph["synonymy_edges"]),
                    "synonymy_status": mechanism_graph["synonymy_status"],
                    "uncovered_passages": len(mechanism_graph["uncovered_passage_ids"]),
                    "hipporag_commit": mechanism_graph["hipporag_commit"],
                    "prompt_hash": mechanism_graph["official_prompt_hash"],
                    "model_name": mechanism_graph["model_name"],
                },
            )

    graph_path = art / "graphs" / "mechanism_g_raw.json"
    if client is not None and allow_full_read and graph_path.exists() and phases & {"all", "comprehend"} and not (art / "comprehension_pilot.json").exists():
        from .hipporag_adapter import load_raw_graph

        raw_graph = load_raw_graph(str(graph_path))
        raw_hash = raw_graph["graph_hash"]
        encoder = tiktoken.get_encoding(config["reading"]["tokenizer"])
        count = _token_counter(encoder)
        overhead = len(encoder.encode((ROOT / "prompts" / "comprehension_system.txt").read_text(encoding="utf-8")))
        # Build one cluster per connected component of this small pilot graph via the analyzer.
        token_by_passage = {
            row["passage_id"]: len(encoder.encode(row["content"])) for row in raw_graph["passages"].values()
        }
        openie_records = [records[pid] for pid in raw_graph["passages"] if pid in records]
        bundle = analyze_openie_records(
            openie_records,
            token_by_passage=token_by_passage,
            input_token_budget=config["reading"]["input_token_budget"],
            source_label="luna_mechanism_pilot",
        )
        store = ComprehensionStore()
        patch_manager = GraphPatchManager(raw_graph, store)
        results = []
        units_seen = 0
        cap = config["budget"]["comprehension_unit_cap"]
        for component in sorted(bundle["relation_components"], key=lambda row: -row["n_nodes"]):
            if units_seen >= cap:
                break
            cluster = materialize_cluster(bundle, component)
            cluster["graph_id"] = raw_hash[:12]
            units = partition_cluster(
                cluster,
                input_token_budget=config["reading"]["input_token_budget"],
                overlap_hops=config["reading"]["overlap_hops"],
                count_tokens=count,
                overhead_tokens=overhead,
            )
            audit = audit_partition(cluster, units, config["reading"]["input_token_budget"])
            for unit in units:
                if units_seen >= cap:
                    break
                try:
                    outcome = evolve_unit(
                        client,
                        cluster,
                        unit,
                        patch_manager,
                        mode="comprehension",
                        reasoning_effort=config["llm"]["comprehension_reasoning_effort"],
                    )
                except BudgetExceeded:
                    outcome = {"unit_id": unit["unit_id"], "status": "stopped_budget"}
                    results.append({"audit": audit, "outcome": outcome})
                    units_seen += 1
                    break
                except Exception as exc:  # noqa: BLE001
                    outcome = {"unit_id": unit["unit_id"], "status": "error", "error": _safe(exc, secret)}
                    _append_failure(failures, {"stage": "comprehension", "error": _safe(exc, secret)})
                results.append({"unit_id": unit["unit_id"], "audit_core_complete": audit["core_coverage_complete"], "outcome": outcome})
                units_seen += 1
                if outcome.get("status") == "stopped_budget":
                    break
        _write_json(art / "comprehension_store.json", store.to_dict())
        _write_json(
            art / "comprehension_pilot.json",
            {
                "raw_hash_unchanged": raw_graph["graph_hash"] == raw_hash,
                "raw_hash": raw_hash,
                "n_active_nodes": len(store.active_nodes()),
                "n_edges": len(store.active_edges()),
                "results": results,
            },
        )
        print("comprehension nodes", len(store.active_nodes()))

    if client is not None and allow_question_time and graph_path.exists() and phases & {"all", "eval", "question"}:
        from .hipporag_adapter import load_raw_graph

        raw_graph = load_raw_graph(str(graph_path))
        store = ComprehensionStore()
        if (art / "comprehension_store.json").exists():
            store.load_dict(_read_json(art / "comprehension_store.json"))
        indexed = set(raw_graph["passages"])
        covered = [
            question
            for question in mechanism_questions
            if question["gold_passage_ids"] and set(question["gold_passage_ids"]) <= indexed
        ]
        heldout_covered = [
            question
            for question in questions
            if question.get("split_role") == "heldout" and question["gold_passage_ids"] and set(question["gold_passage_ids"]) <= indexed
        ]
        embedder = NVEmbedEncoder(config["embedding_model_path"])
        eval_payload = {"covered_update_questions": [q["question_id"] for q in covered], "embedding": {}}
        try:
            embedder.load()
            eval_payload["embedding"] = {"device": embedder.device, "model_path": embedder.model_path, "loaded": True}
            empty = ComprehensionStore()
            retrieval_skills = RetrievalSkillStore()
            answer_skills = AnswerSkillStore()
            if covered and phases & {"all", "eval"}:
                r0 = evaluate_questions(
                    covered,
                    GraphRetriever(raw_graph, empty, embedder, config["retrieval"]),
                    client,
                    retrieval_skills,
                    answer_skills,
                    config["retrieval"]["qa_top_k"],
                )
                r1 = evaluate_questions(
                    covered,
                    GraphRetriever(raw_graph, store, embedder, config["retrieval"]),
                    client,
                    retrieval_skills,
                    answer_skills,
                    config["retrieval"]["qa_top_k"],
                )
                eval_payload["R0"] = {"summary": aggregate_questions(r0), "rows": r0}
                eval_payload["R1"] = {"summary": aggregate_questions(r1), "rows": r1}
                comp_used = sum(1 for row in r1 if row.get("comprehension_ids_in_top10"))
                eval_payload["r1_questions_with_comp_in_top10"] = comp_used
            eval_payload["heldout_fully_covered_by_mechanism_index"] = [q["question_id"] for q in heldout_covered]
            if heldout_covered:
                heldout_rows = evaluate_questions(
                    heldout_covered,
                    GraphRetriever(raw_graph, store, embedder, config["retrieval"]),
                    client,
                    RetrievalSkillStore(),
                    AnswerSkillStore(),
                    config["retrieval"]["qa_top_k"],
                )
                eval_payload["heldout_covered_eval"] = {"summary": aggregate_questions(heldout_rows), "rows": heldout_rows}
            else:
                eval_payload["heldout_covered_eval"] = {
                    "status": "not_run",
                    "reason": "No held-out question has its full gold passage set inside the mechanism index.",
                }
        except Exception as exc:  # noqa: BLE001
            eval_payload["embedding"] = {"loaded": False, "error": _safe(exc, secret)}
            eval_payload["retrieval_status"] = "blocked"
            _append_failure(failures, {"stage": "embedding_or_eval", "error": _safe(exc, secret)})
        _write_json(art / "retrieval_eval.json", eval_payload)

        if allow_question_time and phases & {"all", "question"} and eval_payload.get("embedding", {}).get("loaded") and covered:
            q_results = []
            raw_hash = raw_graph["graph_hash"]
            g_read_snapshot = store.snapshot()
            retrieval_skills = RetrievalSkillStore()
            answer_skills = AnswerSkillStore()
            for question in covered[: config["budget"]["question_time_cap"]]:
                try:
                    retriever = GraphRetriever(raw_graph, store, embedder, config["retrieval"])
                    rollout = rag(
                        question["question"],
                        retriever,
                        client,
                        retrieval_skills,
                        answer_skills,
                        qa_top_k=config["retrieval"]["qa_top_k"],
                    )
                    diagnosis = diagnose(
                        client,
                        question=question,
                        rollout=rollout,
                        raw_graph=raw_graph,
                        reasoning_effort=config["llm"]["analyzer_reasoning_effort"],
                    )
                    others = [item for item in covered if item["question_id"] != question["question_id"]]
                    verification = verify_candidates(
                        question=question,
                        before_rollout=rollout,
                        diagnosis=diagnosis,
                        raw_graph=raw_graph,
                        store=store,
                        retrieval_skills=retrieval_skills,
                        answer_skills=answer_skills,
                        client=client,
                        embedder=embedder,
                        retrieval_config=config["retrieval"],
                        regression_questions=others[:2],
                        qa_top_k=config["retrieval"]["qa_top_k"],
                        raw_hash=raw_hash,
                    )
                    q_results.append(
                        {
                            "question_id": question["question_id"],
                            "rollout_answer": rollout["answer"],
                            "gold_answer_used_only_after_rollout": True,
                            "diagnosis_flags": {
                                "graph_error": diagnosis.get("graph_error"),
                                "retrieval_error": diagnosis.get("retrieval_error"),
                                "answer_error": diagnosis.get("answer_error"),
                            },
                            "verification": verification,
                        }
                    )
                except BudgetExceeded:
                    q_results.append({"question_id": question["question_id"], "status": "stopped_budget"})
                    break
                except Exception as exc:  # noqa: BLE001
                    q_results.append({"question_id": question["question_id"], "status": "error", "error": _safe(exc, secret)})
                    _append_failure(failures, {"stage": "question_time", "error": _safe(exc, secret)})
            _write_json(
                art / "question_time.json",
                {
                    "g_read_version_before": g_read_snapshot["version"],
                    "store_version_after": store.version,
                    "raw_hash": raw_hash,
                    "results": q_results,
                    "retrieval_skills": retrieval_skills.snapshot(),
                    "answer_skills": answer_skills.snapshot(),
                    "heldout_used_for_updates": False,
                },
            )
            _write_json(art / "comprehension_store_after_question_time.json", store.to_dict())

    if client is not None:
        _write_json(art / "ledger.json", client.ledger.snapshot())
    _write_json(
        art / "run_identity.json",
        {
            "hipporag_commit": hipporag_commit(),
            "official_prompt_hash": official_prompt_hash(),
            "requested_llm_deployment": config["llm"]["deployment"],
            "embedding_model_path": config["embedding_model_path"],
            "seed": config["seed"],
            "corpus_hash": corpus_manifest.get("corpus_hash"),
        },
    )
    if phases & {"all", "reports"}:
        write_reports(ROOT)
        print("reports written")


if __name__ == "__main__":
    main()
