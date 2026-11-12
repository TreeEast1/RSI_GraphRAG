#!/usr/bin/env python3
"""Full MuSiQue OpenIE for G_raw. No comprehension and no question-time calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")

from rsi_v4.corpus_manager import load_corpus  # noqa: E402
from rsi_v4.experiment_runner import (  # noqa: E402
    _append_failure,
    _safe,
    _write_json,
    resolve_luna_key,
)
from rsi_v4.hipporag_adapter import openie_one, prompt_manager, write_openie_record  # noqa: E402
from rsi_v4.llm_client import BudgetExceeded, LLMClient, UsageLedger  # noqa: E402

FRESH_CAP = 24000
CHUNK = 80
WORKERS = 8


def main() -> None:
    config = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    art = ROOT / "artifacts"
    out_dir = art / "openie" / "luna"
    failed_dir = art / "openie" / "luna_failed"
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    secret, auth = resolve_luna_key(config, art)
    if not secret:
        raise SystemExit(f"Luna auth failed: {auth}")
    corpus = load_corpus(str(ROOT / "data" / "corpus.jsonl"))
    corpus.sort(key=lambda row: row["corpus_index"])
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
    manager = prompt_manager()
    failures = art / "failures.jsonl"
    pending = [
        row for row in corpus
        if not (out_dir / f"{row['passage_id']}.json").exists()
        and not (failed_dir / f"{row['passage_id']}.json").exists()
    ]
    print(f"g_raw openie pending {len(pending)} of {len(corpus)}", flush=True)
    if not pending:
        _write_json(art / "g_raw_build_status.json", {
            "target_passages": len(corpus),
            "openie_files": len(list(out_dir.glob("*.json"))),
            "failed_files": len(list(failed_dir.glob("*.json"))),
            "written_this_run": 0,
            "stopped_budget": False,
            "complete": True,
            "scope": "openie_only_no_comprehension_no_question_time",
        })
        print("g_raw openie already complete", flush=True)
        return
    written = 0
    stopped = False
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def job(passage: dict) -> str:
        record = openie_one(client, passage, manager, reasoning_effort=config["llm"]["openie_reasoning_effort"])
        write_openie_record(str(out_dir), record)
        return passage["passage_id"]

    for start in range(0, len(pending), CHUNK):
        if stopped or ledger.fresh_calls >= FRESH_CAP:
            stopped = True
            break
        chunk = pending[start : start + CHUNK]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(job, passage): passage for passage in chunk}
            for future in as_completed(futures):
                passage = futures[future]
                try:
                    future.result()
                    written += 1
                except BudgetExceeded:
                    stopped = True
                except Exception as exc:  # noqa: BLE001
                    text = _safe(exc, secret)
                    _append_failure(failures, {"stage": "g_raw_openie", "passage_id": passage["passage_id"], "error": text})
                    if "content_filter" in text or "ResponsibleAIPolicyViolation" in text:
                        failed_dir.joinpath(f"{passage['passage_id']}.json").write_text(
                            json.dumps({"passage_id": passage["passage_id"], "error": "content_filter"}),
                            encoding="utf-8",
                        )
        done = len(list(out_dir.glob("*.json")))
        status = {
            "target_passages": len(corpus),
            "openie_files": done,
            "failed_files": len(list(failed_dir.glob("*.json"))),
            "written_this_run": written,
            "stopped_budget": stopped,
            "fresh_cap": FRESH_CAP,
            "ledger": ledger.snapshot(),
            "response_model": auth.get("response_model"),
            "scope": "openie_only_no_comprehension_no_question_time",
        }
        _write_json(art / "g_raw_build_status.json", status)
        print(f"g_raw progress {done}/{len(corpus)} fresh {ledger.fresh_calls}", flush=True)
        if stopped:
            break
    status = json.loads((art / "g_raw_build_status.json").read_text(encoding="utf-8"))
    status["complete"] = status["openie_files"] + status["failed_files"] >= len(corpus)
    _write_json(art / "g_raw_build_status.json", status)
    print("g_raw openie finished", status["complete"], status["openie_files"], flush=True)


if __name__ == "__main__":
    main()
