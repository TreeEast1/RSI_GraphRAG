"""Luna client. Cache hits against the V4 smoke cache are free. Fresh calls land in V5."""

from __future__ import annotations

import ast
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

V4_SRC = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/experiments/rsi_graphrag_v4/src")
if str(V4_SRC) not in sys.path:
    sys.path.insert(0, str(V4_SRC))

from rsi_v4.hashing import sha256_obj  # noqa: E402

from v5_mech.paths import (  # noqa: E402
    ART,
    HIPPO,
    MAX_FRESH_CALLS,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    V4,
    V4_CACHE,
)


class BudgetExceeded(RuntimeError):
    pass


def load_luna_key() -> str:
    text = (HIPPO / "main_azure.py").read_text(encoding="utf-8")
    module = ast.parse(text)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == "API_KEY":
                    return ast.literal_eval(node.value)
    raise RuntimeError("Luna key not found")


def load_llm_config() -> dict:
    return json.loads((V4 / "configs" / "default.json").read_text(encoding="utf-8"))["llm"]


class Budget:
    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "fresh_calls": 0,
                "http_attempts": 0,
                "cached_calls": 0,
                "v4_cache_hits": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "errors": 0,
                "by_stage": {},
                "response_model_ids": {},
                "caps": {
                    "fresh_calls": MAX_FRESH_CALLS,
                    "prompt_tokens": MAX_INPUT_TOKENS,
                    "completion_tokens": MAX_OUTPUT_TOKENS,
                },
            }
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def assert_room(self) -> None:
        cap_calls = int(self.data.get("caps", {}).get("fresh_calls", MAX_FRESH_CALLS))
        cap_in = int(self.data.get("caps", {}).get("prompt_tokens", MAX_INPUT_TOKENS))
        cap_out = int(self.data.get("caps", {}).get("completion_tokens", MAX_OUTPUT_TOKENS))
        if self.data["fresh_calls"] >= cap_calls:
            raise BudgetExceeded(f"fresh call cap {cap_calls}")
        if self.data["prompt_tokens"] >= cap_in:
            raise BudgetExceeded(f"input token cap {cap_in}")
        if self.data["completion_tokens"] >= cap_out:
            raise BudgetExceeded(f"output token cap {cap_out}")

    def stage(self, name: str) -> dict:
        row = self.data["by_stage"].setdefault(
            name,
            {"fresh_calls": 0, "cached_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "errors": 0},
        )
        return row


def cache_get(path: Path, key: str):
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = conn.execute("select message, metadata from cache where key = ?", (key,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return row[0], json.loads(row[1])


def cache_put(path: Path, key: str, message: str, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=120)
    conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, message TEXT, metadata TEXT)")
    conn.execute(
        "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
        (key, message, json.dumps(metadata, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def request_payload(config: dict, messages: List[dict], max_completion_tokens: int, reasoning_effort: str, disabled: List[str]) -> dict:
    return {
        "deployment": config["deployment"],
        "api_version": config["api_version"],
        "seed": config["seed"],
        "reasoning_effort": reasoning_effort,
        "max_completion_tokens": max_completion_tokens,
        "response_format": None,
        "messages": messages,
        "disabled_params": sorted(disabled),
    }


class LunaClient:
    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.config = load_llm_config()
        self.key = load_luna_key()
        self.cache_path = ART / "llm_cache.sqlite"
        # v5.1 uses a separate ledger so the earlier 60-call pilot is not the cap.
        self.disabled: set = set()

    def complete(self, *, stage: str, messages: List[dict], max_completion_tokens: int, reasoning_effort: str) -> dict:
        payload = request_payload(self.config, messages, max_completion_tokens, reasoning_effort, sorted(self.disabled))
        key = sha256_obj(payload)
        for source, path in (("v4", V4_CACHE), ("v5", self.cache_path)):
            hit = cache_get(path, key)
            if hit is not None:
                text, metadata = hit
                self.budget.data["cached_calls"] += 1
                if source == "v4":
                    self.budget.data["v4_cache_hits"] += 1
                row = self.budget.stage(stage)
                row["cached_calls"] += 1
                self.budget.save()
                return {"text": text, "metadata": metadata, "cache_hit": True, "cache_source": source, "cache_key": key}

        self.budget.assert_room()
        text, metadata = self._call(messages, max_completion_tokens, reasoning_effort)
        metadata["stage"] = stage
        metadata["cache_key"] = key
        cache_put(self.cache_path, key, text, metadata)
        row = self.budget.stage(stage)
        row["fresh_calls"] += 1
        prompt = int(metadata.get("prompt_tokens") or 0)
        completion = int(metadata.get("completion_tokens") or 0)
        reasoning = int(metadata.get("reasoning_tokens") or 0)
        self.budget.data["prompt_tokens"] += prompt
        self.budget.data["completion_tokens"] += completion
        self.budget.data["reasoning_tokens"] += reasoning
        row["prompt_tokens"] += prompt
        row["completion_tokens"] += completion
        row["reasoning_tokens"] += reasoning
        model = metadata.get("response_model")
        if model:
            self.budget.data["response_model_ids"][model] = self.budget.data["response_model_ids"].get(model, 0) + 1
        self.budget.save()
        return {"text": text, "metadata": metadata, "cache_hit": False, "cache_source": "", "cache_key": key}

    def _call(self, messages, max_completion_tokens, reasoning_effort) -> tuple:
        from openai import AzureOpenAI
        import httpx

        client = AzureOpenAI(
            api_version=self.config["api_version"],
            azure_endpoint=self.config["endpoint"],
            api_key=self.key,
            max_retries=0,
            http_client=httpx.Client(timeout=httpx.Timeout(600, read=600)),
        )
        last_error: Optional[Exception] = None
        for attempt in range(4):
            params: Dict[str, Any] = {
                "model": self.config["deployment"],
                "messages": messages,
                "max_completion_tokens": max_completion_tokens,
            }
            if "seed" not in self.disabled:
                params["seed"] = self.config["seed"]
            if reasoning_effort and "reasoning_effort" not in self.disabled:
                params["reasoning_effort"] = reasoning_effort
            self.budget.assert_room()
            self.budget.data["fresh_calls"] += 1
            self.budget.data["http_attempts"] += 1
            self.budget.stage("http").setdefault("attempts", 0)
            self.budget.data["by_stage"].setdefault("http", {"fresh_calls": 0, "cached_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "errors": 0})
            self.budget.data["by_stage"]["http"]["fresh_calls"] += 1
            self.budget.save()
            try:
                response = client.chat.completions.create(**params)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                message = str(exc).lower()
                changed = False
                for name in ("seed", "reasoning_effort"):
                    if name in message and name in params:
                        self.disabled.add(name)
                        changed = True
                if "content_filter" in message or "responsibleaipolicyviolation" in message:
                    self.budget.data["errors"] += 1
                    self.budget.save()
                    raise
                status = getattr(exc, "status_code", None)
                if status in (401, 403):
                    self.budget.data["errors"] += 1
                    self.budget.save()
                    raise
                if changed:
                    continue
                time.sleep(min(30, 2 ** attempt))
                continue
            choice = response.choices[0]
            details = getattr(response.usage, "completion_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", 0) if details is not None else 0
            return choice.message.content or "", {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                "reasoning_tokens": reasoning or 0,
                "finish_reason": choice.finish_reason,
                "response_model": response.model,
                "http_attempts_for_call": attempt + 1,
            }
        self.budget.data["errors"] += 1
        self.budget.save()
        raise RuntimeError(f"Luna call failed: {last_error}")
