"""Azure chat client for GPT-5.6 Luna with cache, backoff, and a hard call budget.

Cached responses do not consume the fresh-call budget. The deployment name is
whatever the environment config requests; the id returned by the API is stored
separately and is never invented by this module.
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .hashing import sha256_obj


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    metadata: Dict[str, Any]
    cache_hit: bool
    stage: str


@dataclass
class UsageLedger:
    max_fresh_calls: int
    fresh_calls: int = 0
    cached_calls: int = 0
    errors: int = 0
    by_stage: Dict[str, Dict[str, float]] = field(default_factory=dict)
    response_model_ids: Dict[str, int] = field(default_factory=dict)

    def stage_row(self, stage: str) -> Dict[str, float]:
        row = self.by_stage.setdefault(
            stage,
            {
                "fresh_calls": 0,
                "cached_calls": 0,
                "errors": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
            },
        )
        return row

    def reserve_fresh(self) -> None:
        if self.fresh_calls >= self.max_fresh_calls:
            raise BudgetExceeded(
                f"fresh LLM call budget exhausted ({self.fresh_calls}/{self.max_fresh_calls})"
            )
        self.fresh_calls += 1

    def add_success(self, stage: str, metadata: Dict[str, Any], cache_hit: bool) -> None:
        row = self.stage_row(stage)
        if cache_hit:
            self.cached_calls += 1
            row["cached_calls"] += 1
        else:
            row["fresh_calls"] += 1
        row["prompt_tokens"] += float(metadata.get("prompt_tokens") or 0)
        row["completion_tokens"] += float(metadata.get("completion_tokens") or 0)
        row["reasoning_tokens"] += float(metadata.get("reasoning_tokens") or 0)
        model = metadata.get("response_model")
        if model and not cache_hit:
            self.response_model_ids[str(model)] = self.response_model_ids.get(str(model), 0) + 1

    def add_error(self, stage: str) -> None:
        self.errors += 1
        self.stage_row(stage)["errors"] += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "max_fresh_calls": self.max_fresh_calls,
            "fresh_calls": self.fresh_calls,
            "cached_calls": self.cached_calls,
            "errors": self.errors,
            "by_stage": self.by_stage,
            "response_model_ids": self.response_model_ids,
        }


class LLMClient:
    def __init__(
        self,
        *,
        cache_path: str,
        deployment: str,
        endpoint: str,
        api_version: str,
        api_key: Optional[str],
        ledger: UsageLedger,
        seed: int = 20261030,
        max_attempts: int = 6,
        request_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.cache_path = cache_path
        self.deployment = deployment
        self.endpoint = endpoint
        self.api_version = api_version
        self.api_key = api_key
        self.ledger = ledger
        self.seed = seed
        self.max_attempts = max_attempts
        self.request_fn = request_fn
        self._lock = threading.Lock()
        self._disabled_params = set()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, message TEXT, metadata TEXT)"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.cache_path, timeout=120)

    def _key(self, payload: Dict[str, Any]) -> str:
        return sha256_obj(payload)

    def _cache_get(self, key: str):
        with self._lock:
            conn = self._connect()
            row = conn.execute("SELECT message, metadata FROM cache WHERE key = ?", (key,)).fetchone()
            conn.close()
        if row is None:
            return None
        return row[0], json.loads(row[1])

    def _cache_put(self, key: str, message: str, metadata: Dict[str, Any]) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                (key, message, json.dumps(metadata, ensure_ascii=False)),
            )
            conn.commit()
            conn.close()

    def complete(
        self,
        *,
        stage: str,
        messages: List[Dict[str, str]],
        max_completion_tokens: int,
        reasoning_effort: Optional[str] = "low",
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResult:
        payload = {
            "deployment": self.deployment,
            "api_version": self.api_version,
            "seed": self.seed,
            "reasoning_effort": reasoning_effort,
            "max_completion_tokens": max_completion_tokens,
            "response_format": response_format,
            "messages": messages,
            "disabled_params": sorted(self._disabled_params),
        }
        key = self._key(payload)
        cached = self._cache_get(key)
        if cached is not None:
            text, metadata = cached
            with self._lock:
                self.ledger.add_success(stage, metadata, cache_hit=True)
            return LLMResult(text=text, metadata=dict(metadata), cache_hit=True, stage=stage)

        with self._lock:
            self.ledger.reserve_fresh()
        try:
            text, metadata = self._call_with_backoff(
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
                response_format=response_format,
            )
        except BudgetExceeded:
            raise
        except Exception:
            with self._lock:
                self.ledger.add_error(stage)
            raise
        metadata["stage"] = stage
        metadata["cache_key"] = key
        self._cache_put(key, text, metadata)
        with self._lock:
            self.ledger.add_success(stage, metadata, cache_hit=False)
        return LLMResult(text=text, metadata=metadata, cache_hit=False, stage=stage)

    def _call_with_backoff(self, **kwargs) -> tuple:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_attempts):
            try:
                return self._call_once(**kwargs)
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                text = str(exc)
                if "content_filter" in text or "ResponsibleAIPolicyViolation" in text:
                    break
                status = getattr(exc, "status_code", None)
                if status in (401, 403):
                    break
                if status is not None and status < 500 and status not in (408, 409, 429):
                    # Parameter rejections are handled inside _call_once. A remaining 4xx is fatal.
                    if "seed" in text or "response_format" in text or "reasoning_effort" in text:
                        pass
                    else:
                        break
                delay = min(60.0, (2 ** attempt) + random.random())
                time.sleep(delay)
        raise RuntimeError(f"LLM call failed after {self.max_attempts} attempts: {last_error}")

    def _call_once(self, *, messages, max_completion_tokens, reasoning_effort, response_format) -> tuple:
        params: Dict[str, Any] = {
            "model": self.deployment,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        }
        if self.seed is not None and "seed" not in self._disabled_params:
            params["seed"] = self.seed
        if reasoning_effort and "reasoning_effort" not in self._disabled_params:
            params["reasoning_effort"] = reasoning_effort
        if response_format and "response_format" not in self._disabled_params:
            params["response_format"] = response_format
        if self.request_fn is not None:
            payload = self.request_fn(params)
        else:
            payload = self._azure_request(params)
        text = payload.get("text") or ""
        metadata = {
            "prompt_tokens": payload.get("prompt_tokens", 0),
            "completion_tokens": payload.get("completion_tokens", 0),
            "reasoning_tokens": payload.get("reasoning_tokens", 0),
            "finish_reason": payload.get("finish_reason"),
            "response_model": payload.get("response_model"),
            "requested_deployment": self.deployment,
        }
        return text, metadata

    def _azure_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY is not set")
        from openai import AzureOpenAI
        import httpx

        client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            max_retries=0,
            http_client=httpx.Client(timeout=httpx.Timeout(600, read=600)),
        )
        try:
            response = client.chat.completions.create(**params)
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            changed = False
            for name in ("seed", "response_format", "reasoning_effort"):
                if name in message and name in params:
                    self._disabled_params.add(name)
                    params.pop(name, None)
                    changed = True
            if changed:
                response = client.chat.completions.create(**params)
            else:
                raise
        choice = response.choices[0]
        details = getattr(response.usage, "completion_tokens_details", None)
        reasoning = 0
        if details is not None:
            reasoning = getattr(details, "reasoning_tokens", 0) or 0
        return {
            "text": choice.message.content or "",
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
            "reasoning_tokens": reasoning,
            "finish_reason": choice.finish_reason,
            "response_model": response.model,
        }


def load_api_key(env_name: str) -> Optional[str]:
    value = os.environ.get(env_name)
    if value:
        return value
    return None
