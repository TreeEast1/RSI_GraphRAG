"""HY4 (Tencent TokenLink) client adapter.

Responsibilities
----------------
* Single place that knows how to talk to the company HY4 endpoint.
* Concurrency limiting, retry/backoff, timeout handling.
* Persistent response cache (sqlite) keyed by model + params + messages so that
  experiments are resumable and a repeated prompt never costs a second call.
* Per-call accounting (tokens, latency, finish reason, errors) written as JSONL.
* Secrets are only read from the environment / .env and never written to logs.

HY4 specifics measured on 2026-11-05:
  * OpenAI-compatible chat/completions
  * reasoning_effort="no_think" disables the built-in thinking trace
  * max_completion_tokens / temperature / response_format={"type":"json_object"}
    are all accepted
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Optional[Path] = None) -> None:
    """Minimal .env loader (does not override already-exported variables)."""
    path = path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


load_env()


@dataclass
class HY4Config:
    base_url: str = os.environ.get("HY4_BASE_URL", "https://tokenlink.woa.com/openapi/v1")
    model: str = os.environ.get("HY4_MODEL", "hy4")
    api_key: str = os.environ.get("HY4_API_KEY", "")
    reasoning_effort: Optional[str] = os.environ.get("HY4_REASONING_EFFORT", "no_think")
    temperature: Optional[float] = 0.0
    max_tokens: int = 4096
    timeout: float = 300.0
    max_retries: int = 14
    # The endpoint returns 429 above ~32-64 simultaneous requests per key.
    # Keep the global budget (processes x concurrency) at or below 24.
    max_concurrency: int = int(os.environ.get("HY4_MAX_CONCURRENCY", "8"))

    @classmethod
    def from_env(cls, **overrides) -> "HY4Config":
        cfg = cls()
        for k, v in overrides.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)
        return cfg


class _ResponseCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
            )
            conn.execute("PRAGMA journal_mode=WAL")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=60)

    def get(self, key: str) -> Optional[Dict]:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: Dict) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )


class HY4Client:
    """Thread-safe wrapper around the OpenAI-compatible HY4 endpoint."""

    def __init__(
        self,
        config: Optional[HY4Config] = None,
        cache_path: Optional[str | Path] = None,
        log_path: Optional[str | Path] = None,
        tag: str = "default",
    ):
        self.cfg = config or HY4Config.from_env()
        if not self.cfg.api_key:
            raise RuntimeError("HY4_API_KEY is not set (see .env).")
        shard = os.environ.get("HY4_CACHE_SHARD", "")
        cache_path = cache_path or (PROJECT_ROOT / "cache" / f"hy4_cache{shard}.sqlite")
        log_path = log_path or (PROJECT_ROOT / "outputs" / "logs" / f"hy4_calls{shard}.jsonl")
        self.client = OpenAI(
            api_key=self.cfg.api_key,
            base_url=self.cfg.base_url,
            timeout=self.cfg.timeout,
            max_retries=0,  # we do our own backoff so we can log attempts
        )
        self.cache = _ResponseCache(cache_path)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._sem = threading.Semaphore(self.cfg.max_concurrency)
        self._log_lock = threading.Lock()
        self.tag = tag
        self.stats = {"calls": 0, "cache_hits": 0, "errors": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "reasoning_tokens": 0, "latency_s": 0.0}

    # ---------------------------------------------------------------- core
    def _cache_key(self, messages: List[Dict], params: Dict) -> str:
        payload = {
            "model": self.cfg.model,
            "base_url": self.cfg.base_url,
            "messages": messages,
            "params": params,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        reasoning_effort: Optional[str] = "__default__",
        use_cache: bool = True,
        tag: Optional[str] = None,
        **extra,
    ) -> Dict[str, Any]:
        """Return {"content", "usage", "finish_reason", "cached", "latency_s"}."""
        params: Dict[str, Any] = {
            "model": self.cfg.model,
            "max_completion_tokens": max_tokens or self.cfg.max_tokens,
        }
        eff = self.cfg.reasoning_effort if reasoning_effort == "__default__" else reasoning_effort
        if eff:
            params["reasoning_effort"] = eff
        temp = self.cfg.temperature if temperature is None else temperature
        if temp is not None:
            params["temperature"] = temp
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        params.update(extra)

        key = self._cache_key(messages, params)
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                with self._log_lock:
                    self.stats["cache_hits"] += 1
                hit = dict(hit)
                hit["cached"] = True
                return hit

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            t0 = time.time()
            try:
                from .ratelimit import acquire as _acquire
                _acquire()
                with self._sem:
                    resp = self.client.chat.completions.create(messages=messages, **params)
                latency = time.time() - t0
                choice = resp.choices[0]
                content = choice.message.content or ""
                usage = resp.usage
                usage_d = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                }
                details = getattr(usage, "completion_tokens_details", None)
                usage_d["reasoning_tokens"] = getattr(details, "reasoning_tokens", None) if details else None
                out = {
                    "content": content,
                    "usage": usage_d,
                    "finish_reason": choice.finish_reason,
                    "model": getattr(resp, "model", self.cfg.model),
                    "cached": False,
                    "latency_s": round(latency, 3),
                }
                self._record(tag or self.tag, params, usage_d, latency, "ok", attempt)
                if use_cache:
                    self.cache.put(key, out)
                return out
            except Exception as exc:  # noqa: BLE001 - endpoint errors vary
                last_err = exc
                wait = min(60.0, (2 ** attempt) * 1.5) * (0.7 + random.random() * 0.6)
                self._record(tag or self.tag, params, None, time.time() - t0,
                             f"error:{type(exc).__name__}", attempt, str(exc)[:300])
                if attempt < self.cfg.max_retries - 1:
                    time.sleep(wait)
        with self._log_lock:
            self.stats["errors"] += 1
        raise RuntimeError(f"HY4 call failed after {self.cfg.max_retries} attempts: {last_err}")

    def complete_text(self, prompt: str, system: Optional[str] = None, **kw) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kw)["content"]

    def complete_json(self, prompt: str, system: Optional[str] = None, **kw) -> Any:
        """Call the model in JSON mode and parse (with one repair retry)."""
        kw.setdefault("json_mode", True)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        out = self.chat(messages, **kw)
        parsed = parse_json_loose(out["content"])
        if parsed is not None:
            return parsed
        # one repair attempt
        repair = messages + [
            {"role": "assistant", "content": out["content"][:4000]},
            {"role": "user", "content": "Your previous reply was not valid JSON. Return ONLY valid JSON, no markdown fences, no commentary."},
        ]
        out2 = self.chat(repair, **kw)
        parsed = parse_json_loose(out2["content"])
        if parsed is None:
            raise ValueError(f"HY4 returned unparseable JSON: {out2['content'][:300]!r}")
        return parsed

    # ---------------------------------------------------------------- misc
    def _record(self, tag, params, usage, latency, status, attempt, error=None):
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tag": tag,
            "model": params.get("model"),
            "reasoning_effort": params.get("reasoning_effort"),
            "max_tokens": params.get("max_completion_tokens"),
            "json_mode": "response_format" in params,
            "latency_s": round(latency, 3),
            "status": status,
            "attempt": attempt,
            "usage": usage,
        }
        if error:
            rec["error"] = error
        with self._log_lock:
            self.stats["calls"] += 1
            self.stats["latency_s"] += latency
            if usage:
                self.stats["prompt_tokens"] += usage.get("prompt_tokens") or 0
                self.stats["completion_tokens"] += usage.get("completion_tokens") or 0
                self.stats["reasoning_tokens"] += usage.get("reasoning_tokens") or 0
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def summary(self) -> Dict:
        return dict(self.stats)


def parse_json_loose(text: str) -> Optional[Any]:
    """Extract the first valid JSON value from a possibly noisy LLM response."""
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    # strip markdown fences
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
            return value
        except json.JSONDecodeError:
            continue
    return None
