"""Shared IO, metrics, and the OpenAI-compatible client. Model identity lives in configs/model.json."""
from __future__ import annotations

import json
import os
import re
import string
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Dict, Iterable, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def abspath(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer(answer: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc((answer or "").lower())))


def em_f1(prediction: str, golds: Iterable[str]) -> Dict[str, float]:
    pred = normalize_answer(prediction)
    golds = [g for g in golds if g]
    if not golds:
        return {"em": 0.0, "f1": 0.0}
    em = max(float(pred == normalize_answer(g)) for g in golds)
    pred_toks = pred.split()
    best = 0.0
    for gold in golds:
        gold_toks = normalize_answer(gold).split()
        if not pred_toks or not gold_toks:
            continue
        common = Counter(pred_toks) & Counter(gold_toks)
        num = sum(common.values())
        if num == 0:
            continue
        precision = num / len(pred_toks)
        recall = num / len(gold_toks)
        best = max(best, 2 * precision * recall / (precision + recall))
    return {"em": em, "f1": best}


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.strip()


def parse_json_object(text: str) -> Optional[dict]:
    raw = strip_thinking(text)
    raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.I).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


class LLMClient:
    def __init__(self, cfg: dict):
        self.model = cfg["model"]
        self.base_url = cfg["base_url"].rstrip("/")
        self.api_key = cfg.get("api_key") or "local"
        self.thinking = bool(cfg.get("thinking", False))
        self.temperature = cfg.get("temperature", 0.0)
        self.max_output_tokens = int(cfg.get("max_output_tokens") or 800)
        self.timeout_s = float(cfg.get("timeout_s") or 180)

    def chat(self, messages: List[dict], max_output_tokens: Optional[int] = None) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_output_tokens or self.max_output_tokens,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        # Local servers must not go through an HTTP proxy. A proxy here returns 502.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        t0 = time.time()
        try:
            with opener.open(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        latency = time.time() - t0
        choice = body["choices"][0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        usage = body.get("usage") or {}
        return {
            "text": text,
            "visible_text": strip_thinking(text),
            "latency_s": latency,
            "prompt_tokens": usage.get("prompt_tokens") or 0,
            "completion_tokens": usage.get("completion_tokens") or 0,
            "finish_reason": choice.get("finish_reason"),
            "model": body.get("model") or self.model,
        }


def load_llm(config_path: Optional[str] = None) -> LLMClient:
    path = config_path or abspath("configs", "model.json")
    return LLMClient(load_json(path))
