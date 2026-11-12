#!/usr/bin/env python3
"""Qwen3.8-27B-FP8 on three 3090s.

3090 is compute capability 8.6, so FP8 kernels are refused and the weights are
dequantized to BF16. CUDA_VISIBLE_DEVICES must be exactly three free GPUs.
This process never opens any other device.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/data2-HDD-SATA-20T/nzq/huggingface_model/Qwen3.8-27B-FP8"
)
SERVED_NAME = os.environ.get("SERVED_NAME", "qwen3.8-27b")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "18183"))
OFFLOAD = os.environ.get(
    "OFFLOAD_FOLDER", "/data2-HDD-SATA-20T/nzq/tmp/qwen38_offload"
)

print(f"loading {MODEL_PATH} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
os.makedirs(OFFLOAD, exist_ok=True)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
n = torch.cuda.device_count()
max_memory = {i: "22GiB" for i in range(n)}
MODEL = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    max_memory=max_memory,
    offload_folder=OFFLOAD,
    trust_remote_code=True,
)
MODEL.eval()
INPUT_DEVICE = next(MODEL.parameters()).device
print(
    "ready",
    SERVED_NAME,
    dict(Counter(str(v) for v in MODEL.hf_device_map.values())),
    "input",
    INPUT_DEVICE,
    flush=True,
)
LOCK = threading.Lock()


def generate(messages, max_tokens: int, temperature: float, enable_thinking: bool):
    inputs = TOKENIZER.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    )
    inputs = {k: v.to(INPUT_DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[1])
    do_sample = temperature is not None and float(temperature) > 0
    gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
    t0 = time.time()
    with torch.no_grad():
        out = MODEL.generate(**inputs, **gen_kwargs)
    latency = time.time() - t0
    new_tokens = out[0][prompt_tokens:]
    text = TOKENIZER.decode(new_tokens, skip_special_tokens=True)
    return text, prompt_tokens, int(new_tokens.shape[0]), latency


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models") or self.path.startswith("/health"):
            self._send(200, {"object": "list", "data": [{"id": SERVED_NAME, "object": "model"}]})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            req = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid json"})
            return
        messages = req.get("messages") or []
        max_tokens = int(req.get("max_tokens") or req.get("max_completion_tokens") or 512)
        temperature = req.get("temperature", 0.0)
        template_kwargs = req.get("chat_template_kwargs") or {}
        enable_thinking = bool(template_kwargs.get("enable_thinking", False))
        try:
            with LOCK:
                text, prompt_tokens, completion_tokens, latency = generate(
                    messages, max_tokens, temperature, enable_thinking
                )
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": str(exc)[:800]})
            return
        print(f"tokens in={prompt_tokens} out={completion_tokens} sec={latency:.2f}", flush=True)
        self._send(200, {
            "id": "chatcmpl-" + uuid.uuid4().hex[:12],
            "object": "chat.completion",
            "model": SERVED_NAME,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "latency_s": latency,
        })


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"listening http://{HOST}:{PORT}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
