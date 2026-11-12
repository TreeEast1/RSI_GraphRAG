#!/usr/bin/env python
"""Collect experiment provenance into outputs/reports/run_manifest.json."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import INDEX_IDENTITY, NVEMBED_PATH, PATHS  # noqa: E402


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def hy4_usage() -> dict:
    out = {"calls": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0, "cache_hits": 0}
    for f in sorted((PATHS["logs"]).glob("hy4_calls*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out["calls"] += 1
            if r.get("status") != "ok":
                out["errors"] += 1
            u = r.get("usage") or {}
            out["prompt_tokens"] += u.get("prompt_tokens") or 0
            out["completion_tokens"] += u.get("completion_tokens") or 0
    return out


def main():
    split = json.loads((PATHS["splits"] / "split.json").read_text()) \
        if (PATHS["splits"] / "split.json").exists() else {}
    manifest = {
        "project": "rsi-graph-rag",
        "index_identity": INDEX_IDENTITY,
        "hipporag_commit": git_commit(PATHS["third_party"]),
        "hipporag_version": "2.0.0a5",
        "dataset": {
            "name": "LoCoMo",
            "file": "data/raw/locomo/locomo10.json",
            "sha256": __import__("hashlib").sha256(
                (ROOT / "data/raw/locomo/locomo10.json").read_bytes()).hexdigest(),
            "split": split,
        },
        "models": {
            "llm": {"name": os.environ.get("HY4_MODEL", "hy4"),
                    "endpoint": os.environ.get("HY4_BASE_URL"),
                    "reasoning_effort": os.environ.get("HY4_REASONING_EFFORT", "no_think")},
            "embedding": {"name": "nvidia/NV-Embed-v2", "path": NVEMBED_PATH, "dim": 4096},
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpus": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                    "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines(),
        },
        "hy4_usage": hy4_usage(),
        "outputs_present": sorted(p.name for p in PATHS["reports"].glob("*.md")),
    }
    out = PATHS["reports"] / "run_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("wrote", out)
    print(json.dumps(manifest["hy4_usage"], indent=1))


if __name__ == "__main__":
    main()
