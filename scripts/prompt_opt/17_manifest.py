#!/usr/bin/env python
"""Write a reproducibility manifest for the prompt-optimization experiments."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.prompt_optimization import comprehension_prompts as CP  # noqa: E402
from src.prompt_optimization import reader_prompts as RP  # noqa: E402
from src.prompt_optimization.runner import sha  # noqa: E402

OUT = ROOT / "outputs" / "prompt_optimization_conv30" / "reports"


def file_sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def main():
    code = [
        "src/retrieval/reader.py", "src/comprehension/prompts.py", "src/comprehension/extract.py",
        "src/evaluation/metrics.py", "src/evaluation/pipeline.py", "src/graph/variants.py",
        "src/prompt_optimization/official_metrics.py", "src/prompt_optimization/reader_prompts.py",
        "src/prompt_optimization/comprehension_prompts.py", "src/prompt_optimization/runner.py",
        "src/prompt_optimization/retrieval_diag.py",
    ]
    logs = {}
    for p in sorted((ROOT / "outputs" / "logs").glob("hy4_calls*.jsonl")):
        n = ok = err = cache = 0
        for line in open(p, encoding="utf-8"):
            n += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(r.get("status", "")).startswith("error"):
                err += 1
            else:
                ok += 1
        logs[p.name] = {"lines": n, "ok": ok, "errors": err}

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {
            "path": "data/raw/locomo/locomo10.json",
            "sha256": file_sha(ROOT / "data/raw/locomo/locomo10.json"),
            "conversation": "conv-30", "n_questions": 105,
        },
        "models": {
            "llm": os.environ.get("HY4_MODEL", "hy4"),
            "llm_base_url": os.environ.get("HY4_BASE_URL"),
            "reasoning_effort": os.environ.get("HY4_REASONING_EFFORT", "no_think"),
            "embedding": "nvidia/NV-Embed-v2 (local, fp16, dim 4096)",
        },
        "generation": {"temperature": 0.0, "reader_max_tokens": 256,
                       "comprehension_max_tokens": 8192},
        "retrieval": {"num_to_retrieve": 20, "linking_top_k": 5,
                      "passage_node_weight": 0.05, "ppr_damping": 0.5,
                      "context_budget_tokens": 4096,
                      "note": "HippoRAG fact reranking is an LLM call per query (official DSPy filter prompt)"},
        "split": {"file": "splits/split_dev40_val65.json", "seed": 42,
                  "n_dev": 40, "n_val": 65},
        "reader_prompts": {p.pid: {"name": p.name, "system_sha256": sha(p.system),
                                   "template_sha256": sha(p.template)} for p in RP.ALL_PROMPTS},
        "comprehension_prompts": {p.pid: {"name": p.name, "instructions_sha256": sha(p.instructions),
                                          "system_sha256": sha(p.system)} for p in CP.ALL},
        "code_sha256": {f: file_sha(ROOT / f) for f in code},
        "hy4_logs": logs,
        "notes": [
            "All original G_raw/C2/C3/C5/C6 results and graph directories were read-only.",
            "New graphs live under outputs/raw_graphs/conv-30/variants/cmpo_*.",
            "Repeats experiment used use_cache=False; all other calls used the SQLite cache.",
            "Oracle conditions never place gold answers in any prompt.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False)[:2500])


if __name__ == "__main__":
    main()
