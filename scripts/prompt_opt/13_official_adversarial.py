#!/usr/bin/env python
"""Official-LoCoMo adversarial protocol: ask the model to choose between
"(a) Not mentioned in the conversation" and "(b) <adversarial answer>".

The multiple-choice text is exactly what the official repo builds for category 5.
Each project Reader prompt (P0/P7) is used with that question, so the comparison
is between our prompts under the official protocol (not the official prompt).

Outputs: reader/official_adversarial/{pid}.jsonl + summary.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import PATHS  # noqa: E402
from src.prompt_optimization.runner import OUT_ROOT, freeze_context, load_conv, session_docs, write_jsonl  # noqa: E402

NOT_MENTIONED = "Not mentioned in the conversation"


def render_for_protocol(spec, context, question):
    if spec is None:  # the official repo's own cat-5 prompt
        return "", (f"{context}\n\nBased on the above context, answer the following question.\n\n"
                    f"Question: {question} Short answer:")
    return spec.system, spec.template.format(context=context, question=question)


def parse_choice(text, key):
    low = str(text or "").strip().lower()
    # strip an "Answer:" prefix produced by the project templates
    low = low.replace("answer:", "").strip()
    if low.startswith("(a)") or low.startswith("a") or "a)" in low[:6]:
        return key["a"], "a"
    if low.startswith("(b)") or low.startswith("b") or "b)" in low[:6]:
        return key["b"], "b"
    if NOT_MENTIONED.lower() in low:
        return NOT_MENTIONED, "not_mentioned"
    return low, "free"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", default="P0,P7")
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--shard", default="oa")
    args = ap.parse_args()

    import os
    os.environ["HY4_CACHE_SHARD"] = args.shard
    from src.adapters.hy4_client import HY4Client

    conv = load_conv("conv-30")
    docs = session_docs(conv)
    base = [json.loads(l) for l in open(PATHS["qa"] / "baselines_hipporag_conv-30.jsonl", encoding="utf-8")]
    adv = [(i, r) for i, r in enumerate(base) if int(r["category"]) == 5]
    # official dataset field is q['answer'] for category 5 (the adversarial answer)
    raw = json.loads((ROOT / "data" / "raw" / "locomo" / "locomo10.json").read_text())
    raw30 = next(x for x in raw if x["sample_id"] == "conv-30")
    adv_answers = {i: q.get("adversarial_answer") or q.get("answer") for i, q in enumerate(raw30["qa"])}

    client = HY4Client(tag="off_adv")
    outdir = OUT_ROOT / "reader" / "official_adversarial"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for pid in args.pids.split(","):
        rows = []
        def one(item):
            i, rec = item
            fz = freeze_context(rec, docs, 4096)
            adv_ans = str(adv_answers.get(i) or "")
            rng = random.Random(1000 + i)
            if rng.random() < 0.5:
                a_text, b_text = NOT_MENTIONED, adv_ans
                key = {"a": NOT_MENTIONED, "b": adv_ans}
            else:
                a_text, b_text = adv_ans, NOT_MENTIONED
                key = {"a": adv_ans, "b": NOT_MENTIONED}
            question = (rec["question"] + " Select the correct answer: (a) {} (b) {}. ").format(a_text, b_text)
            from src.prompt_optimization import reader_prompts as RP
            spec = None if pid == "OFFICIAL" else RP.get(pid)
            system, user = render_for_protocol(spec, fz["context"], question)
            try:
                msgs = [{"role": "user", "content": user}]
                if system:
                    msgs.insert(0, {"role": "system", "content": system})
                out = client.chat(msgs, max_tokens=128, temperature=0.0, tag=f"offadv_{pid}")
                choice, how = parse_choice(out["content"], key)
                correct = int(choice == NOT_MENTIONED)
                return {"pid": pid, "qidx": i, "qid": rec["qid"], "question": rec["question"],
                        "adversarial_answer": adv_ans, "a_text": a_text, "b_text": b_text,
                        "raw": out["content"], "choice": choice, "parse": how, "correct": correct,
                        "usage": out.get("usage"), "finish_reason": out.get("finish_reason"),
                        "cached": out.get("cached"), "error": None}
            except Exception as exc:  # noqa: BLE001
                return {"pid": pid, "qidx": i, "qid": rec["qid"], "question": rec["question"],
                        "adversarial_answer": adv_ans, "raw": "", "choice": "", "parse": "error",
                        "correct": 0, "error": f"{type(exc).__name__}: {exc}"}

        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            rows = list(ex.map(one, adv))
        write_jsonl(outdir / f"{pid}.jsonl", rows)
        n = len(rows)
        summary[pid] = {
            "n": n, "accuracy": float(np.mean([r["correct"] for r in rows])),
            "parse_breakdown": {k: sum(1 for r in rows if r["parse"] == k) for k in ("a", "b", "not_mentioned", "free", "error")},
            "errors": sum(1 for r in rows if r.get("error")),
        }
        print(f"  {pid}: official-protocol adversarial accuracy {summary[pid]['accuracy']:.3f} "
              f"({sum(r['correct'] for r in rows)}/{n})")

    (outdir / "summary.json").write_text(json.dumps({"summaries": summary, "hy4": client.summary(),
                                                     "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                                    indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
