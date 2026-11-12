#!/usr/bin/env python3
"""Task A/B/C capability check on real MuSiQue passages. Does not write the graph."""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    abspath, append_jsonl, em_f1, load_json, load_llm, parse_json_object, read_jsonl, save_json,
)

EXTRACT_SYSTEM = """You extract a knowledge graph from one passage.
Return only a JSON object with this shape:
{"entities": ["..."], "triples": [{"head": "...", "relation": "...", "tail": "..."}]}
Rules:
- Use only facts stated in the passage.
- Do not add world knowledge.
- Keep names, dates, and numbers as they appear.
- Resolve pronouns to the entity they refer to when the passage makes that clear.
- If the passage states no relation, return empty lists."""

PATCH_SYSTEM = """You revise a partial knowledge graph for one passage.
Return only a JSON object:
{"patches": [{"operation": "ADD"|"MODIFY"|"DEACTIVATE", "target_id": "...", "head": "...", "relation": "...", "tail": "...", "evidence": "a short quote from the passage"}],
 "comprehension_chunks": [{"text": "...", "evidence": "a short quote from the passage"}]}
Rules:
- ADD a triple only when the passage states it and the current graph missed it.
- DEACTIVATE target_id when that existing triple is not supported or conflicts with the passage.
- A comprehension chunk may state an implication that is supported by the passage, in one or two sentences.
- Do not invent entities. Prefer fewer accurate updates over a long list.
- Do not answer a question. There is no question."""

QA_SYSTEM = """Answer the question using only the supplied passages.
Return only a JSON object: {"answer": "...", "sufficient": true|false}
If the passages do not contain enough evidence, set sufficient to false and answer to "unknown".
The answer should be a short span, not a sentence, when sufficient is true."""


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()


def supported(head: str, tail: str, passage: str) -> bool:
    hay = norm(passage)
    h = norm(head)
    t = norm(tail)
    if len(h) < 2 or len(t) < 2:
        return False
    return h in hay and t in hay


def as_triples(obj) -> list:
    rows = []
    for item in (obj or {}).get("triples") or []:
        if isinstance(item, dict):
            rows.append({
                "head": str(item.get("head") or "").strip(),
                "relation": str(item.get("relation") or "").strip(),
                "tail": str(item.get("tail") or "").strip(),
            })
        elif isinstance(item, list) and len(item) >= 3:
            rows.append({"head": str(item[0]).strip(), "relation": str(item[1]).strip(), "tail": str(item[2]).strip()})
    return [r for r in rows if r["head"] and r["relation"] and r["tail"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default=abspath("configs", "model.json"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    pilot = load_json(abspath("configs", "pilot.json"))
    model_cfg = load_json(args.model_config)
    client = load_llm(args.model_config)
    questions = load_json(abspath("data", "questions.json"))
    docs = load_json(abspath("data", "documents.json"))
    n = args.limit or pilot["capability_n"]
    sample = questions[:n]
    out_dir = abspath("predictions", "capability", model_cfg["model"].replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    path_a = os.path.join(out_dir, "task_a.jsonl")
    path_b = os.path.join(out_dir, "task_b.jsonl")
    path_c = os.path.join(out_dir, "task_c.jsonl")
    done_a = {r["qid"] for r in read_jsonl(path_a)}
    done_b = {r["qid"] for r in read_jsonl(path_b)}
    done_c = {r["qid"] for r in read_jsonl(path_c)}

    for i, q in enumerate(sample):
        support = q["support_ids"][:2] or q["paragraph_ids"][:1]
        doc = docs[support[0]]
        passage = f"{doc['title']}\n{doc['text']}"
        if q["qid"] not in done_a:
            resp = client.chat([
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": passage},
            ], max_output_tokens=700)
            obj = parse_json_object(resp["visible_text"])
            triples = as_triples(obj) if obj else []
            ok = [t for t in triples if supported(t["head"], t["tail"], passage)]
            append_jsonl(path_a, {
                "qid": q["qid"], "doc_id": doc["doc_id"], "json_ok": obj is not None,
                "n_triples": len(triples), "n_supported": len(ok),
                "n_unsupported": len(triples) - len(ok), "triples": triples,
                "latency_s": resp["latency_s"], "prompt_tokens": resp["prompt_tokens"],
                "completion_tokens": resp["completion_tokens"], "raw": resp["visible_text"][:4000],
            })
            print(f"A {i+1}/{n} json={obj is not None} triples={len(triples)} supported={len(ok)}", flush=True)
        if q["qid"] not in done_b:
            prior = next(r for r in read_jsonl(path_a) if r["qid"] == q["qid"])
            kept = prior["triples"][::2]
            false_triple = {
                "target_id": "false-1",
                "head": doc["title"] or "This entity",
                "relation": "is the capital of",
                "tail": "Narnia",
            }
            shown = [{"target_id": f"t{k}", **t} for k, t in enumerate(kept)] + [false_triple]
            user = "Passage:\n" + passage + "\n\nCurrent triples:\n" + json_block(shown)
            resp = client.chat([
                {"role": "system", "content": PATCH_SYSTEM},
                {"role": "user", "content": user},
            ], max_output_tokens=800)
            obj = parse_json_object(resp["visible_text"]) or {}
            patches = obj.get("patches") if isinstance(obj.get("patches"), list) else []
            chunks = obj.get("comprehension_chunks") if isinstance(obj.get("comprehension_chunks"), list) else []
            adds = [p for p in patches if str(p.get("operation", "")).upper() == "ADD"]
            add_ok = 0
            for p in adds:
                if supported(str(p.get("head") or ""), str(p.get("tail") or ""), passage):
                    add_ok += 1
            detected = False
            for p in patches:
                op = str(p.get("operation", "")).upper()
                if str(p.get("target_id")) == "false-1" and op in {"DEACTIVATE", "MODIFY"}:
                    detected = True
            append_jsonl(path_b, {
                "qid": q["qid"], "json_ok": bool(obj), "n_add": len(adds), "n_add_supported": add_ok,
                "n_chunks": len(chunks), "false_triple_detected": detected,
                "n_deactivate": sum(1 for p in patches if str(p.get("operation", "")).upper() == "DEACTIVATE"),
                "latency_s": resp["latency_s"], "prompt_tokens": resp["prompt_tokens"],
                "completion_tokens": resp["completion_tokens"], "raw": resp["visible_text"][:4000],
            })
            print(f"B {i+1}/{n} add_ok={add_ok}/{len(adds)} false_detected={detected}", flush=True)
        if q["qid"] not in done_c:
            gold_text = "\n\n".join(f"[{did}] {docs[did]['title']}\n{docs[did]['text']}" for did in q["support_ids"])
            distractors = q["distractor_ids"][:2]
            bad_text = "\n\n".join(f"[{did}] {docs[did]['title']}\n{docs[did]['text']}" for did in distractors)
            good = client.chat([
                {"role": "system", "content": QA_SYSTEM},
                {"role": "user", "content": f"Passages:\n{gold_text}\n\nQuestion: {q['question']}"},
            ], max_output_tokens=200)
            bad = client.chat([
                {"role": "system", "content": QA_SYSTEM},
                {"role": "user", "content": f"Passages:\n{bad_text}\n\nQuestion: {q['question']}"},
            ], max_output_tokens=200)
            gobj = parse_json_object(good["visible_text"]) or {}
            bobj = parse_json_object(bad["visible_text"]) or {}
            scores = em_f1(str(gobj.get("answer") or ""), [q["answer"], *q["answer_aliases"]])
            abstained = (bobj.get("sufficient") is False) or norm(str(bobj.get("answer") or "")) in {"unknown", "insufficient", "not enough information"}
            append_jsonl(path_c, {
                "qid": q["qid"], "json_ok_supported": "answer" in gobj, "json_ok_insufficient": "answer" in bobj,
                "em": scores["em"], "f1": scores["f1"], "prediction": gobj.get("answer"),
                "gold": q["answer"], "abstained_when_insufficient": bool(abstained),
                "latency_s": good["latency_s"] + bad["latency_s"],
                "prompt_tokens": (good["prompt_tokens"] or 0) + (bad["prompt_tokens"] or 0),
                "completion_tokens": (good["completion_tokens"] or 0) + (bad["completion_tokens"] or 0),
            })
            print(f"C {i+1}/{n} em={scores['em']} f1={scores['f1']:.2f} abstain={abstained}", flush=True)

    summary = summarize(path_a, path_b, path_c, model_cfg)
    save_json(os.path.join(out_dir, "summary.json"), summary)
    print(json_pretty(summary))


def json_block(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2)


def json_pretty(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2)


def mean(rows, key):
    if not rows:
        return 0.0
    return sum(float(r.get(key) or 0) for r in rows) / len(rows)


def summarize(path_a, path_b, path_c, model_cfg) -> dict:
    a = read_jsonl(path_a)
    b = read_jsonl(path_b)
    c = read_jsonl(path_c)
    triples = sum(r["n_triples"] for r in a)
    supported = sum(r["n_supported"] for r in a)
    adds = sum(r["n_add"] for r in b)
    return {
        "model": model_cfg["model"],
        "thinking": model_cfg.get("thinking", False),
        "n_a": len(a), "n_b": len(b), "n_c": len(c),
        "task_a_json_rate": mean(a, "json_ok"),
        "task_a_triples": triples,
        "task_a_endpoint_support_rate": (supported / triples) if triples else 0.0,
        "task_a_unsupported_triples": sum(r["n_unsupported"] for r in a),
        "task_b_json_rate": mean(b, "json_ok"),
        "task_b_add_support_rate": (sum(r["n_add_supported"] for r in b) / adds) if adds else 0.0,
        "task_b_false_detection_rate": mean(b, "false_triple_detected"),
        "task_b_chunks": sum(r["n_chunks"] for r in b),
        "task_c_json_rate": mean(c, "json_ok_supported"),
        "task_c_em": mean(c, "em"),
        "task_c_f1": mean(c, "f1"),
        "task_c_abstain_rate": mean(c, "abstained_when_insufficient"),
        "latency_s": sum(r["latency_s"] for r in a + b + c),
        "prompt_tokens": sum(r["prompt_tokens"] for r in a + b + c),
        "completion_tokens": sum(r["completion_tokens"] for r in a + b + c),
    }


if __name__ == "__main__":
    main()
