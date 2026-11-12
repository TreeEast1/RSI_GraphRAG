#!/usr/bin/env python
"""Phase 0.1 — audit the *actual* prompts used by the existing pipeline.

No LLM call: the reader prompt is deterministically re-rendered from the frozen
G_raw ranked documents; the comprehension prompt is re-rendered from the C2
triples exactly as ``src.comprehension.extract.comprehend_group`` does.

Outputs under outputs/prompt_optimization_conv30/audit/
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.comprehension.prompts import (COMPREHENSION_INSTRUCTIONS,  # noqa: E402
                                       COMPREHENSION_SYSTEM, EXTRACTION_INSTRUCTIONS,
                                       EXTRACTION_SYSTEM, VERIFY_INSTRUCTIONS,
                                       render_comprehension_prompt)
from src.config import PATHS  # noqa: E402
from src.dataset.locomo import load_locomo, num_tokens  # noqa: E402
from src.evaluation.metrics import select_by_budget  # noqa: E402
from src.retrieval.reader import READER_SYSTEM, READER_TEMPLATE, format_context  # noqa: E402

OUT = ROOT / "outputs" / "prompt_optimization_conv30" / "audit"
CASE_QIDS = [4, 16, 17, 24, 29, 31, 33, 54, 85]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def docs_for(conv):
    return [{"document_id": s.document_id, "session_id": s.session_id,
             "header": f"Session {s.session_id} | {s.date_time_raw}", "content": s.text}
            for s in conv.sessions]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    conv = {c.sample_id: c for c in load_locomo(ROOT / "data" / "raw" / "locomo" / "locomo10.json")}["conv-30"]
    docs = {d["document_id"]: d for d in docs_for(conv)}
    rows = [json.loads(l) for l in open(PATHS["qa"] / "baselines_hipporag_conv-30.jsonl", encoding="utf-8")]

    manifest = {"reader_system": READER_SYSTEM, "reader_template": READER_TEMPLATE,
                "reader_system_sha256": sha(READER_SYSTEM), "reader_template_sha256": sha(READER_TEMPLATE),
                "cases": []}

    for qidx in CASE_QIDS:
        rec = rows[qidx]
        ranked = [docs[d] for d in rec["ranked_docs"] if d in docs]
        ctx_docs = select_by_budget(ranked, 4096, num_tokens)
        context = format_context(ctx_docs)
        user = READER_TEMPLATE.format(context=context, question=rec["question"])
        recomputed_tokens = sum(num_tokens(d["content"]) for d in ctx_docs)
        case = {
            "qid": rec["qid"], "qidx": qidx, "category": rec["category"],
            "question": rec["question"], "gold_answer": rec["gold_answer"],
            "gold_sessions": rec["gold_sessions"], "gold_turns": rec["gold_turns"],
            "ranked_docs": rec["ranked_docs"], "selected_docs": [d["document_id"] for d in ctx_docs],
            "context_docs_order": [[d["document_id"], d["header"]] for d in ctx_docs],
            "record_context_tokens": rec.get("extra", {}).get("context_tokens"),
            "recomputed_context_tokens": recomputed_tokens,
            "record_n_context": rec.get("extra", {}).get("n_context"),
            "recomputed_n_context": len(ctx_docs),
            "reader_system": READER_SYSTEM, "reader_user": user,
            "reader_user_sha256": sha(user), "reader_user_tokens": num_tokens(user),
            "max_tokens": 256, "temperature": 0.0, "reasoning_effort": "no_think",
            "ranking_reconstructs_record": (
                recomputed_tokens == rec.get("extra", {}).get("context_tokens")
                and len(ctx_docs) == rec.get("extra", {}).get("n_context")),
        }
        (OUT / f"reader_prompt_q{qidx:03d}.txt").write_text(
            f"--- SYSTEM ---\n{READER_SYSTEM}\n\n--- USER ---\n{user}\n", encoding="utf-8")
        manifest["cases"].append(case)

    # ---- comprehension prompts: exact re-render for both C2 reading groups
    c2 = json.loads((PATHS["comprehension"] / "conv-30" / "cluster_hybrid_0.5_b8192_triple.json").read_text())
    comp_manifest = {"system": COMPREHENSION_SYSTEM, "system_sha256": sha(COMPREHENSION_SYSTEM),
                     "instructions_sha256": sha(COMPREHENSION_INSTRUCTIONS),
                     "extraction_system_sha256": sha(EXTRACTION_SYSTEM),
                     "extraction_instructions_sha256": sha(EXTRACTION_INSTRUCTIONS),
                     "verify_instructions_sha256": sha(VERIFY_INSTRUCTIONS), "groups": []}
    by_sid = conv.session_map()
    for g in c2["groups"]:
        sids = g["session_ids"]
        parts = []
        for sid in sids:
            s = by_sid[sid]
            lines = [f"=== Session {sid} | {s.date_time_raw} ==="]
            lines += [f"[{t.dia_id}] {t.speaker}: {t.text}" for t in s.turns]
            parts.append("\n".join(lines))
        block = "\n\n".join(parts)
        facts = g["extraction"]["triples"]
        facts_block = "\n".join(f"f{i}: {t['subject']} | {t['predicate']} | {t['object']}"
                                for i, t in enumerate(facts))
        prompt = render_comprehension_prompt(block, facts_block)
        (OUT / f"comp_prompt_group{g['group_id']}.txt").write_text(
            f"--- SYSTEM ---\n{COMPREHENSION_SYSTEM}\n\n--- USER ---\n{prompt}\n", encoding="utf-8")
        comp_manifest["groups"].append({
            "group_id": g["group_id"], "session_ids": sids, "n_facts": len(facts),
            "prompt_tokens": num_tokens(prompt), "prompt_sha256": sha(prompt),
            "sessions_block_tokens": num_tokens(block), "facts_block_tokens": num_tokens(facts_block),
            "sessions_block_head": block[:400], "facts_block_head": facts_block[:400],
        })

    (OUT / "prompt_audit.json").write_text(
        json.dumps({"reader": manifest, "comprehension": comp_manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    print("reader prompt reconstruction vs record:")
    for c in manifest["cases"]:
        print(f"  q{c['qidx']:>3} tokens {c['record_context_tokens']} vs {c['recomputed_context_tokens']} "
              f"n {c['record_n_context']} vs {c['recomputed_n_context']} ok={c['ranking_reconstructs_record']}")
    print("comprehension prompts:")
    for g in comp_manifest["groups"]:
        print(f"  group {g['group_id']} sessions={g['session_ids']} facts={g['n_facts']} "
              f"tokens={g['prompt_tokens']} sha={g['prompt_sha256'][:12]}")


if __name__ == "__main__":
    main()
