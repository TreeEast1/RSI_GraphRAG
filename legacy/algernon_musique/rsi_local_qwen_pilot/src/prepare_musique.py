#!/usr/bin/env python3
"""Freeze a MuSiQue pilot pool. Gold answers are stored for scoring only."""
from __future__ import annotations

import hashlib
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import abspath, load_json, save_json  # noqa: E402


def doc_id(title: str, text: str) -> str:
    digest = hashlib.sha1(f"{title}\n{text}".encode("utf-8")).hexdigest()[:16]
    return f"doc-{digest}"


def main() -> None:
    cfg = load_json(abspath("configs", "pilot.json"))
    qa = load_json(cfg["dataset_path"])
    pool = [row for row in qa if row.get("answerable", True) and row.get("answer")]
    rng = random.Random(cfg["seed"])
    rng.shuffle(pool)
    frozen = pool[: cfg["frozen_pool_n"]]
    read_rows = frozen[: cfg["read_n"]]

    documents = {}
    questions = []
    for qi, row in enumerate(frozen):
        support_ids = []
        distractor_ids = []
        para_ids = []
        for para in row["paragraphs"]:
            title = para.get("title") or ""
            text = para.get("paragraph_text") or ""
            did = doc_id(title, text)
            para_ids.append(did)
            if did not in documents:
                documents[did] = {
                    "doc_id": did,
                    "title": title,
                    "text": text,
                    "n_chars": len(text),
                }
            if para.get("is_supporting"):
                support_ids.append(did)
            else:
                distractor_ids.append(did)
        questions.append({
            "qid": row["id"],
            "pool_index": qi,
            "in_read_set": qi < cfg["read_n"],
            "question": row["question"],
            "answer": row["answer"],
            "answer_aliases": row.get("answer_aliases") or [],
            "support_ids": support_ids,
            "distractor_ids": distractor_ids,
            "paragraph_ids": para_ids,
            "decomposition": [
                {"question": hop.get("question"), "answer": hop.get("answer"),
                 "paragraph_support_idx": hop.get("paragraph_support_idx")}
                for hop in (row.get("question_decomposition") or [])
            ],
        })

    read_doc_ids = []
    seen = set()
    for q in questions:
        if not q["in_read_set"]:
            continue
        for did in q["paragraph_ids"]:
            if did not in seen:
                seen.add(did)
                read_doc_ids.append(did)

    support_covered = sum(1 for q in questions if q["in_read_set"] and q["support_ids"])
    manifest = {
        "seed": cfg["seed"],
        "source": cfg["dataset_path"],
        "frozen_pool_n": len(questions),
        "read_n": cfg["read_n"],
        "retrieval_corpus_docs": len(documents),
        "extraction_docs": len(read_doc_ids),
        "read_questions_with_support": support_covered,
        "full_musique_questions_available": len(qa),
        "limitation": cfg["limits"]["corpus"],
    }
    save_json(abspath("data", "pilot_manifest.json"), manifest)
    save_json(abspath("data", "questions.json"), questions)
    save_json(abspath("data", "documents.json"), documents)
    save_json(abspath("data", "read_doc_ids.json"), read_doc_ids)
    print(json_dumps(manifest))


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
