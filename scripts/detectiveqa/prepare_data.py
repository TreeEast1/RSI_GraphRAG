"""Freeze DetectiveQA questions, prefix-safe chunks and practice/evaluation splits."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detectiveqa.data import (OUT, RAW, audit, build_chunks, load_novels, load_questions,  # noqa: E402
                                  make_splits, sha256, write_jsonl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--lang", default="en", choices=["en", "zh"])
    ap.add_argument("--target-words", type=int, default=200)
    ap.add_argument("--max-words", type=int, default=320)
    args = ap.parse_args()

    novels = load_novels(args.lang, args.raw)
    questions = load_questions(args.lang, args.raw)
    cutoffs = defaultdict(set)
    for q in questions:
        cutoffs[q.novel_id].add(q.cutoff)
    chunks = {novel_id: build_chunks(novel, cutoffs[novel_id], args.target_words, args.max_words)
              for novel_id, novel in novels.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    splits_path = args.out / "splits.json"
    if splits_path.exists():
        splits = json.loads(splits_path.read_text())
        print("splits already frozen; not regenerating")
    else:
        splits = make_splits(questions)
        splits_path.write_text(json.dumps(splits, indent=1))

    write_jsonl(args.out / "questions.jsonl", questions)
    write_jsonl(args.out / "chunks.jsonl", [c for novel_id in sorted(chunks, key=int) for c in chunks[novel_id]])
    files = sorted(p for p in args.raw.rglob("*") if p.is_file()
                   and (f"_{args.lang}" in str(p.parent) or p.name == "README.md"))
    report = audit(novels, questions, chunks, splits)
    report.update({
        "lang": args.lang,
        "chunking": {"target_words": args.target_words, "max_words": args.max_words,
                     "hard_boundaries": "every question cutoff"},
        "sha256": {str(p.relative_to(args.raw)): sha256(p) for p in files},
    })
    (args.out / "data_audit.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "sha256"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
