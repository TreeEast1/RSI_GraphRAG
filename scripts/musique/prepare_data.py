"""Freeze MuSiQue questions, gold mapping and practice/dev/eval splits."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.musique.data import OUT, RAW, load_corpus, load_questions, make_splits, overlap_stats, sha256  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
qs = load_questions()
corpus = load_corpus()
splits_path = OUT / "splits.json"
if splits_path.exists():
    splits = json.load(open(splits_path))
    print("splits already frozen; not regenerating")
else:
    splits = make_splits(qs)
    json.dump(splits, open(splits_path, "w"), indent=1)
with open(OUT / "questions.jsonl", "w") as f:
    for q in qs:
        f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")
audit = {
    "corpus_passages": len(corpus),
    "questions": len(qs),
    "sha256": {n: sha256(RAW / n) for n in ("musique.json", "musique_corpus.json")},
    "splits": {k: len(v) for k, v in splits.items()},
    "overlap": overlap_stats(qs, splits),
    "avg_gold": sum(len(q.gold_pids) for q in qs) / len(qs),
}
json.dump(audit, open(OUT / "data_audit.json", "w"), indent=1)
print(json.dumps(audit, indent=1))
