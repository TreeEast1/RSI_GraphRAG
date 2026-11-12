"""Freeze the MuSiQue corpus and the adaptation / held-out split.

Gold answers and supporting flags are written only to the eval file. Read-time
code opens the corpus file and does not read the eval file.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

HIPPO_SRC = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")
if str(HIPPO_SRC) not in sys.path:
    sys.path.insert(0, str(HIPPO_SRC))

from hipporag.utils.misc_utils import compute_mdhash_id  # noqa: E402

from .hashing import sha256_obj


def passage_content(title: str, text: str) -> str:
    return f"{title}\n{text}"


def passage_id(title: str, text: str) -> str:
    return compute_mdhash_id(passage_content(title, text), prefix="chunk-")


def normalize_question(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def word_set(text: str) -> set:
    return set(normalize_question(text).split())


def jaccard(left: set, right: set) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def largest_remainder(group_sizes: Dict[str, int], target: int) -> Dict[str, int]:
    total = sum(group_sizes.values())
    if total == 0:
        return {key: 0 for key in group_sizes}
    raw = {key: group_sizes[key] * target / total for key in group_sizes}
    floors = {key: int(raw[key]) for key in group_sizes}
    leftover = target - sum(floors.values())
    ranked = sorted(group_sizes, key=lambda key: (raw[key] - floors[key], key), reverse=True)
    for key in ranked[:leftover]:
        floors[key] += 1
    return floors


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def freeze_musique(
    questions_path: str,
    corpus_path: str,
    out_dir: str,
    seed: int = 20261030,
    adaptation_n: int = 600,
    update_n: int = 500,
    dev_n: int = 100,
    heldout_n: int = 400,
) -> Dict:
    del dev_n, heldout_n  # targets are checked against the realized counts below
    out = Path(out_dir)
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    if len(questions) != 1000:
        raise RuntimeError(f"expected 1000 MuSiQue questions, found {len(questions)}")
    if len(corpus) != 11656:
        raise RuntimeError(f"expected 11656 MuSiQue passages, found {len(corpus)}")

    passages = []
    seen = {}
    duplicate_passage_rows = 0
    for index, row in enumerate(corpus):
        title = row["title"]
        text = row["text"]
        content = passage_content(title, text)
        pid = passage_id(title, text)
        if pid in seen:
            duplicate_passage_rows += 1
            continue
        seen[pid] = index
        passages.append(
            {
                "passage_id": pid,
                "corpus_index": index,
                "title": title,
                "text": text,
                "content": content,
                "source": "hipporag_reproduce_musique_corpus",
            }
        )

    by_hop = defaultdict(list)
    eval_rows = []
    exact_groups = defaultdict(list)
    for question in questions:
        hop = len(question["question_decomposition"])
        by_hop[str(hop)].append(question["id"])
        gold_ids = []
        gold_contents = []
        missing = []
        paragraphs = []
        for paragraph in question["paragraphs"]:
            title = paragraph["title"]
            text = paragraph.get("paragraph_text", paragraph.get("text", ""))
            content = passage_content(title, text)
            pid = passage_id(title, text)
            record = {
                "passage_id": pid,
                "title": title,
                "text": text,
                "content": content,
                "is_supporting": bool(paragraph.get("is_supporting")),
                "in_frozen_corpus": pid in seen,
            }
            paragraphs.append(record)
            if record["is_supporting"]:
                gold_contents.append(content)
                if pid in seen:
                    gold_ids.append(pid)
                else:
                    missing.append(pid)
        eval_rows.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "hop": hop,
                "answer": question["answer"],
                "answer_aliases": question.get("answer_aliases") or [],
                "answerable": question.get("answerable"),
                "gold_passage_ids": gold_ids,
                "gold_contents": gold_contents,
                "missing_gold_passage_ids": missing,
                "paragraphs": paragraphs,
                "decomposition": question["question_decomposition"],
            }
        )
        exact_groups[normalize_question(question["question"])].append(question["id"])

    hop_sizes = {hop: len(ids) for hop, ids in by_hop.items()}
    adaptation_by_hop = largest_remainder(hop_sizes, adaptation_n)
    rng = random.Random(seed)
    roles = {}
    for hop, ids in by_hop.items():
        ordered = list(ids)
        rng.shuffle(ordered)
        take = adaptation_by_hop[hop]
        for qid in ordered[:take]:
            roles[qid] = "adaptation"
        for qid in ordered[take:]:
            roles[qid] = "heldout"

    adaptation_hops = defaultdict(list)
    for row in eval_rows:
        if roles[row["question_id"]] == "adaptation":
            adaptation_hops[str(row["hop"])].append(row["question_id"])
    update_by_hop = largest_remainder(
        {hop: len(ids) for hop, ids in adaptation_hops.items()}, update_n
    )
    for hop, ids in adaptation_hops.items():
        ordered = list(ids)
        rng.shuffle(ordered)
        take = update_by_hop[hop]
        for qid in ordered[:take]:
            roles[qid] = "update"
        for qid in ordered[take:]:
            roles[qid] = "dev"

    exact_duplicates = {text: ids for text, ids in exact_groups.items() if len(ids) > 1}
    quarantined = []
    for ids in exact_duplicates.values():
        crosses = any(roles[qid] == "heldout" for qid in ids) and any(
            roles[qid] != "heldout" for qid in ids
        )
        if crosses:
            for qid in ids:
                if roles[qid] == "heldout":
                    roles[qid] = "quarantine_duplicate"
                    quarantined.append(qid)

    similar_pairs = []
    tokens = {row["question_id"]: word_set(row["question"]) for row in eval_rows}
    questions_by_id = {row["question_id"]: row["question"] for row in eval_rows}
    ids = list(questions_by_id)
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            if normalize_question(questions_by_id[left]) == normalize_question(questions_by_id[right]):
                continue
            score = jaccard(tokens[left], tokens[right])
            if score >= 0.9 and roles[left] != roles[right]:
                similar_pairs.append(
                    {
                        "a": left,
                        "b": right,
                        "jaccard": round(score, 4),
                        "role_a": roles[left],
                        "role_b": roles[right],
                    }
                )
    for pair in similar_pairs:
        if pair["jaccard"] < 0.999:
            continue
        for qid, role in ((pair["a"], pair["role_a"]), (pair["b"], pair["role_b"])):
            if role == "heldout":
                roles[qid] = "quarantine_token_identical"
                quarantined.append(qid)

    covered_questions = 0
    missing_gold_slots = 0
    gold_slots = 0
    for row in eval_rows:
        gold_slots += len(row["gold_contents"])
        missing_gold_slots += len(row["missing_gold_passage_ids"])
        if row["gold_passage_ids"] and not row["missing_gold_passage_ids"]:
            covered_questions += 1
        row["split_role"] = roles[row["question_id"]]

    public_rows = [
        {
            "passage_id": row["passage_id"],
            "corpus_index": row["corpus_index"],
            "title": row["title"],
            "text": row["text"],
            "content": row["content"],
        }
        for row in passages
    ]
    corpus_hash = sha256_obj(
        [{"passage_id": row["passage_id"], "content": row["content"]} for row in public_rows]
    )
    role_counts = defaultdict(int)
    hop_role = defaultdict(lambda: defaultdict(int))
    for row in eval_rows:
        role_counts[row["split_role"]] += 1
        hop_role[str(row["hop"])][row["split_role"]] += 1

    manifest = {
        "dataset": "musique",
        "source_questions": questions_path,
        "source_corpus": corpus_path,
        "n_questions": len(eval_rows),
        "n_corpus_rows": len(corpus),
        "n_unique_passages": len(public_rows),
        "duplicate_passage_rows": duplicate_passage_rows,
        "corpus_hash": corpus_hash,
        "seed": seed,
        "role_counts": dict(role_counts),
        "hop_sizes": hop_sizes,
        "hop_by_role": {hop: dict(counts) for hop, counts in hop_role.items()},
        "questions_with_all_gold_in_corpus": covered_questions,
        "gold_passage_slots": gold_slots,
        "gold_passage_slots_missing_from_corpus": missing_gold_slots,
        "exact_duplicate_groups": exact_duplicates,
        "quarantined_duplicate_ids": quarantined,
        "cross_split_near_duplicates_jaccard_0_9": similar_pairs[:50],
        "cross_split_near_duplicate_count": len(similar_pairs),
        "dedup_rule": (
            "Passage ids are HippoRAG chunk hashes of title + newline + text. "
            "Exact normalized-question duplicates that would cross into held-out "
            "are quarantined from held-out. Near-duplicates with token Jaccard >= 0.9 "
            "are recorded and left in their sampled split."
        ),
        "targets": {
            "adaptation": adaptation_n,
            "update": update_n,
            "dev": 100,
            "heldout": 400,
        },
        "heldout_not_for_evolution": True,
    }
    _write_jsonl(out / "corpus.jsonl", public_rows)
    _write_jsonl(out / "questions_eval.jsonl", eval_rows)
    _write_json(out / "split_roles.json", roles)
    _write_json(out / "corpus_manifest.json", manifest)
    return manifest


def load_corpus(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_eval_questions(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def assert_read_time_inputs_have_no_questions(passages: Sequence[dict]) -> None:
    forbidden = {
        "question",
        "answer",
        "answer_aliases",
        "is_supporting",
        "decomposition",
        "gold_passage_ids",
    }
    for row in passages:
        overlap = forbidden & set(row)
        if overlap:
            raise RuntimeError(f"read-time corpus row contains eval fields {sorted(overlap)}")
