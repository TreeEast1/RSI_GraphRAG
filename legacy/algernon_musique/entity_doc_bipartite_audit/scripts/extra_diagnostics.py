#!/usr/bin/env python3
"""Follow-up measurements that do not change the pre-specified thresholds.

Gold colocation for hard groupings is reported after those groupings already exist.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_audit as ra


def cross_coverage(ranked, k: int) -> float:
    n = len(ranked)
    seen = np.zeros(n, dtype=bool)
    for doc, items in enumerate(ranked):
        if not items:
            continue
        seen[doc] = True
        for item in items[:k]:
            seen[item[0]] = True
    return float(seen.mean())


def pair_count_hist(pairs: dict) -> dict:
    hist = Counter()
    for count, _idf in pairs.values():
        hist[int(count)] += 1
    return {
        "n_pairs": len(pairs),
        "count_eq_1": hist[1],
        "count_eq_2": hist[2],
        "count_eq_3": hist[3],
        "count_ge_4": sum(value for key, value in hist.items() if key >= 4),
        "frac_count_eq_1": hist[1] / len(pairs) if pairs else None,
    }


def gold_ids(corpus):
    pid = {row["passage_id"]: i for i, row in enumerate(corpus)}
    questions = []
    with ra.QUESTIONS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            gold = []
            seen = set()
            for item in row.get("gold_passage_ids") or []:
                doc = pid.get(item)
                if doc is None or doc in seen:
                    continue
                seen.add(doc)
                gold.append(doc)
            if len(gold) >= 2:
                questions.append(gold)
    return questions


def colocated(labels, questions) -> int:
    hit = 0
    for gold in questions:
        base = int(labels[gold[0]])
        if all(int(labels[doc]) == base for doc in gold[1:]):
            hit += 1
    return hit


def index_gaps(corpus, questions):
    gaps = []
    adjacent = 0
    for gold in questions:
        indexes = sorted(corpus[doc]["corpus_index"] for doc in gold)
        span = indexes[-1] - indexes[0]
        gaps.append(span)
        if span <= 6:
            adjacent += 1
    return {"span_percentiles": ra.pcts(gaps), "questions_span_le_6": adjacent, "n": len(questions)}


def medium_case(corpus, labels, b1_sets, df, display, tokens, low=6, high=16):
    members = defaultdict(list)
    for doc, label in enumerate(labels.tolist()):
        members[int(label)].append(doc)
    candidates = []
    for docs in members.values():
        if not (low <= len(docs) <= high):
            continue
        titles = {corpus[doc]["title"] for doc in docs}
        if len(titles) < 2:
            continue
        counts = Counter()
        for doc in docs:
            for eid in b1_sets[doc]:
                counts[eid] += 1
        shared = [eid for eid, count in counts.items() if count >= 2]
        if not shared:
            continue
        bridge = max(shared, key=lambda eid: (math.log((len(corpus) + 1) / (int(df[eid]) + 1)), -int(df[eid])))
        candidates.append((int(df[bridge]), len(docs), docs, bridge, shared))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    _dfb, _n, docs, bridge, shared = candidates[0]
    return {
        "n_docs": len(docs),
        "n_titles": len({corpus[doc]["title"] for doc in docs}),
        "tokens": int(tokens[docs].sum()),
        "bridge": {"entity": display[bridge], "df": int(df[bridge])},
        "shared": [{"entity": display[eid], "df": int(df[eid]), "in_group": int(sum(eid in b1_sets[doc] for doc in docs))} for eid in sorted(shared, key=lambda eid: int(df[eid]))[:12]],
        "passages": [
            {
                "passage_id": corpus[doc]["passage_id"],
                "title": corpus[doc]["title"],
                "snippet": ra.snippet(corpus[doc]["text"], 320),
            }
            for doc in docs
        ],
    }


def contains(corpus, passage_id, needle):
    for row in corpus:
        if row["passage_id"] == passage_id:
            return needle.lower() in row["text"].lower() or needle.lower() in row["title"].lower()
    return None


def main() -> None:
    corpus = ra.load_corpus()
    questions = gold_ids(corpus)
    title_ids = np.empty(len(corpus), dtype=np.int32)
    vocab = {}
    for i, row in enumerate(corpus):
        vocab.setdefault(row["title"], len(vocab))
        title_ids[i] = vocab[row["title"]]
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = np.array([len(encoder.encode(row["content"])) for row in corpus], dtype=np.int32)
    out = {"gold_index_gaps": index_gaps(corpus, questions), "sources": {}, "string_checks": {
        "messi_las_palmas": contains(corpus, "chunk-7b2db166f07ccb8e39f5d677bbfef87a", "Las Palmas"),
        "messi_yokohama": contains(corpus, "chunk-7b2db166f07ccb8e39f5d677bbfef87a", "Yokohama"),
        "yaiza_las_palmas": contains(corpus, "chunk-7b27d7072225397ba5ae7da9c1568193", "Las Palmas"),
        "prisoner_macmillan": contains(corpus, "chunk-73adb221395bf6740c2daa28fce2318b", "Macmillan"),
        "surname_macmillan": contains(corpus, "chunk-b0643b987855da5c9e7d62814a10ec9f", "Macmillan"),
    }}
    prepared = {
        "luna_g_raw": ra.load_luna(corpus),
        "hipporag2_gpt4o_mini": ra.load_official(corpus),
    }
    for name, rows in prepared.items():
        graph = ra.intern_source(rows["ner"], rows["triples"])
        n_ent = len(graph["display"])
        n_docs = len(corpus)
        lists = ra.doclists_from_sets(graph["b1_sets"], n_ent)
        df = np.array([len(docs) for docs in lists], dtype=np.int32)
        cap = ra.choose_pair_cap(df)["cap"]
        pairs, meta = ra.build_pairs(lists, df, cap, n_docs)
        mass = ra.doc_idf_mass(lists, df, cap, n_docs)
        neighbors = ra.neighbor_lists(pairs, mass, n_docs)

        def jaccard_of(key, idf_sum):
            left, right = divmod(key, n_docs)
            denom = mass[left] + mass[right] - idf_sum
            return idf_sum / denom if denom > 0 else 0.0

        labels = {
            "shared_count_ge_3": ra.hard_labels_from_pairs(n_docs, pairs, lambda c, s, k, n: c >= 3),
            "shared_count_ge_5": ra.hard_labels_from_pairs(n_docs, pairs, lambda c, s, k, n: c >= 5),
            "jaccard_ge_0.2": ra.hard_labels_from_pairs(n_docs, pairs, lambda c, s, k, n: jaccard_of(k, s) >= 0.2),
            "jaccard_ge_0.4": ra.hard_labels_from_pairs(n_docs, pairs, lambda c, s, k, n: jaccard_of(k, s) >= 0.4),
        }
        out["sources"][name] = {
            "pair_hist": pair_count_hist(pairs),
            "pair_meta": meta,
            "topk_cross_coverage": {str(k): cross_coverage(neighbors["shared_idf"], k) for k in (2, 4, 8, 16)},
            "gold_hard_colocation": {key: colocated(value, questions) for key, value in labels.items()},
            "n_questions": len(questions),
            "medium_jaccard_0_2": medium_case(
                corpus, labels["jaccard_ge_0.2"], graph["b1_sets"], df, graph["display"], tokens
            ),
        }
        print(name, "pairs", meta["enumerated_unique_pairs"], "hist", out["sources"][name]["pair_hist"], flush=True)
    path = ra.OUT / "artifacts" / "extra_diagnostics.json"
    ra.json_dump(path, out)
    print(path)


if __name__ == "__main__":
    main()
