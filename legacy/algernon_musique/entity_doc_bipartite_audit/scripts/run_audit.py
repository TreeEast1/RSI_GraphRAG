#!/usr/bin/env python3
"""Entity–document bipartite audit for the frozen MuSiQue corpus.

Reads existing OpenIE caches only. Does not call a model, embed text, or edit
the original graph. Question gold is consulted only after every threshold below
has already been applied, and it does not change those thresholds.

Pair scores use entities with document frequency <= PAIR_CAP. The cap is a
computation limit so high-DF inverted lists are not expanded into O(df^2)
stored pairs. Connectivity that uses every entity, including hubs, is computed
separately with union-find and is not subject to this cap.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261030
TOKEN_BUDGET = 16000
PAIR_UPDATE_LIMIT = 30_000_000
PAIR_CAPS = (50, 30, 20, 10)
TOPKS = (2, 4, 8, 16)
SHARED_THRESHOLDS = (1, 2, 3, 5)
IDF_THRESHOLDS = (4.0, 6.0, 8.0, 10.0, 12.0)
JACCARD_THRESHOLDS = (0.05, 0.10, 0.20, 0.40)
DF_CUTS = (10, 20, 50, 100)
DF_FRACTIONS = (("0.5pct", 0.005), ("1pct", 0.01), ("2pct", 0.02))
SIZE_BINS = (
    (1, 1, "1"),
    (2, 5, "2-5"),
    (6, 10, "6-10"),
    (11, 20, "11-20"),
    (21, 50, "21-50"),
    (51, 100, "51-100"),
    (101, 500, "101-500"),
    (501, 10**12, "500+"),
)

ROOT = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/experiments")
OUT = ROOT / "entity_doc_bipartite_audit"
CORPUS_PATH = ROOT / "rsi_graphrag_v4" / "data" / "corpus.jsonl"
QUESTIONS_PATH = ROOT / "rsi_graphrag_v4" / "data" / "questions_eval.jsonl"
LUNA_DIR = ROOT / "rsi_graphrag_v4" / "artifacts" / "openie" / "luna"
LUNA_FAILED_DIR = ROOT / "rsi_graphrag_v4" / "artifacts" / "openie" / "luna_failed"
OFFICIAL_OPENIE = (
    ROOT.parent
    / "case_study_musique"
    / "traditional"
    / "graph"
    / "openie_results_ner_gpt-4o-mini.json"
)
OFFICIAL_STATS = (
    ROOT.parent
    / "case_study_musique"
    / "traditional"
    / "graph"
    / "comparison_index_stats.json"
)


def audit_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )
    text = re.sub(r"[^A-Za-z0-9 ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def hippo_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9 ]", " ", str(text).lower()).strip()


def pcts(values) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None,
                "p75": None, "p90": None, "p95": None, "p99": None}
    qs = np.percentile(arr, [50, 75, 90, 95, 99])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(qs[0]),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p75": float(qs[1]),
        "p90": float(qs[2]),
        "p95": float(qs[3]),
        "p99": float(qs[4]),
    }


class UF:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.sz = [1] * n

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]


def load_corpus() -> list:
    rows = []
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: row["corpus_index"])
    return rows


def load_luna(corpus: list) -> dict:
    def read_one(row: dict):
        path = LUNA_DIR / f"{row['passage_id']}.json"
        if not path.exists():
            return row["corpus_index"], [], [], False, None, False
        record = json.loads(path.read_text(encoding="utf-8"))
        mismatch = record.get("content") != row["content"]
        return (
            row["corpus_index"],
            record.get("entities") or [],
            record.get("triples") or [],
            mismatch,
            record.get("response_model"),
            True,
        )

    ner = [None] * len(corpus)
    triples = [None] * len(corpus)
    mismatches = 0
    models = Counter()
    present = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, entities, triple_rows, mismatch, model, ok in pool.map(read_one, corpus):
            ner[index] = entities
            triples[index] = triple_rows
            mismatches += int(mismatch)
            present += int(ok)
            if model:
                models[model] += 1
    failed = sorted(path.name for path in LUNA_FAILED_DIR.glob("*.json")) if LUNA_FAILED_DIR.exists() else []
    return {
        "ner": ner,
        "triples": triples,
        "content_mismatches": mismatches,
        "models": models,
        "present": present,
        "failed_files": failed,
    }


def load_official(corpus: list) -> dict:
    payload = json.loads(OFFICIAL_OPENIE.read_text(encoding="utf-8"))
    by_id = {row["passage_id"]: row for row in corpus}
    ner = [[] for _ in corpus]
    triples = [[] for _ in corpus]
    matched = 0
    extra = 0
    mismatches = 0
    for doc in payload["docs"]:
        row = by_id.get(doc["idx"])
        if row is None:
            extra += 1
            continue
        matched += 1
        ner[row["corpus_index"]] = doc.get("extracted_entities") or []
        triples[row["corpus_index"]] = doc.get("extracted_triples") or []
        if doc.get("passage") != row["content"]:
            mismatches += 1
    missing = len(corpus) - matched
    return {
        "ner": ner,
        "triples": triples,
        "content_mismatches": mismatches,
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "n_docs_in_file": len(payload["docs"]),
    }


def intern_source(ner_rows, triple_rows) -> dict:
    vocab = {}
    display = []
    variants = []
    key_changed = 0
    raw_strings = 0
    invalid_triples = 0
    b1_sets = []
    b2_sets = []
    mention_edges = 0
    mention_multi = 0
    mention_sum = 0
    ner_not_triple_edges = 0
    triple_not_ner_edges = 0

    def intern(raw: str):
        nonlocal key_changed, raw_strings
        if not isinstance(raw, str):
            raw = str(raw)
        raw_strings += 1
        key = audit_key(raw)
        if not key:
            return None
        if hippo_key(raw) != key:
            key_changed += 1
        eid = vocab.get(key)
        if eid is None:
            eid = len(vocab)
            vocab[key] = eid
            display.append(raw.strip())
            variants.append({raw.strip()})
        else:
            if len(raw.strip()) > len(display[eid]):
                display[eid] = raw.strip()
            if len(variants[eid]) < 8:
                variants[eid].add(raw.strip())
        return eid

    for entities, triples in zip(ner_rows, triple_rows):
        counter = Counter()
        for raw in entities or []:
            eid = intern(raw)
            if eid is not None:
                counter[eid] += 1
        b1 = set(counter)
        b2 = set()
        seen_triples = set()
        for triple in triples or []:
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                invalid_triples += 1
                continue
            subject = intern(triple[0])
            obj = intern(triple[2])
            if subject is None or obj is None:
                invalid_triples += 1
                continue
            signature = (subject, audit_key(triple[1]), obj)
            if signature in seen_triples:
                continue
            seen_triples.add(signature)
            b2.add(subject)
            b2.add(obj)
        b1_sets.append(b1)
        b2_sets.append(b2)
        mention_edges += len(counter)
        mention_sum += sum(counter.values())
        mention_multi += sum(1 for count in counter.values() if count > 1)
        ner_not_triple_edges += len(b1 - b2)
        triple_not_ner_edges += len(b2 - b1)

    keys = [""] * len(vocab)
    for key, eid in vocab.items():
        keys[eid] = key
    return {
        "b1_sets": b1_sets,
        "b2_sets": b2_sets,
        "display": display,
        "variants": [sorted(item) for item in variants],
        "keys": keys,
        "key_changed_strings": key_changed,
        "raw_strings": raw_strings,
        "invalid_triples": invalid_triples,
        "mention_edges": mention_edges,
        "mention_multi": mention_multi,
        "mention_sum": mention_sum,
        "ner_not_triple_edges": ner_not_triple_edges,
        "triple_not_ner_edges": triple_not_ner_edges,
    }


def doclists_from_sets(sets: list, n_ent: int) -> list:
    lists = [[] for _ in range(n_ent)]
    for doc_id, entities in enumerate(sets):
        for eid in entities:
            lists[eid].append(doc_id)
    return lists


def component_labels(n_docs: int, lists: list, active: np.ndarray) -> np.ndarray:
    uf = UF(n_docs)
    for eid, docs in enumerate(lists):
        if not active[eid] or len(docs) < 2:
            continue
        base = docs[0]
        for other in docs[1:]:
            uf.union(base, other)
    return np.fromiter((uf.find(i) for i in range(n_docs)), dtype=np.int32, count=n_docs)


def describe_components(labels: np.ndarray, lists: list, active: np.ndarray, title_ids: np.ndarray, n_docs: int) -> dict:
    members = defaultdict(list)
    for doc_id, label in enumerate(labels.tolist()):
        members[label].append(doc_id)
    ent_count = Counter()
    for eid, docs in enumerate(lists):
        if not active[eid] or not docs:
            continue
        ent_count[int(labels[docs[0]])] += 1
    title_sets = {}
    rows = []
    for label, docs in members.items():
        titles = {int(title_ids[doc]) for doc in docs}
        title_sets[label] = titles
        rows.append((len(docs), ent_count[label], len(titles), label, docs))
    rows.sort(key=lambda item: (-item[0], -item[1], item[3]))
    sizes = np.array([item[0] for item in rows], dtype=np.int32)
    ent_sizes = np.array([item[1] for item in rows], dtype=np.int32)
    weighted = np.repeat(sizes, sizes)
    bins = []
    for low, high, name in SIZE_BINS:
        mask = (sizes >= low) & (sizes <= high)
        doc_sum = int(sizes[mask].sum()) if mask.any() else 0
        bins.append({
            "bin": name,
            "n_components": int(mask.sum()),
            "n_docs": doc_sum,
            "doc_frac": doc_sum / n_docs,
        })
    max_row = rows[0]
    min_row = min(rows, key=lambda item: (item[0], item[1], item[3]))
    nonempty = [item for item in rows if item[1] > 0]
    min_nonempty = min(nonempty, key=lambda item: (item[0], item[1], item[3])) if nonempty else min_row
    multi_title_docs = sum(item[0] for item in rows if item[2] > 1)
    single_title_multi = sum(1 for item in rows if item[0] > 1 and item[2] == 1)
    multi_title_components = sum(1 for item in rows if item[2] > 1)
    ranked = []
    for rank, item in enumerate(rows[:200], start=1):
        ranked.append({
            "rank": rank,
            "n_docs": int(item[0]),
            "n_entities": int(item[1]),
            "n_titles": int(item[2]),
            "label": int(item[3]),
        })
    remap = {item[3]: rank for rank, item in enumerate(rows)}
    compact = np.fromiter((remap[int(label)] for label in labels.tolist()), dtype=np.int32, count=n_docs)
    return {
        "n_components": len(rows),
        "unweighted": pcts(sizes),
        "doc_weighted": pcts(weighted),
        "entity_unweighted": pcts(ent_sizes),
        "min_component": {"n_docs": int(min_row[0]), "n_entities": int(min_row[1]), "n_titles": int(min_row[2])},
        "min_nonempty_component": {
            "n_docs": int(min_nonempty[0]),
            "n_entities": int(min_nonempty[1]),
            "n_titles": int(min_nonempty[2]),
        },
        "max_component": {
            "n_docs": int(max_row[0]),
            "n_entities": int(max_row[1]),
            "n_titles": int(max_row[2]),
            "doc_frac": max_row[0] / n_docs,
        },
        "top10": [item["n_docs"] for item in ranked[:10]],
        "top20": [item["n_docs"] for item in ranked[:20]],
        "top5_doc_frac": float(sum(item[0] for item in rows[:5]) / n_docs),
        "top10_doc_frac": float(sum(item[0] for item in rows[:10]) / n_docs),
        "bins": bins,
        "docs_in_multi_title_components": int(multi_title_docs),
        "multi_title_doc_frac": multi_title_docs / n_docs,
        "single_title_multichunk_components": int(single_title_multi),
        "multi_title_components": int(multi_title_components),
        "ranked_top200": ranked,
        "labels": compact,
        "isolated_doc_frac": float(np.mean(sizes[compact] == 1) if False else np.mean(
            np.fromiter((item[0] for item in rows for _ in item[4]), dtype=np.int32, count=n_docs) == 1
        )),
        "_rows": rows,
    }


def fix_isolated(summary: dict, n_docs: int) -> None:
    sizes = np.repeat(
        np.array([item[0] for item in summary["_rows"]], dtype=np.int32),
        np.array([item[0] for item in summary["_rows"]], dtype=np.int32),
    )
    # rows are sorted by size, but repeat above follows row order not doc order.
    # Recompute from labels and ranked sizes.
    label_sizes = np.zeros(summary["n_components"], dtype=np.int32)
    for item in summary["ranked_top200"]:
        if item["rank"] <= summary["n_components"]:
            label_sizes[item["rank"] - 1] = item["n_docs"]
    # ranked_top200 is only 200. Use _rows.
    label_sizes = np.array([item[0] for item in summary["_rows"]], dtype=np.int32)
    summary["isolated_doc_frac"] = float(label_sizes[summary["labels"]].astype(np.int32).mean() and
                                         (label_sizes[summary["labels"]] == 1).mean())
    del sizes


def entity_stats(sets: list, lists: list, title_ids: np.ndarray) -> dict:
    per_doc = np.array([len(item) for item in sets], dtype=np.int32)
    df = np.array([len(docs) for docs in lists], dtype=np.int32)
    title_df = np.zeros(len(lists), dtype=np.int32)
    for eid, docs in enumerate(lists):
        if docs:
            title_df[eid] = len({int(title_ids[doc]) for doc in docs})
    present = df[df > 0]
    return {
        "n_entities": int((df > 0).sum()),
        "n_edges": int(per_doc.sum()),
        "docs_with_zero_entities": int((per_doc == 0).sum()),
        "docs_with_zero_frac": float((per_doc == 0).mean()),
        "entities_per_doc": pcts(per_doc),
        "docs_per_entity": pcts(present),
        "titles_per_entity": pcts(title_df[df > 0]),
        "df1_frac": float((present == 1).mean()) if present.size else None,
        "df_gt1": int((present > 1).sum()),
        "cross_title_entities": int(((title_df > 1) & (df > 0)).sum()),
        "intra_title_only_multi_chunk": int(((df > 1) & (title_df == 1)).sum()),
        "df": df,
        "title_df": title_df,
        "per_doc": per_doc,
    }


def top_entities(df: np.ndarray, title_df: np.ndarray, other_df: np.ndarray, display, variants, keys, k: int = 50) -> list:
    order = np.argsort(-df, kind="mergesort")
    rows = []
    for eid in order:
        if df[eid] <= 0 or len(rows) >= k:
            break
        rows.append({
            "rank": len(rows) + 1,
            "entity": display[eid],
            "key": keys[eid],
            "df": int(df[eid]),
            "title_df": int(title_df[eid]),
            "other_graph_df": int(other_df[eid]) if eid < len(other_df) else 0,
            "n_surface_variants_capped8": len(variants[eid]),
            "surfaces": variants[eid],
        })
    return rows


def df_histogram(df: np.ndarray) -> list:
    present = df[df > 0]
    rows = []
    for low, high, name in ((1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10, "6-10"),
                            (11, 20, "11-20"), (21, 50, "21-50"), (51, 100, "51-100"),
                            (101, 500, "101-500"), (501, 10**12, "500+")):
        count = int(((present >= low) & (present <= high)).sum())
        rows.append({"bin": name, "n_entities": count, "frac": count / len(present) if len(present) else 0.0})
    return rows


def policy_active(df: np.ndarray, threshold):
    active = df > 0
    if threshold is not None:
        active = active & (df <= threshold)
    return active


def summarize_policy(name: str, threshold, labels_info: dict, df: np.ndarray, sets: list, n_docs: int) -> dict:
    active = policy_active(df, threshold)
    had_entity = np.array([len(item) > 0 for item in sets], dtype=bool)
    still_linked = np.zeros(n_docs, dtype=bool)
    for eid, docs in enumerate(labels_info["doclists"]):
        if active[eid]:
            for doc in docs:
                still_linked[doc] = True
    losing = int((had_entity & ~still_linked).sum())
    info = labels_info["info"]
    return {
        "policy": name,
        "threshold": None if threshold is None else float(threshold),
        "n_entities_unlinked": int((~active & (df > 0)).sum()) if threshold is not None else 0,
        "n_entities_linking": int(active.sum()),
        "n_docs_losing_all_links": losing,
        "n_components": info["n_components"],
        "max_docs": info["max_component"]["n_docs"],
        "max_entities": info["max_component"]["n_entities"],
        "max_titles": info["max_component"]["n_titles"],
        "max_doc_frac": info["max_component"]["doc_frac"],
        "p50_unweighted": info["unweighted"]["median"],
        "p95_unweighted": info["unweighted"]["p95"],
        "p99_unweighted": info["unweighted"]["p99"],
        "p50_weighted": info["doc_weighted"]["median"],
        "p95_weighted": info["doc_weighted"]["p95"],
        "p99_weighted": info["doc_weighted"]["p99"],
        "isolated_doc_frac": info["isolated_doc_frac"],
        "docs_in_cc_ge2_frac": 1.0 - info["isolated_doc_frac"],
        "multi_title_doc_frac": info["multi_title_doc_frac"],
        "top5_doc_frac": info["top5_doc_frac"],
        "top10_doc_frac": info["top10_doc_frac"],
    }


def build_components(n_docs, lists, active, title_ids):
    labels = component_labels(n_docs, lists, active)
    info = describe_components(labels, lists, active, title_ids, n_docs)
    label_sizes = np.array([item[0] for item in info["_rows"]], dtype=np.int32)
    info["isolated_doc_frac"] = float((label_sizes[info["labels"]] == 1).mean())
    return info


def choose_pair_cap(df: np.ndarray) -> dict:
    present = df[df > 1]
    considered = []
    for cap in PAIR_CAPS:
        updates = int(sum(int(value) * (int(value) - 1) // 2 for value in present if int(value) <= cap))
        skipped = int((present > cap).sum())
        skipped_slots = int(sum(int(value) * (int(value) - 1) // 2 for value in present if int(value) > cap))
        considered.append({
            "cap": cap,
            "pair_slot_updates": updates,
            "skipped_entities_df_gt_cap": skipped,
            "skipped_pair_slots": skipped_slots,
            "accepted": updates <= PAIR_UPDATE_LIMIT,
        })
    chosen = next(item["cap"] for item in considered if item["accepted"])
    return {"limit": PAIR_UPDATE_LIMIT, "considered": considered, "cap": chosen}


def build_pairs(lists, df: np.ndarray, cap: int, n_docs: int):
    pairs = {}
    skipped_entities = 0
    skipped_slots = 0
    used_entities = 0
    for eid, docs in enumerate(lists):
        degree = int(df[eid])
        if degree < 2:
            continue
        if degree > cap:
            skipped_entities += 1
            skipped_slots += degree * (degree - 1) // 2
            continue
        used_entities += 1
        weight = math.log((n_docs + 1) / (degree + 1))
        ordered = docs
        for left_i, left in enumerate(ordered):
            for right in ordered[left_i + 1:]:
                a, b = (left, right) if left < right else (right, left)
                key = a * n_docs + b
                rec = pairs.get(key)
                if rec is None:
                    pairs[key] = [1, weight]
                else:
                    rec[0] += 1
                    rec[1] += weight
    return pairs, {
        "cap": cap,
        "enumerated_unique_pairs": len(pairs),
        "used_entities": used_entities,
        "skipped_entities_df_gt_cap": skipped_entities,
        "skipped_pair_slots_not_unique": skipped_slots,
    }


def doc_idf_mass(lists, df: np.ndarray, cap: int, n_docs: int) -> np.ndarray:
    mass = np.zeros(n_docs, dtype=np.float64)
    for eid, docs in enumerate(lists):
        degree = int(df[eid])
        if degree < 1 or degree > cap:
            continue
        weight = math.log((n_docs + 1) / (degree + 1))
        for doc in docs:
            mass[doc] += weight
    return mass


def neighbor_lists(pairs: dict, mass: np.ndarray, n_docs: int):
    neigh = [[] for _ in range(n_docs)]
    for key, (count, idf_sum) in pairs.items():
        left, right = divmod(key, n_docs)
        denom = mass[left] + mass[right] - idf_sum
        jaccard = float(idf_sum / denom) if denom > 0 else 0.0
        item_l = (right, int(count), float(idf_sum), jaccard)
        item_r = (left, int(count), float(idf_sum), jaccard)
        neigh[left].append(item_l)
        neigh[right].append(item_r)
    by_count, by_idf, by_jaccard = [], [], []
    for items in neigh:
        by_count.append(sorted(items, key=lambda item: (-item[1], -item[2], item[0])))
        by_idf.append(sorted(items, key=lambda item: (-item[2], -item[1], item[0])))
        by_jaccard.append(sorted(items, key=lambda item: (-item[3], -item[2], item[0])))
    return {"shared_count": by_count, "shared_idf": by_idf, "weighted_jaccard": by_jaccard}


def _group_stats(size_token_member, n_docs: int, total_corpus_tokens: int) -> dict:
    sizes, tokens, membership, covered_tokens = size_token_member
    sizes = np.asarray(sizes, dtype=np.int32)
    tokens = np.asarray(tokens, dtype=np.int64)
    cross = sizes >= 2
    cross_sizes = sizes[cross]
    cross_tokens = tokens[cross]
    covered = membership > 0
    # membership was counted on cross groups only by the caller
    unique_tokens = int(covered_tokens)
    total_group_tokens = int(cross_tokens.sum()) if cross_tokens.size else 0
    return {
        "n_groups_including_singletons": int(sizes.size),
        "n_cross_groups": int(cross.sum()),
        "n_singletons": int((~cross).sum()) if sizes.size else 0,
        "size_all": pcts(sizes),
        "size_cross": pcts(cross_sizes),
        "token_cross": pcts(cross_tokens),
        "frac_cross_groups_le_budget": float((cross_tokens <= TOKEN_BUDGET).mean()) if cross_tokens.size else None,
        "frac_cross_groups_gt_budget": float((cross_tokens > TOKEN_BUDGET).mean()) if cross_tokens.size else None,
        "frac_cross_groups_gt_500_docs": float((cross_sizes > 500).mean()) if cross_sizes.size else None,
        "max_doc_frac": float(cross_sizes.max() / n_docs) if cross_sizes.size else 0.0,
        "coverage_docs_frac": float((membership > 0).mean()),
        "membership_mean": float(membership[membership > 0].mean()) if (membership > 0).any() else 0.0,
        "membership_p95": float(np.percentile(membership[membership > 0], 95)) if (membership > 0).any() else 0.0,
        "membership_max": int(membership.max()) if membership.size else 0,
        "total_cross_group_tokens": total_group_tokens,
        "unique_covered_doc_tokens": unique_tokens,
        "repeat_factor": (total_group_tokens / unique_tokens) if unique_tokens else None,
        "corpus_tokens": total_corpus_tokens,
        "cross_tokens_over_corpus": (total_group_tokens / total_corpus_tokens) if total_corpus_tokens else None,
    }


def summarize_topk(ranked_neighbors, k: int, tokens: np.ndarray) -> dict:
    n_docs = len(ranked_neighbors)
    sizes = np.empty(n_docs, dtype=np.int32)
    group_tokens = np.empty(n_docs, dtype=np.int64)
    membership = np.zeros(n_docs, dtype=np.int32)
    for doc_id, items in enumerate(ranked_neighbors):
        members = [doc_id]
        members.extend(item[0] for item in items[:k])
        sizes[doc_id] = len(members)
        group_tokens[doc_id] = int(tokens[members].sum())
        for member in members:
            membership[member] += 1
    cross = sizes >= 2
    covered = np.zeros(n_docs, dtype=bool)
    for doc_id, items in enumerate(ranked_neighbors):
        if not items:
            continue
        covered[doc_id] = True
        for item in items[:k]:
            covered[item[0]] = True
    unique_tokens = int(tokens[covered].sum())
    return _group_stats((sizes, group_tokens, membership, unique_tokens), n_docs, int(tokens.sum()))


def summarize_hard(labels: np.ndarray, tokens: np.ndarray, n_docs: int) -> dict:
    members = defaultdict(list)
    for doc_id, label in enumerate(labels.tolist()):
        members[label].append(doc_id)
    sizes = []
    group_tokens = []
    membership = np.zeros(n_docs, dtype=np.int32)
    covered = np.zeros(n_docs, dtype=bool)
    for docs in members.values():
        size = len(docs)
        sizes.append(size)
        group_tokens.append(int(tokens[docs].sum()))
        if size >= 2:
            for doc in docs:
                membership[doc] += 1
                covered[doc] = True
    # singletons exist as components; stats helper expects one row per component
    unique_tokens = int(tokens[covered].sum())
    return _group_stats((sizes, group_tokens, membership, unique_tokens), n_docs, int(tokens.sum()))


def hard_labels_from_pairs(n_docs: int, pairs: dict, predicate) -> np.ndarray:
    uf = UF(n_docs)
    for key, (count, idf_sum) in pairs.items():
        if predicate(count, idf_sum, key, n_docs):
            left, right = divmod(key, n_docs)
            uf.union(left, right)
    return np.fromiter((uf.find(i) for i in range(n_docs)), dtype=np.int32, count=n_docs)


def snippet(text: str, limit: int = 380) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def passage_brief(corpus, doc_id: int, entities: set, display, df: np.ndarray, limit_entities: int = 12) -> dict:
    ranked = sorted(entities, key=lambda eid: (-int(df[eid]), display[eid]))
    return {
        "passage_id": corpus[doc_id]["passage_id"],
        "corpus_index": corpus[doc_id]["corpus_index"],
        "title": corpus[doc_id]["title"],
        "snippet": snippet(corpus[doc_id]["text"]),
        "n_entities": len(entities),
        "top_entities": [
            {"entity": display[eid], "df": int(df[eid])}
            for eid in ranked[:limit_entities]
        ],
    }


def select_cases(corpus, title_ids, b1_sets, b2_sets, lists, df, display, keys, pairs, n_docs, comp_rows) -> list:
    cases = []
    title_of = [row["title"] for row in corpus]

    largest = comp_rows[0]
    top_ids = [int(eid) for eid in np.argsort(-df, kind="mergesort") if df[eid] > 0][:15]
    cases.append({
        "category": "largest_component",
        "n_docs": int(largest[0]),
        "n_entities": int(largest[1]),
        "n_titles": int(largest[2]),
        "top_entities": [{"entity": display[eid], "df": int(df[eid]), "key": keys[eid]} for eid in top_ids],
        "note": "The member list is not dumped. These are the fifteen entities with the highest chunk frequency; in a giant component they are the links that hold it together.",
    })

    # Medium components, prefer several titles and a specific bridge.
    medium = []
    for item in comp_rows:
        if 8 <= item[0] <= 16 and item[2] >= 2:
            docs = item[4]
            shared = Counter()
            for doc in docs:
                for eid in b1_sets[doc]:
                    if sum(1 for other in docs if eid in b1_sets[other]) >= 2:
                        shared[eid] += 1
            if not shared:
                continue
            bridge = max(shared, key=lambda eid: (math.log((n_docs + 1) / (int(df[eid]) + 1)), -int(df[eid]), keys[eid]))
            medium.append((int(df[bridge]), item, bridge))
    medium.sort(key=lambda item: (item[0], -item[1][0], item[2]))
    for _, item, bridge in medium[:3]:
        docs = item[4]
        cases.append({
            "category": "medium_component",
            "n_docs": int(item[0]),
            "n_entities": int(item[1]),
            "n_titles": int(item[2]),
            "bridge": {"entity": display[bridge], "df": int(df[bridge]), "key": keys[bridge]},
            "passages": [passage_brief(corpus, doc, b1_sets[doc], display, df) for doc in docs],
        })

    small = []
    for item in comp_rows:
        if item[0] not in (2, 3) or item[2] < 2:
            continue
        docs = item[4]
        inter = set.intersection(*(b1_sets[doc] for doc in docs)) if docs else set()
        if not inter:
            # bridge may be a path rather than one entity shared by all
            counts = Counter(eid for doc in docs for eid in b1_sets[doc])
            inter = {eid for eid, count in counts.items() if count >= 2}
        if not inter:
            continue
        bridge = max(inter, key=lambda eid: (math.log((n_docs + 1) / (int(df[eid]) + 1)), len(keys[eid])))
        if len(keys[bridge]) < 8:
            continue
        small.append((int(df[bridge]), -len(keys[bridge]), item, bridge, sorted(inter)))
    small.sort(key=lambda item: (item[0], item[1]))
    for df_bridge, _, item, bridge, inter in small[:3]:
        docs = item[4]
        cases.append({
            "category": "small_component",
            "n_docs": int(item[0]),
            "n_entities": int(item[1]),
            "n_titles": int(item[2]),
            "bridge": {"entity": display[bridge], "df": int(df[bridge]), "key": keys[bridge]},
            "shared_entities": [{"entity": display[eid], "df": int(df[eid])} for eid in sorted(inter, key=lambda eid: -int(df[eid]))[:8]],
            "passages": [passage_brief(corpus, doc, b1_sets[doc], display, df) for doc in docs],
        })

    # Generic hub pair: intersection is nonempty and every shared entity has DF > 100.
    generic = None
    order = np.argsort(-df, kind="mergesort")
    preferred = ("united states", "china", "english", "france", "europe", "america", "world")
    hub_candidates = [int(eid) for eid in order if df[eid] > 100][:40]
    hub_candidates.sort(key=lambda eid: (0 if any(name in keys[eid] for name in preferred) else 1, -int(df[eid])))
    for eid in hub_candidates:
        sample = []
        seen_titles = set()
        for doc in lists[eid]:
            title = title_of[doc]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            sample.append(doc)
            if len(sample) >= 18:
                break
        for i, left in enumerate(sample):
            for right in sample[i + 1:]:
                inter = b1_sets[left] & b1_sets[right]
                if not inter:
                    continue
                if any(int(df[item]) <= 100 for item in inter):
                    continue
                score = (len(inter), -max(int(df[item]) for item in inter))
                if generic is None or score < generic[0]:
                    generic = (score, left, right, inter)
        if generic is not None and generic[0][0] == 1:
            break
    if generic is not None:
        _, left, right, inter = generic
        cases.append({
            "category": "generic_hub_pair",
            "shared_entities": [
                {"entity": display[eid], "df": int(df[eid])}
                for eid in sorted(inter, key=lambda eid: -int(df[eid]))
            ],
            "passages": [
                passage_brief(corpus, left, b1_sets[left], display, df),
                passage_brief(corpus, right, b1_sets[right], display, df),
            ],
            "same_b2_entity": bool(b2_sets[left] & b2_sets[right]),
        })

    # B1 connects through a rare NER entity; the two chunks share no triple-entity.
    b1_only = None
    for eid, docs in enumerate(lists):
        degree = int(df[eid])
        if degree < 2 or degree > 15 or len(keys[eid]) < 8:
            continue
        sample = docs[:10]
        for left_i, left in enumerate(sample):
            for right in sample[left_i + 1:]:
                if title_of[left] == title_of[right]:
                    continue
                if b2_sets[left] & b2_sets[right]:
                    continue
                score = math.log((n_docs + 1) / (degree + 1))
                if b1_only is None or score > b1_only[0]:
                    inter = {item for item in (b1_sets[left] & b1_sets[right]) if int(df[item]) <= 50}
                    b1_only = (score, left, right, eid, inter)
    if b1_only is not None:
        _, left, right, best, inter = b1_only
        cases.append({
            "category": "ner_link_without_triple_entity_link",
            "bridge": {"entity": display[best], "df": int(df[best]), "key": keys[best]},
            "shared_ner_entities": [
                {"entity": display[eid], "df": int(df[eid]), "in_b2_left": eid in b2_sets[left], "in_b2_right": eid in b2_sets[right]}
                for eid in sorted(inter, key=lambda eid: int(df[eid]))[:8]
            ],
            "passages": [
                passage_brief(corpus, left, b1_sets[left], display, df),
                passage_brief(corpus, right, b1_sets[right], display, df),
            ],
        })

    # Two-hop through exact DF=2 rare entities. Endpoints share neither B1 nor B2 entity.
    adj = defaultdict(list)
    for eid, docs in enumerate(lists):
        if int(df[eid]) != 2 or len(keys[eid]) < 8:
            continue
        left, right = docs
        if title_of[left] == title_of[right]:
            continue
        adj[left].append((right, eid))
        adj[right].append((left, eid))
    two_hop = None
    for middle, neigh in adj.items():
        if len(neigh) < 2:
            continue
        limit = neigh[:10]
        for i, (left, e1) in enumerate(limit):
            for right, e2 in limit[i + 1:]:
                if left == right or title_of[left] in {title_of[right], title_of[middle]} or title_of[right] == title_of[middle]:
                    continue
                if b1_sets[left] & b1_sets[right]:
                    continue
                if b2_sets[left] & b2_sets[right]:
                    continue
                two_hop = (left, middle, right, e1, e2)
                break
            if two_hop:
                break
        if two_hop:
            break
    if two_hop is not None:
        left, middle, right, e1, e2 = two_hop
        cases.append({
            "category": "two_hop_rare_entity",
            "bridges": [
                {"entity": display[e1], "df": int(df[e1])},
                {"entity": display[e2], "df": int(df[e2])},
            ],
            "passages": [
                passage_brief(corpus, left, b1_sets[left], display, df),
                passage_brief(corpus, middle, b1_sets[middle], display, df),
                passage_brief(corpus, right, b1_sets[right], display, df),
            ],
            "endpoints_share_b1": False,
            "endpoints_share_b2": False,
        })
    return cases


def looks_like_alias(left: set, right: set, display) -> bool:
    for a in left:
        ta = set(display[a].lower().split())
        if len(ta) < 2:
            continue
        for b in right:
            tb = set(display[b].lower().split())
            if len(tb) < 2:
                continue
            union = ta | tb
            if not union:
                continue
            if len(ta & tb) / len(union) >= 0.5:
                return True
            shorter, longer = (display[a], display[b]) if len(display[a]) < len(display[b]) else (display[b], display[a])
            if len(shorter) >= 8 and shorter.lower() in longer.lower():
                return True
    return False


def gold_diagnostic(corpus, questions_path, labels_by_policy, ranked_idf, b1_sets, b2_sets, df, display, cap: int) -> dict:
    pid_to_doc = {row["passage_id"]: index for index, row in enumerate(corpus)}
    questions = []
    with open(questions_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            gold = []
            seen = set()
            for pid in row.get("gold_passage_ids") or []:
                doc = pid_to_doc.get(pid)
                if doc is None or doc in seen:
                    continue
                seen.add(doc)
                gold.append(doc)
            if len(gold) >= 2:
                questions.append({
                    "question_id": row["question_id"],
                    "hop": row.get("hop"),
                    "split_role": row.get("split_role"),
                    "gold": gold,
                })
    n_docs = len(corpus)

    def same(labels, gold):
        base = int(labels[gold[0]])
        return all(int(labels[doc]) == base for doc in gold[1:])

    def topk_cover(k, gold):
        needed = set(gold)
        for doc in gold:
            members = {doc}
            members.update(item[0] for item in ranked_idf[doc][:k])
            if needed <= members:
                return True
        return False

    def direct_nonhub(gold):
        for i, left in enumerate(gold):
            for right in gold[i + 1:]:
                inter = b1_sets[left] & b1_sets[right]
                if not any(int(df[eid]) <= cap for eid in inter):
                    return False
        return True

    raw_labels = labels_by_policy["raw"]
    reasons = Counter()
    examples = []
    for question in questions:
        gold = question["gold"]
        if same(raw_labels, gold):
            split = [name for name, labels in labels_by_policy.items() if name != "raw" and not same(labels, gold)]
            reasons["same_raw_component" if not split else "same_raw_but_split_by_listed_filter"] += 1
            if split and len(examples) < 6:
                examples.append({
                    "question_id": question["question_id"],
                    "hop": question["hop"],
                    "split_role": question["split_role"],
                    "reason": "same_raw_but_split_by_hub_filter",
                    "filters_that_split": split,
                    "titles": [corpus[doc]["title"] for doc in gold],
                })
            continue
        if any(len(b1_sets[doc]) == 0 for doc in gold):
            reason = "empty_ner_on_a_gold_passage"
        elif any(looks_like_alias(b1_sets[a], b1_sets[b], display) for i, a in enumerate(gold) for b in gold[i + 1:] if not (b1_sets[a] & b1_sets[b])):
            reason = "possible_alias_or_partial_name"
        else:
            reason = "no_shared_ner_component"
        reasons[reason] += 1
        if len([item for item in examples if item["reason"] == reason]) < 3:
            examples.append({
                "question_id": question["question_id"],
                "hop": question["hop"],
                "split_role": question["split_role"],
                "reason": reason,
                "titles": [corpus[doc]["title"] for doc in gold],
                "passage_ids": [corpus[doc]["passage_id"] for doc in gold],
            })

    by_policy = []
    for name, labels in labels_by_policy.items():
        by_policy.append({
            "policy": name,
            "questions_all_gold_in_one_component": int(sum(same(labels, q["gold"]) for q in questions)),
            "n_questions": len(questions),
        })
    topk_rows = []
    for k in TOPKS:
        topk_rows.append({
            "k": k,
            "questions_covered_by_one_gold_members_topk": int(sum(topk_cover(k, q["gold"]) for q in questions)),
            "n_questions": len(questions),
        })
    direct = int(sum(direct_nonhub(q["gold"]) for q in questions))

    # Baselines use the mean cross-group size of IDF top-8, fixed before this comparison is interpreted.
    top8_sizes = []
    for items in ranked_idf:
        top8_sizes.append(1 + min(8, len(items)))
    group_size = max(2, int(round(float(np.mean(top8_sizes)))))
    rng = random.Random(SEED)
    order = list(range(n_docs))
    rng.shuffle(order)
    random_gid = {doc: index // group_size for index, doc in enumerate(order)}
    seq_gid = {index: index // group_size for index in range(n_docs)}

    def colocated(gid, gold):
        base = gid[gold[0]]
        return all(gid[doc] == base for doc in gold[1:])

    role_counts = Counter(q["split_role"] for q in questions)
    hop_counts = Counter(str(q["hop"]) for q in questions)
    return {
        "used_for_parameter_selection": False,
        "n_questions_with_at_least_two_gold_passages": len(questions),
        "role_counts": dict(role_counts),
        "hop_counts": dict(hop_counts),
        "by_policy": by_policy,
        "topk_idf": topk_rows,
        "questions_all_gold_pairs_share_entity_df_le_cap": direct,
        "pair_cap": cap,
        "baseline_group_size": group_size,
        "random_partition_colocated": int(sum(colocated(random_gid, q["gold"]) for q in questions)),
        "sequential_partition_colocated": int(sum(colocated(seq_gid, q["gold"]) for q in questions)),
        "reason_counts_are_not_accuracy": dict(reasons),
        "examples": examples,
        "note": "Same-component rates on the raw graph are structural connectivity, including connectivity caused by generic hubs. They are not a measure of evidence-group correctness.",
    }


def decide(hub_rows: list, grouping_rows: list) -> dict:
    viable = []
    for row in hub_rows:
        median = row["p50_weighted"]
        if (
            row["max_doc_frac"] <= 0.05
            and row["isolated_doc_frac"] <= 0.20
            and median is not None
            and 2 <= median <= 20
        ):
            viable.append(row)
    if viable:
        viable.sort(key=lambda row: (-row["multi_title_doc_frac"], row["max_doc_frac"], row["policy"]))
        hard = {"status": "viable", "selected_policy": viable[0]["policy"], "rule": "max_doc_frac<=0.05 and isolated_doc_frac<=0.20 and 2<=doc_weighted_median<=20; tie by higher multi-title coverage then smaller max component"}
    else:
        hard = {"status": "no_prespecified_policy_meets_rule", "selected_policy": None, "rule": "max_doc_frac<=0.05 and isolated_doc_frac<=0.20 and 2<=doc_weighted_median<=20"}
    idf_rows = [row for row in grouping_rows if row["method"] == "shared_idf" and row["kind"] == "topk"]
    chosen = None
    for row in idf_rows:
        token_p95 = (row["stats"]["token_cross"] or {}).get("p95")
        if token_p95 is not None and token_p95 <= TOKEN_BUDGET and row["stats"]["membership_mean"] <= 4:
            chosen = row
            break
    if chosen:
        topk = {"status": "meets_prespecified_constraint", "k": chosen["param"], "constraint": "smallest k in 2,4,8,16 with cross-group token p95<=16000 and mean membership<=4"}
    else:
        topk = {"status": "fallback_k4_does_not_meet_constraint", "k": 4, "constraint": "smallest k in 2,4,8,16 with cross-group token p95<=16000 and mean membership<=4"}
    return {"hard_component": hard, "topk_shared_idf": topk, "gold_was_not_used": True}


def public_component(info: dict) -> dict:
    keep = {key: value for key, value in info.items() if not key.startswith("_") and key != "labels"}
    return keep


def analyze(name: str, prepared: dict, corpus: list, title_ids: np.ndarray, tokens: np.ndarray, do_gold: bool) -> dict:
    started = time.time()
    n_docs = len(corpus)
    graph = intern_source(prepared["ner"], prepared["triples"])
    n_ent = len(graph["display"])
    b1_lists = doclists_from_sets(graph["b1_sets"], n_ent)
    b2_lists = doclists_from_sets(graph["b2_sets"], n_ent)
    b1_entity = entity_stats(graph["b1_sets"], b1_lists, title_ids)
    b2_entity = entity_stats(graph["b2_sets"], b2_lists, title_ids)
    active_all_b1 = b1_entity["df"] > 0
    active_all_b2 = b2_entity["df"] > 0
    b1_comp = build_components(n_docs, b1_lists, active_all_b1, title_ids)
    b2_comp = build_components(n_docs, b2_lists, active_all_b2, title_ids)

    # Title-level contraction.
    def title_level(lists, df_arr):
        n_titles = int(title_ids.max()) + 1
        uf = UF(n_titles)
        for eid, docs in enumerate(lists):
            if df_arr[eid] < 2:
                continue
            titles = sorted({int(title_ids[doc]) for doc in docs})
            for other in titles[1:]:
                uf.union(titles[0], other)
        labels = np.array([uf.find(i) for i in range(n_titles)], dtype=np.int32)
        sizes = Counter(labels.tolist())
        # weight titles, and also weight by chunks
        title_sizes = np.array(list(sizes.values()), dtype=np.int32)
        chunk_weight = Counter()
        for doc, title in enumerate(title_ids.tolist()):
            chunk_weight[int(labels[title])] += 1
        weighted_chunks = np.array(list(chunk_weight.values()), dtype=np.int32)
        return {
            "n_titles": n_titles,
            "n_components": len(sizes),
            "max_titles": int(title_sizes.max()),
            "max_title_frac": float(title_sizes.max() / n_titles),
            "max_chunks": int(weighted_chunks.max()),
            "max_chunk_frac": float(weighted_chunks.max() / n_docs),
            "title_size_unweighted": pcts(title_sizes),
            "chunk_weighted": pcts(np.repeat(weighted_chunks, weighted_chunks)),
        }

    policies = [("raw", None)]
    for cut in DF_CUTS:
        policies.append((f"df_gt_{cut}", float(cut)))
    for label, frac in DF_FRACTIONS:
        policies.append((f"df_gt_{label}", frac * n_docs))

    hub_rows = []
    labels_by_policy = {"raw": b1_comp["labels"]}
    policy_components = {}
    for policy_name, threshold in policies:
        if policy_name == "raw":
            info = b1_comp
        else:
            active = policy_active(b1_entity["df"], threshold)
            info = build_components(n_docs, b1_lists, active, title_ids)
            labels_by_policy[policy_name] = info["labels"]
        policy_components[policy_name] = info
        row = summarize_policy(policy_name, threshold, {"info": info, "doclists": b1_lists}, b1_entity["df"], graph["b1_sets"], n_docs)
        hub_rows.append(row)

    cap_info = choose_pair_cap(b1_entity["df"])
    pairs, pair_meta = build_pairs(b1_lists, b1_entity["df"], cap_info["cap"], n_docs)
    mass = doc_idf_mass(b1_lists, b1_entity["df"], cap_info["cap"], n_docs)
    neighbors = neighbor_lists(pairs, mass, n_docs)

    grouping_rows = []
    for method, ranked in neighbors.items():
        for k in TOPKS:
            stats = summarize_topk(ranked, k, tokens)
            grouping_rows.append({"method": method, "kind": "topk", "param": k, "stats": stats})

    def jaccard_of(key, idf_sum):
        left, right = divmod(key, n_docs)
        denom = mass[left] + mass[right] - idf_sum
        return idf_sum / denom if denom > 0 else 0.0

    for threshold in SHARED_THRESHOLDS:
        labels = hard_labels_from_pairs(n_docs, pairs, lambda count, idf_sum, key, n: count >= threshold)
        grouping_rows.append({
            "method": "shared_count",
            "kind": "hard",
            "param": threshold,
            "pair_cap": cap_info["cap"],
            "stats": summarize_hard(labels, tokens, n_docs),
        })
    for threshold in IDF_THRESHOLDS:
        labels = hard_labels_from_pairs(n_docs, pairs, lambda count, idf_sum, key, n, t=threshold: idf_sum >= t)
        grouping_rows.append({
            "method": "shared_idf",
            "kind": "hard",
            "param": threshold,
            "pair_cap": cap_info["cap"],
            "stats": summarize_hard(labels, tokens, n_docs),
        })
    for threshold in JACCARD_THRESHOLDS:
        labels = hard_labels_from_pairs(
            n_docs, pairs, lambda count, idf_sum, key, n, t=threshold: jaccard_of(key, idf_sum) >= t
        )
        grouping_rows.append({
            "method": "weighted_jaccard",
            "kind": "hard",
            "param": threshold,
            "pair_cap": cap_info["cap"],
            "stats": summarize_hard(labels, tokens, n_docs),
        })

    decision = decide(hub_rows, grouping_rows)
    cases = select_cases(
        corpus, title_ids, graph["b1_sets"], graph["b2_sets"], b1_lists, b1_entity["df"],
        graph["display"], graph["keys"], pairs, n_docs, b1_comp["_rows"],
    )
    gold = None
    if do_gold:
        gold = gold_diagnostic(
            corpus, QUESTIONS_PATH, labels_by_policy, neighbors["shared_idf"],
            graph["b1_sets"], graph["b2_sets"], b1_entity["df"], graph["display"], cap_info["cap"],
        )

    # B1 vs B2 entity set comparison on the aligned vocab.
    b1_df = b1_entity["df"]
    b2_df = b2_entity["df"]
    ner_only = int(((b1_df > 0) & (b2_df == 0)).sum())
    triple_only = int(((b2_df > 0) & (b1_df == 0)).sum())
    both = int(((b1_df > 0) & (b2_df > 0)).sum())
    ner_multi_not_triple = int(((b1_df > 1) & (b2_df <= 1)).sum())

    b1_labels = b1_comp["labels"]
    b2_labels = b2_comp["labels"]
    split = 0
    b1_groups = defaultdict(set)
    for doc in range(n_docs):
        b1_groups[int(b1_labels[doc])].add(int(b2_labels[doc]))
    split = sum(1 for value in b1_groups.values() if len(value) > 1)
    multi_b1 = sum(1 for item in b1_comp["_rows"] if item[0] > 1)
    multi_split = 0
    for item in b1_comp["_rows"]:
        if item[0] <= 1:
            continue
        labels = {int(b2_labels[doc]) for doc in item[4]}
        if len(labels) > 1:
            multi_split += 1

    top_b1 = top_entities(b1_df, b1_entity["title_df"], b2_df, graph["display"], graph["variants"], graph["keys"], 50)
    top_b2 = top_entities(b2_df, b2_entity["title_df"], b1_df, graph["display"], graph["variants"], graph["keys"], 50)

    result = {
        "source": name,
        "seconds": time.time() - started,
        "n_chunks": n_docs,
        "normalization": {
            "raw_strings": graph["raw_strings"],
            "strings_whose_audit_key_differs_from_hipporag_text_processing": graph["key_changed_strings"],
            "invalid_or_empty_endpoint_triples": graph["invalid_triples"],
            "mention_edges": graph["mention_edges"],
            "mention_sum": graph["mention_sum"],
            "edges_with_mention_count_gt1": graph["mention_multi"],
        },
        "b1": {
            "entities": {key: value for key, value in b1_entity.items() if key not in {"df", "title_df", "per_doc"}},
            "df_histogram": df_histogram(b1_df),
            "components": public_component(b1_comp),
            "top50": top_b1,
        },
        "b2": {
            "entities": {key: value for key, value in b2_entity.items() if key not in {"df", "title_df", "per_doc"}},
            "df_histogram": df_histogram(b2_df),
            "components": public_component(b2_comp),
            "top50": top_b2,
        },
        "comparison": {
            "ner_only_entities": ner_only,
            "triple_only_entities": triple_only,
            "in_both": both,
            "ner_df_gt1_and_triple_df_le1": ner_multi_not_triple,
            "passage_entity_edges_ner_not_triple": graph["ner_not_triple_edges"],
            "passage_entity_edges_triple_not_ner": graph["triple_not_ner_edges"],
            "b1_components_split_by_b2": split,
            "b1_multidoc_components": multi_b1,
            "b1_multidoc_components_split_by_b2": multi_split,
        },
        "title_level": {
            "b1": title_level(b1_lists, b1_df),
            "b2": title_level(b2_lists, b2_df),
        },
        "hub_filters_b1": hub_rows,
        "pair_index": {**cap_info, **pair_meta},
        "grouping": grouping_rows,
        "decision_before_gold": decision,
        "cases": cases,
        "gold_posthoc": gold,
        "formulas": {
            "idf": "ln((N+1)/(df+1))",
            "shared_idf_sum": "sum of idf over shared entities whose df<=pair_cap",
            "weighted_jaccard": "shared_idf_sum / (idf(A)+idf(B)-shared_idf_sum), A and B exclude entities with df>pair_cap",
            "pair_cap_meaning": "Entities above the cap are omitted from stored candidate pairs and from IDF/Jaccard scores. They remain in the union-find hub-filter experiments.",
        },
    }
    arrays = {
        "b1_labels": b1_comp["labels"],
        "b2_labels": b2_comp["labels"],
        "b1_sizes_by_label": np.array([item[0] for item in b1_comp["_rows"]], dtype=np.int32),
        "b2_sizes_by_label": np.array([item[0] for item in b2_comp["_rows"]], dtype=np.int32),
        "b1_df": b1_df,
        "b2_df": b2_df,
        "b1_per_doc": b1_entity["per_doc"],
        "b2_per_doc": b2_entity["per_doc"],
        "tokens": tokens,
    }
    # passage table inputs
    result["_passage_rows"] = []
    return result, arrays, graph, policy_components


def write_csv(path: Path, rows: list, fieldnames: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_pct(prefix: str, block: dict) -> dict:
    if not block:
        return {}
    return {f"{prefix}_{key}": block.get(key) for key in ("n", "mean", "median", "min", "max", "p75", "p90", "p95", "p99")}


def save_tables(source: str, result: dict, arrays: dict, corpus: list) -> None:
    table_dir = OUT / "tables"
    write_csv(
        table_dir / f"{source}_component_bins.csv",
        [{"graph": graph, **row} for graph in ("b1", "b2") for row in result[graph]["components"]["bins"]],
        ["graph", "bin", "n_components", "n_docs", "doc_frac"],
    )
    write_csv(
        table_dir / f"{source}_component_rank.csv",
        [{"graph": "b1", **row} for row in result["b1"]["components"]["ranked_top200"]]
        + [{"graph": "b2", **row} for row in result["b2"]["components"]["ranked_top200"]],
        ["graph", "rank", "n_docs", "n_entities", "n_titles", "label"],
    )
    write_csv(
        table_dir / f"{source}_top_entities.csv",
        [{"graph": "b1", **{k: v for k, v in row.items() if k != "surfaces"}, "surfaces": " | ".join(row["surfaces"])} for row in result["b1"]["top50"]]
        + [{"graph": "b2", **{k: v for k, v in row.items() if k != "surfaces"}, "surfaces": " | ".join(row["surfaces"])} for row in result["b2"]["top50"]],
        ["graph", "rank", "entity", "key", "df", "title_df", "other_graph_df", "n_surface_variants_capped8", "surfaces"],
    )
    write_csv(table_dir / f"{source}_hub_filters.csv", result["hub_filters_b1"], list(result["hub_filters_b1"][0].keys()))
    grouping_flat = []
    for row in result["grouping"]:
        flat = {
            "method": row["method"],
            "kind": row["kind"],
            "param": row["param"],
            "pair_cap": row.get("pair_cap", result["pair_index"]["cap"]),
        }
        stats = row["stats"]
        for key in (
            "n_cross_groups", "n_singletons", "frac_cross_groups_le_budget", "frac_cross_groups_gt_budget",
            "frac_cross_groups_gt_500_docs", "max_doc_frac", "coverage_docs_frac", "membership_mean",
            "membership_p95", "membership_max", "total_cross_group_tokens", "unique_covered_doc_tokens",
            "repeat_factor", "cross_tokens_over_corpus",
        ):
            flat[key] = stats.get(key)
        flat.update(flatten_pct("size", stats.get("size_cross")))
        flat.update(flatten_pct("tokens", stats.get("token_cross")))
        grouping_flat.append(flat)
    write_csv(table_dir / f"{source}_grouping.csv", grouping_flat, list(grouping_flat[0].keys()))
    b1_size = arrays["b1_sizes_by_label"]
    b2_size = arrays["b2_sizes_by_label"]
    passage_rows = []
    for index, row in enumerate(corpus):
        b1_label = int(arrays["b1_labels"][index])
        b2_label = int(arrays["b2_labels"][index])
        passage_rows.append({
            "passage_id": row["passage_id"],
            "corpus_index": row["corpus_index"],
            "title": row["title"],
            "n_ner_entities": int(arrays["b1_per_doc"][index]),
            "n_triple_entities": int(arrays["b2_per_doc"][index]),
            "b1_component_rank": b1_label,
            "b1_component_docs": int(b1_size[b1_label]),
            "b2_component_rank": b2_label,
            "b2_component_docs": int(b2_size[b2_label]),
        })
    write_csv(
        table_dir / f"{source}_passage_components.csv",
        passage_rows,
        ["passage_id", "corpus_index", "title", "n_ner_entities", "n_triple_entities",
         "b1_component_rank", "b1_component_docs", "b2_component_rank", "b2_component_docs"],
    )
    if result["gold_posthoc"]:
        write_csv(
            table_dir / f"{source}_gold_diagnostic.csv",
            result["gold_posthoc"]["by_policy"] + [
                {"policy": f"topk_idf_{row['k']}", "questions_all_gold_in_one_component": row["questions_covered_by_one_gold_members_topk"], "n_questions": row["n_questions"]}
                for row in result["gold_posthoc"]["topk_idf"]
            ],
            ["policy", "questions_all_gold_in_one_component", "n_questions"],
        )


def save_plots(source: str, result: dict, arrays: dict) -> None:
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    b1_sizes = arrays["b1_sizes_by_label"]
    b2_sizes = arrays["b2_sizes_by_label"]
    # Each component once, not once per doc. labels are ranks 0..C-1 and sizes align.
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels = [row["bin"] for row in result["b1"]["components"]["bins"]]
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, [row["n_components"] for row in result["b1"]["components"]["bins"]], width=w, label="B1 NER")
    ax.bar(x + w / 2, [row["n_components"] for row in result["b2"]["components"]["bins"]], width=w, label="B2 triple entity")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Components")
    ax.set_xlabel("Documents (chunks) in component")
    ax.set_yscale("log")
    ax.set_title(f"{source}: component-size histogram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{source}_component_hist.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for sizes, label in ((b1_sizes, "B1 NER"), (b2_sizes, "B2 triple entity")):
        ordered = np.sort(sizes)[::-1]
        ax.plot(np.arange(1, ordered.size + 1), ordered, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Component rank")
    ax.set_ylabel("Chunks in component")
    ax.set_title(f"{source}: rank–size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{source}_rank_size.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for df, label in ((arrays["b1_df"], "B1 NER"), (arrays["b2_df"], "B2 triple entity")):
        present = np.sort(df[df > 0])[::-1]
        ax.plot(np.arange(1, present.size + 1), present, label=label, linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Entity rank")
    ax.set_ylabel("Document frequency (chunks)")
    ax.set_title(f"{source}: entity document frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{source}_entity_df.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for sizes, label in ((b1_sizes, "B1 NER"), (b2_sizes, "B2 triple entity")):
        ordered = np.sort(sizes)[::-1]
        cum = np.cumsum(ordered) / ordered.sum()
        ax.plot(np.arange(1, ordered.size + 1), cum, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Component rank")
    ax.set_ylabel("Cumulative chunk coverage")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{source}: cumulative chunk coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{source}_cumulative_coverage.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    names = [row["policy"] for row in result["hub_filters_b1"]]
    x = np.arange(len(names))
    ax.plot(x, [row["max_doc_frac"] for row in result["hub_filters_b1"]], marker="o", label="Largest component fraction")
    ax.plot(x, [row["isolated_doc_frac"] for row in result["hub_filters_b1"]], marker="o", label="Isolated chunk fraction")
    ax.plot(x, [row["multi_title_doc_frac"] for row in result["hub_filters_b1"]], marker="o", label="Chunks in multi-title components")
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of chunks")
    ax.set_title(f"{source}: B1 hub-entity filters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{source}_hub_filters.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    idf_rows = [row for row in result["grouping"] if row["method"] == "shared_idf" and row["kind"] == "topk"]
    count_rows = [row for row in result["grouping"] if row["method"] == "shared_count" and row["kind"] == "topk"]
    ks = [row["param"] for row in idf_rows]
    ax.plot(ks, [row["stats"]["token_cross"]["p95"] for row in idf_rows], marker="o", label="IDF sum, token P95")
    ax.plot(ks, [row["stats"]["token_cross"]["p95"] for row in count_rows], marker="o", label="Shared count, token P95")
    ax.axhline(TOKEN_BUDGET, color="black", linewidth=1, linestyle="--", label="16,000-token budget")
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Cross-group input tokens, P95")
    ax.set_title(f"{source}: overlapping evidence-group cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{source}_grouping_tokens.png", dpi=140)
    plt.close(fig)


def draw_cases(source: str, cases: list) -> None:
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    drawn = 0
    for case in cases:
        if case["category"] not in {"small_component", "medium_component", "generic_hub_pair", "two_hop_rare_entity"}:
            continue
        if drawn >= 4:
            break
        passages = case["passages"]
        if case["category"] == "medium_component" and len(passages) > 8:
            continue
        bridges = []
        if "bridge" in case:
            bridges.append(case["bridge"]["entity"])
        bridges.extend(item["entity"] for item in case.get("shared_entities", [])[:6])
        bridges.extend(item["entity"] for item in case.get("bridges", []))
        # unique preserve
        seen = set()
        bridge_names = []
        for name in bridges:
            if name not in seen:
                seen.add(name)
                bridge_names.append(name)
        bridge_names = bridge_names[:8]
        if not bridge_names:
            continue
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        left_y = np.linspace(0.9, 0.1, len(passages))
        right_y = np.linspace(0.9, 0.1, len(bridge_names))
        for y, passage in zip(left_y, passages):
            ax.scatter([0.1], [y], s=80, c="#1f4e79")
            ax.text(0.12, y, passage["title"][:42], va="center", fontsize=8)
        for y, name in zip(right_y, bridge_names):
            ax.scatter([0.78], [y], s=70, c="#b85c38")
            ax.text(0.80, y, name[:42], va="center", fontsize=8)
        for p_i, passage in enumerate(passages):
            names = {item["entity"] for item in passage["top_entities"]}
            for e_i, name in enumerate(bridge_names):
                if name in names:
                    ax.plot([0.1, 0.78], [left_y[p_i], right_y[e_i]], color="#9aa4b2", linewidth=0.8)
        ax.set_axis_off()
        ax.set_title(f"{source}: {case['category']}")
        fig.tight_layout()
        outfile = plot_dir / f"{source}_case_{drawn}_{case['category']}.png"
        fig.savefig(outfile, dpi=140)
        plt.close(fig)
        case["figure"] = str(outfile.relative_to(OUT))
        drawn += 1


def json_dump(path: Path, payload) -> None:
    def convert(obj):
        if isinstance(obj, dict):
            return {str(key): convert(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(value) for value in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "artifacts").mkdir(exist_ok=True)
    print("loading corpus", flush=True)
    corpus = load_corpus()
    titles = [row["title"] for row in corpus]
    title_vocab = {}
    title_ids = np.empty(len(corpus), dtype=np.int32)
    for index, title in enumerate(titles):
        eid = title_vocab.get(title)
        if eid is None:
            eid = len(title_vocab)
            title_vocab[title] = eid
        title_ids[index] = eid
    title_counts = Counter(titles)
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = np.array([len(encoder.encode(row["content"])) for row in corpus], dtype=np.int32)
    chunks_per_title = np.array(list(title_counts.values()), dtype=np.int32)
    source_meta = {
        "corpus_path": str(CORPUS_PATH),
        "n_corpus_rows": len(corpus),
        "n_unique_chunk_ids": len({row["passage_id"] for row in corpus}),
        "n_unique_titles": len(title_vocab),
        "chunks_per_title": pcts(chunks_per_title),
        "titles_with_multiple_chunks": int((chunks_per_title > 1).sum()),
        "chunks_in_multi_chunk_titles": int(sum(count for count in chunks_per_title if count > 1)),
        "token_estimate": "tiktoken cl100k_base on title+newline+text",
        "corpus_tokens": int(tokens.sum()),
        "token_percentiles": pcts(tokens),
        "document_id_status": "Chunk IDs are md5-style hashes of title and text. They do not encode a source document id. The stored title is the MuSiQue/HippoRAG Wikipedia passage title and can be shared by several chunks. All primary graphs use chunks. Title contraction is reported separately and is not treated as an official document id.",
    }
    print("loading luna", flush=True)
    luna = load_luna(corpus)
    print("loading official", flush=True)
    official = load_official(corpus)
    official_stats = json.loads(OFFICIAL_STATS.read_text(encoding="utf-8"))
    print("analyzing luna", flush=True)
    luna_result, luna_arrays, _, _ = analyze("luna_g_raw", luna, corpus, title_ids, tokens, do_gold=True)
    luna_result["load"] = {
        "openie_present": luna["present"],
        "content_mismatches": luna["content_mismatches"],
        "models": luna["models"],
        "failed_files": len(luna["failed_files"]),
    }
    print("analyzing official", flush=True)
    official_result, official_arrays, _, _ = analyze("hipporag2_gpt4o_mini", official, corpus, title_ids, tokens, do_gold=True)
    official_result["load"] = {
        "matched": official["matched"],
        "missing": official["missing"],
        "extra": official["extra"],
        "content_mismatches": official["content_mismatches"],
        "n_docs_in_file": official["n_docs_in_file"],
        "cited_graph_stats": official_stats["graph"],
    }
    for source, result, arrays in (
        ("luna_g_raw", luna_result, luna_arrays),
        ("hipporag2_gpt4o_mini", official_result, official_arrays),
    ):
        print("saving", source, flush=True)
        draw_cases(source, result["cases"])
        save_tables(source, result, arrays, corpus)
        save_plots(source, result, arrays)
        json_dump(OUT / "artifacts" / f"{source}_summary.json", result)
        np.savez_compressed(
            OUT / "artifacts" / f"{source}_arrays.npz",
            b1_sizes=arrays["b1_sizes_by_label"],
            b2_sizes=arrays["b2_sizes_by_label"],
            b1_df=arrays["b1_df"],
            b2_df=arrays["b2_df"],
        )
    json_dump(OUT / "artifacts" / "source_meta.json", source_meta)
    print("done", flush=True)


if __name__ == "__main__":
    main()
