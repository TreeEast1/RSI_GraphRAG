#!/usr/bin/env python3
"""Zero-API-cost cross-document grouping on the frozen MuSiQue corpus.

Hyperparameters live in configs/grid.json and are applied before any gold
supporting passage is read. Gold metrics are attached afterwards.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

AUDIT = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/experiments/entity_doc_bipartite_audit/scripts")
sys.path.insert(0, str(AUDIT))
import run_audit as ra  # noqa: E402

ROOT = Path("/data2-HDD-SATA-20T/nzq/TAG-experiment/experiments/musique_zero_api_grouping")
GRID = json.loads((ROOT / "configs" / "grid.json").read_text(encoding="utf-8"))
SEED = int(GRID["seed"])
TEMPLATE = int(GRID["template_tokens_per_group"])


def log(message: str) -> None:
    print(message, flush=True)


def l2norm(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def labels_to_groups(labels: np.ndarray) -> list:
    buckets = defaultdict(list)
    for doc_id, label in enumerate(labels.tolist()):
        buckets[int(label)].append(doc_id)
    return list(buckets.values())


def exact_knn(matrix: np.ndarray, k: int) -> tuple:
    count = matrix.shape[0]
    neighbors = np.empty((count, k), dtype=np.int32)
    similarity = np.empty((count, k), dtype=np.float32)
    for start in range(0, count, 256):
        stop = min(count, start + 256)
        scores = matrix[start:stop] @ matrix.T
        for offset, global_id in enumerate(range(start, stop)):
            row = scores[offset]
            row[global_id] = -2.0
            chosen = np.argpartition(row, -k)[-k:]
            order = chosen[np.argsort(row[chosen])[::-1]]
            neighbors[global_id] = order
            similarity[global_id] = np.clip(row[order], 0.0, 1.0)
        if start % 2048 == 0:
            log(f"knn {start}/{count}")
    return neighbors, similarity


def knn_edges(neighbors: np.ndarray, similarity: np.ndarray, k: int) -> list:
    best = {}
    for left, row in enumerate(neighbors[:, :k]):
        for slot, right in enumerate(row.tolist()):
            if left == right:
                continue
            pair = (left, right) if left < right else (right, left)
            weight = float(similarity[left, slot])
            if pair not in best or weight > best[pair]:
                best[pair] = weight
    return [(a, b, w) for (a, b), w in best.items()]


def mutual_edges(neighbors: np.ndarray, similarity: np.ndarray, k: int) -> list:
    member = [set(row[:k].tolist()) for row in neighbors]
    best = {}
    for left, row in enumerate(neighbors[:, :k]):
        for slot, right in enumerate(row.tolist()):
            right = int(right)
            if right <= left or left not in member[right]:
                continue
            weight = float(min(similarity[left, slot], similarity[right, list(neighbors[right, :k]).index(left)]))
            best[(left, right)] = max(best.get((left, right), 0.0), weight)
    return [(a, b, w) for (a, b), w in best.items()]


def graph_from_edges(count: int, edges: list):
    import igraph

    graph = igraph.Graph(n=count, directed=False)
    if not edges:
        return graph
    graph.add_edges([(a, b) for a, b, _weight in edges])
    graph.es["weight"] = [weight for _a, _b, weight in edges]
    return graph


def leiden_labels(count: int, edges: list, resolution: float) -> np.ndarray:
    import igraph

    try:
        igraph.random.seed(SEED)
    except Exception:
        pass
    graph = graph_from_edges(count, edges)
    if graph.ecount() == 0:
        return np.arange(count, dtype=np.int32)
    part = graph.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution=resolution,
        n_iterations=2,
    )
    return np.asarray(part.membership, dtype=np.int32)


def other_communities(count: int, edges: list) -> dict:
    import igraph

    graph = graph_from_edges(count, edges)
    out = {}
    if graph.ecount() == 0:
        labels = np.arange(count, dtype=np.int32)
        return {"louvain": labels, "lpa": labels, "infomap": labels}
    try:
        out["louvain"] = np.asarray(
            graph.community_multilevel(weights="weight", resolution=1).membership, dtype=np.int32
        )
    except Exception as exc:  # noqa: BLE001
        out["louvain_error"] = str(exc)
    try:
        out["lpa"] = np.asarray(graph.community_label_propagation(weights="weight").membership, dtype=np.int32)
    except Exception as exc:  # noqa: BLE001
        out["lpa_error"] = str(exc)
    try:
        out["infomap"] = np.asarray(graph.community_infomap(edge_weights="weight", trials=1).membership, dtype=np.int32)
    except Exception as exc:  # noqa: BLE001
        out["infomap_error"] = str(exc)
    return out


class CapUF:
    def __init__(self, tokens: np.ndarray):
        count = len(tokens)
        self.parent = list(range(count))
        self.size = [1] * count
        self.tokens = [int(value) for value in tokens.tolist()]

    def find(self, node: int) -> int:
        parent = self.parent
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def try_union(self, left: int, right: int, max_size: int, max_tokens: int) -> bool:
        ra_id, rb_id = self.find(left), self.find(right)
        if ra_id == rb_id:
            return False
        if self.size[ra_id] + self.size[rb_id] > max_size:
            return False
        if self.tokens[ra_id] + self.tokens[rb_id] > max_tokens:
            return False
        if self.size[ra_id] < self.size[rb_id]:
            ra_id, rb_id = rb_id, ra_id
        self.parent[rb_id] = ra_id
        self.size[ra_id] += self.size[rb_id]
        self.tokens[ra_id] += self.tokens[rb_id]
        return True


def greedy_groups(count: int, edges: list, tokens: np.ndarray, max_size: int, max_tokens: int) -> list:
    uf = CapUF(tokens)
    for left, right, _weight in sorted(edges, key=lambda item: item[2], reverse=True):
        uf.try_union(left, right, max_size, max_tokens)
    labels = np.fromiter((uf.find(i) for i in range(count)), dtype=np.int32, count=count)
    return labels_to_groups(labels)


def recsplit(labels: np.ndarray, matrix: np.ndarray, tokens: np.ndarray, max_size: int, max_tokens: int) -> np.ndarray:
    labels = labels.copy()
    next_id = int(labels.max()) + 1 if len(labels) else 0
    for _pass in range(12):
        changed = False
        for lab in np.unique(labels):
            members = np.where(labels == lab)[0]
            if len(members) <= max_size and int(tokens[members].sum()) <= max_tokens:
                continue
            if len(members) < 4:
                continue
            model = MiniBatchKMeans(
                n_clusters=2,
                random_state=SEED,
                batch_size=min(1024, len(members)),
                n_init=1,
                max_iter=30,
            )
            pred = model.fit_predict(matrix[members])
            if len(set(pred.tolist())) < 2:
                continue
            labels[members[pred == 1]] = next_id
            next_id += 1
            changed = True
        if not changed:
            break
    return labels


def overlap_groups(groups: list, edges: list, tokens: np.ndarray, corpus_tokens: int, multiplier_cap: float, max_copies: int) -> dict:
    count = len(tokens)
    members = [set(group) for group in groups]
    owner = np.empty(count, dtype=np.int32)
    for gid, group in enumerate(groups):
        for doc in group:
            owner[doc] = gid
    copies = np.ones(count, dtype=np.int32)
    current = int(sum(int(tokens[doc].sum()) if False else tokens[np.fromiter(group, dtype=np.int32)].sum() for group in groups))
    considered = 0
    added = 0
    stopped = 0
    for left, right, _weight in sorted(edges, key=lambda item: item[2], reverse=True):
        considered += 1
        left_g, right_g = int(owner[left]), int(owner[right])
        if left_g == right_g:
            continue
        progressed = False
        for src, dest in ((left, right_g), (right, left_g)):
            if copies[src] >= max_copies or src in members[dest]:
                continue
            extra = int(tokens[src])
            if (current + extra) / corpus_tokens > multiplier_cap:
                stopped += 1
                continue
            if len(members[dest]) >= int(GRID["overlap_chunk_cap"]):
                stopped += 1
                continue
            dest_tokens = int(tokens[list(members[dest])].sum()) + extra
            if dest_tokens > int(GRID["group_token_cap_for_overlap"]):
                stopped += 1
                continue
            members[dest].add(src)
            copies[src] += 1
            current += extra
            added += 1
            progressed = True
        if not progressed and (current / corpus_tokens) >= multiplier_cap:
            break
    return {
        "groups": [sorted(group) for group in members],
        "edges_considered": considered,
        "copies_added": added,
        "add_attempts_blocked": stopped,
    }


def find_sentence(text: str, surface: str) -> str | None:
    if not surface or len(surface) < 3:
        return None
    idx = text.lower().find(surface.lower())
    if idx < 0:
        return None
    lefts = [text.rfind(mark, 0, idx) for mark in ".?!"]
    left = max(lefts)
    rights = [text.find(mark, idx) for mark in ".?!"]
    rights = [pos for pos in rights if pos >= 0]
    right = min(rights) + 1 if rights else min(len(text), idx + 240)
    snippet = re.sub(r"\s+", " ", text[left + 1 : right]).strip()
    if len(snippet) < 20:
        return None
    return snippet[:500]


def title_tokens(title: str) -> set:
    return set(re.findall(r"[a-z0-9]+", title.lower()))


def title_score(left: set, right: set, same: bool) -> float:
    if same:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def summarize(name: str, family: str, params: dict, groups: list, tokens: np.ndarray, knn1: np.ndarray, matrix: np.ndarray, title_ids: np.ndarray, seconds: float) -> dict:
    count = len(tokens)
    corpus_tokens = int(tokens.sum())
    sizes = np.array([len(group) for group in groups], dtype=np.int32)
    group_tokens = np.array([int(tokens[group].sum()) if group else 0 for group in groups], dtype=np.int64)
    membership = np.zeros(count, dtype=np.int32)
    covered = np.zeros(count, dtype=bool)
    for group in groups:
        for doc in group:
            membership[doc] += 1
            covered[doc] = True
    passage_tokens = int(group_tokens.sum())
    template_tokens = TEMPLATE * len(groups)
    multi = sizes >= 2
    rng = random.Random(SEED)
    same_title = 0
    cross_title = 0
    multi_groups = 0
    pure_title_groups = 0
    sampled_pairs = 0
    for group in groups:
        if len(group) < 2:
            continue
        multi_groups += 1
        titles = {int(title_ids[doc]) for doc in group}
        if len(titles) == 1:
            pure_title_groups += 1
        pair_indexes = range(len(group))
        pairs = []
        if len(group) <= 12:
            for i, left in enumerate(group):
                for right in group[i + 1 :]:
                    pairs.append((left, right))
        else:
            for _ in range(30):
                left, right = rng.sample(group, 2)
                pairs.append((left, right))
        for left, right in pairs:
            sampled_pairs += 1
            if title_ids[left] == title_ids[right]:
                same_title += 1
            else:
                cross_title += 1
    retained = 0
    for doc, neighbor in enumerate(knn1.tolist()):
        if any(neighbor in group and doc in group for group in _groups_of(doc, groups, membership)):
            retained += 1
    # The helper above is too slow if called naively. Replaced below if needed.
    return {
        "name": name,
        "family": family,
        "params": params,
        "seconds": seconds,
        "n_groups": len(groups),
        "size": ra.pcts(sizes),
        "group_tokens": ra.pcts(group_tokens),
        "singleton_group_frac": float((sizes == 1).mean()) if len(sizes) else None,
        "chunk_coverage": float(covered.mean()),
        "membership_mean": float(membership.mean()),
        "membership_p95": float(np.percentile(membership, 95)),
        "membership_max": int(membership.max()),
        "passage_read_multiplier": passage_tokens / corpus_tokens,
        "template_tokens": template_tokens,
        "total_read_multiplier": (passage_tokens + template_tokens) / corpus_tokens,
        "fraction_chunks_in_size_ge2": float((membership_ge2(groups, count)).mean()),
        "multi_group_pure_title_frac": (pure_title_groups / multi_groups) if multi_groups else None,
        "sampled_pair_same_title_frac": (same_title / sampled_pairs) if sampled_pairs else None,
        "neighbor_retention": None,
        "mean_intra_cosine": intra_cosine(groups, matrix, rng),
        "oversized_groups": int(((sizes > 16) | (group_tokens > 16000)).sum()),
    }


def membership_ge2(groups: list, count: int) -> np.ndarray:
    flags = np.zeros(count, dtype=bool)
    for group in groups:
        if len(group) < 2:
            continue
        for doc in group:
            flags[doc] = True
    return flags


def _groups_of(doc: int, groups: list, membership: np.ndarray):
    # placeholder, not used after rewrite
    return []


def intra_cosine(groups: list, matrix: np.ndarray, rng: random.Random) -> float | None:
    values = []
    for group in groups:
        if len(group) < 2:
            continue
        if len(group) > 24:
            picks = rng.sample(group, 8)
        else:
            picks = group
        block = matrix[picks]
        sims = block @ block.T
        iu = np.triu_indices(len(picks), k=1)
        if len(iu[0]) == 0:
            continue
        values.append(float(sims[iu].mean()))
        if len(values) >= 400:
            break
    if not values:
        return None
    return float(np.mean(values))


def neighbor_retention(groups: list, knn1: np.ndarray) -> float:
    count = len(knn1)
    doc_groups = [set() for _ in range(count)]
    for gid, group in enumerate(groups):
        for doc in group:
            doc_groups[doc].add(gid)
    hit = 0
    for doc, neighbor in enumerate(knn1.tolist()):
        if doc_groups[doc] & doc_groups[int(neighbor)]:
            hit += 1
    return hit / count


def attach_retention(summary: dict, groups: list, knn1: np.ndarray) -> None:
    summary["neighbor_retention"] = neighbor_retention(groups, knn1)
    summary.pop("_slow", None)


def gold_scores(groups: list, questions: list, title_ids: np.ndarray) -> dict:
    count_docs = 1 + max(doc for group in groups for doc in group)
    doc_groups = [set() for _ in range(count_docs)]
    for gid, group in enumerate(groups):
        for doc in group:
            doc_groups[doc].add(gid)
    full = 0
    pair_hit = 0
    pair_total = 0
    cross_hit = 0
    cross_total = 0
    partial = []
    purity = []
    for gold in questions:
        sets = [doc_groups[doc] for doc in gold]
        common = set.intersection(*sets) if sets else set()
        if common:
            full += 1
        counts = Counter()
        for bucket in sets:
            for gid in bucket:
                counts[gid] += 1
        best = max(counts.values()) if counts else 1
        partial.append(best / len(gold))
        if counts:
            gid = max(counts, key=lambda key: (counts[key], -key))
            group_size = sum(1 for doc in gold if False)
            # recover group size from inverse: number of docs is not stored.
            # purity uses gold-only best group size via a parallel list.
        for i, left in enumerate(gold):
            for right in gold[i + 1 :]:
                pair_total += 1
                together = bool(doc_groups[left] & doc_groups[right])
                pair_hit += int(together)
                if title_ids[left] != title_ids[right]:
                    cross_total += 1
                    cross_hit += int(together)
    # purity: for the group with most gold members, fraction of that group that is not in this question's gold
    group_sets = [set(group) for group in groups]
    for gold in questions:
        counts = Counter()
        for doc in gold:
            for gid in doc_groups[doc]:
                counts[gid] += 1
        if not counts:
            continue
        gid = max(counts, key=lambda key: (counts[key], -key))
        group = group_sets[gid]
        purity.append(1.0 - (len(set(gold) & group) / len(group)))
    return {
        "n_questions": len(questions),
        "full_gold_colocation": full,
        "mean_partial_gold_coverage": float(np.mean(partial)) if partial else None,
        "pairwise_gold_colocation": pair_hit / pair_total if pair_total else None,
        "cross_title_pairwise_gold_colocation": cross_hit / cross_total if cross_total else None,
        "cross_title_gold_pairs": cross_total,
        "mean_nongold_fraction_in_best_group": float(np.mean(purity)) if purity else None,
    }


def load_questions(corpus: list) -> list:
    index = {row["passage_id"]: i for i, row in enumerate(corpus)}
    questions = []
    with open(GRID["questions"], encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            gold = []
            seen = set()
            for pid in row.get("gold_passage_ids") or []:
                doc = index.get(pid)
                if doc is None or doc in seen:
                    continue
                seen.add(doc)
                gold.append(doc)
            if len(gold) >= 2:
                questions.append(gold)
    return questions


def eligible(summary: dict) -> bool:
    rule = GRID["unsupervised_selection"]
    size_max = summary["size"]["max"] or 0
    token_max = summary["group_tokens"]["max"] or 0
    return (
        summary["passage_read_multiplier"] <= rule["passage_multiplier_max"]
        and summary["chunk_coverage"] >= rule["coverage_min"]
        and size_max <= rule["max_chunks"]
        and token_max <= rule["max_passage_tokens"]
        and (summary["fraction_chunks_in_size_ge2"] or 0) >= rule["min_fraction_in_groups_of_size_at_least_2"]
    )


def select_unsupervised(rows: list) -> dict:
    pool = [row for row in rows if eligible(row)]
    pool.sort(key=lambda row: (-(row["neighbor_retention"] or 0), -(row["mean_intra_cosine"] or 0), row["name"]))
    bands = {}
    for cap in (1.05, 1.2, 1.5, 2.0):
        band = [
            row for row in rows
            if row["passage_read_multiplier"] <= cap
            and (row["size"]["max"] or 0) <= 16
            and (row["group_tokens"]["max"] or 0) <= 16000
            and row["chunk_coverage"] >= 0.99
        ]
        band.sort(key=lambda row: (-(row["neighbor_retention"] or 0), row["passage_read_multiplier"], row["name"]))
        bands[str(cap)] = band[0]["name"] if band else None
    return {
        "eligible_count": len(pool),
        "selected_by_neighbor_retention": pool[0]["name"] if pool else None,
        "eligible_names": [row["name"] for row in pool[:12]],
        "best_retention_within_multiplier": bands,
        "gold_was_not_used": True,
    }


def prep_ner(prepared: dict, count: int):
    graph = ra.intern_source(prepared["ner"], prepared["triples"])
    lists = ra.doclists_from_sets(graph["b1_sets"], len(graph["display"]))
    df = np.array([len(docs) for docs in lists], dtype=np.int32)
    return graph, lists, df


def entity_edges_from_pairs(pairs: dict, count: int, mode: str) -> list:
    edges = []
    unit = math.log((count + 1) / 2)
    for key, (shared_count, idf_sum) in pairs.items():
        left, right = divmod(key, count)
        if mode == "idf":
            weight = idf_sum / (idf_sum + unit)
        elif mode == "count":
            weight = float(shared_count)
        else:
            raise ValueError(mode)
        edges.append((left, right, weight))
    return edges


def context_edges(corpus, lists, df, display, count: int) -> dict:
    rows = []
    attempted = 0
    failed = 0
    for eid, docs in enumerate(lists):
        degree = int(df[eid])
        if degree < 2 or degree > 15 or len(display[eid]) < 4:
            continue
        idf = math.log((count + 1) / (degree + 1))
        located = []
        for doc in docs:
            attempted += 1
            sentence = find_sentence(corpus[doc]["text"], display[eid])
            if sentence is None:
                failed += 1
                continue
            located.append((doc, sentence))
        if len(located) < 2:
            continue
        rows.append((eid, idf, located))
    if not rows:
        return {"edges": [], "attempted": attempted, "failed": failed}
    vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
    sentences = [sentence for _eid, _idf, located in rows for _doc, sentence in located]
    matrix = vectorizer.fit_transform(sentences)
    edges = {}
    cursor = 0
    for _eid, idf, located in rows:
        block = matrix[cursor : cursor + len(located)]
        cursor += len(located)
        sim = (block @ block.T).toarray()
        norms = np.sqrt(np.clip(np.diag(sim), 1e-8, None))
        sim = sim / norms[:, None] / norms[None, :]
        for i, (left, _sentence) in enumerate(located):
            for j in range(i + 1, len(located)):
                right = located[j][0]
                if left == right:
                    continue
                diverse = idf * (1.0 - float(np.clip(sim[i, j], 0.0, 1.0)))
                pair = (left, right) if left < right else (right, left)
                edges[pair] = edges.get(pair, 0.0) + diverse
    return {
        "edges": [(a, b, w) for (a, b), w in edges.items()],
        "attempted": attempted,
        "failed": failed,
        "located_rate": 1.0 - (failed / attempted if attempted else 0.0),
    }


def field_text(row: dict, entities: list) -> str:
    names = " ".join(entities[:30])
    return f"{row['title']} {row['title']} {names} {row['text']}"


def make_record(name, family, params, groups, tokens, knn1, matrix, title_ids, started) -> tuple:
    summary = summarize(name, family, params, groups, tokens, knn1, matrix, title_ids, time.perf_counter() - started)
    attach_retention(summary, groups, knn1)
    summary["seconds"] = time.perf_counter() - started
    return summary, groups


def main() -> None:
    started_all = time.perf_counter()
    out = ROOT / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "plots").mkdir(exist_ok=True)
    (ROOT / "tables").mkdir(exist_ok=True)
    log("load corpus")
    corpus = ra.load_corpus()
    count = len(corpus)
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = np.array([len(encoder.encode(row["content"])) for row in corpus], dtype=np.int32)
    title_vocab = {}
    title_ids = np.empty(count, dtype=np.int32)
    title_sets = []
    for index, row in enumerate(corpus):
        title_ids[index] = title_vocab.setdefault(row["title"], len(title_vocab))
        title_sets.append(title_tokens(row["title"]))
    log("load official ner")
    official = ra.load_official(corpus)
    graph, lists, df = prep_ner(official, count)
    log("load embeddings")
    table = pq.read_table(GRID["chunk_embeddings"], columns=["hash_id", "content", "embedding"])
    hash_ids = table.column("hash_id").to_pylist()
    contents = table.column("content").to_pylist()
    matrix = np.asarray(table.column("embedding").to_pylist(), dtype=np.float32)
    by_hash = {hash_id: i for i, hash_id in enumerate(hash_ids)}
    order = [by_hash[row["passage_id"]] for row in corpus]
    matrix = l2norm(matrix[order])
    content_mismatch = sum(contents[by_hash[row["passage_id"]]] != row["content"] for row in corpus[:200])
    log(f"embedding content mismatches in first 200: {content_mismatch}")
    log("exact knn")
    neighbors, similarity = exact_knn(matrix, 40)
    knn1 = neighbors[:, 0]
    prior = json.loads(Path(GRID["prior_audit_summary"]).read_text(encoding="utf-8"))
    raw_labels = ra.component_labels(count, lists, df > 0)
    raw_max = int(np.bincount(raw_labels).max())
    prior_max = prior["b1"]["components"]["max_component"]["n_docs"]
    log(f"verify raw max component {raw_max} prior {prior_max}")

    records = []
    store = {}

    def add(name, family, params, groups, started):
        summary, groups = make_record(name, family, params, groups, tokens, knn1, matrix, title_ids, started)
        records.append(summary)
        store[name] = groups
        log(f"{name} groups {summary['n_groups']} max {summary['size']['max']} x{summary['passage_read_multiplier']:.3f} retain {summary['neighbor_retention']:.3f}")

    # E0 baselines
    t0 = time.perf_counter()
    add("E0_raw_entity_cc", "E0", {"filter": None}, labels_to_groups(raw_labels), t0)
    for cut in (10, 50):
        t0 = time.perf_counter()
        labels = ra.component_labels(count, lists, (df > 0) & (df <= cut))
        add(f"E0_entity_cc_df_le_{cut}", "E0", {"max_df": cut}, labels_to_groups(labels), t0)
    t0 = time.perf_counter()
    pairs50, _meta50 = ra.build_pairs(lists, df, 50, count)
    mass50 = ra.doc_idf_mass(lists, df, 50, count)

    def jaccard_of(key, idf_sum, mass=mass50, n_docs=count):
        left, right = divmod(key, n_docs)
        denom = mass[left] + mass[right] - idf_sum
        return idf_sum / denom if denom > 0 else 0.0

    labels = ra.hard_labels_from_pairs(count, pairs50, lambda c, s, key, n: jaccard_of(key, s) >= 0.2)
    add("E0_weighted_jaccard_0.2", "E0", {"threshold": 0.2, "pair_cap": 50}, labels_to_groups(labels), t0)
    t0 = time.perf_counter()
    neigh = [[] for _ in range(count)]
    for key, (shared_count, idf_sum) in pairs50.items():
        left, right = divmod(key, count)
        neigh[left].append((right, idf_sum, shared_count))
        neigh[right].append((left, idf_sum, shared_count))
    for items in neigh:
        items.sort(key=lambda item: (-item[1], -item[2], item[0]))
    for k in (2, 16):
        groups = []
        for doc, items in enumerate(neigh):
            groups.append([doc] + [item[0] for item in items[:k]])
        add(f"E0_idf_topk_{k}", "E0", {"k": k, "pair_cap": 50}, groups, t0)
        t0 = time.perf_counter()
    rng = random.Random(SEED)
    order_docs = list(range(count))
    rng.shuffle(order_docs)
    for size in (4, 8, 16):
        t0 = time.perf_counter()
        add(
            f"E0_random_size_{size}",
            "E0",
            {"size": size},
            [order_docs[i : i + size] for i in range(0, count, size)],
            t0,
        )
        t0 = time.perf_counter()
        add(
            f"E0_sequential_size_{size}",
            "E0",
            {"size": size},
            [list(range(i, min(count, i + size))) for i in range(0, count, size)],
            t0,
        )

    # E7 kmeans
    for target in GRID["kmeans_target_sizes"]:
        t0 = time.perf_counter()
        k = math.ceil(count / target)
        model = MiniBatchKMeans(n_clusters=k, random_state=SEED, batch_size=2048, n_init=1, max_iter=40)
        labels = model.fit_predict(matrix)
        add(f"E7_spherical_kmeans_target_{target}", "E7", {"target_size": target, "k": k, "spherical": True}, labels_to_groups(labels.astype(np.int32)), t0)
    t0 = time.perf_counter()
    k8 = math.ceil(count / 8)
    labels = MiniBatchKMeans(n_clusters=k8, random_state=SEED, batch_size=2048, n_init=1, max_iter=40).fit_predict(matrix)
    split = recsplit(labels.astype(np.int32), matrix, tokens, 8, 16000)
    add("E7_spherical_kmeans_target_8_recsplit_cap8", "E7", {"target_size": 8, "cap_chunks": 8}, labels_to_groups(split), t0)
    t0 = time.perf_counter()
    split16 = recsplit(labels.astype(np.int32), matrix, tokens, 16, 16000)
    add("E7_spherical_kmeans_target_8_recsplit_cap16", "E7", {"target_size": 8, "cap_chunks": 16}, labels_to_groups(split16), t0)

    # E9 knn leiden and other communities
    edge_cache = {}
    for k in GRID["knn_k_values"]:
        edge_cache[k] = knn_edges(neighbors, similarity, k)
        resolutions = (0.5, 1.0, 2.0) if k == 10 else (1.0,)
        for resolution in resolutions:
            t0 = time.perf_counter()
            labels = leiden_labels(count, edge_cache[k], resolution)
            add(f"E9_knn{k}_leiden_res{resolution}", "E9", {"k": k, "resolution": resolution, "mutual": False}, labels_to_groups(labels), t0)
    t0 = time.perf_counter()
    mutual = mutual_edges(neighbors, similarity, 10)
    add("E9_mutual_knn10_leiden_res1", "E9", {"k": 10, "resolution": 1.0, "mutual": True}, labels_to_groups(leiden_labels(count, mutual, 1.0)), t0)
    t0 = time.perf_counter()
    labels = leiden_labels(count, edge_cache[10], 1.0)
    add(
        "E9_knn10_leiden_res1_recsplit_cap8",
        "E9",
        {"k": 10, "resolution": 1.0, "cap_chunks": 8},
        labels_to_groups(recsplit(labels, matrix, tokens, 8, 16000)),
        t0,
    )
    t0 = time.perf_counter()
    others = other_communities(count, edge_cache[10])
    for key in ("louvain", "lpa", "infomap"):
        if key in others:
            add(f"E16_{key}_knn10", "E16", {"graph": "knn10"}, labels_to_groups(others[key]), t0)
            t0 = time.perf_counter()
    community_errors = {key: value for key, value in others.items() if key.endswith("error")}

    # E10 hybrid
    t0 = time.perf_counter()
    pairs30, _meta30 = ra.build_pairs(lists, df, 30, count)
    unit = math.log((count + 1) / 2)
    entity_weight = {}
    for key, (_shared_count, idf_sum) in pairs30.items():
        left, right = divmod(key, count)
        entity_weight[(left, right) if left < right else (right, left)] = idf_sum / (idf_sum + unit)
    cos_weight = {(a, b): w for a, b, w in edge_cache[10]}
    candidate = set(cos_weight) | set(entity_weight)
    title_weight = {}
    for left, right in candidate:
        title_weight[(left, right)] = title_score(title_sets[left], title_sets[right], title_ids[left] == title_ids[right])
    hybrid_edges = {}
    for spec, (alpha, beta, gamma) in GRID["hybrid_weights"].items():
        edges = []
        for pair in candidate:
            if spec == "emb" and pair not in cos_weight:
                continue
            if spec == "ent" and pair not in entity_weight:
                continue
            weight = alpha * cos_weight.get(pair, 0.0) + beta * entity_weight.get(pair, 0.0) + gamma * title_weight[pair]
            if weight > 0:
                edges.append((pair[0], pair[1], weight))
        hybrid_edges[spec] = edges
        add(f"E10_leiden_{spec}", "E10", {"weights": [alpha, beta, gamma], "resolution": 1.0}, labels_to_groups(leiden_labels(count, edges, 1.0)), t0)
        t0 = time.perf_counter()
        add(
            f"E10_greedy_cap8_{spec}",
            "E13",
            {"weights": [alpha, beta, gamma], "max_chunks": 8, "max_tokens": 16000},
            greedy_groups(count, edges, tokens, 8, 16000),
            t0,
        )
        t0 = time.perf_counter()
    add("E10_greedy_cap16_mix", "E13", {"weights": GRID["hybrid_weights"]["mix"], "max_chunks": 16}, greedy_groups(count, hybrid_edges["mix"], tokens, 16, 16000), t0)

    # E12 sparse text
    log("tfidf")
    t0 = time.perf_counter()
    texts = [row["text"] for row in corpus]
    tfidf = TfidfVectorizer(min_df=2, max_df=0.4, stop_words="english", max_features=60000)
    reduced = l2norm(TruncatedSVD(n_components=64, random_state=SEED).fit_transform(tfidf.fit_transform(texts)).astype(np.float32))
    text_neighbors, text_similarity = exact_knn(reduced, 10)
    text_edges = knn_edges(text_neighbors, text_similarity, 10)
    add("E12_tfidf_svd_knn10_leiden", "E12", {"svd": 64}, labels_to_groups(leiden_labels(count, text_edges, 1.0)), t0)
    t0 = time.perf_counter()
    add("E12_tfidf_greedy_cap8", "E12", {"max_chunks": 8}, greedy_groups(count, text_edges, tokens, 8, 16000), t0)
    log("bm25")
    t0 = time.perf_counter()
    token_rows = [re.findall(r"[a-z0-9]+", f"{row['title']} {row['text']}".lower()) for row in corpus]
    df_terms = Counter()
    tfs = []
    for tokens_row in token_rows:
        tf = Counter(tokens_row)
        tfs.append(tf)
        df_terms.update(tf.keys())
    avgdl = sum(len(row) for row in token_rows) / count
    postings = defaultdict(list)
    for doc, tf in enumerate(tfs):
        for term in tf:
            degree = df_terms[term]
            if 2 <= degree <= 80:
                postings[term].append(doc)
    k1, b = 1.5, 0.75
    bm25_edges = []
    idf_terms = {term: math.log(1 + (count - freq + 0.5) / (freq + 0.5)) for term, freq in df_terms.items() if 2 <= freq <= 80}
    for doc, tf in enumerate(tfs):
        ranked_terms = sorted(((value, term) for term, value in tf.items() if term in idf_terms), reverse=True)[:8]
        scores = defaultdict(float)
        dl = len(token_rows[doc])
        for _tf_value, term in ranked_terms:
            tf_value = tf[term]
            denom = tf_value + k1 * (1 - b + b * dl / avgdl)
            weight = idf_terms[term] * (tf_value * (k1 + 1)) / denom
            for other in postings[term]:
                if other != doc:
                    scores[other] += weight
        if not scores:
            continue
        top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:10]
        for other, weight in top:
            pair = (doc, other) if doc < other else (other, doc)
            bm25_edges.append((pair[0], pair[1], float(weight)))
    best = {}
    for left, right, weight in bm25_edges:
        best[(left, right)] = max(best.get((left, right), 0.0), weight)
    if best:
        max_w = max(best.values())
        bm25_final = [(a, b, w / max_w) for (a, b), w in best.items()]
    else:
        bm25_final = []
    add("E12_bm25_greedy_cap8", "E12", {"k1": k1, "b": b, "max_chunks": 8}, greedy_groups(count, bm25_final, tokens, 8, 16000), t0)

    # E13 remaining capacity variants on embedding knn
    for cap in (4, 16):
        t0 = time.perf_counter()
        add(f"E13_embknn10_cap{cap}", "E13", {"max_chunks": cap, "graph": "embknn10"}, greedy_groups(count, edge_cache[10], tokens, cap, 16000), t0)
    for token_cap in (4000, 8000, 16000):
        t0 = time.perf_counter()
        add(
            f"E13_embknn10_tokens_{token_cap}",
            "E13",
            {"max_tokens": token_cap, "max_chunks": 10**6, "graph": "embknn10"},
            greedy_groups(count, edge_cache[10], tokens, 10**6, token_cap),
            t0,
        )

    # E15 context diversity versus plain idf, both capacity 8
    t0 = time.perf_counter()
    idf_edges = entity_edges_from_pairs(pairs30, count, "idf")
    add("E15_idf_greedy_cap8", "E15", {"pair_cap": 30, "max_chunks": 8}, greedy_groups(count, idf_edges, tokens, 8, 16000), t0)
    log("context sentences")
    t0 = time.perf_counter()
    context = context_edges(corpus, lists, df, graph["display"], count)
    add("E15_context_diversity_cap8", "E15", {"pair_cap_df": 15, "mention_located_rate": context.get("located_rate"), "mentions_attempted": context["attempted"], "mentions_failed": context["failed"]}, greedy_groups(count, context["edges"], tokens, 8, 16000), t0)

    # E14 limited overlap on three frozen disjoint graphs
    for base_name, edge_list in (
        ("E13_embknn10_cap8", edge_cache[10]),
        ("E10_greedy_cap8_mix", hybrid_edges["mix"]),
        ("E12_bm25_greedy_cap8", bm25_final),
    ):
        # emb cap 8 was named E13 only for 4 and 16; build cap 8 if absent
        if base_name not in store:
            t0 = time.perf_counter()
            add(base_name, "E13", {"max_chunks": 8}, greedy_groups(count, edge_list, tokens, 8, 16000), t0)
        for budget, copies in ((1.2, 2), (1.5, 3)):
            t0 = time.perf_counter()
            packed = overlap_groups(store[base_name], edge_list, tokens, int(tokens.sum()), budget, copies)
            add(
                f"E14_{base_name}_m{budget}_c{copies}",
                "E14",
                {"base": base_name, "multiplier_cap": budget, "max_copies": copies, "copies_added": packed["copies_added"], "blocked": packed["add_attempts_blocked"]},
                packed["groups"],
                t0,
            )

    selection = select_unsupervised(records)
    (out / "unsupervised_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    log(f"unsupervised selection {selection['selected_by_neighbor_retention']}")

    # Gold only after the selection file exists.
    questions = load_questions(corpus)
    cross_title_questions = 0
    for gold in questions:
        if any(title_ids[gold[0]] != title_ids[other] for other in gold[1:]):
            cross_title_questions += 1
    for summary in records:
        summary.update({f"gold_{key}": value for key, value in gold_scores(store[summary["name"]], questions, title_ids).items()})
    raw_gold = next(row["gold_full_gold_colocation"] for row in records if row["name"] == "E0_raw_entity_cc")
    top16 = next(row["gold_full_gold_colocation"] for row in records if row["name"] == "E0_idf_topk_16")
    jacc = next(row["gold_full_gold_colocation"] for row in records if row["name"] == "E0_weighted_jaccard_0.2")
    verification = {
        "prior_b1_max_docs": prior_max,
        "recomputed_b1_max_docs": raw_max,
        "prior_raw_full_gold": next(row["questions_all_gold_in_one_component"] for row in prior["gold_posthoc"]["by_policy"] if row["policy"] == "raw"),
        "recomputed_raw_full_gold": raw_gold,
        "prior_topk16_full_gold": next(row["questions_covered_by_one_gold_members_topk"] for row in prior["gold_posthoc"]["topk_idf"] if row["k"] == 16),
        "recomputed_topk16_full_gold": top16,
        "recomputed_jaccard02_full_gold": jacc,
        "questions": len(questions),
        "questions_with_any_cross_title_gold_pair": cross_title_questions,
        "embedding_content_mismatches_first_200": content_mismatch,
        "community_errors": community_errors,
        "mention_context": {"attempted": context["attempted"], "failed": context["failed"], "located_rate": context.get("located_rate")},
    }
    log(f"verify gold raw {raw_gold} top16 {top16} jaccard {jacc}")

    # Luna robustness for entity-using methods selected or central.
    log("luna robustness")
    luna = ra.load_luna(corpus)
    luna_graph, luna_lists, luna_df = prep_ner(luna, count)
    luna_rows = []
    luna_raw = labels_to_groups(ra.component_labels(count, luna_lists, luna_df > 0))
    luna_df10 = labels_to_groups(ra.component_labels(count, luna_lists, (luna_df > 0) & (luna_df <= 10)))
    luna_pairs, _meta = ra.build_pairs(luna_lists, luna_df, 30, count)
    luna_edges = entity_edges_from_pairs(luna_pairs, count, "idf")
    luna_cap = greedy_groups(count, luna_edges, tokens, 8, 16000)
    luna_context = context_edges(corpus, luna_lists, luna_df, luna_graph["display"], count)
    luna_bridge = greedy_groups(count, luna_context["edges"], tokens, 8, 16000)
    for name, groups in (
        ("Luna_E0_raw_entity_cc", luna_raw),
        ("Luna_E0_entity_cc_df_le_10", luna_df10),
        ("Luna_E15_idf_greedy_cap8", luna_cap),
        ("Luna_E15_context_diversity_cap8", luna_bridge),
    ):
        summary, _groups = make_record(name, "Luna", {}, groups, tokens, knn1, matrix, title_ids, time.perf_counter())
        summary.update({f"gold_{key}": value for key, value in gold_scores(groups, questions, title_ids).items()})
        luna_rows.append(summary)

    # Cases from the unsupervised winner and the best gold results inside 1.05x and 1.5x.
    def band_winner(cap: float) -> str | None:
        pool = [
            row for row in records
            if row["passage_read_multiplier"] <= cap
            and (row["size"]["max"] or 99) <= 16
            and (row["group_tokens"]["max"] or 10**9) <= 16000
            and row["chunk_coverage"] >= 0.99
        ]
        if not pool:
            return None
        pool.sort(key=lambda row: (-row["gold_full_gold_colocation"], row["passage_read_multiplier"], row["name"]))
        return pool[0]["name"]

    case_methods = []
    for name in (
        selection["selected_by_neighbor_retention"],
        band_winner(1.05),
        band_winner(1.5),
    ):
        if name and name not in case_methods:
            case_methods.append(name)
    cases = []
    b1_sets = graph["b1_sets"]
    display = graph["display"]
    for name in case_methods:
        ranked = []
        for group in store[name]:
            if not 3 <= len(group) <= 8:
                continue
            if len({corpus[doc]["title"] for doc in group}) < 2:
                continue
            block = matrix[group]
            sims = block @ block.T
            iu = np.triu_indices(len(group), 1)
            score = float(sims[iu].mean()) if len(iu[0]) else 0.0
            ranked.append((score, group))
        ranked.sort(key=lambda item: item[0], reverse=True)
        picks = ranked[:3]
        if len(ranked) > 6:
            picks.append(ranked[len(ranked) // 2])
        for score, group in picks:
            shared = Counter()
            for doc in group:
                for eid in b1_sets[doc]:
                    shared[eid] += 1
            bridges = [eid for eid, freq in shared.items() if freq >= 2]
            bridges.sort(key=lambda eid: (math.log((count + 1) / (int(df[eid]) + 1)), -int(df[eid])), reverse=True)
            cases.append({
                "method": name,
                "mean_cosine": score,
                "passages": [
                    {
                        "passage_id": corpus[doc]["passage_id"],
                        "title": corpus[doc]["title"],
                        "snippet": ra.snippet(corpus[doc]["text"], 280),
                    }
                    for doc in group
                ],
                "shared_entities": [
                    {"entity": display[eid], "df": int(df[eid]), "chunks_in_group": int(shared[eid])}
                    for eid in bridges[:8]
                ],
            })

    payload = {
        "verification": verification,
        "selection_before_gold": selection,
        "methods": records,
        "luna": luna_rows,
        "band_winners_by_gold_after_freeze": {"1.05": band_winner(1.05), "1.5": band_winner(1.5), "2.0": band_winner(2.0)},
        "seconds_total": time.perf_counter() - started_all,
        "skipped": {
            "E8_dense_hierarchical": "Full average/complete linkage needs a dense distance matrix. Sparse alternative is the capacity-constrained greedy merge on the KNN graph (E13).",
            "E11_neural_mention_embedding": "The only local embedding model on disk is NV-Embed-v2. Whole-passage chunk embeddings from it are reused. Mention-window neural encoding was not run.",
            "E16_spectral_dense": "Spectral clustering on a dense 11656x11656 matrix was not run. Label propagation, Louvain, Infomap and Leiden run on the sparse KNN graph.",
            "E16_metis": "metis is not installed.",
            "community_errors": community_errors,
        },
    }
    (out / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    write_table(records, ROOT / "tables" / "methods.csv")
    write_table(luna_rows, ROOT / "tables" / "luna_robustness.csv")
    plot_all(records)
    log(f"done in {payload['seconds_total']:.1f}s")


def write_table(rows: list, path: Path) -> None:
    import csv

    fields = [
        "name", "family", "n_groups", "passage_read_multiplier", "total_read_multiplier",
        "chunk_coverage", "membership_mean", "membership_max", "neighbor_retention",
        "mean_intra_cosine", "singleton_group_frac", "fraction_chunks_in_size_ge2",
        "multi_group_pure_title_frac", "sampled_pair_same_title_frac", "oversized_groups", "seconds",
        "size_mean", "size_p50", "size_p90", "size_p95", "size_p99", "size_max",
        "tokens_mean", "tokens_p50", "tokens_p90", "tokens_p95", "tokens_p99", "tokens_max",
        "gold_full_gold_colocation", "gold_mean_partial_gold_coverage", "gold_pairwise_gold_colocation",
        "gold_cross_title_pairwise_gold_colocation", "gold_mean_nongold_fraction_in_best_group",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields}
            for prefix, block in (("size", row["size"]), ("tokens", row["group_tokens"])):
                for stat in ("mean", "p50", "p90", "p95", "p99", "max"):
                    source = "median" if stat == "p50" else stat
                    flat[f"{prefix}_{stat}"] = block.get(source)
            writer.writerow(flat)


def plot_all(rows: list) -> None:
    plot_dir = ROOT / "plots"
    families = sorted({row["family"] for row in rows})
    colors = {family: plt.cm.tab20(i / max(1, len(families))) for i, family in enumerate(families)}
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for row in rows:
        ax.scatter(row["size"]["p95"], row["size"]["max"], s=28, color=colors[row["family"]], label=row["family"])
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Group size P95")
    ax.set_ylabel("Max group size")
    ax.set_title("Group size")
    fig.tight_layout()
    fig.savefig(plot_dir / "size_p95_vs_max.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for row in rows:
        ax.scatter(row["passage_read_multiplier"], row["gold_full_gold_colocation"], s=28, color=colors[row["family"]])
    ax.set_xlabel("Passage read multiplier")
    ax.set_ylabel("Questions with all gold chunks in one group")
    ax.set_title("Full gold co-location vs reading cost")
    fig.tight_layout()
    fig.savefig(plot_dir / "gold_vs_multiplier.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for row in rows:
        ax.scatter(row["group_tokens"]["p95"], row["gold_pairwise_gold_colocation"], s=28, color=colors[row["family"]])
    ax.set_xscale("log")
    ax.set_xlabel("Group passage tokens P95")
    ax.set_ylabel("Pairwise gold co-location")
    ax.set_title("Pairwise gold co-location vs group tokens")
    fig.tight_layout()
    fig.savefig(plot_dir / "pairwise_vs_tokens.png", dpi=140)
    plt.close(fig)

    focus = [
        row for row in rows
        if row["name"] in {
            "E7_spherical_kmeans_target_8_recsplit_cap8",
            "E9_knn10_leiden_res1_recsplit_cap8",
            "E10_greedy_cap8_mix",
            "E13_embknn10_cap8",
            "E15_context_diversity_cap8",
            "E0_weighted_jaccard_0.2",
        }
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for row in focus:
        # histogram cannot be drawn without raw sizes; use a bar of p50/p95/max
        ax.plot([row["size"]["median"], row["size"]["p95"], row["size"]["max"]], marker="o", label=row["name"][:42])
    ax.set_xticks([0, 1, 2], ["P50", "P95", "Max"])
    ax.set_ylabel("Chunks in group")
    ax.legend(fontsize=7)
    ax.set_title("Selected group-size summaries")
    fig.tight_layout()
    fig.savefig(plot_dir / "selected_size_summary.png", dpi=140)
    plt.close(fig)

    overlap = [row for row in rows if row["family"] == "E14" or row["name"] in {"E13_embknn10_cap8", "E10_greedy_cap8_mix", "E12_bm25_greedy_cap8"}]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for row in overlap:
        ax.scatter(row["passage_read_multiplier"], row["gold_full_gold_colocation"], s=36)
        ax.annotate(row["name"].replace("E14_", "").replace("E13_", "")[:28], (row["passage_read_multiplier"], row["gold_full_gold_colocation"]), fontsize=6)
    ax.set_xlabel("Passage read multiplier")
    ax.set_ylabel("Full gold co-location")
    ax.set_title("Disjoint groups vs limited overlap")
    fig.tight_layout()
    fig.savefig(plot_dir / "overlap_tradeoff.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
