"""Unsupervised document grouping strategies (experiment B).

All strategies take the same inputs -- session documents, per-session entity
sets, per-session embeddings -- and return *reading groups*: lists of session
ids whose union fits inside a token budget.  No question or gold information is
ever used here.

Strategies
----------
B0 random          : random packing (multiple seeds)
B1 temporal        : consecutive sessions packed in time order
B2 entity          : shared-entity similarity (jaccard / IDF-weighted)
B3 embedding       : cosine similarity on NV-Embed-v2 session vectors
B4 hybrid          : w * entity + (1 - w) * embedding
B5 community       : bipartite document-entity graph + Louvain communities
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class GroupingConfig:
    budget_tokens: int = 8192
    min_group_size: int = 1
    allow_overlap: bool = False
    seed: int = 0
    entity_weight: float = 0.5
    similarity: str = "idf"  # jaccard | idf
    community_resolution: float = 1.0
    split_large: bool = True


@dataclass
class GroupingResult:
    method: str
    conv_id: str
    budget_tokens: int
    groups: List[List[str]] = field(default_factory=list)
    params: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "conv_id": self.conv_id,
            "budget_tokens": self.budget_tokens,
            "params": self.params,
            "groups": self.groups,
            "n_groups": len(self.groups),
            "total_reading_tokens": self.params.get("total_reading_tokens"),
        }


# ---------------------------------------------------------------- similarities
def entity_idf(entity_sets: Dict[str, Set[str]]) -> Dict[str, float]:
    n = len(entity_sets)
    df: Dict[str, int] = {}
    for ents in entity_sets.values():
        for e in ents:
            df[e] = df.get(e, 0) + 1
    return {e: math.log((n + 1) / (c + 1)) + 1.0 for e, c in df.items()}


def entity_similarity(a: Set[str], b: Set[str], idf: Dict[str, float], mode: str = "idf") -> float:
    inter = a & b
    if not inter:
        return 0.0
    if mode == "jaccard":
        return len(inter) / len(a | b)
    # IDF-weighted cosine over the entity sets.
    num = sum(idf.get(e, 1.0) for e in inter)
    den_a = math.sqrt(sum(idf.get(e, 1.0) ** 2 for e in a)) or 1.0
    den_b = math.sqrt(sum(idf.get(e, 1.0) ** 2 for e in b)) or 1.0
    return num / (den_a * den_b)


def cosine_sim_matrix(vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vn = v / norms
    return vn @ vn.T


def similarity_matrix(
    doc_ids: Sequence[str],
    entity_sets: Dict[str, Set[str]],
    embeddings: Optional[np.ndarray],
    cfg: GroupingConfig,
) -> np.ndarray:
    n = len(doc_ids)
    sim = np.zeros((n, n), dtype=np.float32)
    if cfg.similarity in ("jaccard", "idf"):
        idf = entity_idf(entity_sets)
        for i in range(n):
            for j in range(i + 1, n):
                s = entity_similarity(entity_sets.get(doc_ids[i], set()),
                                      entity_sets.get(doc_ids[j], set()), idf, cfg.similarity)
                sim[i, j] = sim[j, i] = s
    if cfg.similarity == "embedding" or (embeddings is not None and cfg.similarity.startswith("hybrid")):
        esim = cosine_sim_matrix(embeddings)
        if cfg.similarity == "embedding":
            sim = esim
        else:
            w = cfg.entity_weight
            sim = w * sim + (1 - w) * esim
    np.fill_diagonal(sim, 1.0)
    return sim


# --------------------------------------------------------------- group packing
def pack_by_budget(
    order: Sequence[str],
    token_sizes: Dict[str, int],
    budget: int,
    min_size: int = 1,
) -> List[List[str]]:
    """Greedy sequential packing in the given order."""
    groups: List[List[str]] = []
    cur: List[str] = []
    used = 0
    for doc in order:
        t = token_sizes[doc]
        if cur and used + t > budget:
            if len(cur) >= min_size:
                groups.append(cur)
            cur, used = [], 0
        cur.append(doc)
        used += t
    if cur:
        groups.append(cur)
    return groups


def constrained_agglomerative(
    doc_ids: Sequence[str],
    sim: np.ndarray,
    token_sizes: Dict[str, int],
    budget: int,
    min_size: int = 1,
) -> List[List[str]]:
    """Merge clusters greedily by descending inter-cluster mean similarity."""
    clusters: Dict[int, List[str]] = {i: [doc_ids[i]] for i in range(len(doc_ids))}
    sizes = {i: token_sizes[doc_ids[i]] for i in range(len(doc_ids))}

    pairs = []
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            pairs.append((float(sim[i, j]), i, j))
    pairs.sort(reverse=True)

    parent = {i: i for i in range(len(doc_ids))}
    members = {i: [i] for i in range(len(doc_ids))}
    tokens = {i: sizes[i] for i in range(len(doc_ids))}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, i, j in pairs:
        if s <= 0:
            break
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if tokens[ri] + tokens[rj] > budget:
            continue
        # greedy union by size
        if len(members[ri]) < len(members[rj]):
            ri, rj = rj, ri
        parent[rj] = ri
        members[ri].extend(members[rj])
        tokens[ri] += tokens[rj]
        members.pop(rj, None)

    groups = [[doc_ids[i] for i in m] for m in members.values()]
    # order sessions chronologically inside a group for readable prompts
    order = {d: k for k, d in enumerate(doc_ids)}
    for g in groups:
        g.sort(key=lambda d: order[d])
    return groups


def split_oversized(groups: List[List[str]], token_sizes: Dict[str, int], budget: int) -> List[List[str]]:
    out = []
    for g in groups:
        g = sorted(g, key=lambda d: token_sizes[d])
        cur, used = [], 0
        for d in g:
            if cur and used + token_sizes[d] > budget:
                out.append(cur)
                cur, used = [], 0
            cur.append(d)
            used += token_sizes[d]
        if cur:
            out.append(cur)
    return out


# ------------------------------------------------------------------ strategies
def group_random(doc_ids, token_sizes, cfg: GroupingConfig) -> List[List[str]]:
    order = list(doc_ids)
    random.Random(cfg.seed).shuffle(order)
    groups = pack_by_budget(order, token_sizes, cfg.budget_tokens, cfg.min_group_size)
    order_idx = {d: k for k, d in enumerate(doc_ids)}
    return [sorted(g, key=lambda d: order_idx[d]) for g in groups]


def group_temporal(doc_ids, token_sizes, cfg: GroupingConfig) -> List[List[str]]:
    return pack_by_budget(list(doc_ids), token_sizes, cfg.budget_tokens, cfg.min_group_size)


def group_similarity(doc_ids, token_sizes, entity_sets, embeddings, cfg: GroupingConfig) -> List[List[str]]:
    sim = similarity_matrix(doc_ids, entity_sets, embeddings, cfg)
    groups = constrained_agglomerative(doc_ids, sim, token_sizes, cfg.budget_tokens, cfg.min_group_size)
    if cfg.split_large:
        groups = split_oversized(groups, token_sizes, cfg.budget_tokens)
    return groups


def group_community(
    doc_ids: Sequence[str],
    token_sizes: Dict[str, int],
    entity_sets: Dict[str, Set[str]],
    cfg: GroupingConfig,
) -> List[List[str]]:
    """Bipartite doc-entity graph -> Louvain communities -> budget packing."""
    import networkx as nx

    idf = entity_idf(entity_sets)
    g = nx.Graph()
    for d in doc_ids:
        g.add_node(f"doc::{d}", bipartite="doc")
    for d in doc_ids:
        for e in entity_sets.get(d, set()):
            if idf.get(e, 0.0) < 1.0:
                continue  # suppress hub entities that appear everywhere
            node = f"ent::{e}"
            if node not in g:
                g.add_node(node, bipartite="ent")
            g.add_edge(f"doc::{d}", node, weight=1.0 / math.sqrt(1 + idf.get(e, 1.0)))

    communities = nx.community.louvain_communities(g, weight="weight", seed=cfg.seed,
                                                   resolution=cfg.community_resolution)
    # keep only communities that contain documents; docs may be absent if isolated
    assigned: Dict[str, int] = {}
    groups: List[List[str]] = []
    for ci, comm in enumerate(communities):
        docs = [n.split("::", 1)[1] for n in comm if n.startswith("doc::")]
        if not docs:
            continue
        docs.sort(key={d: k for k, d in enumerate(doc_ids)}.__getitem__)
        groups.append(docs)
        for d in docs:
            assigned[d] = ci
    # isolated docs
    leftovers = [d for d in doc_ids if d not in assigned]
    if leftovers:
        groups.extend(pack_by_budget(leftovers, token_sizes, cfg.budget_tokens, cfg.min_group_size))
    if cfg.split_large:
        groups = split_oversized(groups, token_sizes, cfg.budget_tokens)
    return groups


def build_grouping(
    method: str,
    conv_id: str,
    doc_ids: List[str],
    token_sizes: Dict[str, int],
    entity_sets: Dict[str, Set[str]],
    embeddings: Optional[np.ndarray],
    cfg: GroupingConfig,
) -> GroupingResult:
    if method == "random":
        groups = group_random(doc_ids, token_sizes, cfg)
    elif method == "temporal":
        groups = group_temporal(doc_ids, token_sizes, cfg)
    elif method in ("entity_jaccard", "entity_idf"):
        cfg = GroupingConfig(**{**cfg.__dict__, "similarity": method.split("_")[1]})
        groups = group_similarity(doc_ids, token_sizes, entity_sets, None, cfg)
    elif method == "embedding":
        cfg = GroupingConfig(**{**cfg.__dict__, "similarity": "embedding"})
        groups = group_similarity(doc_ids, token_sizes, entity_sets, embeddings, cfg)
    elif method.startswith("hybrid"):
        w = float(method.split("_")[1]) if "_" in method else cfg.entity_weight
        cfg = GroupingConfig(**{**cfg.__dict__, "similarity": "hybrid", "entity_weight": w})
        groups = group_similarity(doc_ids, token_sizes, entity_sets, embeddings, cfg)
    elif method == "community":
        groups = group_community(doc_ids, token_sizes, entity_sets, cfg)
    else:
        raise ValueError(f"unknown grouping method {method}")

    groups = [g for g in groups if g]
    total = sum(token_sizes[d] for g in groups for d in g)
    res = GroupingResult(method=method, conv_id=conv_id, budget_tokens=cfg.budget_tokens, groups=groups)
    res.params = {
        "budget_tokens": cfg.budget_tokens,
        "seed": cfg.seed,
        "similarity": cfg.similarity,
        "entity_weight": cfg.entity_weight,
        "total_reading_tokens": total,
        "reading_cost_ratio": total / max(1, sum(token_sizes.values())),
        "overlap_docs": sum(len(g) for g in groups) - len(set(d for g in groups for d in g)),
    }
    return res
