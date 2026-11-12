"""Unsupervised reading-group construction for QASPER papers.

Two groupings, both QA-label-free:

* ``section``: contiguous sections merged until the 4096-8192-token reading
  budget is reached. This is the primary, pre-registered grouping.
* ``semantic``: k-means over paragraph embeddings with a small overlap band;
  paragraphs whose second-best centroid is nearly as close as the best are
  included in both groups. Exploratory secondary grouping.

Neither grouping changes the physical graph nodes: groups only decide what the
read-time LLM sees together.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class ReadingGroup:
    group_id: str
    paper_id: str
    method: str
    passage_ids: List[str]
    n_tokens: int

    def to_dict(self) -> Dict:
        return {"group_id": self.group_id, "paper_id": self.paper_id, "method": self.method,
                "passage_ids": self.passage_ids, "n_tokens": self.n_tokens}


def section_groups(paper, min_tokens: int = 4096, max_tokens: int = 8192) -> List[ReadingGroup]:
    """Merge consecutive sections into 4096-8192-token reading groups."""
    by_sec: Dict[int, List] = {}
    for p in paper.paragraphs:
        by_sec.setdefault(p.section_index, []).append(p)
    secs = [by_sec[k] for k in sorted(by_sec)]
    groups: List[List] = []
    cur: List = []
    cur_tok = 0
    for sec in secs:
        sec_tok = sum(p.n_tokens for p in sec)
        # a single oversized section is split by paragraphs
        if sec_tok > max_tokens:
            if cur:
                groups.append(cur)
                cur, cur_tok = [], 0
            part, part_tok = [], 0
            for p in sec:
                if part and part_tok + p.n_tokens > max_tokens:
                    groups.append(part)
                    part, part_tok = [], 0
                part.append(p)
                part_tok += p.n_tokens
            if part:
                groups.append(part)
            continue
        if cur and cur_tok + sec_tok > max_tokens:
            groups.append(cur)
            cur, cur_tok = [], 0
        cur.extend(sec)
        cur_tok += sec_tok
        if cur_tok >= min_tokens:
            groups.append(cur)
            cur, cur_tok = [], 0
    if cur:
        # avoid a tiny tail group: merge into the previous one if it fits
        if groups and cur_tok < 400 and sum(p.n_tokens for p in groups[-1]) + cur_tok <= max_tokens:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    out = []
    for i, g in enumerate(groups):
        out.append(ReadingGroup(f"{paper.paper_id}::g{i:02d}", paper.paper_id, "section",
                                [p.passage_id for p in g], sum(p.n_tokens for p in g)))
    return out


def semantic_groups(paper, embedder, target_tokens: int = 5000, max_tokens: int = 8192,
                    overlap_sim: float = 0.03, seed: int = 0) -> List[ReadingGroup]:
    """k-means over paragraph embeddings, ordered into overlapping groups."""
    ids = [p.passage_id for p in paper.paragraphs]
    texts = [p.text for p in paper.paragraphs]
    toks = {p.passage_id: p.n_tokens for p in paper.paragraphs}
    if len(ids) <= 2:
        return [ReadingGroup(f"{paper.paper_id}::g00", paper.paper_id, "semantic",
                             ids, sum(toks.values()))]
    X = np.asarray(embedder.encode_documents(texts), dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    total = sum(toks.values())
    k = max(1, min(len(ids), int(np.ceil(total / target_tokens))))
    # deterministic k-means++-lite: init from evenly spaced quantiles
    rng = np.random.default_rng(seed)
    init_idx = [int(x) for x in np.linspace(0, len(ids) - 1, k)]
    C = X[init_idx].copy()
    for _ in range(20):
        sim = X @ C.T
        assign = sim.argmax(1)
        newC = []
        for j in range(k):
            m = assign == j
            if m.sum() == 0:
                newC.append(C[j])
            else:
                v = X[m].mean(0)
                newC.append(v / (np.linalg.norm(v) + 1e-9))
        C = np.stack(newC)
    sim = X @ C.T
    groups: List[List[str]] = [[] for _ in range(k)]
    for i, pid in enumerate(ids):
        best = int(sim[i].argmax())
        groups[best].append(pid)
        if k > 1:
            second = int(np.argsort(-sim[i])[1])
            if sim[i][second] >= sim[i][best] - overlap_sim:
                groups[second].append(pid)
    # merge very small clusters into the largest cluster when it fits the budget
    big = max(range(k), key=lambda j: sum(toks[p] for p in groups[j])) if k else 0
    for j in range(k):
        if j == big or not groups[j]:
            continue
        if sum(toks[p] for p in groups[j]) < 1000 and sum(toks[p] for p in groups[big]) + sum(toks[p] for p in groups[j]) <= max_tokens:
            groups[big].extend(groups[j])
            groups[j] = []
    # cap group size, drop lowest-similarity members first
    out = []
    for j, g in enumerate(groups):
        if not g:
            continue
        g = sorted(set(g), key=lambda pid: ids.index(pid))
        while sum(toks[p] for p in g) > max_tokens and len(g) > 1:
            worst = min(g, key=lambda pid: sim[ids.index(pid)][j])
            g.remove(worst)
        out.append(ReadingGroup(f"{paper.paper_id}::s{j:02d}", paper.paper_id, "semantic",
                                g, sum(toks[p] for p in g)))
    out.sort(key=lambda gg: ids.index(gg.passage_ids[0]))
    for i, gg in enumerate(out):
        gg.group_id = f"{paper.paper_id}::s{i:02d}"
    return out
