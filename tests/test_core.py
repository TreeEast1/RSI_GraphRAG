"""Core correctness tests (no GPU / no network required).

Run: python -m pytest tests -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.locomo import load_locomo, make_splits  # noqa: E402
from src.evaluation.metrics import (evidence_coverage, exact_match, f1_score,
                                    recall_at_k, select_by_budget)  # noqa: E402
from src.grouping.methods import GroupingConfig, build_grouping  # noqa: E402


@pytest.fixture(scope="module")
def convs():
    return load_locomo(ROOT / "data" / "raw" / "locomo" / "locomo10.json")


def test_session_and_turn_mapping(convs):
    for c in convs:
        assert len(c.sessions) > 0
        for s in c.sessions:
            for t in s.turns:
                assert t.dia_id.startswith(f"D{s.session_id}:")
                assert t.session_id == s.session_id
        # no cross-conversation leakage
        docs = c.documents()
        assert all(d["document_id"].startswith(c.sample_id + "::") for d in docs)


def test_evidence_maps_to_existing_turns(convs):
    c = convs[0]
    all_dia = {t.dia_id for s in c.sessions for t in s.turns}
    for q in c.qa:
        for d in q.evidence_turns:
            assert d in all_dia
        for s in q.evidence_sessions:
            assert any(x.session_id == s for x in c.sessions)


def test_split_is_deterministic_and_disjoint(convs):
    a = make_splits(convs, 3, seed=20240501)
    b = make_splits(convs, 3, seed=20240501)
    assert a == b
    assert set(a["dev"]).isdisjoint(a["test"])
    assert len(a["dev"]) + len(a["test"]) == len(convs)


def test_metrics_basics():
    assert exact_match("Dancing.", ["dancing"]) == 1.0
    assert f1_score("by dancing", ["dancing"]) == pytest.approx(2 / 3, abs=1e-6)
    assert recall_at_k(["a", "b"], {"b"}, 1) == 0.0
    assert recall_at_k(["a", "b"], {"b"}, 2) == 1.0


def test_select_by_budget():
    from src.dataset.locomo import num_tokens
    docs = [{"document_id": "d1", "content": "word " * 100},
            {"document_id": "d2", "content": "word " * 100}]
    sel = select_by_budget(docs, 150, num_tokens)
    assert len(sel) == 1


def test_evidence_coverage():
    t2s = {"D1:1": "c::s01", "D2:1": "c::s02"}
    cov = evidence_coverage({"c::s01"}, ["D1:1", "D2:1"], t2s)
    assert cov["full"] == 0.0
    assert cov["partial"] == 0.5


def test_groupings_respect_budget(convs):
    c = convs[1]
    doc_ids = [s.document_id for s in c.sessions]
    toks = {s.document_id: s.n_tokens for s in c.sessions}
    ents = {d: {"Alice", f"e{i}"} for i, d in enumerate(doc_ids)}
    import numpy as np
    emb = np.random.RandomState(0).randn(len(doc_ids), 8).astype("float32")
    for method in ["random", "temporal", "entity_idf", "embedding", "hybrid_0.5", "community"]:
        res = build_grouping(method, c.sample_id, doc_ids, toks, ents, emb,
                             GroupingConfig(budget_tokens=4096, seed=0))
        for g in res.groups:
            assert g, "empty group"
            # single oversized docs are allowed, but never two docs over budget
            if len(g) > 1:
                assert sum(toks[d] for d in g) <= 4096
        assert set(d for g in res.groups for d in g) <= set(doc_ids)


def test_cache_and_manifest_guard(tmp_path):
    """G_raw must not be silently overwritten: variants use separate dirs."""
    from src.graph.variants import working_dir_of
    a = working_dir_of(tmp_path / "raw", "hy4", "/x/NV-Embed-v2")
    b = working_dir_of(tmp_path / "raw" / "variants" / "v1", "hy4", "/x/NV-Embed-v2")
    assert a != b
