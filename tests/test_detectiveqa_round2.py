"""Round-two DetectiveQA logic tests (no GPU / no network)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detectiveqa import round2  # noqa: E402
from src.detectiveqa.data import Novel, Paragraph, Question, build_chunks  # noqa: E402
from src.detectiveqa.readtime import Packet, relation_to_triple, validate_relation  # noqa: E402
from src.detectiveqa.round2 import (bootstrap_paired_by_novel, cutoff_bucket, gold_reach,  # noqa: E402
                                    permute_options, recent_chunks, retrieval_metrics, rrf)


def sample_question():
    return Question("human_anno:1:00", "1", "human_anno", 0, "Who lied?",
                    {"A": "Alice", "B": "Bob", "C": "Carol", "D": "Dan"}, "C", 9, 9, [2, 6])


class Round2Test(unittest.TestCase):
    def test_permutation_tracks_gold(self):
        q, mapping = permute_options(sample_question(), 1)
        self.assertEqual(mapping[q.answer], "C")
        self.assertEqual(sorted(q.options.values()), ["Alice", "Bob", "Carol", "Dan"])

    def test_metrics_distinguish_any_and_all_gold(self):
        reach = gold_reach(["a", "x"], ["a", "b"])
        self.assertEqual(reach["gold_any_in_context"], 1.0)
        self.assertEqual(reach["gold_all_in_context"], 0.0)
        metrics = retrieval_metrics(["x", "b", "a"], ["a", "b"], ks=(2, 3))
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertEqual(metrics["all@3"], 1.0)

    def test_recent_is_prefix_safe(self):
        novel = Novel("1", "x", "y", [Paragraph(i, f"word {i}") for i in range(1, 13)])
        chunks = build_chunks(novel, {5, 9}, target_words=2, max_words=3)
        original = round2.tokens
        round2.tokens = lambda text: len(text.split())
        try:
            self.assertTrue(all(c.end_pid < 9 for c in recent_chunks(chunks, 9, 10000)))
        finally:
            round2.tokens = original

    def test_rrf_and_bucket(self):
        self.assertEqual(rrf([["a", "b"], ["b", "c"]])[0], "b")
        self.assertEqual(cutoff_bucket(0.91), "late_ge_90")

    def test_cluster_bootstrap_reports_wins(self):
        rows = [{"novel_id": "1", "a": {"correct": 0}, "b": {"correct": 1}},
                {"novel_id": "2", "a": {"correct": 1}, "b": {"correct": 1}}]
        result = bootstrap_paired_by_novel(rows, "a", "b", n=20)
        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["losses"], 0)

    def test_v3_rejects_type_name_predicate_and_uses_fixed_status(self):
        novel = Novel("1", "x", "y", [Paragraph(i, f"Alice saw Bob {i}") for i in range(1, 13)])
        chunks = build_chunks(novel, {5, 9}, target_words=1000, max_words=2000)
        packet = Packet("1:000", "1", "Alice", [chunks[0].chunk_id, chunks[1].chunk_id], 9, 20)
        raw = {"subject": "Alice", "predicate": "motive_or_interest", "object": "Bob",
               "relation_type": "motive_or_interest", "source_chunk_ids": packet.chunk_ids,
               "evidence": [{"chunk_id": packet.chunk_ids[0], "quote": "Alice saw Bob 1"},
                            {"chunk_id": packet.chunk_ids[1], "quote": "Alice saw Bob 5"}]}
        self.assertEqual(validate_relation(raw, packet, chunks)[1], "generic_predicate")
        raw["predicate"] = "wants money from"
        relation, reason = validate_relation(raw, packet, chunks)
        self.assertEqual(reason, "")
        relation["fixed_status"] = "suspected"
        self.assertEqual(relation_to_triple(relation)[1], "suspected wants money from")


if __name__ == "__main__":
    unittest.main()
