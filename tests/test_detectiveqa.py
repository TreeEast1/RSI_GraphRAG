"""DetectiveQA prefix-safety and memory-gate tests (no GPU / no network)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detectiveqa.data import (Novel, Paragraph, Question, build_chunks, gold_chunk_ids,  # noqa: E402
                                  make_splits, visible_chunk_ids)
from src.detectiveqa.evolution import accept_update, memory_gate, skill_gate  # noqa: E402
from src.detectiveqa.readtime import Packet, extra_by_latest_chunk, validate_relation  # noqa: E402


def novel():
    return Novel("1", "Case", "Author", [Paragraph(i, f"Paragraph {i} Alice saw Bob near the library.")
                                          for i in range(1, 13)])


def question(qid="human_anno:1:00", source="human_anno", cutoff=8, clues=None, novel_id="1"):
    return Question(qid, novel_id, source, 0, "Who lied?", {"A": "Alice", "B": "Bob", "C": "Carol", "D": "Dan"},
                    "A", cutoff, cutoff, clues or [2, 6], ["clue"], "reason", {})


class DetectiveQATest(unittest.TestCase):
    def test_chunks_respect_every_cutoff(self):
        chunks = build_chunks(novel(), {5, 9}, target_words=1000, max_words=2000)
        self.assertEqual([(c.start_pid, c.end_pid) for c in chunks], [(1, 4), (5, 8), (9, 12)])
        self.assertEqual(visible_chunk_ids(chunks, 9), [chunks[0].chunk_id, chunks[1].chunk_id])
        self.assertNotIn(chunks[2].chunk_id, visible_chunk_ids(chunks, 9))

    def test_gold_chunk_mapping_deduplicates(self):
        chunks = build_chunks(novel(), {9}, target_words=1000, max_words=2000)
        self.assertEqual(gold_chunk_ids(question(clues=[2, 3, 10], cutoff=11), chunks),
                         [chunks[0].chunk_id, chunks[1].chunk_id])

    def test_split_keeps_human_questions_out_of_updates(self):
        qs = [question("human_anno:1:00"), question("AIsup_anno:1:00", "AIsup_anno", cutoff=5),
              question("AIsup_anno:2:00", "AIsup_anno", novel_id="2"),
              question("AIsup_anno:3:00", "AIsup_anno", novel_id="3")]
        splits = make_splits(qs, cross_dev_novels=1)
        self.assertEqual(splits["eval_human"], ["human_anno:1:00"])
        self.assertEqual(splits["within_practice"], ["AIsup_anno:1:00"])
        self.assertTrue(set(splits["cross_dev_novels"]).isdisjoint(splits["cross_practice_novels"]))

    def test_relation_requires_two_quoted_chunks_and_attaches_late(self):
        chunks = build_chunks(novel(), {5, 9}, target_words=1000, max_words=2000)
        packet = Packet("1:000", "1", "Alice", [chunks[0].chunk_id, chunks[2].chunk_id], 13, 100)
        raw = {"subject": "Alice", "predicate": "lied about", "object": "Bob",
               "relation_type": "contradiction_or_deception",
               "source_chunk_ids": packet.chunk_ids,
               "evidence": [{"chunk_id": packet.chunk_ids[0], "quote": "Alice saw Bob"},
                            {"chunk_id": packet.chunk_ids[1], "quote": "near the library"}],
               "status": "supported", "confidence": 0.8}
        relation, reason = validate_relation(raw, packet, chunks)
        self.assertEqual(reason, "")
        self.assertEqual(relation["attach_chunk_id"], chunks[2].chunk_id)
        self.assertEqual(list(extra_by_latest_chunk([relation])), [chunks[2].chunk_id])
        raw["evidence"][1]["quote"] = "invented quote"
        self.assertEqual(validate_relation(raw, packet, chunks)[1], "quote_not_found")

    def test_memory_gate_blocks_future_and_option_leaks(self):
        chunks = build_chunks(novel(), {5, 9}, target_words=1000, max_words=2000)
        q = question(cutoff=9)
        memory = {"subject": "Alice", "predicate": "was near", "object": "library",
                  "source_chunk_ids": [chunks[0].chunk_id, chunks[1].chunk_id],
                  "evidence": [{"chunk_id": chunks[0].chunk_id, "quote": "Alice saw Bob"},
                               {"chunk_id": chunks[1].chunk_id, "quote": "near the library"}]}
        self.assertIsNone(memory_gate(memory, q, chunks, [c.chunk_id for c in chunks[:2]]))
        future = dict(memory, source_chunk_ids=[chunks[0].chunk_id, chunks[2].chunk_id])
        self.assertEqual(memory_gate(future, q, chunks, [c.chunk_id for c in chunks]), "future_evidence")
        q.options["A"] = "Alice was near library"
        self.assertEqual(memory_gate(memory, q, chunks, [c.chunk_id for c in chunks[:2]]), "option_text_leak")

    def test_skill_gate_and_acceptance_rule(self):
        q = question()
        leaked = {"title": "Check Alice", "trigger": "when asked who lied",
                  "procedure": "Prefer Alice when two statements conflict.", "checks": []}
        self.assertEqual(skill_gate(leaked, q), "specific_name_leak")
        safe = {"title": "Separate claims", "trigger": "deception questions",
                "procedure": "Separate observed events from claims, then test every claim against independent evidence.",
                "checks": ["Do not treat a speaker claim as fact."]}
        self.assertIsNone(skill_gate(safe, q))
        result = accept_update({"correct": 0, "evidence_recall": 0}, {"correct": 1, "evidence_recall": .5},
                               [1, 0, 1], [1, 1, 1])
        self.assertTrue(result.accepted)
        result = accept_update({"correct": 0}, {"correct": 1}, [1, 1], [1, 0])
        self.assertEqual(result.reason, "dev_regression")


if __name__ == "__main__":
    unittest.main()
