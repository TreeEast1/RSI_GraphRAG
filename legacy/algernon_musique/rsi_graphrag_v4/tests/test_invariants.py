import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")

from rsi_v4.answer_skill_store import AnswerSkillStore  # noqa: E402
from rsi_v4.comprehension_store import ComprehensionStore  # noqa: E402
from rsi_v4.evolution_analyzer import assert_update_role  # noqa: E402
from rsi_v4.graph_partitioner import audit_partition, partition_cluster  # noqa: E402
from rsi_v4.graph_patch_manager import GraphPatchManager  # noqa: E402
from rsi_v4.hipporag_adapter import build_raw_graph, load_raw_graph, save_raw_graph  # noqa: E402
from rsi_v4.llm_client import BudgetExceeded, LLMClient, UsageLedger  # noqa: E402
from rsi_v4.rag import reader_messages  # noqa: E402
from rsi_v4.retrieval import GraphRetriever  # noqa: E402


def _chain_cluster():
    nodes = [{"id": f"e{i}", "content": f"ent{i}"} for i in range(6)]
    triples = []
    passages = []
    for index in range(5):
        triples.append(
            {
                "triple_id": f"t{index}",
                "s": f"ent{index}",
                "p": "next",
                "o": f"ent{index + 1}",
                "s_id": f"e{index}",
                "o_id": f"e{index + 1}",
                "passage_ids": [f"p{index}"],
            }
        )
        passages.append(
            {
                "passage_id": f"p{index}",
                "title": "T",
                "text": "x" * 40,
                "content": "T\n" + ("x" * 40),
            }
        )
    return {
        "cluster_id": "c0",
        "parent_component_id": "c0",
        "graph_id": "raw-test",
        "nodes": nodes,
        "triples": triples,
        "passages": passages,
    }


class _MapEmbedder:
    def encode(self, texts, instruction=""):
        rows = []
        for text in texts:
            if "hidden bridge" in text:
                rows.append([1.0, 0.0, 0.0, 0.0])
            else:
                rows.append([0.0, 1.0, 0.0, 0.0])
        return np.array(rows, dtype=np.float32)


class Invariants(unittest.TestCase):
    def test_partition_coverage_and_overlap_ids(self):
        cluster = _chain_cluster()
        units = partition_cluster(
            cluster,
            input_token_budget=3,
            overlap_hops=1,
            count_tokens=lambda text: text.count("PASSAGE "),
            overhead_tokens=0,
        )
        audit = audit_partition(cluster, units, 3)
        self.assertTrue(audit["core_coverage_complete"], audit)
        self.assertFalse(audit["invalid_passages"])
        self.assertFalse(audit["lost_triple_ids"])
        self.assertGreater(audit["n_units"], 1)
        seen = []
        for unit in units:
            self.assertFalse(set(unit["core_node_ids"]) & set(unit["overlap_node_ids"]))
            seen.extend(unit["core_node_ids"])
        self.assertEqual(sorted(seen), [f"e{i}" for i in range(6)])
        no_overlap = partition_cluster(
            cluster,
            input_token_budget=3,
            overlap_hops=0,
            count_tokens=lambda text: text.count("PASSAGE "),
            overhead_tokens=0,
        )
        hop0 = audit_partition(cluster, no_overlap, 3)
        self.assertGreaterEqual(audit["boundary_retention_rate"], hop0["boundary_retention_rate"])

    def test_raw_graph_hash_survives_comprehension(self):
        record = {
            "passage_id": "chunk-fixed",
            "title": "Alpha",
            "text": "alpha city signed the long treaty with beta",
            "content": "Alpha\nalpha city signed the long treaty with beta",
            "entities": ["alpha city", "beta"],
            "triples": [["alpha city", "signed", "treaty with beta"]],
        }
        graph = build_raw_graph([record], status="test", model_name="fake", corpus_hash="c")
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "g_raw.json")
            save_raw_graph(graph, path)
            before = Path(path).read_bytes()
            store = ComprehensionStore()
            manager = GraphPatchManager(graph, store)
            entity_ids = list(graph["entities"])
            passage_id = record["passage_id"]
            # The graph builder rewrites the passage id to the hash of content when records
            # already carry an id. Use the id actually stored.
            passage_id = next(iter(graph["passages"]))
            span = "alpha city signed the long treaty"
            accepted, rejected = manager.validate_and_apply(
                [
                    {
                        "op": "ADD",
                        "semantic_content": "alpha city and beta are bound by the signed treaty",
                        "source_entity_ids": entity_ids[:1],
                        "source_triple_ids": [next(iter(graph["triples"]))],
                        "source_passage_ids": [passage_id],
                        "evidence_spans": [{"passage_id": passage_id, "span": span}],
                        "reasoning_description": "the triple names the treaty but not the binding between the two parties",
                        "links": [{"target_id": entity_ids[0], "relation": "bound by treaty"}],
                        "confidence": 0.7,
                    }
                ],
                created_by="read_time",
            )
            self.assertEqual(len(accepted), 1, rejected)
            self.assertEqual(graph["graph_hash"], manager.raw_hash)
            self.assertEqual(Path(path).read_bytes(), before)
            load_raw_graph(path)
            bad_accepted, bad_rejected = manager.validate_and_apply(
                [
                    {
                        "op": "ADD",
                        "semantic_content": "a fabricated claim that is long enough to pass the length rule",
                        "source_entity_ids": ["entity-missing"],
                        "source_passage_ids": [passage_id],
                        "evidence_spans": [{"passage_id": passage_id, "span": "not in the passage at all"}],
                    }
                ],
                created_by="read_time",
            )
            self.assertEqual(bad_accepted, [])
            self.assertEqual(len(store.active_nodes()), 1)
            self.assertTrue(bad_rejected)
            snap = store.snapshot()
            store.deactivate(store.active_nodes()[0]["comprehension_id"], "test")
            store.restore(snap)
            self.assertEqual(len(store.active_nodes()), 1)
            self.assertEqual(Path(path).read_bytes(), before)

    def test_skill_changes_reader_and_rejects_answers(self):
        store = AnswerSkillStore()
        skill, reason = store.propose(
            {
                "trigger_condition": "capital city question",
                "strategy_description": "answer with the city only",
                "executable_guidance": "Give only the city name, without a sentence around it.",
            },
            question="What is the capital?",
            answers=["Paris"],
            source_case_id="q1",
        )
        self.assertEqual(reason, "")
        store.stage(skill)
        store.commit(skill["skill_id"], "q1", True)
        messages = reader_messages(
            "What is the capital?",
            [{"kind": "passage", "doc_id": "p", "title": "T", "text": "body text", "content": "T\nbody text"}],
            store.select("What is the capital?"),
        )
        self.assertIn("Give only the city name", messages[0]["content"])
        leaked, why = store.propose(
            {
                "trigger_condition": "capital city question",
                "strategy_description": "remember Paris",
                "executable_guidance": "The answer is Paris for this kind of question.",
            },
            question="What is the capital?",
            answers=["Paris"],
            source_case_id="q1",
        )
        self.assertIsNone(leaked)
        self.assertIn("gold", why)

    def test_heldout_cannot_update_and_groups_match(self):
        with self.assertRaises(PermissionError):
            assert_update_role("heldout")
        with self.assertRaises(PermissionError):
            assert_update_role("dev")
        assert_update_role("update")
        record = {
            "passage_id": "chunk-a",
            "title": "A",
            "text": "alpha lives in northland",
            "content": "A\nalpha lives in northland",
            "entities": ["alpha"],
            "triples": [["alpha", "lives in", "northland"]],
        }
        left = build_raw_graph([record], status="a", model_name="m", corpus_hash="h")
        right = build_raw_graph([record], status="b", model_name="m", corpus_hash="h")
        self.assertEqual(left["graph_hash"], right["graph_hash"])
        self.assertEqual(ComprehensionStore().snapshot()["version"], ComprehensionStore().snapshot()["version"])

    def test_comprehension_node_is_retrievable(self):
        record = {
            "passage_id": "chunk-a",
            "title": "A",
            "text": "alpha lives in northland today",
            "content": "A\nalpha lives in northland today",
            "entities": ["alpha", "northland"],
            "triples": [["alpha", "lives in", "northland"]],
        }
        graph = build_raw_graph([record], status="test", model_name="m", corpus_hash="h")
        empty = ComprehensionStore()
        embedder = _MapEmbedder()
        before = GraphRetriever(graph, empty, embedder, {"linking_top_k": 5, "passage_node_weight": 0.05, "damping": 0.5})
        before_docs = before.retrieve(["where is the hidden bridge"], top_k=5)["docs"]
        self.assertTrue(all(doc["kind"] == "passage" for doc in before_docs))
        store = ComprehensionStore()
        manager = GraphPatchManager(graph, store)
        passage_id = next(iter(graph["passages"]))
        entity_id = next(iter(graph["entities"]))
        accepted, rejected = manager.validate_and_apply(
            [
                {
                    "op": "ADD",
                    "semantic_content": "the hidden bridge connects alpha to the other note",
                    "source_entity_ids": [entity_id],
                    "source_triple_ids": [next(iter(graph["triples"]))],
                    "source_passage_ids": [passage_id],
                    "evidence_spans": [{"passage_id": passage_id, "span": "alpha lives in northland today"}],
                    "reasoning_description": "the passage does not name the hidden bridge connection",
                    "links": [{"target_id": entity_id, "relation": "connected across notes"}],
                }
            ],
            created_by="read_time",
        )
        self.assertEqual(len(accepted), 1, rejected)
        after = GraphRetriever(graph, store, embedder, {"linking_top_k": 5, "passage_node_weight": 0.05, "damping": 0.5})
        result = after.retrieve(["where is the hidden bridge"], top_k=5)
        self.assertIn(store.active_nodes()[0]["comprehension_id"], result["comprehension_ids_in_top10"])
        self.assertEqual(graph["graph_hash"], before.raw_graph["graph_hash"])

    def test_llm_cache_does_not_spend_budget(self):
        calls = {"n": 0}

        def request_fn(params):
            calls["n"] += 1
            return {
                "text": "{\"ok\": true}",
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "reasoning_tokens": 1,
                "finish_reason": "stop",
                "response_model": "unit-test-model",
            }

        with tempfile.TemporaryDirectory() as tmp:
            ledger = UsageLedger(max_fresh_calls=1)
            client = LLMClient(
                cache_path=str(Path(tmp) / "cache.sqlite"),
                deployment="gpt-5.6-luna",
                endpoint="https://example.invalid",
                api_version="2024-12-01-preview",
                api_key="test",
                ledger=ledger,
                request_fn=request_fn,
            )
            client.complete(stage="probe", messages=[{"role": "user", "content": "hi"}], max_completion_tokens=10)
            client.complete(stage="probe", messages=[{"role": "user", "content": "hi"}], max_completion_tokens=10)
            self.assertEqual(calls["n"], 1)
            self.assertEqual(ledger.fresh_calls, 1)
            with self.assertRaises(BudgetExceeded):
                client.complete(
                    stage="probe",
                    messages=[{"role": "user", "content": "another"}],
                    max_completion_tokens=10,
                )


if __name__ == "__main__":
    unittest.main()
