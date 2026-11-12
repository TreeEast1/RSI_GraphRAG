"""Re-run the real RAG function on an isolated candidate state."""

from __future__ import annotations

from typing import List, Sequence

from .evolution_analyzer import assert_update_role, score_row
from .graph_patch_manager import GraphPatchManager, assert_raw_unchanged
from .retrieval import GraphRetriever


def _better(before: dict, after: dict) -> bool:
    if after["em"] > before["em"]:
        return True
    if after["em"] == before["em"] and after.get("Recall@5", 0) > before.get("Recall@5", 0) + 1e-9:
        return True
    return False


def _regressed(before_rows: Sequence[dict], after_rows: Sequence[dict]) -> List[str]:
    before_map = {row["question_id"]: row for row in before_rows}
    dropped = []
    for row in after_rows:
        previous = before_map.get(row["question_id"])
        if previous and row["em"] + 1e-9 < previous["em"]:
            dropped.append(row["question_id"])
    return dropped


def verify_candidates(
    *,
    question: dict,
    before_rollout: dict,
    diagnosis: dict,
    raw_graph: dict,
    store,
    retrieval_skills,
    answer_skills,
    client,
    embedder,
    retrieval_config: dict,
    regression_questions: Sequence[dict],
    qa_top_k: int,
    raw_hash: str,
) -> dict:
    assert_update_role(question.get("split_role", ""))
    assert_raw_unchanged(raw_graph, raw_hash)
    base_store = store.snapshot()
    base_retrieval = retrieval_skills.snapshot()
    base_answer = answer_skills.snapshot()
    before_score = score_row(question, before_rollout)
    answers = [question.get("answer") or ""] + list(question.get("answer_aliases") or [])
    candidates = []

    graph_ops = diagnosis.get("graph_ops") or []
    if graph_ops and diagnosis.get("graph_error"):
        candidates.append({"kind": "graph", "ops": graph_ops})
    if diagnosis.get("retrieval_error") and diagnosis.get("retrieval_skill"):
        skill, reason = retrieval_skills.propose(
            diagnosis["retrieval_skill"],
            question=question["question"],
            answers=answers,
            source_case_id=question["question_id"],
        )
        if skill:
            candidates.append({"kind": "retrieval_skill", "skill": skill})
        else:
            candidates.append({"kind": "retrieval_skill", "rejected_reason": reason})
    if diagnosis.get("answer_error") and diagnosis.get("answer_skill"):
        skill, reason = answer_skills.propose(
            diagnosis["answer_skill"],
            question=question["question"],
            answers=answers,
            source_case_id=question["question_id"],
        )
        if skill:
            candidates.append({"kind": "answer_skill", "skill": skill})
        else:
            candidates.append({"kind": "answer_skill", "rejected_reason": reason})

    def run_question(item: dict) -> dict:
        retriever = GraphRetriever(raw_graph, store, embedder, retrieval_config)
        rollout = rag_call(item, retriever)
        return score_row(item, rollout), rollout

    def rag_call(item, retriever):
        from .rag import rag

        return rag(
            item["question"],
            retriever,
            client,
            retrieval_skills,
            answer_skills,
            qa_top_k=qa_top_k,
        )

    regression_before = []
    for other in regression_questions:
        if other["question_id"] == question["question_id"]:
            continue
        baseline_score, _baseline_rollout = run_question(other)
        regression_before.append(baseline_score)

    results = []
    helpful = []
    for candidate in candidates:
        store.restore(base_store)
        retrieval_skills.restore(base_retrieval)
        answer_skills.restore(base_answer)
        if "rejected_reason" in candidate:
            results.append({"candidate": candidate, "decision": "rejected_before_rerun"})
            continue
        if candidate["kind"] == "graph":
            manager = GraphPatchManager(raw_graph, store)
            accepted, rejected = manager.validate_and_apply(candidate["ops"], created_by="question_time")
            if not accepted:
                results.append({"candidate": {"kind": "graph"}, "decision": "rejected_validation", "rejected": rejected})
                store.restore(base_store)
                continue
        elif candidate["kind"] == "retrieval_skill":
            retrieval_skills.stage(candidate["skill"])
        else:
            answer_skills.stage(candidate["skill"])
        after_score, after_rollout = run_question(question)
        regression_after = []
        for other in regression_questions:
            if other["question_id"] == question["question_id"]:
                continue
            other_score, _other_rollout = run_question(other)
            regression_after.append(other_score)
        dropped = _regressed(regression_before, regression_after)
        improved = _better(before_score, after_score)
        decision = "accepted" if improved and not dropped else "rejected"
        results.append(
            {
                "candidate": {key: candidate[key] for key in candidate if key != "ops"},
                "before": before_score,
                "after": after_score,
                "improved": improved,
                "regressed_ids": dropped,
                "decision": decision,
                "after_answer": after_rollout.get("answer"),
                "comprehension_ids_in_top10": after_rollout.get("comprehension_ids_in_top10"),
            }
        )
        if decision == "accepted":
            helpful.append(candidate)
        store.restore(base_store)
        retrieval_skills.restore(base_retrieval)
        answer_skills.restore(base_answer)

    chosen = helpful[0] if helpful else None
    if chosen:
        if chosen["kind"] == "graph":
            manager = GraphPatchManager(raw_graph, store)
            manager.validate_and_apply(chosen["ops"], created_by="question_time")
        elif chosen["kind"] == "retrieval_skill":
            retrieval_skills.stage(chosen["skill"])
            retrieval_skills.commit(chosen["skill"]["skill_id"], question["question_id"], True)
        else:
            answer_skills.stage(chosen["skill"])
            answer_skills.commit(chosen["skill"]["skill_id"], question["question_id"], True)
    assert_raw_unchanged(raw_graph, raw_hash)
    return {
        "question_id": question["question_id"],
        "before": before_score,
        "n_candidates": len(candidates),
        "results": results,
        "committed": None if chosen is None else chosen["kind"],
        "raw_hash": raw_hash,
    }
