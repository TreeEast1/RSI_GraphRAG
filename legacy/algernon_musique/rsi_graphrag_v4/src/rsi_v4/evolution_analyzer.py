"""Question-time diagnosis. Gold is visible here and must not be passed back into RAG."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .hipporag_adapter import extract_json_object

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "analyzer_system.txt"


def assert_update_role(role: str) -> None:
    if role != "update":
        raise PermissionError(f"refusing question-time update for split role {role}")


def diagnose(client, *, question: dict, rollout: dict, raw_graph: dict, reasoning_effort: str = "medium") -> dict:
    assert_update_role(question.get("split_role", ""))
    passages = []
    by_id = raw_graph["passages"]
    for passage_id in question.get("gold_passage_ids") or []:
        if passage_id in by_id:
            passages.append(by_id[passage_id])
    for doc in rollout.get("retrieved_docs") or []:
        if doc["doc_id"] in by_id and all(item["passage_id"] != doc["doc_id"] for item in passages):
            passages.append(by_id[doc["doc_id"]])
    passage_block = "\n\n".join(
        f"PASSAGE {item['passage_id']}\n{item['content']}" for item in passages[:8]
    )
    entity_ids = sorted(
        {
            edge["entity_id"]
            for edge in raw_graph["context_edges"]
            if edge["passage_id"] in {item["passage_id"] for item in passages}
        }
    )[:40]
    user = "\n".join(
        [
            f"Question: {question['question']}",
            f"Rollout answer: {rollout.get('answer')}",
            f"Gold answer: {question.get('answer')}",
            f"Reader document ids: {rollout.get('reader_doc_ids')}",
            f"Retrieved documents: {rollout.get('retrieved_docs')}",
            f"Entity ids available for links: {entity_ids}",
            "Passages:",
            passage_block,
        ]
    )
    response = client.complete(
        stage="question_time_analyzer",
        messages=[
            {"role": "system", "content": PROMPT_PATH.read_text(encoding="utf-8")},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=3500,
        reasoning_effort=reasoning_effort,
    )
    try:
        payload = extract_json_object(response.text)
    except Exception as exc:  # noqa: BLE001
        payload = {"graph_error": False, "retrieval_error": False, "answer_error": False, "parse_error": str(exc)[:300]}
    payload["response_model"] = response.metadata.get("response_model")
    payload["prompt_tokens"] = response.metadata.get("prompt_tokens")
    payload["completion_tokens"] = response.metadata.get("completion_tokens")
    payload["question_id"] = question["question_id"]
    return payload


def score_row(question: dict, rollout: dict) -> dict:
    from .metrics import exact_match, passage_recall, token_f1

    ranked = [doc["doc_id"] for doc in rollout.get("retrieved_docs") or []]
    recall = passage_recall(question.get("gold_passage_ids") or [], ranked, (1, 3, 5, 10))
    answers = [question.get("answer") or ""] + list(question.get("answer_aliases") or [])
    return {
        **recall,
        "em": exact_match(rollout.get("answer") or "", answers),
        "f1": token_f1(rollout.get("answer") or "", answers),
        "hop": question.get("hop"),
        "question_id": question["question_id"],
    }
