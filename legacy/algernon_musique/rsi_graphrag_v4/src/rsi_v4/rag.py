"""Query-time RAG. Gold answers are not an argument of this function."""

from __future__ import annotations

import json
from typing import List, Sequence

from .hipporag_adapter import extract_json_object, prompt_manager
from .retrieval import GraphRetriever


def parse_answer(text: str) -> str:
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip().splitlines()[0].strip()
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def render_docs(docs: Sequence[dict]) -> str:
    blocks = []
    for doc in docs:
        if doc.get("kind") == "comprehension":
            blocks.append(f"Comprehension Note: {doc['doc_id']}\n{doc['content']}")
        else:
            title = doc.get("title") or ""
            text = doc.get("text") or doc.get("content") or ""
            blocks.append(f"Wikipedia Title: {title}\n{text}")
    return "\n\n".join(blocks)


def reader_messages(question: str, docs: Sequence[dict], answer_skills: Sequence[dict]) -> list:
    user = render_docs(docs) + f"\n\nQuestion: {question}\nThought: "
    messages = prompt_manager().render(name="rag_qa_musique", prompt_user=user)
    if answer_skills:
        policies = "\n\n".join(skill["executable_guidance"] for skill in answer_skills)
        messages[0]["content"] += "\n\nAnswer policies selected for this question:\n" + policies
    blob = json.dumps(messages)
    if "gold answer" in blob.lower() or "reference answer" in blob.lower():
        raise RuntimeError("reader prompt contains a gold-answer marker")
    return messages


def decomposition_messages(question: str, retrieval_skills: Sequence[dict]) -> list:
    policies = "\n\n".join(
        f"Trigger: {skill['trigger_condition']}\nDo: {skill['executable_guidance']}"
        for skill in retrieval_skills
    )
    return [
        {
            "role": "system",
            "content": (
                "Rewrite the question into at most three retrieval subqueries using the skills. "
                'Return JSON {"subqueries": ["..."]}. Do not answer the question.'
            ),
        },
        {"role": "user", "content": f"Skills:\n{policies}\n\nQuestion:\n{question}"},
    ]


def rag(
    question: str,
    retriever: GraphRetriever,
    client,
    retrieval_skills,
    answer_skills,
    *,
    qa_top_k: int = 5,
    reader_effort: str = "low",
) -> dict:
    selected_retrieval = retrieval_skills.select(question, k=2) if retrieval_skills is not None else []
    queries = [question]
    decomposition = None
    if selected_retrieval:
        decomposition_response = client.complete(
            stage="retrieval_skill_decompose",
            messages=decomposition_messages(question, selected_retrieval),
            max_completion_tokens=800,
            reasoning_effort=reader_effort,
        )
        decomposition = decomposition_response.text
        try:
            payload = extract_json_object(decomposition_response.text)
            parsed = [item.strip() for item in payload.get("subqueries") or [] if isinstance(item, str) and item.strip()]
            if parsed:
                queries = parsed[:3]
        except Exception:
            queries = [question]
    retrieval = retriever.retrieve(queries, top_k=max(qa_top_k, 10))
    reader_docs = retrieval["docs"][:qa_top_k]
    selected_answer = answer_skills.select(question, k=2) if answer_skills is not None else []
    messages = reader_messages(question, reader_docs, selected_answer)
    response = client.complete(
        stage="qa_reader",
        messages=messages,
        max_completion_tokens=1500,
        reasoning_effort=reader_effort,
    )
    return {
        "question": question,
        "queries": queries,
        "decomposition": decomposition,
        "retrieval_skill_ids": [skill["skill_id"] for skill in selected_retrieval],
        "answer_skill_ids": [skill["skill_id"] for skill in selected_answer],
        "retrieved_docs": [
            {
                "doc_id": doc["doc_id"],
                "kind": doc["kind"],
                "score": doc["score"],
                "title": doc.get("title", ""),
            }
            for doc in retrieval["docs"][:10]
        ],
        "reader_doc_ids": [doc["doc_id"] for doc in reader_docs],
        "comprehension_ids_in_top10": retrieval["comprehension_ids_in_top10"],
        "retrieval_traces": retrieval["traces"],
        "reader_output": response.text,
        "answer": parse_answer(response.text),
        "reader_metadata": {
            "prompt_tokens": response.metadata.get("prompt_tokens"),
            "completion_tokens": response.metadata.get("completion_tokens"),
            "response_model": response.metadata.get("response_model"),
            "cache_hit": response.cache_hit,
        },
    }
