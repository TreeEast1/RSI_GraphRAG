#!/usr/bin/env python3
"""Traditional graph, fixed-window read-time patches, and the paired QA eval.

Baseline and evolved share documents, NV-Embed vectors, top-k, the raw question,
and the reader prompt. Gold answers are never placed in a read-time prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    abspath, append_jsonl, em_f1, load_json, load_llm, parse_json_object, read_jsonl, save_json,
)

EXTRACT_SYSTEM = """You extract a knowledge graph from one passage.
Return only a JSON object:
{"entities": ["..."], "triples": [{"head": "...", "relation": "...", "tail": "..."}]}
Use only facts stated in the passage. Do not add outside knowledge.
Keep dates, numbers, and names as written. Resolve pronouns when the passage is clear."""

READ_SYSTEM = """You revise a knowledge graph after reading a window of documents.
Return only a JSON object:
{"patches": [
  {"operation": "ADD"|"MODIFY"|"DEACTIVATE",
   "target_id": "existing id or empty for ADD",
   "head": "...", "relation": "...", "tail": "...",
   "source_document_ids": ["doc-..."],
   "evidence": "verbatim span from one cited document",
   "comprehension_text": "optional one-sentence implication, or empty"}
],
 "comprehension_chunks": [
  {"text": "one or two sentences of new cross-fact understanding",
   "source_document_ids": ["doc-..."],
   "evidence": "verbatim spans that support the chunk"}
]}
Rules:
- Propose an ADD only when the cited documents support both entities and the relation.
- A useful ADD may compose facts across the window. It still must not introduce names or numbers absent from the cited documents.
- DEACTIVATE only when a cited document contradicts an existing triple.
- Do not restate triples that are already present.
- Do not answer an external question. No gold answer is provided.
- Prefer a small set of high-value updates. Empty lists are acceptable."""

QA_SYSTEM = """Answer the question using only the passages.
Return only a JSON object: {"answer": "..."}
Use a short answer span. If the passages are insufficient, answer "unknown"."""


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()


def triple_key(head: str, relation: str, tail: str) -> str:
    return " | ".join(norm(x) for x in (head, relation, tail))


def _haystack(docs: dict, source_ids: list) -> str:
    return norm("\n".join(docs[sid]["title"] + "\n" + docs[sid]["text"] for sid in source_ids if sid in docs))


def in_docs(span: str, docs: dict, source_ids: list) -> bool:
    needle = norm(span)
    if len(needle) < 2:
        return False
    hay = _haystack(docs, source_ids)
    if needle in hay:
        return True
    # Quotes often differ by a few tokens. Accept a long span when most of its words are copied.
    words = [w for w in re.findall(r"[a-z0-9']+", needle) if len(w) > 3]
    if len(words) < 4:
        return False
    return sum(w in hay for w in words) / len(words) >= 0.75


def entity_in_docs(entity: str, docs: dict, source_ids: list) -> bool:
    if in_docs(entity, docs, source_ids) and norm(entity) in _haystack(docs, source_ids):
        return True
    toks = [t for t in re.findall(r"[A-Za-z0-9]+", entity or "") if len(t) > 2]
    if not toks:
        return False
    hay = _haystack(docs, source_ids)
    return all(t.casefold() in hay for t in toks)


def content_overlap(text: str, docs: dict, source_ids: list) -> float:
    words = [w for w in re.findall(r"[A-Za-z0-9']+", text or "") if len(w) > 3]
    if not words:
        return 0.0
    hay = norm("\n".join(docs[sid]["text"] for sid in source_ids if sid in docs))
    hit = sum(1 for w in words if w.casefold() in hay)
    return hit / len(words)


def load_vectors():
    data = np.load(abspath("data", "embeddings.npz"), allow_pickle=False)
    keys = list(data["keys"].tolist())
    vecs = data["vecs"].astype(np.float32)
    return {k: vecs[i] for i, k in enumerate(keys)}


def cosine(a, b) -> float:
    return float(np.dot(a, b))


def traditional_graph(docs: dict) -> dict:
    triples = []
    for row in read_jsonl(abspath("predictions", "triples.jsonl")):
        if row.get("error"):
            continue
        for t in row.get("triples") or []:
            triples.append(t)
    return {"triples": triples, "docs": docs}


def validate_patch(patch: dict, docs: dict, active: dict) -> dict:
    op = str(patch.get("operation") or "").upper()
    sources = [s for s in (patch.get("source_document_ids") or []) if s in docs]
    evidence = str(patch.get("evidence") or "")
    head = str(patch.get("head") or "").strip()
    relation = str(patch.get("relation") or "").strip()
    tail = str(patch.get("tail") or "").strip()
    out = dict(patch)
    out["operation"] = op
    out["source_document_ids"] = sources
    out["validation_status"] = "rejected"
    out["reject_reason"] = ""
    if op not in {"ADD", "MODIFY", "DEACTIVATE"}:
        out["reject_reason"] = "bad_operation"
        return out
    if not sources:
        out["reject_reason"] = "missing_source"
        return out
    if len(norm(evidence)) < 12 or not in_docs(evidence, docs, sources):
        out["reject_reason"] = "evidence_not_in_source"
        return out
    if op == "DEACTIVATE":
        target = str(patch.get("target_id") or "")
        if target not in active:
            out["reject_reason"] = "unknown_target"
            return out
        out["before"] = {k: active[target][k] for k in ("head", "relation", "tail")}
        out["validation_status"] = "accepted"
        return out
    if not (head and relation and tail):
        out["reject_reason"] = "empty_triple"
        return out
    if not entity_in_docs(head, docs, sources) or not entity_in_docs(tail, docs, sources):
        out["reject_reason"] = "entity_not_in_source"
        return out
    key = triple_key(head, relation, tail)
    if any(triple_key(t["head"], t["relation"], t["tail"]) == key for t in active.values()):
        out["reject_reason"] = "duplicate"
        return out
    if op == "MODIFY":
        target = str(patch.get("target_id") or "")
        if target not in active:
            out["reject_reason"] = "unknown_target"
            return out
        out["before"] = {k: active[target][k] for k in ("head", "relation", "tail")}
    out["after"] = {"head": head, "relation": relation, "tail": tail}
    out["validation_status"] = "accepted"
    return out


def validate_chunk(chunk: dict, docs: dict) -> dict:
    sources = [s for s in (chunk.get("source_document_ids") or []) if s in docs]
    text = str(chunk.get("text") or "").strip()
    evidence = str(chunk.get("evidence") or "")
    out = {"text": text, "source_document_ids": sources, "evidence": evidence,
           "validation_status": "rejected", "reject_reason": ""}
    if not sources:
        out["reject_reason"] = "missing_source"
        return out
    if not (40 <= len(text) <= 800):
        out["reject_reason"] = "bad_length"
        return out
    if len(norm(evidence)) < 12 or not in_docs(evidence, docs, sources):
        out["reject_reason"] = "evidence_not_in_source"
        return out
    overlap = content_overlap(text, docs, sources)
    out["overlap"] = overlap
    if overlap < 0.45:
        out["reject_reason"] = "low_lexical_overlap"
        return out
    out["validation_status"] = "accepted"
    return out


def stage_extract(client, docs, read_ids) -> None:
    import fcntl
    lock_path = abspath("predictions", "extract.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    lockf = open(lock_path, "a")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("extract lock held; not starting a second writer", flush=True)
        return
    path = abspath("predictions", "triples.jsonl")
    done = {r["doc_id"] for r in read_jsonl(path)}
    for i, did in enumerate(read_ids):
        if did in done:
            continue
        doc = docs[did]
        passage = f"Document {did}\nTitle: {doc['title']}\n{doc['text']}"
        try:
            resp = client.chat([
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": passage},
            ], max_output_tokens=700)
            obj = parse_json_object(resp["visible_text"])
            triples = []
            if obj:
                raw = obj.get("triples") or []
                for item in raw:
                    if isinstance(item, dict):
                        head, relation, tail = item.get("head"), item.get("relation"), item.get("tail")
                    elif isinstance(item, list) and len(item) >= 3:
                        head, relation, tail = item[0], item[1], item[2]
                    else:
                        continue
                    head, relation, tail = str(head or "").strip(), str(relation or "").strip(), str(tail or "").strip()
                    if not (head and relation and tail):
                        continue
                    if not entity_in_docs(head, docs, [did]) or not entity_in_docs(tail, docs, [did]):
                        status = "rejected"
                    else:
                        status = "accepted"
                    tid = "t-" + hashlib.sha1(f"{did}|{head}|{relation}|{tail}".encode()).hexdigest()[:12]
                    triples.append({
                        "triple_id": tid, "head": head, "relation": relation, "tail": tail,
                        "source_ids": [did], "status": status, "origin": "traditional",
                    })
            accepted = [t for t in triples if t["status"] == "accepted"]
            append_jsonl(path, {
                "doc_id": did, "json_ok": obj is not None, "triples": accepted,
                "n_rejected_unsupported": sum(1 for t in triples if t["status"] == "rejected"),
                "latency_s": resp["latency_s"], "prompt_tokens": resp["prompt_tokens"],
                "completion_tokens": resp["completion_tokens"],
            })
            print(f"extract {i+1}/{len(read_ids)} {did} json={obj is not None} kept={len(accepted)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            append_jsonl(path, {"doc_id": did, "error": str(exc)[:500], "triples": []})
            print(f"extract ERROR {did}: {exc}", flush=True)


def stage_readtime(client, docs, read_ids, window: int) -> None:
    path = abspath("graph_patches", "readtime.jsonl")
    done_windows = {r["window_index"] for r in read_jsonl(path)}
    by_doc = {}
    for row in read_jsonl(abspath("predictions", "triples.jsonl")):
        by_doc[row["doc_id"]] = [t for t in row.get("triples") or [] if t.get("status") == "accepted"]
    active = {}
    for did in read_ids:
        for t in by_doc.get(did, []):
            active[t["triple_id"]] = t
    n_windows = (len(read_ids) + window - 1) // window
    accepted_triples_path = abspath("graph_patches", "accepted_triples.jsonl")
    accepted_chunks_path = abspath("graph_patches", "accepted_chunks.jsonl")
    # Rebuild accepted side files from the patch log so resume does not duplicate them.
    if done_windows:
        open(accepted_triples_path, "w").close()
        open(accepted_chunks_path, "w").close()
    for w in range(n_windows):
        window_ids = read_ids[w * window:(w + 1) * window]
        if w in done_windows:
            for rec in read_jsonl(path):
                if rec["window_index"] != w:
                    continue
                _apply_saved(rec, active, accepted_triples_path, accepted_chunks_path)
            continue
        shown = []
        for did in window_ids:
            for t in by_doc.get(did, []):
                if t["triple_id"] in active:
                    shown.append(t)
        # A small historical neighborhood: triples sharing an entity string with this window.
        window_ents = set()
        for t in shown:
            window_ents.add(norm(t["head"]))
            window_ents.add(norm(t["tail"]))
        history = []
        for t in active.values():
            if t["triple_id"] in {x["triple_id"] for x in shown}:
                continue
            if norm(t["head"]) in window_ents or norm(t["tail"]) in window_ents:
                history.append(t)
            if len(history) >= 24:
                break
        doc_block = []
        for did in window_ids:
            doc = docs[did]
            doc_block.append(f"[{did}] {doc['title']}\n{doc['text']}")
        triple_block = []
        for t in shown + history:
            triple_block.append({
                "target_id": t["triple_id"], "head": t["head"], "relation": t["relation"],
                "tail": t["tail"], "source_ids": t["source_ids"],
            })
        user = "Documents:\n" + "\n\n".join(doc_block) + "\n\nCurrent triples:\n" + _dumps(triple_block)
        try:
            resp = client.chat([
                {"role": "system", "content": READ_SYSTEM},
                {"role": "user", "content": user},
            ], max_output_tokens=1100)
            obj = parse_json_object(resp["visible_text"]) or {}
        except Exception as exc:  # noqa: BLE001
            append_jsonl(path, {"window_index": w, "doc_ids": window_ids, "error": str(exc)[:500],
                                "patches": [], "chunks": []})
            print(f"readtime ERROR window {w}: {exc}", flush=True)
            continue
        raw_patches = obj.get("patches") if isinstance(obj.get("patches"), list) else []
        raw_chunks = obj.get("comprehension_chunks") if isinstance(obj.get("comprehension_chunks"), list) else []
        checked_patches = []
        for patch in raw_patches:
            if not isinstance(patch, dict):
                continue
            # Restrict citations to the current window so a patch cannot claim unseen documents.
            cited = [s for s in (patch.get("source_document_ids") or []) if s in window_ids]
            patch = dict(patch)
            patch["source_document_ids"] = cited
            checked = validate_patch(patch, docs, active)
            checked_patches.append(checked)
            if checked["validation_status"] != "accepted":
                continue
            if checked["operation"] == "DEACTIVATE":
                active.pop(checked["target_id"], None)
            elif checked["operation"] == "MODIFY":
                old = active.pop(checked["target_id"], None)
                if old:
                    new_id = "t-" + hashlib.sha1(
                        f"mod|{w}|{checked['after']['head']}|{checked['after']['relation']}|{checked['after']['tail']}".encode()
                    ).hexdigest()[:12]
                    new_t = dict(old)
                    new_t.update(checked["after"])
                    new_t["triple_id"] = new_id
                    new_t["origin"] = "readtime_modify"
                    new_t["source_ids"] = checked["source_document_ids"]
                    new_t["replaces"] = checked["target_id"]
                    active[new_id] = new_t
                    append_jsonl(accepted_triples_path, new_t)
            else:
                tid = "t-" + hashlib.sha1(
                    f"add|{w}|{checked['after']['head']}|{checked['after']['relation']}|{checked['after']['tail']}".encode()
                ).hexdigest()[:12]
                new_t = {
                    "triple_id": tid, **checked["after"], "source_ids": checked["source_document_ids"],
                    "status": "accepted", "origin": "readtime_add",
                }
                active[tid] = new_t
                append_jsonl(accepted_triples_path, new_t)
        checked_chunks = []
        for chunk in raw_chunks:
            if not isinstance(chunk, dict):
                continue
            chunk = dict(chunk)
            chunk["source_document_ids"] = [s for s in (chunk.get("source_document_ids") or []) if s in window_ids]
            checked = validate_chunk(chunk, docs)
            if checked["validation_status"] == "accepted":
                checked["chunk_id"] = "c-" + hashlib.sha1(checked["text"].encode()).hexdigest()[:12]
                append_jsonl(accepted_chunks_path, checked)
            checked_chunks.append(checked)
        rec = {
            "window_index": w, "doc_ids": window_ids, "json_ok": bool(obj),
            "n_proposed": len(raw_patches), "n_accepted": sum(p["validation_status"] == "accepted" for p in checked_patches),
            "n_rejected": sum(p["validation_status"] != "accepted" for p in checked_patches),
            "n_chunks_accepted": sum(c["validation_status"] == "accepted" for c in checked_chunks),
            "patches": checked_patches, "chunks": checked_chunks,
            "latency_s": resp["latency_s"], "prompt_tokens": resp["prompt_tokens"],
            "completion_tokens": resp["completion_tokens"],
        }
        append_jsonl(path, rec)
        print(f"window {w+1}/{n_windows} accepted={rec['n_accepted']} rejected={rec['n_rejected']} chunks={rec['n_chunks_accepted']}", flush=True)


def _apply_saved(rec, active, triples_path, chunks_path) -> None:
    for checked in rec.get("patches") or []:
        if checked.get("validation_status") != "accepted":
            continue
        if checked["operation"] == "DEACTIVATE":
            active.pop(checked.get("target_id"), None)
        elif checked["operation"] == "ADD" and checked.get("after"):
            tid = "t-" + hashlib.sha1(
                f"add|{rec['window_index']}|{checked['after']['head']}|{checked['after']['relation']}|{checked['after']['tail']}".encode()
            ).hexdigest()[:12]
            new_t = {"triple_id": tid, **checked["after"], "source_ids": checked["source_document_ids"],
                     "status": "accepted", "origin": "readtime_add"}
            active[tid] = new_t
            append_jsonl(triples_path, new_t)
        elif checked["operation"] == "MODIFY" and checked.get("after"):
            old = active.pop(checked.get("target_id"), None)
            if old:
                new_id = "t-" + hashlib.sha1(
                    f"mod|{rec['window_index']}|{checked['after']['head']}|{checked['after']['relation']}|{checked['after']['tail']}".encode()
                ).hexdigest()[:12]
                new_t = dict(old)
                new_t.update(checked["after"])
                new_t["triple_id"] = new_id
                new_t["origin"] = "readtime_modify"
                new_t["source_ids"] = checked["source_document_ids"]
                new_t["replaces"] = checked["target_id"]
                active[new_id] = new_t
                append_jsonl(triples_path, new_t)
    for checked in rec.get("chunks") or []:
        if checked.get("validation_status") == "accepted":
            if "chunk_id" not in checked:
                checked["chunk_id"] = "c-" + hashlib.sha1(checked["text"].encode()).hexdigest()[:12]
            append_jsonl(chunks_path, checked)


def _dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)


def rank_passages(question: str, qvec, docs: dict, triples: list, chunks: list, vectors, top_k: int):
    scores = {}
    reasons = {}
    for did in docs:
        key = f"passage:{did}"
        if key not in vectors or qvec is None:
            continue
        score = cosine(qvec, vectors[key])
        scores[did] = score
        reasons[did] = {"kind": "passage", "score": score, "id": did}
    for t in triples:
        vec = vectors.get(f"triple:{t['triple_id']}")
        if vec is None:
            continue
        score = cosine(qvec, vec)
        for did in t.get("source_ids") or []:
            if did not in scores:
                continue
            if score > scores[did]:
                scores[did] = score
                reasons[did] = {"kind": "triple", "score": score, "id": t["triple_id"],
                                "text": f"{t['head']} {t['relation']} {t['tail']}", "origin": t.get("origin")}
    for c in chunks:
        vec = vectors.get(f"chunk:{c['chunk_id']}")
        if vec is None:
            continue
        score = cosine(qvec, vec)
        for did in c.get("source_document_ids") or []:
            if did not in scores:
                continue
            if score > scores[did]:
                scores[did] = score
                reasons[did] = {"kind": "chunk", "score": score, "id": c["chunk_id"], "text": c.get("text", "")[:240]}
    ranked = sorted(scores, key=lambda d: scores[d], reverse=True)
    return ranked[:top_k], {did: reasons[did] for did in ranked[:top_k]}


def recall_at(ranked_ids, support_ids, ks) -> dict:
    support = set(support_ids)
    out = {}
    for k in ks:
        hit = len(support.intersection(ranked_ids[:k]))
        out[f"recall@{k}"] = hit / len(support) if support else 0.0
        out[f"hit@{k}"] = float(hit > 0)
    return out


def stage_eval(client, name: str, questions, docs, triples, chunks, ks, top_k: int) -> None:
    vectors = load_vectors()
    path = abspath("results", f"{name}.jsonl")
    done = {r["qid"] for r in read_jsonl(path)}
    for i, q in enumerate(questions):
        if q["qid"] in done:
            continue
        qvec = vectors.get(f"query:{q['qid']}")
        ranked, reasons = rank_passages(q["question"], qvec, docs, triples, chunks, vectors, max(ks))
        metrics = recall_at(ranked, q["support_ids"], ks)
        retrieved = set(ranked[:top_k])
        context_parts = [
            f"[{did}] {docs[did]['title']}\n{docs[did]['text']}" for did in ranked[:top_k] if did in docs
        ]
        # Baseline passes no chunks, so this slot stays empty and the prompt text is unchanged.
        # Evolved may append comprehension notes tied to the retrieved passages.
        shown_chunks = []
        for chunk in chunks:
            if retrieved.intersection(chunk.get("source_document_ids") or []):
                shown_chunks.append(chunk)
        for chunk in shown_chunks[:4]:
            context_parts.append(f"[comprehension {chunk.get('chunk_id')}] {chunk.get('text')}")
        context = "\n\n".join(context_parts)
        try:
            resp = client.chat([
                {"role": "system", "content": QA_SYSTEM},
                {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {q['question']}"},
            ], max_output_tokens=180)
            obj = parse_json_object(resp["visible_text"]) or {}
            answer = str(obj.get("answer") or resp["visible_text"])
            usage = {"latency_s": resp["latency_s"], "prompt_tokens": resp["prompt_tokens"],
                     "completion_tokens": resp["completion_tokens"], "json_ok": "answer" in obj}
        except Exception as exc:  # noqa: BLE001
            answer = ""
            usage = {"error": str(exc)[:400], "latency_s": 0, "prompt_tokens": 0, "completion_tokens": 0, "json_ok": False}
        scores = em_f1(answer, [q["answer"], *q["answer_aliases"]])
        row = {
            "qid": q["qid"], "question": q["question"], "gold": q["answer"], "prediction": answer,
            "support_ids": q["support_ids"], "ranked_ids": ranked, "reasons": reasons,
            "shown_chunk_ids": [c.get("chunk_id") for c in shown_chunks[:4]],
            **metrics, **scores, **usage,
        }
        append_jsonl(path, row)
        print(f"{name} {i+1}/{len(questions)} R5={metrics.get('recall@5', 0):.2f} em={scores['em']} f1={scores['f1']:.2f}", flush=True)


def active_traditional():
    triples = []
    for row in read_jsonl(abspath("predictions", "triples.jsonl")):
        triples.extend(t for t in row.get("triples") or [] if t.get("status") == "accepted")
    return triples


def active_evolved(base_triples):
    active = {t["triple_id"]: dict(t) for t in base_triples}
    # Replays are already materialized in accepted_triples, but deactivations live only in the log.
    for rec in read_jsonl(abspath("graph_patches", "readtime.jsonl")):
        for patch in rec.get("patches") or []:
            if patch.get("validation_status") != "accepted":
                continue
            if patch["operation"] == "DEACTIVATE":
                active.pop(patch.get("target_id"), None)
            elif patch["operation"] == "MODIFY" and patch.get("after"):
                old = active.pop(patch.get("target_id"), None)
                if old:
                    new_id = "t-" + hashlib.sha1(
                        f"mod|{rec['window_index']}|{patch['after']['head']}|{patch['after']['relation']}|{patch['after']['tail']}".encode()
                    ).hexdigest()[:12]
                    new_t = dict(old)
                    new_t.update(patch["after"])
                    new_t["triple_id"] = new_id
                    new_t["source_ids"] = patch.get("source_document_ids") or old.get("source_ids")
                    new_t["origin"] = "readtime_modify"
                    new_t["replaces"] = patch.get("target_id")
                    active[new_id] = new_t
            elif patch["operation"] == "ADD" and patch.get("after"):
                tid = "t-" + hashlib.sha1(
                    f"add|{rec['window_index']}|{patch['after']['head']}|{patch['after']['relation']}|{patch['after']['tail']}".encode()
                ).hexdigest()[:12]
                active[tid] = {
                    "triple_id": tid, **patch["after"],
                    "source_ids": patch.get("source_document_ids") or [],
                    "status": "accepted", "origin": "readtime_add",
                }
    chunks = [c for c in read_jsonl(abspath("graph_patches", "accepted_chunks.jsonl")) if c.get("validation_status") == "accepted"]
    # accepted_chunks may duplicate on resume; unique by chunk_id
    uniq = {}
    for c in chunks:
        uniq[c["chunk_id"]] = c
    return list(active.values()), list(uniq.values())


def stage_summary(ks) -> None:
    base = read_jsonl(abspath("results", "baseline.jsonl"))
    evo = read_jsonl(abspath("results", "evolved.jsonl"))
    by_q = {r["qid"]: r for r in evo}

    def avg(rows, key):
        return sum(r.get(key) or 0 for r in rows) / len(rows) if rows else 0.0

    cases = []
    for b in base:
        e = by_q.get(b["qid"])
        if not e:
            continue
        delta = (e.get("recall@5") or 0) - (b.get("recall@5") or 0)
        if abs(delta) < 1e-9 and abs((e.get("f1") or 0) - (b.get("f1") or 0)) < 1e-9:
            continue
        cases.append({
            "qid": b["qid"], "question": b["question"], "gold": b["gold"],
            "baseline_prediction": b.get("prediction"), "evolved_prediction": e.get("prediction"),
            "baseline_recall@5": b.get("recall@5"), "evolved_recall@5": e.get("recall@5"),
            "baseline_f1": b.get("f1"), "evolved_f1": e.get("f1"),
            "baseline_ranked": b.get("ranked_ids"), "evolved_ranked": e.get("ranked_ids"),
            "evolved_reasons": e.get("reasons"),
            "recall_delta@5": delta,
        })
    cases.sort(key=lambda r: r["recall_delta@5"], reverse=True)
    patches = read_jsonl(abspath("graph_patches", "readtime.jsonl"))
    extract_rows = read_jsonl(abspath("predictions", "triples.jsonl"))
    summary = {
        "n_baseline": len(base),
        "n_evolved": len(evo),
        "baseline": {f"recall@{k}": avg(base, f"recall@{k}") for k in ks} | {
            "hit@5": avg(base, "hit@5"), "em": avg(base, "em"), "f1": avg(base, "f1"),
            "qa_latency_s": sum(r.get("latency_s") or 0 for r in base),
            "qa_prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in base),
            "qa_completion_tokens": sum(r.get("completion_tokens") or 0 for r in base),
        },
        "evolved": {f"recall@{k}": avg(evo, f"recall@{k}") for k in ks} | {
            "hit@5": avg(evo, "hit@5"), "em": avg(evo, "em"), "f1": avg(evo, "f1"),
            "qa_latency_s": sum(r.get("latency_s") or 0 for r in evo),
            "qa_prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in evo),
            "qa_completion_tokens": sum(r.get("completion_tokens") or 0 for r in evo),
        },
        "traditional_triples_accepted": sum(len(r.get("triples") or []) for r in extract_rows),
        "traditional_triples_rejected_unsupported": sum(r.get("n_rejected_unsupported") or 0 for r in extract_rows),
        "extract_prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in extract_rows),
        "extract_completion_tokens": sum(r.get("completion_tokens") or 0 for r in extract_rows),
        "extract_latency_s": sum(r.get("latency_s") or 0 for r in extract_rows),
        "patches_proposed": sum(r.get("n_proposed") or 0 for r in patches),
        "patches_accepted": sum(r.get("n_accepted") or 0 for r in patches),
        "patches_rejected": sum(r.get("n_rejected") or 0 for r in patches),
        "chunks_accepted": sum(r.get("n_chunks_accepted") or 0 for r in patches),
        "readtime_latency_s": sum(r.get("latency_s") or 0 for r in patches),
        "readtime_prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in patches),
        "readtime_completion_tokens": sum(r.get("completion_tokens") or 0 for r in patches),
        "n_recall_improved": sum(1 for c in cases if c["recall_delta@5"] > 0),
        "n_recall_regressed": sum(1 for c in cases if c["recall_delta@5"] < 0),
        "improved_cases": [c for c in cases if c["recall_delta@5"] > 0][:8],
        "regressed_cases": [c for c in cases if c["recall_delta@5"] < 0][:8],
        "answer_changed_cases": [c for c in cases if (c["baseline_f1"] != c["evolved_f1"])][:8],
    }
    save_json(abspath("results", "summary.json"), summary)
    print(_dumps(summary)[:4000])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["extract", "readtime", "baseline", "evolved", "summary"])
    parser.add_argument("--model-config", default=abspath("configs", "model.json"))
    args = parser.parse_args()
    pilot = load_json(abspath("configs", "pilot.json"))
    docs = load_json(abspath("data", "documents.json"))
    read_ids = load_json(abspath("data", "read_doc_ids.json"))
    questions = [q for q in load_json(abspath("data", "questions.json")) if q["in_read_set"]]
    if args.stage == "summary":
        stage_summary(pilot["recall_ks"])
        return
    client = load_llm(args.model_config)
    if args.stage == "extract":
        stage_extract(client, docs, read_ids)
    elif args.stage == "readtime":
        stage_readtime(client, docs, read_ids, pilot["window_size"])
    elif args.stage == "baseline":
        stage_eval(client, "baseline", questions, docs, active_traditional(), [], pilot["recall_ks"], pilot["retrieval_top_k"])
    else:
        base = active_traditional()
        triples, chunks = active_evolved(base)
        stage_eval(client, "evolved", questions, docs, triples, chunks, pilot["recall_ks"], pilot["retrieval_top_k"])


if __name__ == "__main__":
    main()
