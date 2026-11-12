#!/usr/bin/env python3
"""Encode texts with the existing local NV-Embed-v2 snapshot.

Run with the echorag interpreter from configs/pilot.json. CUDA_VISIBLE_DEVICES
must be set by the caller so this does not touch busy GPUs.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from transformers import AutoModel

sys.path.insert(0, os.path.dirname(__file__))
from common import abspath, load_json, read_jsonl  # noqa: E402


def encode_texts(model, texts, instruction: str, max_length: int, batch_size: int) -> np.ndarray:
    rows = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        with torch.no_grad():
            emb = model.encode(batch, instruction=instruction, max_length=max_length)
        if isinstance(emb, torch.Tensor):
            emb = emb.float().cpu().numpy()
        rows.append(np.asarray(emb, dtype=np.float32))
        print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    if not rows:
        return np.zeros((0, 1), dtype=np.float32)
    vecs = np.concatenate(rows, axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


def load_cache(path: str):
    if not os.path.exists(path):
        return [], np.zeros((0, 1), dtype=np.float32)
    data = np.load(path, allow_pickle=False)
    return list(data["keys"].tolist()), data["vecs"]


def save_cache(path: str, keys, vecs) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, keys=np.array(keys), vecs=vecs.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", choices=["corpus", "triples", "chunks", "queries"], required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    cfg = load_json(abspath("configs", "pilot.json"))
    path = cfg["embedding_model_path"]
    print("loading", path, flush=True)
    model = AutoModel.from_pretrained(
        path, trust_remote_code=True, local_files_only=True, torch_dtype=torch.float16,
    )
    model = model.cuda().eval()
    cache_path = abspath("data", "embeddings.npz")
    keys, vecs = load_cache(cache_path)
    index = {k: i for i, k in enumerate(keys)}
    todo_keys = []
    todo_texts = []

    if args.what == "corpus":
        docs = load_json(abspath("data", "documents.json"))
        for did, doc in docs.items():
            key = f"passage:{did}"
            if key not in index:
                todo_keys.append(key)
                todo_texts.append(f"{doc['title']}\n{doc['text']}")
        instruction = ""
    elif args.what == "queries":
        for q in load_json(abspath("data", "questions.json")):
            if not q["in_read_set"]:
                continue
            key = f"query:{q['qid']}"
            if key not in index:
                todo_keys.append(key)
                todo_texts.append(q["question"])
        instruction = cfg["query_instruction"]
    elif args.what == "triples":
        for row in read_jsonl(abspath("predictions", "triples.jsonl")):
            for triple in row.get("triples") or []:
                key = f"triple:{triple['triple_id']}"
                if key not in index:
                    todo_keys.append(key)
                    todo_texts.append(f"{triple['head']} {triple['relation']} {triple['tail']}")
        instruction = ""
    else:
        for row in read_jsonl(abspath("graph_patches", "accepted_chunks.jsonl")):
            key = f"chunk:{row['chunk_id']}"
            if key not in index:
                todo_keys.append(key)
                todo_texts.append(row["text"])
        for row in read_jsonl(abspath("graph_patches", "accepted_triples.jsonl")):
            key = f"triple:{row['triple_id']}"
            if key not in index:
                todo_keys.append(key)
                todo_texts.append(f"{row['head']} {row['relation']} {row['tail']}")
        instruction = ""

    print(f"what={args.what} missing={len(todo_keys)} cached={len(keys)}", flush=True)
    if todo_keys:
        new_vecs = encode_texts(model, todo_texts, instruction, cfg["max_embed_length"], args.batch_size)
        if len(keys) == 0:
            keys = todo_keys
            vecs = new_vecs
        else:
            if vecs.shape[1] != new_vecs.shape[1]:
                raise RuntimeError(f"dim mismatch {vecs.shape} vs {new_vecs.shape}")
            keys = list(keys) + todo_keys
            vecs = np.concatenate([vecs, new_vecs], axis=0)
        save_cache(cache_path, keys, vecs)
    print("cache", cache_path, "n", len(keys), flush=True)


if __name__ == "__main__":
    main()
