# RSI GraphRAG V4 Pipeline

正式顺序是：冻结全量语料，传统 OpenIE 建图，关系图划分阅读单元，理解层补丁，Question-Time 诊断，再用原 RAG 重跑验证。G_raw 只读。Held-out 不进入更新。

## 模块

- `corpus_manager` 冻结 passage id，并写出不含答案的语料。评测字段只在 `questions_eval.jsonl`。
- `hipporag_adapter` 调用本地 HippoRAG commit 的官方 NER / triple / QA prompt，生成不可变 G_raw。
- `graph_analyzer` 分别统计关系图和“关系边 + passage context 边”的连通分量，并用 Leiden 给超预算分量打社区标签。
- `graph_partitioner` 在分量内部递归切分，并加 hop overlap。Overlap 只复用实体 id。
- `comprehension_evolver` / `graph_patch_manager` 只写 G_comp。证据 span 必须是原文子串。
- `retrieval.py` 把理解 chunk 作为文档节点、理解边作为图边，走 fact linking + DPR reset + PPR。
- `rag.py` 的 `rag()` 不接收 gold。`evolution_analyzer` 在 rollout 之后才读 gold。
- `rag_verifier` 在隔离快照上重跑 `rag()`，再决定提交或回滚。

## 本次身份

- HippoRAG commit: c0a8b6e27f340d7703d83c48c671d2cb5223d712
- 请求的 LLM deployment: gpt-5.6-luna
- 语料 hash: 85652f89891c6f309e8f0d1dda8b3a9d1f67835e6664d20b44930e2010f239a7
- 新鲜 LLM 调用: 53

复现：`PYTHONPATH=src /data2-HDD-SATA-20T/nzq/env/hipporag/bin/python -m rsi_v4.experiment_runner --phase all`
