# 数据来源审计

本审计只读已有语料和 OpenIE 缓存。没有重跑 NER、三元组抽取、embedding 或 QA，也没有改原索引。

统计脚本是 `scripts/run_audit.py`，补充诊断是 `scripts/extra_diagnostics.py`。数字来自这两次运行的 `artifacts/` 与 `tables/`。

## 语料单位是 chunk，不是原始文档

冻结语料：

`experiments/rsi_graphrag_v4/data/corpus.jsonl`

来源清单 `corpus_manifest.json` 记录它由 HippoRAG 的 `musique_corpus.json` 冻结，11,656 行，11,656 个不重复 passage，重复行 0，`corpus_hash` 为 `85652f89891c6f309e8f0d1dda8b3a9d1f67835e6664d20b44930e2010f239a7`。

每个 chunk 的 ID 是 `title + 换行 + text` 的哈希，前缀 `chunk-`。哈希里没有原始文档 ID。能从语料行恢复的来源标签只有 `title`，也就是 MuSiQue / HippoRAG 使用的 Wikipedia 段落标题。

| 项目 | 数量 |
| --- | ---: |
| Chunk | 11,656 |
| 不同 title | 9,838 |
| 一个 title 对应多个 chunk | 647 个 title，共 2,465 个 chunk |
| 每个 title 的 chunk 数 | 中位数 1，P95 2，P99 6，最大 28 |

因此下面所有二部图的左侧节点都是 chunk。把同一 title 收成一个节点后的结果单独报告，不把 11,656 叫作文档数。Title 收缩之后巨型簇仍然在，所以这个结论不是切段造成的。

全文 token 用 `tiktoken cl100k_base` 估算，计的是 `title + 换行 + text`。全库 1,348,035 token，每篇中位数 101，P95 243，最大 599。这不是 Luna 或 gpt-4o-mini 的官方计费 tokenizer。

## 两套 OpenIE，分开统计

当前仓库里有两套对齐到同一批 chunk ID 的全量 OpenIE。

### A. RSI 项目里的 G_raw：Luna

目录：`experiments/rsi_graphrag_v4/artifacts/openie/luna/`

- 11,622 个 JSON，模型字段都是 `gpt-5.6-luna-2026-07-09`
- 34 个内容过滤失败写在 `luna_failed/`
- 与冻结语料逐篇比对，`content` 不一致的篇数是 0
- 每篇记录里 `entities` 是 NER 列表，`triples` 是随后的三元组
- 建图代码 `build_raw_graph` 会保留只出现在 NER 里的实体，并标 `ner_only`
- Passage–entity 边只来自三元组的主语和宾语
- 同义词边需要实体 embedding。这次 G_raw 的状态是 `skipped_no_embedding`，磁盘上没有 Luna 同义词边
- 也没有现成的 Entity → Chunk 倒排文件。本审计在内存里从 JSON 重建，不写回原目录

38 个 chunk 在 B1 和 B2 上都没有实体：34 个是过滤失败，另外 4 个 OpenIE 文件的实体和三元组都空。

### B. 已完成的 HippoRAG 2 索引：gpt-4o-mini

目录：`TAG-experiment/case_study_musique/traditional/graph/`

OpenIE 文件 `openie_results_ner_gpt-4o-mini.json` 有 11,656 篇，chunk ID 与冻结语料一一对应，没有多余 ID，正文与语料不一致的篇数是 0。

`comparison_index_stats.json` 记录的图规模：

| 对象 | 数量 |
| --- | ---: |
| Passage 节点 | 11,656 |
| Phrase 节点 | 102,194 |
| 抽取三元组 | 128,834 |
| 带 passage 端点的三元组计数 | 148,721 |
| 同义词三元组 | 1,331,727 |
| 图中总三元组 | 1,609,282 |
| 实体 embedding | 102,194 |
| Fact embedding | 128,834 |
| Chunk embedding | 11,656 |

官方建图代码把 phrase 节点限制在三元组主语和宾语上（`extract_entity_nodes(chunk_triples)`）。Passage–phrase 边由 `add_passage_edges` 连接这些三元组实体。Fact 是三元组字符串的 embedding，不是本审计使用的文档–实体边。同义词边由实体 embedding 的近邻生成，阈值和 top-k 在 HippoRAG 配置里，本审计不把它们算进连通分量。

用本审计的规范化计数，三元组端点实体是 102,050，passage–实体边是 148,656。和上面的 102,194、148,721 相差不到 0.2%，差在规范化：本审计做了 NFKC、去重音和空白折叠，HippoRAG 的 `text_processing` 只做小写并删去非字母数字。官方 OpenIE 里有 26,888 / 379,410 个原始字符串因此得到不同的 key。没有用 embedding 合并别名。

NER 唯一实体是 66,088，不是 102,194。Phrase 节点不等于 NER 全集。

Luna 与 gpt-4o-mini 的 fact / entity embedding 没有混用。同义词边、fact embedding 和 passage embedding 都没有进入二部图。

## 规范化与边的定义

实体 key：NFKC，去掉组合重音，小写，非字母数字改为空格，再折叠连续空白。同一 chunk 里同一个 key 只留一条边，mention count 是 NER 列表中的重复次数。gpt-4o-mini 的 116,079 条 NER 边里，mention count 大于 1 的只有 34 条。

B1 的右侧节点是 NER 实体。B2 的右侧节点是三元组主语和宾语。连通分量只用对应的 chunk–实体边。Fact 边、同义词边和相似度边没有加入。

未确认的同形异义保持同一个字符串节点，并在案例里单独标出。`Macmillan` 出版社和 `MacMillan` 姓氏就是这样被大小写折叠到一起的。
