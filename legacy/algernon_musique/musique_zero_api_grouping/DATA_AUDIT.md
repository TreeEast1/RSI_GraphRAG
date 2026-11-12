# 数据复核

本轮没有调用远程 LLM 或 Embedding API，也没有改 G_raw、OpenIE 或向量文件。参数在 `configs/grid.json`。分组脚本是 `scripts/run_benchmark.py`。Gold 在 `artifacts/unsupervised_selection.json` 写完之后才读取。

## 语料和缓存

冻结语料仍是 `experiments/rsi_graphrag_v4/data/corpus.jsonl`：11,656 个 chunk，9,838 个 title。Chunk ID 不含原始文档 ID。Title 只是 Wikipedia 条目标题，647 个 title 对应多个 chunk。

主方法用 gpt-4o-mini 的 NER：`case_study_musique/traditional/graph/openie_results_ner_gpt-4o-mini.json`，11,656 篇全部对齐。Chunk 向量是同一目录下的 `chunk_embeddings/vdb_chunk.parquet`，11,656 行，hash 就是 chunk ID。抽查前 200 篇，向量库正文与语料 `content` 不一致的篇数是 0。向量在分组前做了 L2 归一化，相似度是余弦。

实体向量库没有用来建边。它的节点是三元组端点，属于 Triple Extraction 之后的信息。三元组关系也没有用来建边，只在上一轮审计里作为 B2 对照。

Luna NER 只用于四条实体方法的稳健性复跑。Embedding 分组不依赖 NER，两套缓存上结果相同。

Token 仍用 `tiktoken cl100k_base` 统计 `title + 换行 + text`。全库 1,348,035 token。每个组另计 200 token 模板，单独报告，不混进原文阅读倍数。

## 上一轮数字

用同一套 NER 规范化重算：

| 量 | 上一轮记录 | 本轮重算 |
| --- | ---: | ---: |
| B1 最大分量 | 11,422 | 11,422 |
| 原始分量完整 gold 同组 | 996 / 1,000 | 996 / 1,000 |
| Jaccard ≥ 0.20 完整 gold 同组 | 32 / 1,000 | 32 / 1,000 |

IDF Top-16 的上一轮数字是 483 / 1,000，本轮统一口径是 544 / 1,000。差在定义。上一轮要求某个 gold chunk 自己的 top-k 组盖住其余 gold。本轮按任务口径，任一组分量包含全部 gold 即算覆盖，所以重叠组会略高。互斥分组上两种口径一致，原始分量和 Jaccard 已经对上。

1,000 道有至少两个 gold chunk 的题里，977 道含跨 title 的 gold 对。Gold 下标跨度的中位数是 1,424；跨度 ≤ 15 的有 378 道。语料文件里的相邻关系本身就能解释一部分同组现象。
