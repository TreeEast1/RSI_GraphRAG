# Graph Construction Audit

## 冻结语料

- 问题 1000，语料行 11656，唯一 passage 11656。
- 划分: {'update': 500, 'heldout': 399, 'dev': 100, 'quarantine_token_identical': 1}。
- hop × role: {'2': {'update': 259, 'heldout': 206, 'dev': 52, 'quarantine_token_identical': 1}, '3': {'update': 158, 'heldout': 127, 'dev': 31}, '4': {'update': 83, 'heldout': 66, 'dev': 17}}。
- 金证据全部落在冻结语料中的问题: 1000 / 1000。
- 缺失的金 passage 槽位: 0 / 2648。
- 精确重复题组: 1，隔离出 held-out 的题: ['2hop__781079_9237']。
- 跨划分、Jaccard ≥ 0.9 的近重复: 5。
- 去重规则: Passage ids are HippoRAG chunk hashes of title + newline + text. Exact normalized-question duplicates that would cross into held-out are quarantined from held-out. Near-duplicates with token Jaccard >= 0.9 are recorded and left in their sampled split.

## 历史 OpenIE

本地 `openie_results_ner_gpt-4o-mini.json` 和 Llama-3.3-70B 文件都不是 GPT-5.6 Luna，也不是同一套 prompt 记录，因此没有当作本轮 G_raw。下面的结构数字只描述 gpt-4o-mini 历史抽取。

```json
{
  "source_label": "historical_gpt-4o-mini_openie_not_luna",
  "n_records": 11656,
  "official_idx_mismatches": 0,
  "n_relation_nodes": 101631,
  "n_relation_edges": 120624,
  "n_ner_entities": 66277,
  "n_isolated_ner_entities": 6005,
  "isolated_ner_ratio": 0.0906045837922658,
  "n_triples": 125846,
  "n_uncovered_passages": 1,
  "uncovered_passage_ratio": 8.579272477693891e-05,
  "relation_component_count": 5781,
  "relation_component_size_histogram": {
    "1": 3,
    "2-5": 5281,
    "6-20": 494,
    "21-100": 2,
    "101-1000": 0,
    "1000+": 1
  },
  "relation_largest_component_ratio": 0.8287923960208992,
  "relation_largest_component_nodes": 84231,
  "full_graph_definition": "relation edges plus passage-context edges; synonymy not included",
  "full_component_count": 321,
  "full_component_size_histogram": {
    "1": 1,
    "2-5": 25,
    "6-20": 290,
    "21-100": 4,
    "101-1000": 0,
    "1000+": 1
  },
  "full_largest_component_nodes": 110212,
  "top_predicates": [
    [
      "is",
      4640
    ],
    [
      "is located in",
      3971
    ],
    [
      "is a",
      2863
    ],
    [
      "is part of",
      1651
    ],
    [
      "is in",
      1644
    ],
    [
      "includes",
      1572
    ],
    [
      "located in",
      1251
    ],
    [
      "has",
      1231
    ],
    [
      "features",
      1171
    ],
    [
      "was",
      1057
    ],
    [
      "won",
      970
    ],
    [
      "born on",
      766
    ],
    [
      "released in",
      738
    ],
    [
      "was born in",
      681
    ],
    [
      "died on",
      628
    ],
    [
      "is also known as",
      561
    ],
    [
      "was born on",
      525
    ],
    [
      "written by",
      522
    ],
    [
      "is from",
      522
    ],
    [
      "was a",
      507
    ],
    [
      "plays",
      496
    ],
    [
      "starring",
      476
    ],
    [
      "played for",
      413
    ],
    [
      "contains",
      406
    ],
    [
      "directed by",
      406
    ],
    [
      "was released in",
      404
    ],
    [
      "died in",
      394
    ],
    [
      "appeared in",
      383
    ],
    [
      "stars",
      381
    ],
    [
      "produced by",
      368
    ]
  ],
  "n_clusters_after_leiden": 5781,
  "clusters_over_budget": 1,
  "leiden_skipped_components_over_8000_nodes": 1,
  "leiden_errors": 0
}
```

## 最大连通分量的 Leiden

最大关系分量 84231 个节点，Leiden 社区 133 个，最大社区 3864 （占该分量 0.04587384692096734）。
社区规模直方图: {'1000+': 23, '101-1000': 97, '21-100': 5, '6-20': 8}
这仍是 gpt-4o-mini 历史图，不是 Luna G_raw。

## 切分试点

This partition pilot uses the historical gpt-4o-mini OpenIE graph. It is not the Luna G_raw. stress_800 forces splits so overlap can be measured; budget_16000 is the actual reading budget.

- 展开的分量: 25。
- 大于 250 个节点、没有展开成阅读单元的分量: 1。
- 16000 token 预算: {"core_coverage_all": true, "mean_boundary_retention": {"0": 1.0, "1": 1.0, "2": 1.0}, "mean_units": {"0": 1.0, "1": 1.0, "2": 1.0}, "mean_overlap_mention_ratio": {"0": 0.0, "1": 0.0, "2": 0.0}, "max_input_tokens": {"0": 2198, "1": 2198, "2": 2198}}
- 800 token 压力切分: {"core_coverage_all": true, "mean_boundary_retention": {"0": 0.44, "1": 0.7650023310023311, "2": 0.7650023310023311}, "mean_units": {"0": 2.8, "1": 2.8, "2": 2.8}, "mean_overlap_mention_ratio": {"0": 0.0, "1": 0.13779237794556143, "2": 0.16654331442028308}, "max_input_tokens": {"0": 796, "1": 799, "2": 799}}

## Luna OpenIE

- 请求 deployment `gpt-5.6-luna`，API 返回模型 id: `gpt-5.6-luna-2026-07-09`。
- 探针 passage `chunk-0f19b42d483dc05cd5a62ea7dffa0864`，实体 15，三元组 19。
- 全量 OpenIE 投影调用数: 23312。
- 投影 prompt / completion tokens: 14301912.0 / 5583224.0。
- 美元: None。No Luna price was configured. Tokens are reported without a dollar conversion.
- 正式语料前缀 OpenIE: {"requested_new": 3, "written_new": 3, "cached_files": 4, "stopped_budget": false, "errors": [], "target_passages": 4, "full_corpus_passages": 11656, "note": "Corpus-order prefix of the frozen corpus. Not selected from question gold."}
- 机制试点 OpenIE: {"requested_new": 20, "written_new": 20, "cached_files": 24, "stopped_budget": false, "question_ids": ["2hop__77134_88165"], "n_passages_requested": 20, "n_paragraphs_missing_from_frozen_corpus": 0, "warning": "Mechanism pool is the paragraph union of a seeded update-split sample, including distractors. It is not the formal full-corpus G_raw. Held-out questions were not used to choose it."}
- 机制图: {"graph_hash": "7cf3ea5138959b6eb303480d5c19b46f5850c254b245bf01971d11d0e9ed20c6", "n_passages": 20, "n_entities": 219, "n_triples": 244, "n_context_edges": 236, "n_synonymy_edges": 0, "synonymy_status": "skipped_no_embedding", "uncovered_passages": 0, "hipporag_commit": "c0a8b6e27f340d7703d83c48c671d2cb5223d712", "prompt_hash": "be84b66e9f5f43bbfb95ae0f4e4d7caaf33248ec1b0ae51a88842c59975915fa", "model_name": "gpt-5.6-luna"}
- commit c0a8b6e27f340d7703d83c48c671d2cb5223d712，prompt hash be84b66e9f5f43bbfb95ae0f4e4d7caaf33248ec1b0ae51a88842c59975915fa。
