# legacy / algernon_musique：MuSiQue 阶段的代码与结果

来源：algernon `/data2-HDD-SATA-20T/nzq/TAG-experiment/{experiments, rsi_local_qwen_pilot}`（2026-11-12 拉取）。
汇总分析见 [experiments/01_musique_wiki/README.md](../../experiments/01_musique_wiki/README.md)。

| 子项目（新 → 旧） | 日期 | 问题 | 入口报告 | 代码 |
|---|---|---|---|---|
| [`musique_zero_api_grouping/`](musique_zero_api_grouping/) | 11-03 | 不调 API，哪种分组能让 gold 同组？ | [FINAL_REPORT.md](musique_zero_api_grouping/FINAL_REPORT.md) | `scripts/run_benchmark.py`、`configs/grid.json` |
| [`entity_doc_bipartite_audit/`](entity_doc_bipartite_audit/) | 11-03 | NER 二部图能否在抽三元组前分组？ | [FINAL_REPORT.md](entity_doc_bipartite_audit/FINAL_REPORT.md) | `scripts/run_audit.py`、`extra_diagnostics.py` |
| [`rsi_graphrag_v5_mechanism_search/`](rsi_graphrag_v5_mechanism_search/) | 11-01 → 11-02 | 哪种生成 + 验证机制能产生真正的跨段落理解？ | [READ_TIME_MECHANISM_SEARCH_FINAL.md](rsi_graphrag_v5_mechanism_search/reports/READ_TIME_MECHANISM_SEARCH_FINAL.md) | `scripts/`（generate / v52 / v53 / ppr_audit） |
| [`rsi_graphrag_v4/`](rsi_graphrag_v4/) | 10-29 → 10-30 | Luna 全量 G_raw + 分区 + 两套阅读协议 | [FINAL_EXPERIMENT_RESULTS.md](rsi_graphrag_v4/reports/FINAL_EXPERIMENT_RESULTS.md) | `scripts/`、`configs/`、`tests/` |
| [`rsi_local_qwen_pilot/`](rsi_local_qwen_pilot/) | 10-27 → 10-28 | 本地 Qwen3.5-9B 能否承担 Read-Time？ | [LOCAL_QWEN_READTIME_PILOT.md](rsi_local_qwen_pilot/reports/LOCAL_QWEN_READTIME_PILOT.md) | `src/`、`scripts/` |

## 各子项目的其它报告

| 子项目 | 报告 |
|---|---|
| zero_api_grouping | [DATA_AUDIT](musique_zero_api_grouping/DATA_AUDIT.md) · [METHODS](musique_zero_api_grouping/METHODS.md) · [GROUPING_RESULTS](musique_zero_api_grouping/GROUPING_RESULTS.md) · [GOLD_COVERAGE](musique_zero_api_grouping/GOLD_COVERAGE.md) · [COST_ANALYSIS](musique_zero_api_grouping/COST_ANALYSIS.md) · [CASE_STUDIES](musique_zero_api_grouping/CASE_STUDIES.md) |
| bipartite_audit | [DATA_SOURCE_AUDIT](entity_doc_bipartite_audit/DATA_SOURCE_AUDIT.md) · [BIPARTITE_GRAPH_REPORT](entity_doc_bipartite_audit/BIPARTITE_GRAPH_REPORT.md) · [HUB_ENTITY_ANALYSIS](entity_doc_bipartite_audit/HUB_ENTITY_ANALYSIS.md) · [GROUPING_COMPARISON](entity_doc_bipartite_audit/GROUPING_COMPARISON.md) · [CASE_STUDIES](entity_doc_bipartite_audit/CASE_STUDIES.md) |
| v5 | [MECHANISM_SEARCH_LOG](rsi_graphrag_v5_mechanism_search/reports/MECHANISM_SEARCH_LOG.md) · [V51_REFINEMENT](rsi_graphrag_v5_mechanism_search/reports/READ_TIME_V51_REFINEMENT.md) · [V52_RETRIEVAL_DIAGNOSTIC](rsi_graphrag_v5_mechanism_search/reports/READ_TIME_V52_RETRIEVAL_DIAGNOSTIC.md) · [V53_STRUCTURAL_COMPREHENSION](rsi_graphrag_v5_mechanism_search/reports/READ_TIME_V53_STRUCTURAL_COMPREHENSION.md) |
| v4 | [FINAL_PIPELINE_REPORT](rsi_graphrag_v4/reports/FINAL_PIPELINE_REPORT.md) · [GRAPH_CONSTRUCTION_AUDIT](rsi_graphrag_v4/reports/GRAPH_CONSTRUCTION_AUDIT.md) · [READ_TIME_PROTOCOL_PILOT](rsi_graphrag_v4/reports/READ_TIME_PROTOCOL_PILOT.md) · [READ_TIME_EVOLUTION_REPORT](rsi_graphrag_v4/reports/READ_TIME_EVOLUTION_REPORT.md) · [QUESTION_TIME_EVOLUTION_REPORT](rsi_graphrag_v4/reports/QUESTION_TIME_EVOLUTION_REPORT.md) · [SMOKE_READ_TIME_CASES](rsi_graphrag_v4/reports/SMOKE_READ_TIME_CASES.md) |
| local_qwen | [QUESTION_TIME_INTERFACE](rsi_local_qwen_pilot/reports/QUESTION_TIME_INTERFACE.md) · `results/summary.json` |

## 没有拉取的内容

- `rsi_graphrag_v4/artifacts/openie`（48MB，Luna OpenIE 原始输出，还在服务器上）。
- algernon 上更早的 CogitoRAG 相关目录：`case_study_musique`、`LogitRerank`、`qdm_analysis`、`significance_analysis_20260730`、`CogitoRAG-openreview`、`Multi-Hop`。需要时可以补拉。
- 各自的 `.venv` 和模型权重。
