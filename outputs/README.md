# outputs 索引：每个子目录对应哪个实验

所有结果都是从 devcloud `/group/40150/licochen/Memory/RSI_GraphRAG/outputs` 拉下来的原始产物（2026-11-12）。
没有入库的内容：`*.partial.jsonl`（断点续跑的中间文件）、`logs/hy4_calls*.jsonl`（逐调用日志）、`detectiveqa/chunks.jsonl`（26MB，可以重新生成）、图工作目录（parquet / pickle / sqlite）。

| 目录 | 实验 | 从哪里开始看 | 内容 |
|---|---|---|---|
| [`detectiveqa/`](detectiveqa/) | **04 DetectiveQA** | [PILOT_REPORT.md](detectiveqa/PILOT_REPORT.md) → [round2/ROUND2_REPORT.md](detectiveqa/round2/ROUND2_REPORT.md) | Round 1：数据审计、建图统计、prefix 安全检查、检索 / Reader pilot。`round2/`：Read-Time v3、关系通道、人工审计、逐题 case 表、LLM judge |
| [`qasper_readtime/`](qasper_readtime/) | **03 QASPER** | [INDEX.md](qasper_readtime/INDEX.md) → [FINAL_REPORT.md](qasper_readtime/FINAL_REPORT.md) | 数据集审计、read-time 产物（facts / insights / units）、M0–M6 检索、QA、机制追踪、case |
| [`reports/`](reports/) | **02 LoCoMo 阶段 A** | [START_HERE.md](reports/START_HERE.md) → [FINAL_TABLES.md](reports/FINAL_TABLES.md) | 各实验报告、汇总 JSON、`metrics.csv`、conv-30 case |
| [`qa/`](qa/) | 02 LoCoMo 阶段 A | — | 逐题预测，命名为 `<实验>_<变体>_<conv>.jsonl`（baselines / readtime / readtime_test / readtime_ablation / qtime / qtime_test_* / smoke） |
| [`groupings/`](groupings/) | 02 LoCoMo 阶段 A | — | 每段对话 9 种分组 × 2 预算的组划分 |
| [`comprehension/`](comprehension/) | 02 LoCoMo 阶段 A | — | 每段对话的联合抽取三元组和理解层关系（C2/C3/C5/C6） |
| [`skills/`](skills/) | 02 LoCoMo 阶段 A | — | 第一版 Skill Evolution 的冻结库（category → 查询后缀） |
| [`smoke/`](smoke/) | 02 LoCoMo 阶段 A | — | HY4 OpenIE 冒烟输出 |
| [`prompt_optimization_conv30/`](prompt_optimization_conv30/) | **02 LoCoMo 阶段 B** | [reports/FINAL_REPORT.md](prompt_optimization_conv30/reports/FINAL_REPORT.md) | `audit/`（prompt 审计、错误归因、官方口径对比）、`reader/`（P0–P8、oracle、官方对抗协议）、`comprehension/`（COMP-P0..P5 抽取、检索、no-rerank）、`joint/`（Graph × Reader）、`cases/` |
| [`question_evolution_conv30/`](question_evolution_conv30/) | **02 LoCoMo 阶段 C** | [README.md](question_evolution_conv30/README.md) → [FINAL_REPORT.md](question_evolution_conv30/FINAL_REPORT.md) | RSI 闭环的全部代码（`qecode/`）、技能记忆、快照、E0–E6 / R0–R5 预测、错误分析 |
| [`musique/`](musique/) | 01 MuSiQue（11-11 devcloud 准备） | `data_audit.json` | 数据审计、practice/dev/eval 切分、问题文件 |
| [`logs/`](logs/) | 全部 | — | 各阶段运行日志（不含逐调用日志） |

MuSiQue 早期（algernon）的结果在 [`../legacy/algernon_musique/`](../legacy/algernon_musique/README.md)。
