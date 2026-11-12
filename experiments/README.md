# 实验索引（最新在上）

每个数据集一个目录，README 结构统一：

| 章节 | 内容 |
|---|---|
| ① 数据集 | 规模、例题、是否有金标准、每题 gold 数、外部工作数字 |
| ② 实验方法 | 框架图、怎么做"理解" |
| ③ 各模块结果 | 每个模块的数字 |
| ④ Case 分析 | 典型正例和负例 |
| ⑤ 已完成 / 未完成 | 哪些还待验证 |
| ⑥ 问题归因 | 是数据集的问题，还是方法的问题 |
| ⑦ 复现 + 文件索引 | 命令和产物位置 |

| # | 时间 | 数据集 | 状态 | 核心问题 | 结论 | 目录 |
|---|---|---|---|---|---|---|
| **04** | **11-11 → 11-12** | **DetectiveQA**（英文侦探小说） | Pilot + Round 2 完成 | 长篇叙事里的跨文档线索能否被理解层召回？ | 图没过门槛；`recent` .805 = oracle；gold 标注不可靠；关系 precision 66.7% | [04_detectiveqa/](04_detectiveqa/README.md) |
| 03 | 11-10 | QASPER（科研论文） | 阶段 D 完成 | 理解层的 triple / unit 能否提升论文内跨段证据检索？ | 图 < dense；unit 只覆盖 58% 的题，等权 RRF 会把未覆盖题的正确证据挤掉 | [03_qasper/](03_qasper/README.md) |
| 02 | 11-05 → 11-10 | LoCoMo（长对话） | 阶段 A/B/C 完成 | 分组 / 联合抽取 / 理解层 / Question-Time / RSI 有没有用？ | 检索 ↑、QA 不动；Reader 是瓶颈；数据集太小、太饱和 | [02_locomo/](02_locomo/README.md) |
| 01 | 10-27 → 11-03 | MuSiQue / Wiki | 审计 + 机制搜索完成 | 阅读时能否把相关文档聚在一起，再做跨文档理解？ | 片段之间没有连续世界；分组无法让 gold 同组 | [01_musique_wiki/](01_musique_wiki/README.md) |

```mermaid
flowchart LR
  M["01 MuSiQue<br/>片段碎、没有连续世界"] -->|需要长篇连续文本| L["02 LoCoMo<br/>池太小、1 题 ≈ 1pp"]
  L -->|需要更多文档和题目| Q["03 QASPER<br/>段间关系弱、图 < dense"]
  Q -->|需要强跨文档依赖| D["04 DetectiveQA<br/>结构合适，但标注和切分要修"]
```

方法细节见 [docs/METHOD.md](../docs/METHOD.md)，外部数字见 [docs/REFERENCE_NUMBERS.md](../docs/REFERENCE_NUMBERS.md)。
