# DetectiveQA 主线：审计结论、数据集选择与 GPU 实验计划

日期：2026-11-11  
状态：CPU 机上已完成数据审计、切块、划分、防泄漏代码和单元测试；**尚未建图，也未运行任何 GPU/LLM 实验**。

## 1. 结论

主数据集改为 **DetectiveQA 英文版**，不再使用 MuSiQue。

原因不是它“更难”，而是它的结构正好对应我们想证明的机制：

- 同一部小说内部存在大量远距离线索，答案需要组合人物、动机、时间、证词、矛盾和物证。
- 每题都有人工或 AI 辅助标注的线索段落号，可以计算 gold evidence coverage。
- 每题有 `answer_position`，能严格限制系统只读答案揭示之前的文本，防止“剧透式检索”。
- 同一小说有多道题，适合研究 Question-Time 记忆是否能帮助后续问题；不同小说之间又能检验技能是否可迁移。

MuSiQue 的段落大多是彼此独立的 Wikipedia 片段。它适合测多跳检索，但很难说明系统“理解了一个连续世界，并长出了新的关系”。

## 2. 已有实验的主要问题

1. **数据集不匹配。** LoCoMo 单对话太短，R@20 接近饱和；QASPER 的论文内段落关系较弱，而且 Reader 是瓶颈。
2. **Comprehension Triple 进不了检索链路。** conv-30 上 99/105 题存在触及 gold 的关系，但只有 5 题进入 top-5 fact，最终只兑现 1 题。
3. **关系挂载位置不合理。** 旧实现把关系挂到它引用的每个 session/chunk。用于增量阅读时，这会让后读到的信息影响更早位置的检索。
4. **验证会 fail-open。** 旧版 verification 失败或无裁决时保留候选；汇总文件还使用未验证的 `relations`。
5. **Question-Time 反思直接看到 gold evidence。** 旧 reflection prompt 同时提供 gold answer、gold evidence 和 retrieved context，容易把标签信息改写成经验。
6. **技能验证样本太小。** 43 道 evaluation 题上 1 题相当于 2.3pp，且 temperature=0 下 Reader 仍有明显抖动。

## 3. DetectiveQA 数据审计

本地数据来自 Hugging Face `Phospheneser/DetectiveQA`，许可为 Apache-2.0。英文版有 62 部小说、600 道四选一题。

| 项目 | 数值 |
|---|---:|
| 人工标注题 | 154 题 / 23 部小说 |
| AI 辅助标注题 | 446 题 / 62 部小说 |
| 每题线索段落中位数 | 5 |
| 线索跨度中位数 | 645 个段落 |
| 线索跨度 P90 | 2,220 个段落 |
| 答案位置前可见文本中位数 | 约 49.8k 英文词 |
| 切块后总块数 | 14,166（目标 200 词，最多 320 词） |
| 每题可见块中位数 | 170 |
| 每题 gold 块中位数 | 4 |
| 晚于或等于答案位置的线索 | 21 / 3,803（训练和评测时都会删掉） |

切块在每个问题的答案位置前强制断开，所以 gold clue 不会因为整块截断而丢失。

已知风险：

- 部分英文译名与小说正文不一致，例如同一角色有多个拼写。必须报告 dense / graph / reader 是否因译名失败。
- 15 个选项有 `200:` 或时间前缀；代码只清理明确的 `200:` 噪声，不改动真实时间。
- 小说可能被模型记住。需要加 `question-only` 和 `options-only` 对照，衡量参数记忆和选项偏差。
- 同一小说中，后续题与前面题经常共享线索：人工题中有 48/154 与更早的 AI 题共享精确线索段落，58/154 共享相邻 5 段内的线索。这能支持局部记忆复用，但也必须防止把前题答案直接转成后题提示。

## 4. 冻结划分

代码：`src/detectiveqa/data.py`，脚本：`scripts/detectiveqa/prepare_data.py`。

| split | 用途 | 规模 |
|---|---|---:|
| `eval_human` | 最终评测，永不更新记忆 | 154 |
| `within_practice` | 与人工题同小说的 AI 题，用于测同一故事内记忆迁移 | 161 |
| `cross_practice` | AI-only 小说，用于学习跨小说技能 | 208 / 29 部 |
| `cross_dev` | AI-only 小说，用于技能接受和负迁移检查 | 77 / 10 部 |

同一小说内的记忆只有在 `memory.available_at <= question.cutoff` 时才能用于该题。112/154 道人工题前面至少有一道同小说练习题，可用于测 within-novel learning curve。

## 5. RQ1：Read-Time Comprehension 新方案

### 5.1 低成本跨文档阅读包

先运行官方 HippoRAG2 OpenIE，不改其实体抽取、三元组抽取和图检索。

然后用官方 NER 结果构造阅读包：

1. 找出在小说中反复出现、但不是高频泛化词的实体。
2. 对每个实体按故事早、中、晚期抽样 4-6 个块。
3. 同一个包同时用于“联合 OpenIE”对照和“Comprehension Extraction”。

这样，二者的差异只来自抽取目标，而不是看到的上下文不同。

### 5.2 Comprehension Triple 的定义

关系必须同时满足：

- 至少引用两个不同块，且每个块都提供逐字证据。
- 不能是一条普通事实的改写。
- 只在所有证据都读完以后才进入图谱：关系挂到最晚的证据块上。
- 用 `claimed / suspected / contradicts` 区分说法、怀疑与事实，防止把证词当真相。

关系类型限定为：身份/别名、在场机会、动机利益、作案方法、矛盾欺骗、时间因果、状态变化。

### 5.3 进入检索链路

旧实验的核心失败不是“没有生成关系”，而是关系竞争不过普通 fact。DetectiveQA 主实验会同时报告：

- 关系是否进入 top-5 fact candidates。
- 是否经 PPR 把 gold chunk 排进 top-k。
- 是否改变 reader evidence 与答案。

默认先跑 no-LLM-reranker 版本，因为 LoCoMo 上它更利于新增关系进入 PPR；官方 LLM reranker 作为补充对照。

### 5.4 对照

| ID | 方法 |
|---|---|
| R0 | Dense retrieval |
| R1 | 官方 HippoRAG2 |
| R2 | R1 + 同阅读包联合 OpenIE |
| R3 | R1 + 同阅读包 Comprehension Triple |
| R4 | R3 + 验证器 |
| R5 | R3，按数量匹配 R2 |

主要指标：Any/Recall/All Evidence@5/10/20、QA accuracy、新关系进入 top-5 fact 的比例、gold chunk rank 改善、LLM token。

## 6. RQ2：Question-Time Self-Evolution 新方案

### 6.1 三类经验

- **Comprehension Memory：** 当前小说中跨块关系。只能引用当前题已检索到、且答案位置前可见的块。
- **Retrieval Skill：** 可跨小说复用的查询规划，例如“动机题同时找利益、冲突与隐瞒”。
- **Reasoning Skill：** 可跨小说复用的判断规则，例如“证词是 claim，要用独立观察验证”。

### 6.2 反思输入

反思时可以知道“答错了”，但不提供 gold clue 位置，也不让 gold option 文本进入记忆。

知识记忆必须通过：

- 证据块必须已在本题检索结果中；
- 证据块必须早于问题 cutoff；
- quote 必须能在块内找到；
- 关系不能包含选项文本。

技能必须不含当前问题和选项中的人名、选项文本或答案字母。

### 6.3 受控递归验证

每个候选最多一次修订、两次 RAG 重试：

1. 原题重试：比较 answer、gold evidence recall、retrieved facts。
2. 开发集重放：选同类型和随机题，检查负迁移。
3. 只有“局部有提升，且开发集无回退”的候选才能长期保留。

当前题变好只作为局部证据。长期保留必须看独立开发题。

### 6.4 对照

| ID | 方法 |
|---|---|
| Q0 | 静态 GraphRAG |
| Q1 | 仅更新 Comprehension Memory |
| Q2 | 仅更新 Retrieval/Reasoning Skill |
| Q3 | 记忆和技能共同更新 |
| V0 | 不验证，直接写入 |
| V1 | 原题单次验证 |
| V2 | 原题 + 开发集受控递归验证 |

最终在冻结记忆后评估 `eval_human`。同时报告 within-novel 和 cross-novel 两种泛化。

## 7. 已完成代码

- `src/detectiveqa/data.py`：读取小说、题目和线索；按答案位置切块；冻结划分；审计。
- `src/detectiveqa/graph.py`：每部小说一张 HippoRAG 图；查询时屏蔽未来 passage/fact/entity，并按可见来源重算图边权重。
- `src/detectiveqa/readtime.py`：阅读包构造、关系溯源校验、关系挂载到最晚证据块。
- `src/detectiveqa/evolution.py`：记忆/技能泄漏门控和接受规则。
- `src/detectiveqa/prompts.py`：第一版 prompt，尚需 smoke test。
- `scripts/detectiveqa/*.py`：数据准备、建图、前缀安全检索评测。
- `tests/test_detectiveqa.py`：6 个无 GPU 测试，已通过。

## 8. GPU 机执行顺序

先只跑 6 部人工标注小说做 pilot：

```bash
cd /group/40150/licochen/Memory/RSI_GraphRAG
source .venv/bin/activate
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken HF_HUB_OFFLINE=1
python -m unittest tests/test_detectiveqa.py
python scripts/detectiveqa/prepare_data.py
python scripts/detectiveqa/build_raw_graphs.py --split pilot
python scripts/detectiveqa/run_retrieval.py --split eval_human --novels 27,79,117,118,121,124
```

pilot 规模：6 部小说、1,381 个块、41 道人工评测题、37 道同小说练习题。

继续扩大前必须检查：

1. Graph 是否稳定优于 Dense，至少在 Recall@10 或 All@20 上有明确差异。
2. 未来块是否完全没有出现在检索结果中。
3. 失败案例是否主要来自检索缺失，而不是选项偏差或参数记忆。
4. Comprehension prompt 是否能在 20 个阅读包上产生可溯源、非冗余关系。

## 9. 未完成事项

- 还没有在 GPU 机运行 HippoRAG OpenIE 与 NV-Embed。
- Read-Time 提取脚本、Question-Time 闭环 runner 还没接到 LLM 调用。
- Prompt 只完成第一版，尚未做真实 case smoke test。
- 没有任何 DetectiveQA 检索或 QA 结果，报告中不能引用实验收益。
