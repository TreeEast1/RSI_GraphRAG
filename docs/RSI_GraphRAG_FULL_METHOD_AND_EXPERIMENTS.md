# RSI GraphRAG / CogitoRAG —— 完整方法、实验与指标报告

> **文档用途**：供外部审阅（reviewer / 审核模型）逐项核查方法定义、指标口径、实验覆盖度、结论强度与已知缺陷。
> **编写原则**：
> 1. **只写真实跑过的实验**，未运行的项显式标注 `Not Run`；
> 2. 所有数字来自 `outputs/` 下的落盘产物（逐题 jsonl / summary json / 日志），不写估计值；
> 3. 凡"文档与代码不一致"处，**以代码为准并显式记录差异**（见 §11.3）；
> 4. 单题可翻转量级、置信区间、样本量一律给出，避免把小样本噪声写成方法收益。

**项目路径**：`/data/workspace/group/licochen/Memory/RSI_GraphRAG`
**执行周期**：2026-11-05 ~ 2026-11-10（CST）
**文档生成**：2026-11-11

---

## 目录

- [0. 一页纸摘要](#0-一页纸摘要)
- [1. 项目定位与研究问题](#1-项目定位与研究问题)
- [2. 统一环境、模型与工程约束](#2-统一环境模型与工程约束)
- [3. 数据集](#3-数据集)
- [4. 指标定义（严格口径）](#4-指标定义严格口径)
- [5. 阶段 A：LoCoMo 全链路（P0/P1/P2）](#5-阶段-alocomo-全链路p0p1p2)
- [6. 阶段 B：conv-30 Prompt 优化](#6-阶段-bconv-30-prompt-优化)
- [7. 阶段 C：conv-30 Skill 自进化（RSI 闭环）](#7-阶段-cconv-30-skill-自进化rsi-闭环)
- [8. 阶段 D：QASPER Read-Time](#8-阶段-dqasper-read-time)
- [9. 跨阶段综合与结论](#9-跨阶段综合与结论)
- [10. 成本与工程账](#10-成本与工程账)
- [11. 局限、噪声与实现风险](#11-局限噪声与实现风险)
- [12. 复现](#12-复现)
- [13. 附录](#13-附录)

---

## 0. 一页纸摘要

### 0.1 做了什么

四个阶段，累计 **4 个方法族 × 2 个数据集**，全部使用同一 LLM（HY4，`no_think`）、同一 embedding（NV-Embed-v2）、同一 reader 模板族。

| 阶段 | 名称 | 数据集 / 范围 | 规模 | 状态 |
|---|---|---|---|---|
| **A** | LoCoMo 全链路 | LoCoMo，10 conversation | 1,986 题（dev 502 / test 1,484） | 完成 |
| **B** | Read-Time Comprehension Prompt 优化 + Reader Prompt 优化 | LoCoMo conv-30 | 105 题（81 可答 / 24 对抗） | 完成 |
| **C** | Question-Time Skill 自进化（RSI 闭环） | LoCoMo conv-30 | adaptation 43 / development 19 / evaluation 43 | 完成 |
| **D** | QASPER Read-Time | QASPER v0.3 validation，100 篇 | 372 题（333 可答且有映射文本证据） | 完成 |

### 0.2 五个主结论

1. **检索可被系统性改善，答案几乎不跟随。**
   阶段 A（test 1,484 题）：所有图谱变体把 R@5 从 .864 提到 .874–.885、证据覆盖从 .831 提到 .838–.849；但 8 个联合组合的 EM 全在 .236–.244、F1 全在 .472–.494，差异全部在噪声内。

2. **理解层（Comprehension）本身没有增量，"一次读全"才有。**
   C3（聚类+理解）EM .238 < C2（聚类联合抽取）.241；C6（全文+理解）.243 ≈ C5（全文联合抽取）.245。把新增关系数量从 16 降到 10（按 confidence 截断）全部检索指标不变 → **规模不是驱动因素**。

3. **在 conv-30 上，真正的瓶颈是 Reader，不是检索。**
   Oracle（把金标证据整段给 Reader）：原始 Prompt F1 .387 ≈ 正常检索 .390（**给了正确证据也不变好**）；优化后的 P7 在同 Oracle 上 F1 .603 vs 正常检索 .521。同一个图、同一批证据，仅换 Reader Prompt：全 105 题 F1 .390 → .521（+13.1pp）。

4. **QASPER 上"理解"的收益取决于表示与覆盖，不是"理解质量"。**
   Triple 表示（M3）中性偏正（All@5 +1.1pp，CI 跨 0）；Unit 独立索引（M4）整体 **−14.4pp** 显著负迁移；机制上是 **覆盖不足（仅 58% 的题 gold 被任一 unit 触及）+ 等权 RRF 在未覆盖题上挤掉正确证据**，而非 unit 排不上（733 个 unit 中仅 21 个"触及 gold 但从未进 top-5"）。

5. **自进化（RSI）闭环机制可跑通，但在小样本上不产生正收益。**
   Answer Skill：43 题 adaptation 中新增 3 条 skill、拒绝 8 条候选；evaluation 43 题上 E2（进化 answer）F1 .508 < E0（固定 P7）.563。Retrieval Skill：dev 快照选择直接否决了唯一学到的 skill（部署 0 条）。检索层退化主要来自 QDM 的 dense 合并（R@5 .932 → .886）。

### 0.3 最有价值的科研信号

**"证据已进入上下文，答案依然错误"比"再强一点的检索"更值得做。**
- LoCoMo temporal 类：证据覆盖 .875、F1 仅 .238；
- LoCoMo multi-hop 类：证据覆盖从 .274 提到 .441（+16.7pp），F1 只 +4.0pp；
- conv-30：69/81 题证据完整，其中 **53 题仍答错**；
- QASPER：Oracle gold context 的 Answer F1 仅 .487（dense .464），检索层差异几乎没有传导空间。

---

## 1. 项目定位与研究问题

### 1.1 出发点

项目名 **RSI GraphRAG / CogitoRAG**，框架表述为 **Read-Time Comprehension Evolution × Question-Time Retrieval Skill**，底座是官方 HippoRAG2（未改动其 `index()` / `retrieve()`），通过官方扩展点注入自研模块。

三条研究直觉：

1. **Read-Time**：传统 OpenIE 在每个文档（session / paragraph）内独立抽三元组，会漏掉"跨文档才能得到"的知识（状态变化、时间顺序、指代统一、因果/使能关系）。若在**阅读时**做跨文档联合理解，图谱应更接近"理解后的知识"，从而改善后续检索。
2. **Question-Time**：问题到来时按需拆解（QDM）、诊断缺失关系链（Retrieval Skill）、校验并修订答案（Answer Skill），应能补上 Read-Time 静态图谱无法覆盖的查询形态。
3. **Evolution**：把 Question-Time 的技能写成可累积、可版本化、可淘汰的 **Skill Memory**（不改模型权重），即 RSI 闭环。

### 1.2 四个阶段的分工（为什么不是一个大实验）

| 阶段 | 回答的问题 | 为什么单独做 |
|---|---|---|
| A | 在标准 benchmark 上，Read-Time / Question-Time 是否有效？ | 需要全量 dev/test 与固定预算，先建立可信基线 |
| B | 如果无效，是"知识不够"还是"Reader 不会用"？ | LoCoMo 全量上两者混杂，需在单对话上做 Oracle / 无重排 / 数量控制等隔离实验 |
| C | Question-Time 的"进化"能否作为独立策略层？ | 需要严格的 adaptation / development / evaluation 三分与冻结流程 |
| D | 换一个"池更大、证据更多"的数据集，结论是否变化？ | LoCoMo 单对话仅 1.1–2.2 万 token、19 篇文档，检索天花板太低 |

### 1.3 全程假设与最终判定

| 假设 | 内容 | 最终判定 | 关键证据 |
|---|---|---|---|
| **H1** | 实体/语义无监督分组能让相关 session 在有限阅读预算内共同出现 | **弱成立** | entity_idf 跨会话共置 .431 vs 随机 .304（8192 预算），但未转化为端到端收益 |
| **H2** | 聚类后联合抽取比逐 session 抽取更容易发现跨文档关系 | **成立** | C2 证据 .837 vs G_raw .819；test R@5 .875 vs .864 |
| **H3** | 专门的理解层能产生普通三元组无法表达的可验证语义关联 | **不成立**（LoCoMo）/ **中性偏正**（QASPER） | C6≈C5、C3<C2；QASPER M3 All@5 +1.1pp（CI 跨 0） |
| **H4** | 理解图谱能提高多证据检索 Recall 与 QA 正确率 | **检索成立、QA 不成立** | test 证据 +0.7~1.8pp；EM/F1 差异 ≤±0.5pp |
| **H5** | Question-Time Retrieval Skill 能补全遗漏的关系链 | **检索弱成立、QA 不成立** | Q2 R@5 +1.7pp（dev）/ +0.6pp（test）；QA 持平或负向 |
| **H6** | Read-Time 与 Question-Time 互补 | **不成立** | test 上检索最好的 R1Q2 在 QA 上不如 R1Q1/R1Q0 |
| **H7** | LoCoMo 上直接读全文可能很强，复杂聚类未必必要 | **成立** | Full-Context F1 .493 ≈ 最好图检索 .492；全文变体 ≥ 聚类变体 |
| **H8**（B 阶段新增） | 当前瓶颈在 Reader，而非图谱 | **成立** | Oracle+P0 F1 .387 ≈ 正常 .390；Oracle+P7 .603 |
| **H9**（B 阶段新增） | 新增关系的价值取决于"是否进入 top-5 fact 候选" | **成立（机制）** | 99/105 题有触及金标的关系 → 5/105 进入 fact 候选 → 1/105 兑现 |
| **H10**（C 阶段新增） | 错误反思生成的技能能在新题上泛化 | **不成立（否定结果）** | 28/43 题命中技能，仅 1 有帮助、1 有害、26 中性 |
| **H11**（D 阶段新增） | Unit（自足短句）比 Triple 更适合做跨段理解的载体 | **整体不成立、覆盖子集成立** | 覆盖子集 All@5 .567→.627；未覆盖子集 .514→.089 |

---

## 2. 统一环境、模型与工程约束

### 2.1 硬件与系统

| 项 | 值 |
|---|---|
| GPU | 4 × NVIDIA H20，97,871 MiB/卡（QASPER 单卡运行，`CUDA_VISIBLE_DEVICES=0`） |
| CPU / 内存 | 384 核 / 2,265 GB |
| 磁盘 | overlay 12 TB（本地 ~4 GB/s）；项目盘为 Ceph（小文件与冷读极慢） |
| 系统 | TencentOS 3.2，Linux 5.4.241，Python 3.10.14 |

### 2.2 软件与模型身份

| 组件 | 版本 / 路径 | 说明 |
|---|---|---|
| HippoRAG | `OSU-NLP-Group/HippoRAG` @ `d5c8329422e0a0b834a15874545cb6a74b4f9b26`（v2.0.0a5） | **未修改** `index()` / `retrieve()` / PPR / 同义边 |
| PyTorch / CUDA | 2.5.1+cu124 / 12.4 | |
| transformers | 4.45.2 | |
| LLM | HY4（Tencent TokenLink），`https://tokenlink.woa.com/openapi/v1`，`HY4_MODEL=hy4`，`reasoning_effort=no_think`，`temperature=0` | 全项目**唯一** LLM，所有方法共用 |
| Embedding | `nvidia/NV-Embed-v2`，fp16，dim 4096，本地 `/root/models/NV-Embed-v2` | 加载 ~5.7s；`config.json._name_or_path` 被打为本地路径（远程代码会按 repo id 重载 tokenizer） |
| 索引身份串 | `rsigraph` ＋ `hy4-no_think` ＋ `nvembed-v2` ＋ `v1`（竖线拼接） | 写入 HippoRAG 索引清单，用于缓存失效判定 |
| tokenizer（预算用） | `tiktoken cl100k_base` | **只用于预算，不用于计费** |

### 2.3 数据与代码身份

| 项 | 值 |
|---|---|
| LoCoMo 文件 | `data/raw/locomo/locomo10.json`，sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`，2,805,274 bytes |
| LoCoMo split | seed **20240501**，dev = `conv-30, conv-44, conv-48`，test = 其余 7 个（见 `data/splits/split.json`） |
| QASPER 文件 | `qasper-dev-v0.3.json`，sha256 前缀 `2ae7ee62a65b1c42` |
| QASPER 样本 | seed **20261110**，100 篇，冻结于 `outputs/qasper_readtime/dataset/sample.json`（在任一检索/QA 评测前写入） |
| split 生成逻辑 | `load_locomo` 先按 `sample_id` 字典序排序，再 `random.Random(20240501).shuffle()`，取前 3 排序后为 dev（`src/dataset/locomo.py:226-239`） |

### 2.4 API 配额、限速与缓存（影响可复现性，必须记录）

**配额实测**（HY4 是公司共享池，容量随他人负载波动）：

| 时间 | 1 QPS | 2 QPS | 4 QPS | 5 QPS | 8 QPS |
|---|---|---|---|---|---|
| 11-06 02:00 | 90% 成功 | 7% | — | — | — |
| 11-06 11:00 | 100% | 100% | 100% | — | 62% 429 |
| 11-06 16:00 | 100% | 100% | 100% | 100% | — |
| 11-06 17:00 | — | — | — | — | 大面积 429 |

累计观测 **22,159 次 429**（占 61,818 次记录调用的 36%）。所有方法使用同一模型、同一 prompt，**限速只影响耗时，不影响公平性**。

**工程机制**（`src/adapters/`）：

| 机制 | 实现 |
|---|---|
| 内容寻址缓存 | `sha256(json(model, base_url, messages, params, sort_keys=True))` → sqlite（WAL，`INSERT OR REPLACE`）；键含 base_url 与全部生成参数，换端点/改参数即失效（`hy4_client.py:135-144`） |
| 重试 | 客户端 `max_retries=14`，退避 `min(60, 1.5·2^attempt)·U(0.7,1.3)`；HippoRAG 适配层另套 14 次，仅对 429/timeout/502/503/504 重试（`hipporag_adapter.py:43-62`） |
| 跨进程限速 | `src/adapters/ratelimit.py`：文件锁 + **预留式（reservation）**全局间隔（非令牌桶），`slot = max(now, last + 1/qps)`；QPS 可热更新（写 `cache/rate_qps_{group}.txt`，无需重启 worker） |
| 池分组 | `HY4_RATE_FILE` 取 `poolA` / `poolB` / `qasper` / `qe_main`；未指定时按 `sha1(api_key)[:8]` 自动分桶 |
| 调用日志 | `outputs/logs/hy4_calls{shard}.jsonl`，字段 `ts/tag/model/reasoning_effort/max_tokens/json_mode/latency_s/status/attempt/usage{prompt,completion,total,reasoning}`；**密钥不写日志** |
| 图谱变体隔离 | 变体目录只复制 raw 的 embedding store / `openie_state.json` / manifest，**不复制 `graph.pickle`**，再 `force_index_from_scratch=True` 重跑官方 `index()`；G_raw 永不被修改（`src/graph/variants.py`） |

**公平性声明**：所有对比方法共用同一 reader（`src/retrieval/reader.py`）、同一 top-20 截断、同一 4096-token 上下文预算。Full-Context 单独报告，不与固定预算方法混比。

---

## 3. 数据集

### 3.1 LoCoMo（主数据集，阶段 A/B/C）

**规模**（`outputs/reports/DATASET_AUDIT.md`）：

| 指标 | 值 |
|---|---|
| Conversation | 10 |
| Session（= 文档数） | 272 |
| Turn | 5,882 |
| QA | 1,986（dev 502 / test 1,484） |
| Session token | 均值 684.8，中位 642，最大 1,397，总 186,262 |
| 单对话全文 token | **11,435 – 21,979** |
| 跨 session 多证据 QA | 328（16.5%） |
| 平均每题证据条数 | 1.42 |
| 证据无法映射的 dia_id | 8 |
| 无 evidence 的 QA | 4 |

**类别分布**：

| category | 名称 | 全量 | dev | test | conv-30 |
|---|---|---|---|---|---|
| 1 | multi-hop | 282 | 62 | 220 | 11 |
| 2 | temporal | 321 | 92 | 229 | 26 |
| 3 | open-domain | 96 | 17 | 79 | **0** |
| 4 | single-hop | 841 | 224 | 617 | 44 |
| 5 | adversarial | 446 | 107 | 339 | 24 |

> **结构性前提**：单个 conversation 全文仅 1.1–2.2 万 token，HY4 上下文 1M 完全放得下。这是"聚类没有必要""Full-Context 已接近最好方法"的直接原因。conv-30 更是极端：19 篇文档 = 检索池全量，R@20 恒为 100%，且**一道 open-domain 都没有**。

**每对话明细**：

| conversation | split | sessions | turns | QA | 平均 session token | max | min |
|---|---|---|---|---|---|---|---|
| conv-26 | test | 19 | 419 | 199 | 771.1 | 1,168 | 434 |
| conv-30 | dev | 19 | 369 | 105 | 601.8 | 960 | 373 |
| conv-41 | test | 32 | 663 | 193 | 686.8 | 1,157 | 432 |
| conv-42 | test | 29 | 629 | 260 | 655.7 | 1,231 | 318 |
| conv-43 | test | 29 | 680 | 242 | 735.3 | 1,257 | 379 |
| conv-44 | dev | 28 | 675 | 158 | 750.2 | 1,180 | 451 |
| conv-47 | test | 31 | 689 | 190 | 657.7 | 1,082 | 417 |
| conv-48 | dev | 30 | 681 | 239 | 654.2 | 1,397 | 367 |
| conv-49 | test | 25 | 509 | 196 | 653.0 | 921 | 469 |
| conv-50 | test | 30 | 568 | 204 | 683.8 | 1,294 | 366 |

**防泄漏**：`session_summary` / `observation` / `event_summary` / gold answer / gold evidence **从未进入任何索引或生成路径**。LoCoMo 的这些字段是人工摘要，若进入索引会构成泄漏；本项目分组与抽取只使用原始对话文本（`src/dataset/locomo.py`）。

### 3.2 QASPER（阶段 D）

**语料与候选池**：

| 指标 | 值 |
|---|---|
| papers / questions | 281 / 1,005（answerable 945，unanswerable 60） |
| 有映射 gold 的 answerable 题 | 888 |
| 每篇段落数 | mean 46.95，P10 22.0，**P50 43.0**，P90 70.0，max 281 |
| 每篇 token | mean 4,680.7，P50 4,438，P90 6,636 |
| 段落 token | mean 99.7，P50 86，P90 193 |
| gold 段落数/题/标注者 | mean 1.75，P50 1.0，**P90 3.0**，max 19 |
| 单段落 gold / 多段落 gold | 63.2% / 36.8% |
| 跨章节 | 19.0% |
| 仅图表证据 | 3.5% |
| top-3/5/10 占池 | 8.4% / **14.0%** / 27.8% |

**证据映射**（级联，方法记在 `method` 字段）：`FLOAT SELECTED → float` → 空串 → section 名（heading）→ 归一化精确匹配（exact，2,426）→ 最多 4 段连续拼接（concat）→ 双向子串（substring）→ token Jaccard ≥ 0.85（fuzzy，253）→ unmatched。文本证据映射率 **100.0%**，未匹配 0 条；`FLOAT SELECTED`（253）与 heading（129）不映射到文本段落，**图表证据从文本检索指标中排除但保留在 QA 中**。

**多标注者处理**：每个标注者的映射段落集合去重 → 完全相同的标注者集合去重 → 指标**先在每个标注者集合上计算再平均**，绝不对 union 强行要求同时召回（避免"替代证据集"被误判为漏召）。

**冻结样本**：uniform random sample without replacement，`sha1(f"{seed}:{paper_id}")` 排序取前 100（seed 20261110），规则在任何检索/QA 评测前写入。100 篇 / **372 题**，其中 **333 道** answerable 且有映射文本证据，141 道多证据，56 道跨章节。样本池 mean 48.09 / P10 22.9 / P90 68.2。

**按论文长度分层**（五分段）：多证据比例随长度上升（最长段 42.7%），跨章节比例稳定在 18–22%。这支持把多证据/跨章节子集作为主要区分条件。

### 3.3 两个数据集的可比性（关键）

| 维度 | LoCoMo | QASPER | 含义 |
|---|---|---|---|
| 检索单元 | session（19–32 篇/对话） | paragraph（P50 43，P90 70 篇/论文） | QASPER 池大 2–3 倍，但仍不是"大语料" |
| 检索限制 | 全局（10 个对话各自建图） | **严格限制在题目所属论文内** | QASPER 排除了跨论文干扰 |
| 平均证据条数 | 1.42 | 1.75（单段落 63.2%） | 都是"单/少证据"为主 |
| 多证据题占比 | 16.5% | 36.8% | QASPER 区分度更高 |
| 上下文长度 | 全文 11k–22k token | 全文 mean 4,681 token | 两者 Full-Context 都装得下 |
| 检索天花板 | test R@20 ≈ .986 | dense Any@5 **.740** | LoCoMo 已饱和；QASPER 单证据也接近饱和 |
| 图 vs dense | 图明显更好（R@5 .864 vs .781） | **dense 更好**（Any@5 .740 vs .692） | **强结论：图检索的优势不是普适的** |

---

## 4. 指标定义（严格口径）

> 这一节是审核重点。**同一套数字在不同的 normalize 下可以差 3–4pp**，且分页/分母不同会产生"看起来矛盾"的两套结果。所有正式表格均标注口径。

### 4.1 检索指标（LoCoMo）

实现在 `src/evaluation/metrics.py` + `pipeline.py`，对 top-20 截断的 session 排名计算：

| 指标 | 公式 | 备注 |
|---|---|---|
| Recall@k | top-k 与 gold 的交集大小 ÷ gold session 数 | gold 为空 → `nan`（聚合时跳过） |
| R@20 | 同上 k=20 | LoCoMo 上近乎恒为 1（池只有 19–32 篇） |
| MRR | `1 / i`，`i` = 首个命中 session 的 **1-based** 排名；无命中 0 | 对 top-1 质量敏感 |
| nDCG@10 | `DCG/IDCG`，`DCG = Σ_{i≤10, ranked[i]∈gold} 1/log2(i+1)`，`IDCG = Σ_{i=1..min(n_gold,10)} 1/log2(i+1)`（n_gold = 该题 gold session 数） | 二值增益 |
| session_full@B | `1[gold_sessions ⊆ selected_ids]` | 全部 gold session 都要进上下文 |
| **evidence_full@B** | turn 级：`1[∀ gold_turn, turn_to_session[t] ∈ selected]` | **主证据指标** |
| evidence_partial@B | `hit ÷ gold_turn 数` | |

**上下文装填规则（易误读，必须写清）**：
`select_by_budget` 按排名顺序贪心装填；若 `used + n_tokens(doc) > budget`，**默认 `all_or_nothing=True` → `continue`（跳过该篇，继续尝试后面更短的文档）**，而非 `break`。
即 context 可能是**非连续**的排名子集（例：q16 选中 s02,s05,s09,s11,s15,s18）。命名与语义相反，见 §11.3。

### 4.2 QA 指标（项目口径 = SQuAD 风格）

| 项 | 实现 |
|---|---|
| `normalize_answer` | 小写 → **逐字符删除标点（不插空格）** → 只删冠词 `a` / `an` / `the`（**不删 "and"**）→ 空白归一（`metrics.py:21-25`） |
| EM | 规范化后**字符串相等**，多 gold 取 any |
| F1 | token 多重集交集 `Σ min(cp, cg)` → P/R → F1；多 gold 取 **max**；**不做词干化**、不按逗号拆子答案 |
| **分母口径** | **EM/F1 只统计 category 1–4（可答）**；category 5 单独按"正确拒答率"计 |
| 拒答判定 | `pred.lower()` 命中 `no answer / cannot / not mentioned / unknown / unanswerable / not stated / no information available` 之一（**子串匹配**） |
| cat-5 计分 | `em = f1 = float(refused)`，因此 cat-5 的 em/f1 数值 = 该组拒答率 |

> 因此 `outputs/reports/metrics.csv` 里的 `em`/`f1`（全题口径，cat-5 按拒答率计入）**不等于**报告里的 `EM(可答)`。例：HippoRAG dev 全题 em .4084 / f1 .5873，而 EM(可答) .2532 / F1(可答) .4806。**两套数都对，口径不同**。

### 4.3 官方 LoCoMo 口径（复刻，用于对照）

来自 `snap-research/locomo` 的 `task_eval/evaluation.py`（2026-11-09 抓取），逐行重写于 `src/prompt_optimization/official_metrics.py`。与项目口径的 6 处差异：

1. `normalize_answer` **先删逗号**（不插空格）→ 小写 → 去标点 → 去 `a/an/the/and`；
2. EM = **token 集合相等**（顺序无关），不是字符串相等；
3. F1 对每个 token 施加 **Porter stemmer**；
4. category 1（multi-hop）用 `f1_multi`：双方按 `,` 切分，对**每个 gold 子答案**取与任一 pred 子答案的最大 F1，再对 gold 子答案取均值；
5. category 3 只取 gold 中 `;` 之前部分；
6. category 5 是**二选一多选**（`(a) Not mentioned in the conversation (b) <诱导答案>`），严格口径只认 `no information available` / `not mentioned`。

**影响量级**（conv-30，预测不变只换口径）：项目 F1 .390 → 官方 F1 **.426（+3.6pp）**；EM 几乎不变（.210 vs .210）。

**cat-5 三种协议**（同一批 24 题）：

| 输入 Prompt | 官方多选协议准确率 |
|---|---|
| 官方 `QA_PROMPT_CAT_5` | 87.5%（21/24） |
| 项目 P0 | 83.3%（20/24） |
| 项目 P7 | 87.5%（21/24） |

> **协议错配警告**：若用官方严格关键词（"not mentioned"）去判我们输出的 "No answer"，严格拒答率会算成 **0%**。这是措辞错配，不是模型失败。所有报告同时给"项目口径"与"官方口径"两套数。

### 4.4 检索机制专用指标（阶段 B/C/D）

| 指标 | 定义 | 用途 |
|---|---|---|
| `fallback_dense` | rerank 后**没有任何 fact 存活** → 整题退化为 `dense_passage_retrieval`（top-1 分数被 min-max 归一化成 1.0） | 诊断"没进 PPR" |
| `n_fact_candidates_before_rerank` / `n_facts_after_rerank` | 官方 `rerank_facts` 的中间状态（`retrieval_diag.py`） | 区分排序变化来自 PPR 还是 fallback |
| 新关系进 top-5 fact 题数 | 新增关系文本与每题最终 top-5 fact 做规范化匹配 | **H9 漏斗** |
| multi-evidence complete | 多证据题的 gold session 全部进 4096 上下文 | 阶段 C 主指标之一 |
| answer support | 预测中长度 >2 的 token 落在上下文 token 集合内的比例（纯词形，无词干/同义） | 幻觉代理指标 |

### 4.5 QASPER 指标

| 指标 | 公式 | 备注 |
|---|---|---|
| Any-Evidence@k | `1[top-k 与 s 的交集非空]` | **逐标注者集合算，再平均**（不是对 union 算）；union 视图仅作次要报告 |
| All-Evidence@k | `1[s ⊆ top-k]` | 主区分指标（多证据题） |
| Recall@k | `top-k 与 s 的交集大小 ÷ s 的大小` | |
| MRR / nDCG@10 | 同 LoCoMo 定义（1-based，二值增益） | |
| Cov@4096 | 按排名贪心装填段落（装不下 **跳过继续**），对每个标注者集合算 `full = 1[s ⊆ selected]` 再平均 | 与 LoCoMo 的 evidence_full@4096 同构 |
| Answer F1 | 每个标注者算 token-F1，**取 max**；EM = 任一标注者规范化后相等 | 题级平均（不是标注者级） |
| Evidence F1 | 预测段落原文集合 vs 原始 evidence 字符串集合的 P/R/F1，取 max | **对空白/裁剪差异敏感**（见 §11.3） |
| bootstrap | 单方法 CI：对指标向量 2,000 次有放回重采样取 2.5/97.5 分位（**非配对**）；方法间差值：**按题配对**后对差值向量重采样 2,000 次 | |

---

## 5. 阶段 A：LoCoMo 全链路（P0/P1/P2）

### 5.1 文档粒度与基线

- **1 Session = 1 document**（272 篇），保留 turn 级 `dia_id` 映射用于 turn 级证据覆盖。
- `document_id = f"{conv_id}::s{sid:02d}"`；session 文本 = `[Session N | <date>]` + 逐轮 `speaker: text`（header 只在文内出现一次，`locomo.py:77-80`）。

| 基线 | 定义 |
|---|---|
| **Dense RAG** | 问题 → NV-Embed-v2 查询向量 → session 向量 cosine top-20 → Reader |
| **HippoRAG2 G_raw** | 官方 OpenIE（NER→三元组）→ entity/fact/passage 图 + 同义边 → fact rerank + PPR → top-20 session → Reader |
| **Full-Context QA** | 整个对话（11k–22k token）一次性给 Reader，无检索（单独报告，非等预算） |

### 5.2 分组（Experiment B）

**9 种无监督方法 × 2 预算（4096/8192）**。实体集合来自 G_raw 的官方 OpenIE NER 输出（**不额外调用 LLM**）；文档向量来自 NV-Embed-v2（空 instruction）。

| 方法 | 实现要点 |
|---|---|
| `random`（5 seeds） | 随机洗牌后按预算贪心打包，组内再按原始顺序重排 |
| `temporal` | 按 session 时间顺序连续打包 |
| `entity_jaccard` | 共享实体 Jaccard + 约束层次聚类 |
| `entity_idf` | **IDF 加权余弦**：`idf(e)=ln((n+1)/(df(e)+1))+1`；`sim = Σ_{e∈A∩B} idf(e) / (‖A‖_idf·‖B‖_idf)` |
| `embedding` | L2 归一化后内积 cosine |
| `hybrid_{0.25,0.5,0.75}` | `sim = w·sim_entity + (1-w)·sim_emb` |
| `community` | doc–entity 二部图 + Louvain（`resolution=1.0`）；**IDF hub 抑制**：`idf(e) < 1.0` 的实体被跳过（即 df ≥ n 的停用词级实体），边权 `1/sqrt(1+idf(e))` |

**打包规则（统一）**：贪心顺序装箱；仅当 `cur` 非空且 `used + t > budget` 时封箱（**单个超预算文档仍自成一组**，不丢弃）；`constrained_agglomerative` 只合并 token 和不超预算的两簇；Louvain 产出的超预算社区用 `split_oversized` 按 token 升序切分。

**分组指标**（`grouping/evaluate.py`）：

| 指标 | 定义 |
|---|---|
| gold_full_coverage | 每题 `best = max_g`（gold 与组 g 的交集大小 ÷ gold 大小）取 1.0 的比例（**需要单一组覆盖全部 gold session**） |
| gold_partial_coverage | 上述 `best` 的均值 |
| cross_session_colocation | 仅 gold session 数 > 1 的题，`best == 1.0` 的比例 |
| duplicate_reading_tokens | `Σ组内 token − Σ总 token` |
| reading_cost_ratio | `total_reading_tokens / Σ doc tokens` |

### 5.3 Read-Time 变体（Experiment C）

| 名称 | 阅读组织 | 抽取 | 理解层 | 图 |
|---|---|---|---|---|
| C0/C1 | 单 session | 官方 OpenIE | 无 | G_raw |
| C2 | 聚类阅读组（hybrid_0.5 / 8192） | 联合抽取（显式事实） | 无 | G_raw + 组内三元组 |
| C3 | 同 C2 | 联合抽取 | **有**（跨会话关系） | G_raw + 三元组 + 关系 |
| C3v | 同 C3 | 同 C3 | 有 + LLM 校验 | 校验后子集（conv-30 无产物，未参与比较） |
| C5 | 全对话 1 组 | 联合抽取 | 无 | G_raw + 全文三元组 |
| C6 | 全对话 1 组 | 联合抽取 | 有 | G_raw + 全文三元组 + 关系 |

**公平性设计**：C1 与 C0 使用**同一官方 prompt、同一 HY4、单 session**，因此等价，不单独报告；C2/C3 与 C5/C6 共用**同一抽取 prompt 与同一 schema**，只改输入 session 数 → `C2−G_raw` 隔离"联合抽取"，`C3−C2` / `C6−C5` 隔离"理解层"。

**理解层输出 schema**：

```
subject / predicate / object / relation_type / source_dia_ids /
supporting_fact_ids / evidence / derivation(Explicit|Composed|Inferred) / confidence
```

硬约束：每条关系**至少引用两个不同 session 的 dia_id**（否则丢弃，`needs_two_sessions`）；`evidence` 为逐字引用（每条 ≤25 词）；`derivation=Composed` 表示无需世界知识、`Inferred` 表示需常识/语义推理。

**图变体构建步骤**（`src/graph/variants.py`，这是本项目最关键的可信性设计）：

1. 建 `{conv}/variants/{name}`，复制 raw 工作目录**除 `graph.pickle` 外**的一切（embedding parquet、`chunk_metadata.json`、`openie_state.json`、index manifest）；
2. `llm_cache` 用**符号链接**指向 raw（避免重复 OpenIE 调用；失败退化 copytree）；
3. 用 `compute_mdhash_id(text, "chunk-")` 计算 chunk id（内容寻址），在 `openie_state.json` 的 `payload["docs"]` 中找到 `idx == chunk_id` 的行，把新三元组 append 到 **`extracted_triples`**，并把 subject/object 补进 **`extracted_entities`**（去重，`skipped_duplicate` 计数）；
4. 覆盖全部 session 构造 `Chunk` → `open_hipporag(force_index_from_scratch=True)` → 官方 `index(chunks)`。因为 OpenIE state 已覆盖所有 chunk，**不会触发任何 LLM 抽取**，只对新实体/事实算 embedding；
5. 溯源写 `payload["read_time_injection"] = [{note, n_added, docs}]`；文件原子写（`.tmp` + `os.replace`）。

**图规模（conv-30，已双向核对 `openie_state.json` 与 `get_graph_info()`）**：

| 图 | phrase 节点 | passage 节点 | 总节点 | 抽取三元组 | 含 passage 的三元组 | 同义边 | 总边数 |
|---|---|---|---|---|---|---|---|
| G_raw | 441 | 19 | 460 | 600 | 545 | 741 | 1,733 |
| C2 | 552 | 19 | 571 | 787 | 688 | 1,090 | 2,343 |
| C3 | 570 | 19 | 589 | 813 | 742 | 1,108 | 2,434 |
| C5 | 532 | 19 | 551 | 735 | 657 | 901 | 2,095 |
| C6 | 559 | 19 | 578 | 753 | 723 | 938 | 2,213 |

规模增幅（按抽取三元组）：C2 +31%、C3 +36%、C5 +23%、C6 +26%；passage 节点恒为 19 → 变化全部来自新增知识。
> ⚠️ `outputs/reports/MASTER_REPORT.md` §6.5 的"conv-30 505/750"是**行错位**（505/750 实为 conv-26）。本表为核对后的值，见 `CONV30_READTIME_ANALYSIS.md` §1.1。

### 5.4 Question-Time（Experiment E）

| 名称 | 流程 | 关键参数 |
|---|---|---|
| Q0 | 原始问题 → 检索 → Reader | top-20 |
| Q1 | **QDM** 判定是否拆解 → 子问题各自检索 → RRF 合并 → Reader | 子问题 ≤4（截断 `subqs[:4]`）；解析失败 → fail-closed（不拆） |
| Q2 | Q1 + **Retrieval Skill**（观察 top-5 检索结果，诊断缺失关系链，最多补 2 个查询）→ 再 RRF | `max_new=2`；证据摘要每条截 280 字符 |
| Q3 | Q2 + **Answer Skill**（证据校验、修订、可拒答） | 输出 `{answer, answerable, support}`；`answerable=False` 不强制改写拒答 |

**RRF**：`score(d) = Σ_list 1/(k + rank + 1)`，`k=60`，`rank` 为 **0-based**（第 1 名贡献 `1/61`），返回 top-20。

**Skill Evolution（第一版，E4）**：不是 LLM 自动生成，而是 **`(category, query_suffix, observed_benefit)` 网格搜索**——在 dev 上对每个 category 评估 3 种检索扩展后缀（temporal / person / place），保留每类最佳正收益后缀，冻结到 `outputs/skills/frozen_skill_library.json`，再原样应用到 test。

### 5.5 联合（Experiment F）

`R{0,1} × Q{0,1,2,3}`：Read-Time（G_raw 或 C6）× Question-Time，test 全量 2×4；conv-30 另做 2×2（R0/R1 × Q0/Q3）。

### 5.6 结果：基线（全量 1,986 题）

| method | n | EM(可答) | F1(可答) | R@5 | R@20 | 证据@4096 | 证据@8192 | 拒答率(cat5) |
|---|---|---|---|---|---|---|---|---|
| Dense RAG | 1,986 | .221 | .436 | .791 | .980 | .742 | .879 | .946 |
| **HippoRAG2 G_raw** | 1,986 | .244 | .483 | **.863** | .987 | **.828** | **.916** | .953 |
| Full-Context | 1,986 | **.245** | **.491** | N/A | N/A | N/A | N/A | .915 |

**Dev / Test 拆分**：

| method | split | n | EM | F1 | R@5 | 证据@4096 |
|---|---|---|---|---|---|---|
| Dense | dev | 502 | .241 | .451 | .822 | .771 |
| Dense | test | 1,484 | .214 | .431 | .781 | .732 |
| HippoRAG | dev | 502 | .253 | .481 | .858 | .819 |
| HippoRAG | test | 1,484 | .240 | .484 | .864 | .831 |
| Full-Context | dev | 502 | .261 | .486 | N/A | N/A |
| Full-Context | test | 1,484 | .239 | .493 | N/A | N/A |

**按类别（全量 1,986 题）——本报告最重要的一张表**：

| category | n | Dense R@5 | HippoRAG R@5 | Dense 证据@4096 | HippoRAG 证据@4096 | Dense F1 | HippoRAG F1 |
|---|---|---|---|---|---|---|---|
| 1 multi-hop | 282 | .559 | **.676** | .274 | **.441** | .353 | **.393** |
| 2 temporal | 321 | .815 | **.888** | .773 | **.875** | .214 | **.238** |
| 3 open-domain | 96 | .615 | **.668** | .506 | **.584** | .155 | **.172** |
| 4 single-hop | 841 | .847 | **.936** | .850 | **.937** | .580 | **.642** |
| 5 adversarial | 446 | .852 | .863 | .857 | .881 | .946 | .953 |

- multi-hop：证据覆盖 .274 → .441（**+16.7pp**），F1 仅 .353 → .393（+4.0pp），EM .092 → .103。
- temporal：证据 .875、F1 **.238** —— 证据几乎都在，模型答不对相对时间。
- **检索层的巨大改善没有等比例传导到答案层。**

**按对话（HippoRAG）**：

| conv | n | EM | F1 | R@5 |
|---|---|---|---|---|
| conv-26 | 199 | .317 | .510 | .888 |
| conv-30 | 105 | .381 | .520 | .903 |
| conv-41 | 193 | .415 | .668 | .859 |
| conv-42 | 260 | .373 | .555 | .849 |
| conv-43 | 242 | .479 | .639 | .866 |
| conv-44 | 158 | .437 | .623 | .827 |
| conv-47 | 190 | .426 | .605 | .928 |
| conv-48 | 239 | .402 | .593 | .860 |
| conv-49 | 196 | .398 | .558 | .829 |
| conv-50 | 204 | .392 | .591 | .836 |

跨对话 EM 极差 .317–.479（**16pp**）→ 任何 1–2pp 的整体提升都应视为噪声级别。这条校准贯穿全文。

### 5.7 结果：分组（Experiment B）

均值覆盖 dev+test 全部参与对话：

| method | budget | runs | gold full | gold partial | cross-session 共置 | 平均组 token | 最大组 token |
|---|---|---|---|---|---|---|---|
| random (5 seeds) | 4096 | 15 | .858 | .927 | .113 | 3,528 | 4,096 |
| random (5 seeds) | 8192 | 15 | **.883** | .947 | .304 | 6,421 | 8,107 |
| temporal | 4096 | 3 | .871 | .932 | .226 | 3,528 | 3,998 |
| temporal | 8192 | 3 | .883 | .948 | .305 | 6,421 | 8,099 |
| entity_jaccard | 4096 | 3 | .869 | .936 | .163 | 3,528 | 3,989 |
| entity_jaccard | 8192 | 3 | .886 | .948 | .249 | 6,421 | 7,978 |
| **entity_idf** | 4096 | 3 | .870 | .935 | .195 | 3,210 | 3,956 |
| **entity_idf** | 8192 | 3 | **.905** | **.958** | **.431** | 6,421 | 8,173 |
| embedding | 4096 | 3 | .867 | .933 | .180 | 3,528 | 4,086 |
| embedding | 8192 | 3 | .884 | .946 | .259 | 6,421 | 8,137 |
| hybrid_0.5 | 4096 | 3 | .867 | .933 | .180 | 3,528 | 4,086 |
| hybrid_0.5 | 8192 | 10 | .881 | .946 | .281 | 6,644 | 8,184 |
| community | 4096/8192 | 3 | .854 | .918 | .086 | 1,430 | 3,778 |

**结论**：
1. 8192 预算下**随机分组 gold 覆盖已达 .883**，所有聚类方法 .881–.905，差距 ≤2.2pp 且在单对话方差内 → **聚类没有必要**（H7 成立）。
2. 唯一稳定信号是 `entity_idf` 的跨会话共置 **.431 vs 随机 .304**，但它没有转化为端到端 QA 优势（C2/C3 不优于 C5/C6）。
3. `community` 最差（.854）：生成 10–13 个小社区且未按预算合并，粒度与阅读预算不匹配 —— 实现层面的发现。
4. 4096 预算下结论相同（gold full .858–.871）。

### 5.8 结果：Read-Time（Experiment C）

**Dev（3 对话，502 题，含 Reader）**：

| 方法 | EM(可答) | F1(可答) | R@5 | R@20 | 证据@4096 | 证据@8192 |
|---|---|---|---|---|---|---|
| Dense | .241 | .451 | .822 | .985 | .771 | .908 |
| **G_raw（C0/C1）** | .253 | .481 | .858 | .986 | .819 | .922 |
| Full-Context | .261 | .486 | N/A | N/A | N/A | N/A |
| C2 聚类联合抽取 | .253 | .488 | .868 | .990 | .837 | .924 |
| C3 聚类+理解 | .261 | .492 | .873 | .990 | .849 | .924 |
| **C5 全文联合抽取** | **.271** | .500 | .881 | .988 | .845 | .926 |
| C6 全文+理解 | .268 | **.501** | **.883** | .988 | .847 | .924 |

**Test（7 对话，1,484 题，含 Reader）**：

| 方法 | EM(可答) | F1(可答) | R@5 | R@20 | 证据@4096 | 证据@8192 |
|---|---|---|---|---|---|---|
| Dense | .214 | .431 | .781 | .978 | .732 | .869 |
| **G_raw** | .240 | .484 | .864 | .987 | .831 | .913 |
| Full-Context | .239 | **.493** | N/A | N/A | N/A | N/A |
| C2 | .241 | .490 | .875 | .985 | .838 | .915 |
| C3 | .238 | .484 | **.881** | .986 | .842 | .917 |
| **C5** | **.245** | .489 | .874 | .987 | .841 | .915 |
| C6 | .243 | .492 | **.881** | .988 | **.849** | .916 |

**结论**：
1. **H2 成立**：只做联合抽取（C2/C5）就拿到几乎全部检索增益（C5 vs G_raw：dev R@5 .858→.881、test .864→.874）。
2. **H3 不成立（LoCoMo）**：C3 相对 C2 的 EM 在 test **−0.3pp**；C6 相对 C5 的 EM **−0.2pp**。
3. **H7 成立**：全文（C5/C6）≥ 聚类（C2/C3）；Full-Context（.493）≈ 最好图检索（.492）。
4. **H4 检索成立、QA 不成立**：test 证据 +0.7~1.8pp、R@5 +1.0~1.7pp，但 EM/F1 差异全部 ≤±0.5pp 且方向不一致；逐对话 F1 方差（C6 vs G_raw）：+0.2 / −1.2 / +2.6 / −0.1 / +3.1 / +0.6 / −1.4 pp（4 升 3 降）。

### 5.9 结果：Read-Time 消融（Composed vs Inferred，dev 502 题）

| 图谱 | EM(可答) | F1(可答) | R@5 | 证据@4096 | 证据@8192 |
|---|---|---|---|---|---|
| G_raw | .253 | .481 | .858 | .819 | .922 |
| **Composed-only** | **.266** | **.501** | .869 | .845 | .926 |
| Inferred-only | .253 | .494 | **.875** | .841 | .922 |
| C3（两者合并） | .261 | .492 | .873 | **.849** | .924 |

- **Composed（可验证的组合）带来最好的 QA**：EM +1.3pp、F1 +2.0pp vs G_raw，且优于合并的 C3。
- **Inferred 主要提升检索**（R@5 .875 最高）但对 EM 无贡献；合并后 F1 比 Composed-only 低 0.9pp。
- 解读：**"可验证的组合"有效，"语义推断"更像检索噪声。**

### 5.10 结果：Question-Time（Experiment E）

**Dev（3 对话，502 题）**：

| 策略 | EM(可答) | F1(可答) | R@5 | R@20 | 证据@4096 | 证据@8192 |
|---|---|---|---|---|---|---|
| Q0 原始 | .253 | .481 | .858 | .986 | .819 | .922 |
| Q1 QDM | **.258** | **.487** | .863 | .986 | .827 | .924 |
| Q2 +Retrieval Skill | **.258** | **.487** | **.875** | .991 | **.835** | .928 |
| Q3 +Answer Skill | **.258** | .483 | .873 | .991 | .829 | .922 |

**Test（7 对话，1,484 题）**：

| 策略 | EM(可答) | F1(可答) | R@5 | R@20 | 证据@4096 | 证据@8192 |
|---|---|---|---|---|---|---|
| Q0 | **.240** | .484 | .864 | .987 | .831 | .913 |
| Q1 | .237 | .483 | .860 | .987 | .826 | .907 |
| Q2 | .236 | **.489** | **.870** | .987 | .831 | .915 |
| Q3 | .238 | .472 | .868 | .986 | **.833** | .911 |

**行为统计**：

| 指标 | dev Q2（502） | test Q2（1,484） |
|---|---|---|
| QDM 判定需要拆解 | 44（8.8%） | 180（12.1%） |
| Retrieval Skill 触发 | 463（**92.2%**） | 1,372（**92.5%**） |
| 追加查询数 | 639（1.27/题） | 1,868（1.26/题） |

**结论**：
- **H5 只"检索弱成立"**：dev Q2 R@5 +1.7pp、证据 +1.6pp；test Q2 R@5 +0.6pp、证据 0.0pp。
- **QA 完全持平或负向**：dev EM +0.5pp（各策略相同）、F1 +0.2~0.6pp；test EM −0.2~−0.4pp、F1 −1.2~+0.5pp。
- **Retrieval Skill 严重过度触发（92%）**：它没有学会"证据已经够了"。这是设计缺陷，也是后续（阶段 C）加 sufficiency gate 的直接动机。
- **Q3 在 conv-30（105 题）上的 +3.7pp F1 是单对话波动**：conv-44/conv-48 与 test 全量均不复现，test 上 Q3 反而 −1.2~−1.3pp。

### 5.11 结果：联合矩阵（Experiment F）

**conv-30（105 题）2×2**（EM/F1 含 cat5 口径，与本表外其它表不同，见标注）：

| Read-Time ↓ / Q → | 原始查询 | Q3 |
|---|---|---|
| G_raw | EM .381 / F1 .520 / R@5 .903 / 证据 .886 | EM **.390** / F1 **.539** / R@5 .917 / 证据 .905 |
| C3 | EM **.400** / F1 .530 / R@5 .908 / 证据 .905 | EM .390 / F1 .527 / R@5 **.941** / 证据 **.933** |

**test 完整 2×4（1,484 题）**，格式 `EM / F1 / R@5 / 证据@4096`：

| Read-Time ↓ / Q → | Q0 原始 | Q1 QDM | Q2 +Retrieval Skill | Q3 +Answer Skill |
|---|---|---|---|---|
| **G_raw** | .240/.484/.864/.831 | .237/.483/.860/.826 | .236/.489/.870/.831 | .238/.472/.868/.833 |
| **C6 全文理解** | .243/.492/.881/.849 | **.244/.494**/.876/.843 | .239/.491/**.885**/**.851** | .238/.479/.884/.852 |

- **检索单调改善**：R@5 .864 → .885、证据 .831 → .852。
- **QA 完全跟不上**：8 格 EM 全在 .236–.244、F1 全在 .472–.494；最优 R1Q1 仅比基线 R0Q0 高 0.4pp EM / 1.0pp F1。
- **联合不是最好**：检索最好的 R1Q2/R1Q3 在 QA 上不如 R1Q1/R1Q0 → **H6 不成立**：两条路线在改善同一个中间量（召回），而召回已接近天花板（R@20 .986）。

### 5.12 结果：Skill Evolution（第一版 E4）

dev 学习（240 题抽样，每类评估 3 种检索扩展后缀），冻结后应用于 test 的 **1,137 道有 gold 证据的题**（严格同批可比）：

| 类别 | 后缀在该类别子集上的 R@5（temporal / person / place） | 选中 |
|---|---|---|
| multi-hop | .621 / .641 / .621 | none |
| temporal | .914 / .905 / **.925** | **place** |
| open-domain | .576 / .576 / .605 | none |
| single-hop | **.922** / .909 / .922 | **temporal** |

| | R@5 | 证据@4096 |
|---|---|---|
| 基线（不加技能） | .8664 | .8691 |
| 冻结库应用后 | **.8723** | **.8745** |
| 变化 | **+0.59pp** | **+0.54pp** |

三组对话均为正（+0.80 / +0.66 / +0.16 pp）。

> **方法学说明（必须随结论一起报告）**：选择规则是"该类别后缀的 R@5 > **全局**基线 .809 且提升 >0.2pp"。由于 multi-hop 子集基线本身只有 ~.62，它的后缀虽高于 .809 也被判为"无扩展"——这不是与类别自身基线比较，是**偏宽松的阈值规则**。test 上的 +0.59pp 是对冻结库效果的无偏测量，但"为什么选中这两个技能"的解释力有限。且**只验证了检索层，未跑 Reader**（无 EM/F1）。

### 5.13 结果：Comprehension 质量审计（717 条关系）

分类规则：A=与 G_raw 重复，B=同义改写（字符相似度 ≥0.85），C=单 session 可得，D=跨 session 且有双 session 证据，E=低置信推断（confidence <0.6）。

| 类别 | 含义 | 数量 | 占比 |
|---|---|---|---|
| A | 与 G_raw 完全重复 | **0** | 0% |
| B | 同义改写 | 1 | 0.1% |
| C | 单 session 可得 | **0** | 0% |
| **D** | 跨 session 且有双 session 证据 | **688** | **95.9%** |
| E | 低置信推断 | 28 | 3.9% |

逐对话（部分）：conv-43 的 155 条中 E=14（该对话噪声最大）；conv-26/30/42/47/49 全部 D=100%。

> **审计局限（重要）**：A=0 需谨慎——审计用词面匹配（exact + 0.85 字符相似度），而官方 OpenIE 与我们的抽取 prompt 措辞差异很大，**语义重复可能被低估**。E 类 3.9% 与消融中"Inferred 只提升检索、不提升 EM"一致。

### 5.14 机制与案例（阶段 A）

> 关键前提：triple / relation **只进入图，不进入 reader prompt**（reader 只读 session 原文）。因此"关系起作用"只能通过检索排序生效，不存在文本泄漏。

**案例 1（正例，Comprehension 唯一硬正例）q16**
`When did Jon go to a fair to get more exposure for his dance studio?` / Gold `24 April, 2023` / 证据 s10·D10:1。

| 方法 | Top-5 | 金标 s10 名次 | Ev@4096 | 答案 |
|---|---|---|---|---|
| G_raw | s05, s09, s11, s02, s18 | 9 | 0 | `No answer` ❌ |
| C2 | s18, s05, s01, s02, s17（**dense 回退**，top1=1.0） | 8 | 0 | `No answer` ❌ |
| **C3** | s08, s11, s18, **s10**, s15 | **4** | **1** | `24 April, 2023` ✅ |
| C5 | s18, s05, s01, s02, s17（dense 回退） | 8 | 0 | `No answer` ❌ |
| **C6** | s08, s11, **s10**, s18, s05 | **3** | **1** | `24 April, 2023` ✅ |

起作用的关系：`Jon --attended_fair_before--> worked_on_opening dance studio`（Composed，conf 0.88，sessions [10,15]，`temporal_ordering`）。**消融验证**：只保留 Composed 的 C3-composed 复现名次 4 与正确答案；只保留 Inferred 的 C3-inferred 退回 C2 的 dense 回退（名次 8、`No answer`）→ 这条 Composed 关系是**唯一作用源**。
反直觉细节：C2 的失败不是"排错"，而是**根本没进 PPR**（top-1 分数恰为 1.0 = dense fallback 的 min-max 归一化特征）。

**案例 2（检索改善但答案未兑现）q31**
`How long did it take for Jon to open his studio?` / Gold `six months`（需 s1 + s15）。关系 `Jon's dance studio --opened_after--> Jon searched for ideal spot`（Composed .93，[2,15]）把 s15 从第 4 拉到第 1，但 "six months" 在 s15 需与 s1 的时间做差 → Reader 仍答 `No answer`。**检索到位、多跳算术不到位。**

**案例 3（负例：不是理解层的错）q54**
`How is Gina's store doing?`（短、泛、无实体）。G_raw：s04 第 1、Ev=1。C2/C3：s04 掉到第 9、**dense 回退**、Ev=0。但 C2 与 C3 完全相同 → 这不是 Comprehension 的错，是"**往图里塞知识**"这个动作改变了 fact 候选池的副作用。C5/C6 不受影响。

**案例 4（负例：全文变体把多证据挤出预算）q29**
`Which cities has Jon visited?` Gold `Paris, Rome`（s2+s15）。G_raw：s2 第 1 / s15 第 3 → Ev=1、EM=1。**C5/C6：s15 掉到第 9 → Ev=0**，答案退化成 `Paris`。原因：全文抽取新增三元组把 s15 挤下去，且没有任何新关系能 bridge s15。

**案例 5（噪声标尺，必须看）**
同一批产物中存在可量化的 reader 抖动：**C3 vs C3-composed 有 98/105 题上下文集合完全相同，其中 6 题（q0,q24,q33,q47,q74,q85）EM 不同，4 题"连顺序都相同"仍 EM 不同**。→ **任何 ≤2 题的 EM 翻转都不能单独作为方法有效的证据**；阶段 A 只把有检索层独立佐证的 q16 记为 Comprehension 硬正例。

**"机会 → 兑现"漏斗（conv-30）**：

| 口径 | C3 | C6 |
|---|---|---|
| 金标 session 上至少挂 1 条新关系的题 | **100 / 105** | 99 / 105 |
| 其中关系文本与问题有词面重叠 | **90 / 105** | 75 / 105 |
| 实际发生金标名次改善 | 3 | 3 |
| 实际把答案由错变对（可归因） | **1（q16）** | **1（q16）** |

**90 题有机会 → 1 题兑现**，这是阶段 A 最重要的机制数字，也是阶段 B 直接追查的对象。

---

## 6. 阶段 B：conv-30 Prompt 优化

**范围**：仅 LoCoMo `conv-30`（19 sessions / **105 题** / 81 可答 / 24 对抗）。
**动机**：阶段 A 出现"90 题有机会 → 1 题兑现"，但无法区分是"知识不够"还是"Reader 不会用"。本阶段在同一图、同一 context 上做**隔离变量**实验。

### 6.1 三处审计（先定位问题，再改）

#### 6.1.1 Reader Prompt 审计

实际发送内容（`src/retrieval/reader.py`）：

```
[System]
You are a careful reading-comprehension assistant. You answer questions about a long
conversation between two speakers using only the retrieved context. Use the dates and session
numbers to reason about time when needed; resolve relative expressions such as 'yesterday' or
'last week' against the session date. If the context truly does not contain the answer, reply
exactly: No answer

[User]
Retrieved context:
{context}

Question: {question}

Format rules for the final line:
- It must start with 'Answer: ' and contain only the answer span itself.
- No explanation, no full sentence, no quotes, no trailing punctuation.
- For a date, give it exactly as written in the context.
- If the answer is not in the context, write 'Answer: No answer'.

Thought: think briefly about where the answer is in the context.
Answer:
```

调用参数：`max_completion_tokens=256`、`temperature=0.0`、`reasoning_effort=no_think`；解析用 `Answer:\s*(.*)` 取第一行。

| 审计问题 | 结论 | 依据 |
|---|---|---|
| 是否强制短答案 | 是（"contain only the answer span itself"+"No explanation"） | 105 题平均答案 4.5 词 |
| 是否 `max_tokens=256` 截断 | **否** | 105 题 completion 最大 35，`finish_reason` 全为 stop |
| `no_think` 是否生效 | 是 | `reasoning_tokens` 全为 null |
| Session 日期是否进上下文 | 是，**且重复一次** | `format_context` 加一层 header，`Session.text` 自带一层；多花 ~12 token/篇 |
| turn id（`D10:1`）是否可见 | **不可见** | reader context 无 dia_id；dia_id 只出现在 comprehension 的 `sessions_block` |
| 超预算处理 | **跳过整篇继续装**，故 context 可能非连续 | `select_by_budget(all_or_nothing=True)` |
| 文档顺序 | 按检索名次，**非时间序** | — |

#### 6.1.2 Comprehension Prompt 审计（结构性缺陷，全部有数据支持）

| # | 缺陷 | 证据 |
|---|---|---|
| 1 | `temporal_ordering` 与其它类型并列 → 模型产出集中在已饱和题型 | C3 26 条中 **17 条（65%）** 是 temporal_ordering；而 cat2 的 R@5 已 96.2% |
| 2 | 不约束 subject/object/predicate 长度 | 10/26 谓词 ≥5 词，平均 5.2，最长 12 |
| 3 | 不要求关系两端对齐图中已有实体 | 只有 **7/26（27%）** 两端都是已有实体；C6 仅 1/18 |
| 4 | 只检查"跨 2 session"，不检查"组合是否带来新知识" | 26 条全部通过 `needs_two_sessions`，但多数只是把两条事实改写成事件短语；真正需要 3 个 session 的仅 4 条 |
| 5 | 不检查关系是否可被未来问题检索到 | 新关系只出现在 5–11/105 题的 top-5 fact（§6.6） |
| 6 | 无冲突/时间错误校验 | 例：`Jon --lost_job_before--> started dream business` 与 `lost_job_enabled` 语义近乎重复 |
| 7 | verify 分支默认不启用，且看不到被引用的对话原文 | `verify_candidates` 只给 subject/predicate/object/evidence，模型无法核对证据 |

#### 6.1.3 检索路径审计（本项目最重要的工程发现）

官方 `HippoRAG.retrieve()` 的真实链条：

```
query embedding
  → get_fact_scores()：与全部 fact embedding 点积，取 linking_top_k = 5 个候选
  → rerank_facts()：【逐题一次 LLM 调用】（DSPy filter prompt）过滤，len_after_rerank = 5
  → 若过滤结果为空 → 整题退化为 dense_passage_retrieval（top-1 分数被 min-max 归一化为 1.0）
  → 否则用保留 fact 的 subject/object 作 PPR 种子 → Personalized PageRank → top-20 passage
```

**推论**：新增 Comprehension 关系必须先进入 **top-5 fact 候选**并通过 LLM filter，才可能影响 PPR。**"有没有进 PPR"往往比"关系内容好不好"更决定结果。**

为此在 `src/prompt_optimization/retrieval_diag.py` 复现了官方 retrieve 的完整中间状态（`facts_before_rerank` / `facts_after_rerank` / `fallback_dense` / top facts），**未改官方代码**。一致性校验：G_raw 的 ranked_docs 与历史记录逐题一致 **93/105**，C2 的 R@3/R@5/MRR/Evidence 与历史**完全一致**（.8413/.9032/.7586/.8952）；剩余差异来自 GPU/embedding 的批次级数值非确定性（单题检索 vs 批量检索）。

### 6.2 评测口径差异（先量化，避免把口径当方法）

| 方法 | 项目 EM | 项目 F1 | 官方 F1(1–4) | 官方严格拒答率 | 官方多选协议 |
|---|---|---|---|---|---|
| G_raw (P0) | .210 | .390 | **.426** | 0.0% | **83.3%（20/24）** |
| C2 | .222 | .404 | .433 | 0.0% | — |
| C3 | .247 | .415 | **.445** | 0.0% | — |
| C5 | .210 | .405 | .442 | 0.0% | — |
| C6 | .198 | .401 | .437 | 0.0% | — |

- 官方口径把 F1 抬高约 **2.9–4.1pp**（主要来自 cat4 词干化：`Hoodies→hoodie`、`excitement→excited`）。
- 官方 cat-5 严格拒答率 **0%** 是协议错配（我们输出 "No answer"，官方只认 "not mentioned"）。
- 项目 EM 与官方 EM 在本数据上几乎相同（.210 vs .210）→ **换 EM 口径救不回这些题**。

**真实差异案例**（官方 F1 − 项目 F1）：

| qid | cat | gold | pred | 项目 F1 | 官方 F1 |
|---|---|---|---|---|---|
| q77 | 4 | Hoodies | hoodie | .00 | **1.00** |
| q2 | 4 | by dancing | dance | .00 | **.67** |
| q76 | 4 | excitement | excited and positive | .00 | **.67** |
| q49 | 4 | cozy and comfortable | cozy and inviting | .67 | **.50**（反过来更低） |
| q25 | 1 | one-on-one mentoring…, workshops and classes | dance classes and workshops, one-on-one… | .55 | .47 |
| q29 | 1 | Paris, Rome | Paris and Rome | EM 0 / F1 .67 | EM 1 / cat1 F1 .667 |

→ 同一答案在三种口径下可以同时是 **0 / 1 / 0.667**。项目 EM 对"顺序、and、词形、日期粒度"都非常脆。

### 6.3 错误归因（G_raw，105 题，规则化 + 人工核对）

| 指标 | 数值 |
|---|---|
| 可回答题 | 81 |
| 正确（EM=1） | 17（21.0%） |
| 错误 | 64 |
| 证据完整进入 4096 | 69/81（**85.2%**） |
| 证据完全缺失 | 12/81（14.8%） |
| 答案截断（completion ≥250） | **0** |
| 可回答题被拒答 | 15（其中 **13 题证据完整**） |
| 对抗题正确拒答 | 23/24（95.8%） |

主错误分布（同题可带次要标签）：

| 标签 | 含义 | 题数 | 占错误 |
|---|---|---|---|
| E | 答案表达问题（语义接近但过短/措辞不同/部分） | 23 | 35.9% |
| F | **过度拒答**（证据完整却 No answer） | 13 | 20.3% |
| A | 检索错误（金标 session 不在上下文） | 11 | 17.2% |
| C | 时间推理错误（证据在，日期/跨度算错） | 9 | 14.1% |
| G | 错误推断/幻觉 | 4 | 6.2% |
| D | 多事实整合错误 | 3 | 4.7% |
| B | 证据不完整 | 1 | 1.6% |
| H | 评测器/语义等价（次要） | 16 | 25.0% |

按类别：

| 类别 | n | 正确 | 主错误分布 |
|---|---|---|---|
| cat1 multi-hop | 11 | 4 | A:4, D:1, F:1, B:1 |
| cat2 temporal | 26 | 6 | **C:9, F:9**, A:2 |
| cat3 open-domain | 0 | — | — |
| cat4 single-hop | 44 | 7 | **E:23**, A:5, G:4, F:3, D:2 |

**结论**：**「找到了证据但没答对」比「没找到证据」重要得多**——69 题证据完整中有 53 题仍错。

### 6.4 Oracle / Full-Context 诊断（本阶段最重要的对照）

| 上下文 | Prompt | n | EM | F1 | 官方F1 | 过拒率 | 平均 context token |
|---|---|---|---|---|---|---|---|
| R1 正常 G_raw 检索 | P0 | 81 | .210 | .390 | .426 | 18.5% | 3.9k（4096 预算） |
| R1 正常 G_raw 检索 | P7 | 81 | .235 | .521 | .546 | 8.6% | 3.9k |
| R2 Gold Evidence Oracle | P0 | 81 | **.185** | **.387** | .402 | 17.3% | 761 |
| R2 Gold Evidence Oracle | P7 | 81 | **.272** | **.603** | .627 | 6.2% | 761 |
| R3 Full Conversation | P0 | 105 | .198 | .360 | .400 | 17.3% | 11,435 |
| R3 Full Conversation | P7 | 105 | .259 | .546 | .577 | 4.9% | 11,796 |

三条解读：

1. **P0 在 Oracle 上没有变好，甚至略差**（.390 → .387 F1）→ 把正确证据直接给它，它仍答不对 ⇒ **瓶颈不是检索**。
2. **P7 在 Oracle 上显著变好**（.521 → .603 F1）→ 优化后的 Prompt 能利用更干净的证据，但正常检索下只能兑现一部分 ⇒ **检索仍有约 4–8pp 的可兑现空间**。
3. **Full Conversation 比 Oracle 差**（P7：.546 vs .603；P0：.360 vs .387）→ 更多上下文带来干扰，"把整段对话塞进去"不是解法。
   （注意：R2 平均仅 761 token、R3 11.4k token，**不是等预算比较**。）

### 6.5 Reader Prompt 实验（P0–P8）

**设计**：固定 G_raw 的 4096-token 检索上下文（逐题校验 token 数与历史一致）；P0 复用历史答案（零调用），P1–P8 每题一次真实调用，统一 `max_tokens=256`、`temperature=0`、`no_think`。
**划分**：dev 40 题（34 可答 + 6 对抗）/ val 65 题（47 可答 + 18 对抗），seed 42；**`reader/pilot/selection.json` 在跑 val 之前就已写入**（防选择性报告）。

| ID | 设计意图 | 关键改动 |
|---|---|---|
| P0 | 基线（现网 prompt） | 短 span + 严格拒答 |
| P1 | 简洁但完整 | 允许短句 → **失败**（答案 25.3 词，EM 0） |
| P2 | 先整合再回答 | 要求先在思考里整合 → 长答案，失败 |
| P3 | 时间敏感 | 强化时间规则、弱化拒答 → cat2 大涨但对抗拒答 0% |
| P4 | 拒答校准 | 只在证据不足时拒答 → EM 反而降 |
| P5 | 按问题类型 | 按题型切换格式 → 失败（长答案） |
| P6 | 官方 LoCoMo 风格 | "short phrase + exact words"，日期可 approximate → F1 高、EM 不升 |
| **P7** | **综合（P0+P3+P6）** | 短 span/逗号列表 + 时间规则（只给日期）+ 列表完整 + 拒答校准 + 跨 session 组合 + 禁止编造 |
| P8 | P7 + 前提检查 | 增加"若题目预设了未发生的事 → No answer" |

**P7 正文（实际部署的 Reader，阶段 C 复用）**：

```
[System]
You are a careful reading-comprehension assistant answering questions about a long conversation
between two speakers. Use only the retrieved context and answer with exact words from the context
whenever possible. Each session header gives the date of that conversation. If the context truly
does not contain the answer, reply exactly: No answer

[User]
Retrieved context:
{context}

Question: {question}

Rules:
1. Give only the answer span or a comma-separated list of spans, using exact words from the
   context. Do not write a sentence and do not explain your reasoning.
2. If the question asks which items, events, people, or places, list every supported item
   separated by commas.
3. If the question asks why, give all supported reasons as a short phrase or as
   comma-separated short phrases.
4. If the question asks when, output only the date, with no surrounding words such as
   "around", "approximately", or "in the conversation". The date may be the session date when
   the event is described as "yesterday", "last week", or "next month" relative to it.
5. If the question asks how long, output only the duration computed from the supported start
   and end dates.
6. Combine facts from different sessions when needed. Do not refuse when the required
   evidence is present; only write "No answer" if it is genuinely absent from the context
   and cannot be simply derived from it.
7. Never invent facts or dates.

Answer:
```

**结果：dev 40 题**

| Prompt | EM | F1 | 官方F1 | 过拒率 | 对抗拒答 | 平均词数 |
|---|---|---|---|---|---|---|
| P0 | .235 | .394 | .423 | 26.5% | 100.0% | 3.7 |
| P1 | .000 | .178 | .193 | 20.6% | 83.3% | 25.3 |
| P2 | .029 | .220 | .239 | 23.5% | 83.3% | 21.1 |
| P3 | .206 | .444 | .450 | 2.9% | **0.0%** | 21.2 |
| P4 | .147 | .364 | .403 | 20.6% | 83.3% | 12.8 |
| P5 | .088 | .345 | .349 | 20.6% | 83.3% | 13.0 |
| P6 | .147 | .523 | **.584** | 11.8% | 83.3% | 5.3 |
| **P7** | **.294** | **.600** | **.618** | 11.8% | 83.3% | 8.4 |
| P8 | .206 | .504 | .544 | 20.6% | **100.0%** | 8.9 |

**结果：val 65 题**（P1–P6 未在 val 上跑）

| Prompt | EM | F1 | 官方F1 | 过拒率 | 对抗拒答 |
|---|---|---|---|---|---|
| P0 | .191 | .387 | .428 | 12.8% | 94.4% |
| **P7** | .191 | **.465** | **.494** | 6.4% | 94.4% |
| P8 | .191 | .464 | .482 | 8.5% | 94.4% |

**全 105 题 + 分题型**：

| 数据 | Prompt | EM | F1 | 官方F1 | Multi-hop F1 | Temporal F1 | 拒答率 |
|---|---|---|---|---|---|---|---|
| all 105 | P0 | .210 | .390 | .426 | .489 | .292 | .185 |
| all 105 | **P7** | **.235** | **.521** | **.546** | **.508** | **.818** | .086 |
| all 105 | P8 | .198 | .481 | .508 | .403 | .741 | .136 |
| dev 40 | P7 | .294 | .600 | .618 | .508 | .806 | .118 |
| val 65 | P7 | .191 | .465 | .494 | — | .829 | .064 |

| 类别 | n | P0 EM | P7 EM | P0 F1 | P7 F1 |
|---|---|---|---|---|---|
| cat1 multi-hop | 11 | 36.4% | 27.3% | .489 | .508 |
| cat2 temporal | 26 | 23.1% | **42.3%** | .292 | **.818** |
| cat4 single-hop | 44 | 15.9% | 11.4% | .423 | .349 |
| cat5 adversarial | 24 | 拒答 95.8% | 拒答 91.7% | — | — |

**关键现象**：
- **P1/P2/P5 全部失败**，原因不是内容错，而是答案变成 20–25 词长句/复述 → 项目 EM 对长答案几乎必然判 0。**"要求完整"若不与"只输出 span / 逗号列举"绑定，会直接毁掉分数。**
- **P3 把 cat2 从 EM 25%/F1 25% 提到 42%/55%，过拒率 50%→0%**，但对抗题**全部不再拒答（0%）** → 验证了"时间题被过度拒答"，也暴露了去掉拒答约束的副作用。
- **P6（官方风格）F1 最高但 EM 不升**：答案短、精确，但日期常带 "around"、列表用 "and" 而非逗号。
- **P7 是唯一同时抬高 EM/F1/官方F1 的方案**；收益来自三处：时间规则、拒答校准、列表完整。
- **val 暴露真实代价**：Temporal 翻倍（EM 21.4%→42.9%、F1 32.9%→82.9%），但 Single-hop EM −9.1pp（答案变长）。**不能用"全面提升"表述。**
- **P8 恢复对抗拒答（100%）但 dev EM 从 .294 压到 .206** → 过严拒答的代价，未采用。

**输出波动（同 prompt / 同 context / temperature=0，禁用缓存，14 题代表集 ×3 次真实调用）**：

| Prompt | 词面不稳定率 | 得分不稳定率 | 平均两两 Jaccard | 三次 EM | 多数票 EM |
|---|---|---|---|---|---|
| P0 | 14.3% | 14.3% | 0.976 | .214 / .286 / .357 | .286 |
| P7 | 28.6% | **0.0%** | 0.886 | .143 / .143 / .143 | .143 |

→ P0 的单题 EM 可在 3 次运行间翻转（2 题）；**P7 措辞会变但得分稳定**（这是 P7 的一个实际优势）。
→ **105 题上 1–2 题的差异属于噪声。**

**`max_tokens=256` 截断敏感性**：P0 零截断（最大 35 token）；P7 在长上下文下 1–2 题截断（q50/q68/q3），用 512 重跑后 **4 行 EM 全部 0→0**，F1 仅 +0.02~+0.22，对总 F1 影响 ≤0.2pp。但 P7 的"完整枚举"规则确实会诱发 **runaway enumeration**（这是下一版要修的缺陷）。

### 6.6 Comprehension Prompt 实验（COMP-P0–P5）

**设计**：固定 C2 的三元组集合（conv-30：228 条三元组），**只替换关系生成 prompt**，其余（schema、建图流程、检索参数）完全一致 → 严格隔离"关系质量"。

| ID | 设计意图 | 关键约束 |
|---|---|---|
| COMP-P0 | 原 Prompt（C3） | 无长度/实体约束，temporal_ordering 与其它并列 |
| COMP-P1 | Evidence-grounded composition | 只保留必须组合两 session 的关系；短实体短谓词；temporal_ordering 降为最低 |
| COMP-P2 | Cross-session bridge | 把对话读成连续历史，优先 plan→outcome / event→consequence / state before-after / identity unification |
| COMP-P3 | Event & state tracking | 先建时间线，再输出状态转移 |
| **COMP-P4** | **Retrieval-oriented** | **面向"未来未知问题"构建可检索关系**：短实体（≤6 词）+ 1–4 词具体谓词 + 两端尽量用已有实体 + 禁止把答案塞进谓词 |
| COMP-P5 | Generate → verify | 最多 12 条候选 → 带"被引用 turn 原文"的 verifier 逐条审查 |

**COMP-P4 硬约束原文要点**（部署到阶段 C 的检索底座即为 `COMP-P4_top5`）：

```
1. 至少两个来自不同 session 的 source_dia_ids
2. subject/object：短、具体、可读的名词短语（<=6 词），优先使用原子事实中已出现的名字
3. predicate：短具体动词短语（1-4 词），小写加下划线；禁止 related_to / associated_with / >4 词
4. evidence：每个被引 session 一句 <=25 词的逐字引用
5. relation_type ∈ {causal, enabling_link, state_change, goal_outcome, event_dependency,
   identity_unification, temporal_ordering}；仅当顺序本身就是新信息时才用 temporal_ordering
6. derivation ∈ {Composed, Inferred}；confidence 0-1
7. 不得输出已是原子事实的关系，不得输出同义重复
8. 若无合格关系，返回 {"relations": []}
```

**生成质量**：

| Prompt | 原始候选 | 格式合法 | 最终关系 | temporal_ordering | 谓词≥5词 | 两端已有实体 | 平均置信度 |
|---|---|---|---|---|---|---|---|
| COMP-P0 (C3) | — | — | 26 | **17** | **10** | 7（27%） | .883 |
| COMP-P1 | 19 | 14 | 14 | 0 | 0 | 1（7%） | .811 |
| COMP-P2 | 23 | 19 | 19 | 0 | 0 | 1（5%） | .794 |
| COMP-P3 | 30 | 27 | 27 | 0 | 0 | 1（4%） | .834 |
| COMP-P4 | 20 | 16 | 16 | 2 | 0 | 1（6%） | .861 |
| COMP-P5 | 24 | 18 | **7** | 1 | 0 | 0 | .879 |

- 所有被拒候选的唯一原因是 `needs_two_sessions`（模型只引用了 1 个 session）；无格式错误。
- **P5 的 verifier 从 18 条只保留 7 条**，大量拒绝"因果过度声称"，说明 verifier 在起作用，但也可能误杀。
- 新 Prompt 关系更短更具体，但**两端仍很少命中已有实体**——这是未解决的问题。

**检索结果（105 题）**：

| 图 | 新增关系 | R@1 | R@3 | R@5 | MRR | nDCG@10 | Ev@4096 | 丢失证据题 | 新增证据题 | R@5 丢失 | R@5 新增 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G_raw | 0 | .6198 | .8238 | .8984 | .7619 | .8067 | .8857 | 2 | 1 | 2 | 2 |
| C2（无理解） | 0 | .5944 | .8413 | .9032 | .7586 | .8026 | .8952 | — | — | — | — |
| COMP-P0（原版） | 26 | .5944 | .8317 | .9079 | .7575 | .8041 | .9048 | 0 | 1 | 1 | 2 |
| COMP-P1 | 14 | .5921 | .8532 | .9103 | .7547 | .8034 | .8952 | 1 | 1 | 1 | 2 |
| COMP-P2 | 19 | .5944 | .8317 | .9079 | .7598 | .8064 | .8952 | 0 | 0 | 0 | 1 |
| COMP-P3 | 27 | .5944 | .8460 | .9079 | .7579 | .8043 | .8952 | 1 | 1 | 0 | 1 |
| **COMP-P4** | 16 | .5944 | **.8508** | **.9127** | **.7638** | **.8100** | **.9048** | **0** | **1** | **0** | **1** |
| COMP-P5 | 7 | .5944 | .8413 | .9032 | .7582 | .8022 | .8952 | 0 | 0 | 0 | 0 |

- 最好的 **COMP-P4**：R@3 +0.95pp、R@5 +0.95pp、MRR +0.52pp、nDCG +0.74pp、Ev +0.95pp（**恰好 +1 题 q16**），且**零负迁移**。
- P1/P3 都是"q31 进、q24 出"（一进一出，Ev 不变）；P2/P5 完全没改变 Evidence。
- **没有系统性排序改善**：所有 MRR 变化都在 ±0.5pp 内。

**分层分析（P4 vs C2，这张表比总分更能说明问题）**：

| 分层 | n | C2 Ev@4096 | P4 Ev@4096 | C2 R@5 | P4 R@5 | 证据增加 | 证据丢失 |
|---|---|---|---|---|---|---|---|
| 原始检索已成功（G_raw Ev=1） | 93 | .989 | .989 | .975 | .975 | 0 | **0** |
| 原始完全失败（Ev=0 且 partial=0） | 7 | .286 | **.429** | .286 | **.429** | 1 | 0 |
| 原始部分证据（0<partial<1） | 5 | .000 | .000 | .433 | .433 | 0 | 0 |
| 证据已在上下文但答案错误 | 54 | .981 | .981 | .963 | .963 | 0 | 0 |
| **真正跨 session 多证据问题（cat1）** | **11** | **.455** | **.455** | .621 | .621 | **0** | **0** |

→ 三条硬事实：
1. **P4 零负迁移**（93 道原始成功题证据覆盖无一下降）。
2. **P4 的全部收益来自 7 道"原始完全失败"题中的 1 道（q16）**。
3. **11 道真正跨 session 的多证据题，Evidence 完全没变（.455 → .455）**——**为"跨 session 理解"设计的 Prompt，救回的是一道单 session 的时间题，不是多跳题。** 这是本阶段对研究假设最重要的负面证据。

### 6.7 机制追踪：新关系到底有没有进入检索？

把每题最终 top-5 fact 与新增关系做规范化匹配（`scripts/prompt_opt/06b_trace_facts.py`）：

| 图 | 关系数 | 新关系出现在 top-5 fact 的题数 | 命中次数 | 被用到的关系数 |
|---|---|---|---|---|
| COMP-P1 | 14 | 8 / 105 | 8 | 4 |
| COMP-P2 | 19 | 7 / 105 | 7 | 5 |
| COMP-P3 | 27 | 11 / 105 | 12 | 11 |
| **COMP-P4** | 16 | **5 / 105** | 5 | 4 |
| COMP-P5 | 7 | 3 / 105 | 3 | 1 |

COMP-P4 的 q16 命中链：关系 `Jon's fair attendance --generated_leads_for--> Jon's dance studio`（s10+s14，conf .86）→ q16 的 gold s10 从 C2 的第 8 名（dense fallback）升到**第 1 名**，Ev 0→1。**这是全阶段唯一"新关系 → top-5 fact → PPR → 证据覆盖改善"的完整链路。**

对照：P3 的 27 条里有 11 条进过 top-5 fact（次数更多），但**都没有把 gold session 拉进 4096 预算** → **命中次数多 ≠ 命中正确题目**。

**漏斗（H9 的定量形式）**：`99/105 题有触及金标的关系 → 5/105 进入最终 fact 候选 → 1/105 兑现为证据覆盖`。

### 6.8 数量控制（把"关系更好"与"图更大"分开）

把 COMP-P4 的 16 条按模型自报 `confidence` 降序保留前 10 条（`COMP-P4-c10`），其余设置完全相同：

| 图 | 关系数 | R@3 | R@5 | MRR | nDCG@10 | Ev@4096 | dense fallback |
|---|---|---|---|---|---|---|---|
| C2 | 0 | .8413 | .9032 | .7586 | .8026 | .8952 | 16 |
| COMP-P4 | 16 | .8508 | .9127 | .7638 | .8100 | .9048 | 15 |
| **COMP-P4-c10** | **10** | **.8508** | **.9127** | .7665 | .8101 | **.9048** | 15 |

去掉 6 条低置信关系后**所有主要检索指标完全不变**，q16 的 s10 仍然第 1、Ev=1。
三个自然对照：P3（27 条）Ev 与 C2 相同；P5（7 条）Ev 完全不变；P1（14）与 P3（27）都是 q31 进/q24 出。

→ **关系数量与检索收益无单调关系；关键是"是否进入 top-5 fact 候选并指向正确 passage"。**
（残余混杂：被删关系也可能影响 reranker 的输入候选，本实验只报告观测结果。）

### 6.9 追加消融：关闭 LLM fact reranker

动机：官方 reranker 是逐题 LLM 调用。我们在 `retrieval_diag.py` 加 `use_llm_rerank=False` 分支：**直接用 embedding 分数取 top-k fact 作 PPR 种子，0 次 LLM 调用**。两种设置：`top5`（与 reranker 最终 fact 数一致）、`top20`（放大候选窗）。

**检索（105 题，无 LLM）**：

| 图 / 设置 | R@3 | R@5 | MRR | nDCG@10 | Ev@4096 | dense fallback |
|---|---|---|---|---|---|---|
| G_raw（官方 rerank） | .8238 | .8984 | **.7619** | **.8067** | .8857 | 14 |
| G_raw / no-rerank top5 | .8111 | .8865 | .7173 | .7769 | .8762 | **0** |
| G_raw / no-rerank top20 | .6087 | .7437 | .5431 | .6211 | .7619 | 0 |
| C2（官方 rerank） | .8413 | .9032 | **.7586** | **.8026** | .8952 | 16 |
| C2 / no-rerank top5 | .8405 | .9103 | .7164 | .7732 | .8952 | **0** |
| C2 / no-rerank top20 | .7190 | .8127 | .6225 | .6913 | .8095 | 0 |
| COMP-P4（官方 rerank） | .8508 | .9127 | **.7638** | **.8100** | .9048 | 15 |
| **COMP-P4 / no-rerank top5** | .8500 | **.9294** | .7224 | .7797 | **.9143** | **0** |
| COMP-P4 / no-rerank top20 | .7214 | .8056 | .6303 | .6991 | .7905 | 0 |

**结论 1：top5 无重排不是灾难，也不是免费午餐。**
R@3/R@5/Ev 基本保持（P4 R@5 +1.7pp、Ev +0.95pp），但 **MRR/nDCG 稳定下降 3–4pp** → LLM reranker 的价值在"把最对的那条 fact 排到第一"，不在"多召回"。dense fallback 从 14–16 题降到 **0**（不存在"过滤为空"路径）。

**结论 2：top20 明确更差**（G_raw R@5 .898→.744）→ 更多 embedding-only fact 作种子会让 PPR 主题漂移。**"多给种子"不是解法。**

**结论 3：关闭 reranker 反而让 Comprehension 的收益翻倍。**
正常重排：C2 Ev .8952 → P4 .9048（+1 题）；无重排 top5：.8952 → **.9143（+2 题 q16、q57）**，R@5 .9103 → **.9294（+1.9pp）**。
逐题追踪：无重排把 q57 的 s07、q44 的 s09 拉到第 1；代价是 q23 的 R@5 .5→.25。
→ **LLM reranker 在 q57/q44 上压制了本可用的证据**，这解释了"P4 的收益为什么只有 q16"。

**QA（同 105 题、同 4096 预算、P0/P7 两个 Reader）**：

| 图 | Reader | EM | F1 | 官方F1 | 过拒率 | 对抗拒答 |
|---|---|---|---|---|---|---|
| G_raw + rerank | P0 | .210 | .390 | .426 | .185 | .958 |
| G_raw + rerank | P7 | .235 | .521 | .546 | .086 | .917 |
| G_raw + no-rerank top5 | P0 | .185 | .388 | .418 | .222 | .917 |
| G_raw + no-rerank top5 | P7 | .235 | .531 | **.569** | .062 | .875 |
| P4 + rerank | P0 | .210 | .387 | .425 | .198 | .958 |
| P4 + rerank | P7 | .247 | .524 | .539 | .086 | .917 |
| **P4 + no-rerank top5** | **P0** | .222 | .417 | .445 | .173 | .958 |
| **P4 + no-rerank top5** | **P7** | **.259** | **.539** | **.569** | .062 | .917 |

- P0 在 G_raw 上被无重排**拖累**（EM .210→.185）；P7 在 G_raw 上持平/略好，在 P4 上**全面变好**。
- 最佳组合：**P4 + no-rerank top5 + P7 = EM .259 / F1 .539 / 官方 F1 .569**，相对 E3（.247/.524/.539）为 +1.2pp EM / +1.5pp F1 / +3.0pp 官方 F1。
- 分题型：cat2 Temporal EM .500 / F1 .869（正常 E3 为 .462/.831），cat1/cat4 EM 不变 → 收益仍集中在 Temporal。
- 与 E3 的逐题 EM 差异只有 q11（0→1）→ **+1.2pp EM 仍是 1 题量级，在噪声带内**；官方 F1 +3pp 更值得关注。
- 工程上：无重排每图约 50 秒、**0 次 LLM 调用**；有重排每图 105 次调用且受共享配额影响。

### 6.10 联合实验（Graph × Reader）

| 实验 | 图谱 | Reader | EM | F1 | 官方F1 | 过拒率 | 对抗拒答 | 平均词数 |
|---|---|---|---|---|---|---|---|---|
| E0 | G_raw | P0 | .210 | .390 | .426 | 18.5% | 95.8% | 4.5 |
| E1 | G_raw | P7 | .235 | .521 | .546 | 8.6% | 91.7% | 12.2 |
| E2 | COMP-P4 | P0 | .210 | .387 | .425 | 19.8% | 95.8% | 4.6 |
| **E3** | **COMP-P4** | **P7** | **.247** | **.524** | .539 | 8.6% | 91.7% | 13.0 |
| 附加 | C2 | P7 | .235 | .515 | .530 | 9.9% | 91.7% | 12.4 |

| 实验 | cat1 EM / F1 | cat2 EM / F1 | cat4 EM / F1 |
|---|---|---|---|
| E0 | 36.4% / .489 | 23.1% / .292 | 15.9% / .423 |
| E1 | 27.3% / .508 | 42.3% / .818 | 11.4% / .349 |
| E2 | 36.4% / .568 | 19.2% / .254 | 18.2% / .421 |
| E3 | 27.3% / .522 | **46.2% / .831** | 11.4% / .344 |

**三个可分离的问题**：

1. **Prompt 本身带来多少？** E1−E0（图与上下文完全不变）= **EM +2.5pp、F1 +13.1pp、官方 F1 +12.0pp**；Temporal EM +19.2pp / F1 +52.6pp；代价 Single-hop EM −4.5pp / F1 −7.4pp。**全阶段最大且最稳的单项收益，但不是"全面提升"。**
2. **新 Comprehension Prompt 带来多少？** E2−E0 = EM 0 / F1 −0.3pp（表面无收益，但 q16 由 `No answer` 变正确，同时 4 个与图无关的 reader 翻转 q0/q11/q26/q78 把总数抹平）。用同一 Reader 比较更干净：**E3 vs C2+P7 = EM .247 vs .235（+1.2pp）、F1 .524 vs .515（+0.9pp）**。→ **Comprehension 的独立收益 = 1 道题**。
3. **两者互补吗？** E3−E1 = EM +1.2pp / F1 +0.3pp；E3−E2 = EM +3.7pp / F1 +13.7pp。**基本可加，且 Reader 贡献占绝大多数**；但 E3 官方 F1（.539）略低于 E1（.546）→ 不能宣称 E3 全面优于 E1。

**联合证据链（q16，唯一完整闭环）**：
G_raw+P0：s10 第 9、Ev=0、`No answer`（错）→ P4+P0（E2）：s10 第 1、Ev=1、`24 April, 2023`（对）→ P4+P7（E3）：同样正确。
**关键限定：在 P4 图上 P0 也能答对 → P7 不是必要条件。Comprehension 修的是"证据到达"，Reader 修的是"证据→答案"，两者解决不同故障。**

### 6.11 案例（阶段 B，真实数据）

**正例 1：Comprehension 独立增益（q16）** — 见 §6.7 完整链路。

**正例 2：Reader Prompt 修复时间拒答（5 题，证据都已完整）**：

| qid | Gold | P0 | P7 | 机制 |
|---|---|---|---|---|
| q7 | 29 January, 2023 | No answer | 29 January, 2023 | session header 就是答案 |
| q13 | 16 March, 2023 | No answer | 16 March, 2023 | 同上 |
| q19 | 10 May, 2023 | yesterday | 10 May, 2023 | "yesterday" 锚定 session date |
| q30 | 20 June, 2023 | tomorrow | 20 June, 2023 | "tomorrow" 锚定 |
| q38 | 21 July 2023 | Last Friday | 21 July, 2023 | 相对时间锚定 |

**负例 1：Comprehension 把证据挤出预算（q24，P1/P3）**
Gold `fair, networking events, dance competition`（s8/s10/s16）。C2：`s16,s18,s08,s04,s14`、Ev=1、答案正确。COMP-P1/P3：s10 掉到第 7，4096 只装下 `s16,s18,s08,s05,s14,s04`，**s10 被挤出**（Ev 0）。→ 负迁移的机制不是"关系内容错"，而是"多出来的实体/边改变了 fact 候选与 PPR 权重"。

**负例 2：P7 的列表枚举不稳定（q24 / q4）**
q24：P0 `a fair, networking events, a dance competition`（EM 1）；P7 `networking events, a fair, the last networking event`（EM 0，漏 dance competition、多一个错项）。
q4：P0 F1 .32；P7 把 5 条同义证据用逗号拼成一句（F1 .20）。
→ P7 的"完整"规则被模型理解为"把所有相关句子都列出来"。

**负例 3：q54 是"加知识"的副作用，不是 Comprehension 的错** — 见 §5.14 案例 3。

**负例 4：对抗题前提误判（q85）**
`What is Gina's attitude towards participating in the dance festival?`（Gold：不可回答）。G_raw+P0 `No answer`（对）；P4 图上 P7 回答 Gina 很兴奋（错）。P8 的前提检查能修，但代价是 dev EM −8.8pp；官方多选协议下 P0 20/24、P7 21/24 → **更多是"回答格式"问题，不是知识缺失**。

**负例 5：日期 off-by-one（q33）**
Gold `20 June, 2023`，证据 D16:6 说 "yesterday"，session 16 日期是 21 June。P7 答 `21 June, 2023`（用 session 日期没减 1 天）→ 相对时间规则没有覆盖"事件日期 ≠ session 日期"。

**正例 3：官方口径下的措辞差异（不是模型错）**：q77 `Hoodies` vs `hoodie`（项目 EM/F1 = 0/0，官方 F1 = 1.00）；q2 `by dancing` vs `dance`；q76 `excitement` vs `excited and positive`。

### 6.12 阶段 B 结论（十条，按贡献排序）

1. **当前 EM/F1 低的主要原因**：① Reader 的短答案/拒答策略（13 题证据完整却拒答，EM 上限损失约 16pp）；② 时间推理（cat2 另 9 题日期算错，F1 仅 .292）；③ 答案表达/部分正确（23 题）；④ 检索缺失（12/81 题，只解释约 15% 的错误）；⑤ 评测口径（+3–4pp，非主因）；⑥ 输出波动（1–2 题，噪声）。
2. **仅改 QA Prompt 的收益**：全 105 题 EM +2.5pp、F1 +13.1pp、官方 F1 +12.0pp；**题型依赖极强**（Temporal 大涨、Single-hop 变差），dev 上更高、val 上 EM 持平 → 不能宣称"一个 Prompt 全面解决"。
3. **Oracle 水平**：P0 Oracle ≈ 正常检索（.387 vs .390）；P7 Oracle .603 > 正常 .521 → **Reader 优化后检索仍有 4–8pp 可兑现**，但 Oracle 也只有 27.2% EM → Reader 与评测仍是限制。
4. **Comprehension 是否产生普通抽取无法提供的知识**：是，但极少且方向出乎预期——**救回的是单 session 时间题，不是 11 道跨 session 多证据题（Ev .455→.455）**。
5. **是否改善检索而非仅增大图**：改善存在但高度局部（+1 题），且**数量不是驱动因素**（16→10 条指标不变；P3 的 27 条反而无收益）。
6. **新机制候选（可写成方法论）**：把"**是否进入 top-5 fact 候选**"作为 Comprehension 的质量指标；把生成目标从"知识正确"改为"在 top-k fact 竞争中可被检索到"（`99 有机会 → 5 进候选 → 1 兑现` 的漏斗比关系数量更有解释力）。
7. **优化优先级**：Reader > 检索的 fact 候选/reranker > Comprehension 的规模化。
8. **reranker 不是纯收益**：提高 top-1（MRR +3–4pp），但会在 q57/q44 上压制正确证据，并把 P4 的 Ev 收益从 +2 题吃成 +1 题。
9. **若论文指标是 EM/F1/Evidence**：关掉 LLM reranker + embedding top5 + COMP-P4 + P7 是当前最好组合（.259/.539/.569），且省掉每图 105 次调用；**若指标是 top-1/MRR**，保留 reranker。**top20 一律不要用。**
10. **本阶段全部结论只在 conv-30 上成立**，是探索性实验，不能作为跨对话泛化证明。

---

## 7. 阶段 C：conv-30 Skill 自进化（RSI 闭环）

**目标**：验证 Question-Time 的"进化"能否作为**独立策略层**——不改模型权重，只累积 prompt 文本形式的技能记忆（Skill Memory），并有版本、合并、淘汰与回滚。

**固定底座**：检索使用冻结的 **`COMP-P4_top5`** 排序，即 **COMP-P4 图 + embedding top-5 fact 直接作 PPR 种子（关闭官方 LLM fact reranker）**，4096-token Reader context，Reader = **P7**。
（选它的原因见 §6.9：该组合在 105 题上 EM/F1/官方 F1 最高，且每图省掉 105 次 rerank 调用。注意 `_top5` 后缀的含义是"无 LLM 重排的 embedding top-5"，不是"top-5 fact"的一般说法。）

### 7.1 数据划分（在任何技能学习之前生成）

seed **20261110**，按 category 分层（**只分 4 类**，category 3 完全不入划分）：

| 子集 | n | multi-hop(1) | temporal(2) | single-hop(4) | adversarial(5) |
|---|---|---|---|---|---|
| adaptation | 43 | 4 | 11 | 18 | 10 |
| development | 19 | 2 | 5 | 8 | 4 |
| evaluation | 43 | 5 | 10 | 18 | 10 |

实现细节：每类先 `sorted()` 再 `rng.shuffle()`，按配额切片并 `assert len == na+nd+ne`；adaptation 的**学习顺序**再单独 shuffle 一次（集合划分顺序与学习顺序分离）。
隔离保证：① 划分在学任何技能前生成；② 两个进化循环只遍历 `adaptation_order`；③ 反思/验证只用 adaptation 题 + 冻结检索；④ 快照选择只读 dev 评测文件；⑤ evaluation 仅在最终评测阶段被读取。

> **caveat**：conv-30 已被前序轮次研究过，evaluation 是**内部探索性测试**，不是独立最终 benchmark。

### 7.2 Answer Skill 进化循环（完整状态机）

单步顺序（`qecode/answer_evolve.py`）：

```
1. select_skills(active, question, cat, k=2)         # 技能检索
2. answer()                                          # 有选中技能→带技能作答；无→复用冻结 P7 缓存行
3. 判分 + 记 answer_support
4. 反事实归因 record_marginal()                      # 只在 E0错→本轮对 / E0对→本轮错 时计分
5. wrong = (em == 0) or (cat != 5 and f1 < 0.5)      # cat-5 只看 EM
6. 反思：LLM 输出 error_axis / error_type / diagnosis / generalizable_pattern /
   update_skill_id / candidate_skill                 # error_type 强制 12 值中恰好一个
7. 【硬门】若 gold evidence 未进上下文 → 强制丢弃候选（"证据没检出来"不产生 answer skill）
8. sanitizer（见 §7.3）
9. verify_candidate：4 个 gate（见下表）
10. 一次自修复：被拒后用 previous + failures 再反思一次，重新 sanitize + 验证
11. 写 memory：update_skill_id 命中→append 拼接 action、合并 patterns、version+1、
    confidence = max(0.5, c*0.9)；否则 add()（success=0/failure=0, confidence=0.55, v1）
12. 决策标签：correct / added / updated / rejected / rejected_sanitize / no_candidate
13. 快照与回滚：step % 10 == 0 或末步 → 写快照 + 跑 dev 评测；
    若 dev F1 < best - 0.05 → 从 best 快照 restore，n_rollbacks += 1
14. 收尾：merge_duplicates(threshold=0.90) → 落盘 memory
```

**error_type 枚举（11 值，prompt 原文强制"恰好选一个"）**：`evidence_not_retrieved / evidence_not_used / temporal_arithmetic / temporal_granularity / multi_fact_omission / over_verbose / format_mismatch / wrong_refusal / unsupported_inference / wrong_target_or_premise / other`。
配套规则：`evidence_not_retrieved` → 强制 `error_axis="retrieval"` 且 `candidate_skill=null`；语义对但格式不同 → 提格式规则；否则 `error_axis="answer"` 且只提一条可泛化的规则。

**四个验证 gate**：

| gate | 通过条件 | 基线口径 |
|---|---|---|
| `fix_pass` | 源题 `F1 ≥ baseline_F1 + 0.10` **或**（候选 EM=1 且基线 EM=0） | 源题用"当前记忆下真实错的草稿" |
| `similar_pass` | ≤3 个同类别相似题（embedding 余弦，无 embedder 用 Jaccard）中 `hard_neg` 必须为 0（基线 EM=1 → 候选 EM=0） | replay 题用**无 skill 的 P7 缓存** |
| `dissimilar_pass` | 从其它 3 个 answerable 类别各取 1 题（优先基线答对的），不得 EM 1→0 | 同上 |
| `adversarial_pass` | ≤2 个 cat-5（优先已见过的）不得"基线拒答 → 候选不拒答" | 同上 |

`accept = fix_pass && similar_pass && dissimilar_pass && adversarial_pass`。`leak_pass` 在验证里**硬编码 True**（泄漏由前置 sanitizer 负责）。

**长期信誉机制**：`failure_count ≥ 3 且 failure_count ≥ success_count + 2` → 自动 `disable`；普通 `record_usage` 用 `(success+1)/(n+2)` 更新 confidence；`merge_duplicates` 仅合并同类 category 集合且相似度 ≥0.90 者，保留 version 高者，被并入者 `status=disabled`。

**技能检索与渲染**：
`select_skills` 打分 = `2.0`（有类别约束）/ `0.8`（无）+ `1.2·pattern_overlap`（token 覆盖 ≥0.34 才计）+ `0.6·Jaccard(question, trigger+title)` + `1.5·max(0, cos(embedding))` + `0.3·confidence`；`score ≥ 1.0` 才入选，取 top-k。
**k 的口径不一致**：dev/adaptation/E-matrix 用 **k=2**，ReAct/重复实验用默认 **k=4**（见 §11.3）。

**渲染格式**：默认 **action-only**：`- [skill_id vN] <action_instructions>`；`QE_INCLUDE_TRIGGER=1` 时 trigger-aware：`- [id vN] Apply only when: <trigger> Rule: <action>`。

### 7.3 Sanitizer（防泄漏的关键，逐条列出）

**Answer 候选**（`sanitize_candidate`）：非 dict → 拒；`action_instructions` <10 字符 → 拒、>700 字符 → 按词边界截断；`categories` 只保留 `{1,2,4,5}`；patterns 截前 4；`title≤90`、`trigger≤240`、`expected_output_format≤180`、`evidence_requirements≤180`、`failure_modes` 前 4 项各 ≤120。

**`leak_check`（answer）**：
1. `sensitive_terms(qidx)`：从 `question + gold + adversarial_answer` 抽 `[A-Za-z][A-Za-z'\-]+|\d+`，剔除三张白名单（约 200 个通用词如 job/business/passion、疑问词、月份表）与长度 ≤2 的词；
2. 三条判敏感：① 纯数字且长度 >2（日期/年份/计数）；② 含大写字母（专有名词），`jon`/`gina` 豁免；③ 非数字且长度 >4（长实词）；
3. 命中判 fail；额外 `\b(19|20)\d{2}\b` → `explicit_year`；`len(gold) > 4` 且 gold 小写整串是候选子串 → `gold_substring`。

**`leak_check_retrieval`（更宽松，因查询模板本就含通用动词）**：只禁 `question + gold` 中的**大写开头词**（排除 what/when/where/who/which/why/how）、`(19|20)\d{2}`、`\d{3,}`、以及归一化后长度 >4 的 gold 整串作为子串；结构上模板必须含 `{question}` 或 `{entities}`（否则 `no_placeholders`），且 `categories` 必须包含源题 category（否则 `category_mismatch`）。

### 7.4 Retrieval Skill 进化 + sufficiency gate

- **主干**：base 排序永远是冻结的 `COMP-P4_top5`；技能**只追加查询**（占位符仅 `{question}` 与 `{entities}`，每个技能至多前 4 条模板、实际取前 3 条）。
- **pipeline 模式**：`base | qdm | qdm_suff | full`。`base` 直接返回；否则 QDM（1 次 LLM）→ 需要拆解则对前 **3** 条子问题各 dense top-10 并与 base 做 RRF；`qdm_suff/full` 再做 **sufficiency 检查**（+1 次 LLM），**仅当 `not sufficient` 且库非空**才选技能（k=2）。
- **sufficiency 输出契约**：`{"missing": [{"query","why"}], "sufficient": bool}`，`max_new=2`，`sufficient` 缺省回退 `not missing`。语义：检索到的会话"提到了题面实体但缺连接关系"即算 missing。**评测期只见问题与检索结果，从不用 gold。**
- **轮次上限**：QDM 一轮 + 一次技能/missing 合并轮 = **至多两轮**（结构性上限，代码里没有计数器强制）。
- **是否需要学习**：`cat != 5 and gold_sessions 非空 and (evidence_full@4096 != 1 or recall@5 != 1 or (n_evidence >= 2 and multi_evidence_complete != 1))`。
- **验证（纯本地，0 次 LLM，只做 dense replay）**：
  - schema 前置（category 必须含源题类别；至少一条含占位符的模板）；
  - `fix_pass`（三选一）：`evidence_full@4096` 0→1 / `evidence_partial@4096` 提升 ≥0.25 / `recall@5` 提升 ≥0.5；
  - 负迁移检查：同类别、已见过、Jaccard 最近的 3 题上，`evidence_full 1→非1` 或 `recall@5` 下降 >0.30 任一条即拒。

### 7.5 Question-Time ReAct + Grounded Verifier（R0–R5）

| 变体 | 检索 | Answer 层 | verifier |
|---|---|---|---|
| R0 | base（冻结 ranked，4096） | P7，无技能 | 否 |
| R1 | base | P7 | 是 |
| R2 | qdm_suff + **静态** retrieval 库 | P7 | 是 |
| R3 | base | **进化** Answer Skills | 否 |
| R4 | base | 进化 Answer Skills | 是 |
| R5 | qdm_suff + **进化** retrieval 库 | 进化 Answer Skills | 是 |

**verifier 契约**：输入仅 `question + 同一份 4096 上下文 + 候选答案`（system 明示"never use outside knowledge"）；输出 `supported / support_quote(≤160) / missing_parts[] / date_error / wrong_target / over_enumeration / should_refuse / issues[](≤6) / verdict ∈ {ok, revise, refuse}`。
**执行规则**：`verdict == "refuse"` 或（`should_refuse` 且原答案未拒答）→ 强制 `No answer`；`verdict == "revise"` 且（有 issues 或 missing_parts）→ 追加 `REVISE_SUFFIX` 修正**一次**（上限 1 次）。

### 7.6 Baseline（P0 / P7 / P8，conv-30 105 题）

| Reader | EM | F1 | 官方 F1 | 对抗拒答 | answer support |
|---|---|---|---|---|---|
| P0 | .222 | .417 | .445 | .958 | .967 |
| **P7** | **.259** | **.539** | **.569** | .917 | .970 |
| P8 | .222 | .504 | .529 | .958 | .985 |

（注：此处 P7 与阶段 B 的 `COMP-P4_top5 + P7` 是同一配置（无 LLM 重排），故数值一致（.259/.539/.569）；阶段 B 的 G_raw 结果是**有** LLM 重排的另一配置（.235/.521/.546），两者不可混比。）

**按子集**（EM / F1 / 官方F1 / 对抗拒答）：

| Reader | 子集 | n | EM | F1 | 官方F1 | adv refusal |
|---|---|---|---|---|---|---|
| P7 | adaptation | 43 | .182 | .463 | .484 | .900 |
| P7 | development | 19 | .333 | .654 | .677 | 1.000 |
| P7 | **evaluation** | 43 | **.303** | **.563** | **.606** | .900 |

**按类别（全 105 题，EM / F1）**：multi-hop P7 .273/.511；temporal P7 .500/.869；single-hop P7 **.114/.352**。
→ P7 的画像很清晰：**时间题强、单跳弱**（单跳变长导致 EM 掉）。

### 7.7 结果：E0–E6 联合矩阵（evaluation，n=43）

| Exp | Answer 层 | Retrieval 层 | EM | F1 | 官方F1 | adv refusal | Ev@4096 | multi-ev | R@5 | calls/题 |
|---|---|---|---|---|---|---|---|---|---|---|
| **E0** | P7 | base | **.303** | **.563** | **.606** | .900 | **.909** | **.714** | .932 | 1.00 |
| E1 | 静态 answer | base | .273 | .423 | .436 | .900 | .909 | .714 | .932 | 1.00 |
| E2 | 进化 answer | base | .273 | .508 | .574 | .900 | .909 | .714 | .932 | 1.00 |
| E3 | P7 | 静态 retrieval | .303 | .545 | .576 | .900 | .818 | .429 | .803 | 3.00 |
| E4 | P7 | 进化 retrieval | .273 | .546 | .602 | .900 | .848 | .429 | .886 | 3.00 |
| E5 | 进化 answer | 静态 retrieval | .303 | .537 | .556 | .900 | .818 | .429 | .803 | 3.00 |
| E6 | 进化 answer | 进化 retrieval | .273 | .506 | .589 | .900 | .848 | .429 | .886 | 3.00 |
| EB | 长静态 prompt | base | .303 | .488 | .523 | .900 | .909 | .714 | .932 | 1.00 |

> E4/E6 实际**部署 0 个学到的 retrieval skill**（dev 快照选择否决了唯一那个），因此它们测的是 "QDM + sufficiency 而无技能"；E3/E5 用**手写静态** retrieval 库。

**按类别（F1 / 官方F1 / n）**：

| Exp | multi-hop (5) | temporal (10) | single-hop (18) |
|---|---|---|---|
| E0 | .567 / .543 | **.940 / .940** | .353 / .437 |
| E2 | .382 / .543 | .680 / .680 | **.447 / .524** |
| E3 | .476 / .455 | .907 / .907 | .364 / .426 |
| E4 | .377 / .543 | .940 / .940 | .373 / .431 |
| E6 | .377 / .543 | .680 / .680 | .445 / .551 |
| EB | .567 / .543 | .600 / .600 | .403 / .474 |

**读法**：
- **E0（固定 P7）在所有主指标上最强**；每一层技能都至少拉低其中一项。
- E2 优于 E1（手写规则）与 EB（长静态 prompt）→ **进化出的规则比手写规则"伤害更小"**，但都没超过 P7。
- E3–E6 显示 **QDM + dense 合并本身就会破坏强图排序**：R@5 .932 → .80–.89、evidence-full .909 → .82–.85。
- 检索层内部：**静态 4 规则库（E3，R@5 .803）比 QDM-only（E4，.886）更差** → 手写检索查询制造排序噪声而非召回证据。
- 43 题上**一题 = 2.3pp EM**，E1/E4/E6 的差异大于此，但整体结论仍为负。

### 7.8 结果：ReAct + Verifier（evaluation）

| 变体 | EM | F1 | 官方F1 | adv refusal | answer support | 修订率 | calls/题 |
|---|---|---|---|---|---|---|---|
| R0 | .273 | .562 | .631 | .900 | .999 | 0.000 | 1.00 |
| R1 | .273 | .549 | .617 | .900 | .999 | .395 | 2.14 |
| R2 | **.303** | .520 | .544 | .900 | .999 | .535 | 4.26 |
| R3 | .273 | .484 | .553 | .900 | .999 | 0.000 | 1.00 |
| R4 | .273 | .490 | .558 | .900 | .999 | .326 | 2.07 |
| R5 | .242 | .425 | .445 | .900 | 1.000 | .442 | 4.12 |

- **verifier 修订 33–54% 的答案但不提升 F1**（R1−R0：F1 −0.013，+1.14 calls/题；R5−R3：−0.060，+3.12 calls）。
- R5（全进化系统）**最差** → answer 与 retrieval 两层同时过度应用。
- verifier 是 grounded check（只给问题/证据/候选答案），**能压制幻觉但不能创造缺失证据**。

### 7.9 Answer Skill：学到的技能与实际效果

adaptation 43 题的结果：**16 题无需新技能即答对**；技能 **新增 3 / 更新 0**；候选**拒绝 8 / 接受 3**（验证通过率 3/11）；sanitizer 拒绝 6（全部 leak-check）；自修复 5 次 / 接受 0；merge 0 / rollback 0。

**反思驱动的错误类型**：

| error_type | 次数 |
|---|---|
| format_mismatch | 5 |
| over_verbose | 4 |
| evidence_not_retrieved | 4 |
| temporal_granularity | 4 |
| multi_fact_omission | 4 |
| temporal_arithmetic | 3 |
| evidence_not_used | 1 |
| wrong_target_or_premise | 1 |
| wrong_refusal | 1 |

**dev 快照选择**：

| feedback 题数 | active skills | dev EM | dev F1 | dev 官方F1 | adv refusal |
|---|---|---|---|---|---|
| 0 | 0 | .333 | .654 | .677 | 1.000 |
| 10 | 3 | .400 | .662 | **.687** | 1.000 |
| 20 / 30 / 40 / 43 | 3 | .400 | .662 | .687 | 1.000 |

选中 **step 43**（选择准则：`(official_f1, project_f1, step)` 取 max，即以 **dev 官方 F1 为主键**）。注意快照实际落盘为 **0/10/20/30/40/43**（`%10==0` 命中 40），报告文字写"0/10/20/30/43"是笔误。

**入选的 3 条技能（原文）**：

| skill_id | 类别 | success/failure | conf | 创建步 | 后续被选（其中答对） |
|---|---|---|---|---|---|
| `ans_use_explicit_caused_state_wording` | [4] | 1/0 | .67 | 3 | 16（2） |
| `ans_return_concise_explicit_purpose_for_why_` | [4] | 1/0 | .67 | 8 | 14（2） |
| `ans_preserve_temporal_granularity_for_relati` | [2] | 0/0 | .55 | 10 | 7（2） |

> 1. Prefer the exact adjective or state in an explicit causative phrase over nearby metaphors, summaries, or paraphrases. Quote or minimally return that stated state, and avoid substituting broader interpretations.
> 2. Use the shortest explicit purpose phrase tied directly to the asked action. Omit later explanations, expected benefits, pronouns, and paraphrases unless needed to preserve meaning.
> 3. Resolve relative time expressions from the evidence timestamp, but answer only at the granularity explicitly supported. Do not convert month-level expressions into exact calendar days or add unsupported precision.

**ablation**：把每条技能的 trigger 条件渲染进 Reader prompt（而非只给 action）→ dev 官方 F1 **.683** vs 部署版 **.687** → 未部署（但这是"dev 选择偏好"，不等于证明 trigger 渲染无用）。

**evaluation 上的技能命中分析（E2）**：

| 项 | 数量 |
|---|---|
| 至少命中 1 条技能的题 | **28 / 43** |
| 有帮助（E0 错 → E2 对） | **1** |
| 有害（E0 对 → E2 错） | **1** |
| 中性 | **26** |
| 无技能命中且错 | 5 |

| skill | uses | helpful | harmful | neutral |
|---|---|---|---|---|
| `ans_use_explicit_caused_state_wording` | 18 | 1 | 0 | 17 |
| `ans_return_concise_explicit_purpose_for_why_` | 18 | 1 | 0 | 17 |
| `ans_preserve_temporal_granularity_for_relati` | 10 | 0 | **1** | 9 |

**解读**：**复用频繁（16/14/7 次）但成功复用稀少（每技能 2 次）**；技能按 category 触发，被应用到许多触发条件并不匹配的题上。唯一有害命中是"把确切日期改写成相对短语"——这是**具体的过度泛化案例**（q38：gold `21 July 2023`，E2 答 `Last Friday`）。

**E0→E2 的逐题变化**：fixed 1（q71 `amazing`）、worsened 2（q18 枚举失控、q38 时间粒度）、仍错 23。
**证据在但拒答**：E2/E6 各 2 题（q9 multi-hop、q67 single-hop → `No answer`）。

### 7.10 Retrieval Skill：学到 1 条，被 dev 否决

| 项 | 值 |
|---|---|
| adaptation 题数 | 43 |
| base + QDM/sufficiency 后证据已完整 | **38** |
| 新增技能 / 更新 / 拒绝 | 1 / 0 / 2 |
| 技能层捞回 gold 的题 | 1 |
| 技能层丢失 gold 的题 | 1 |
| 原始 memory 中 active 技能 | 1 |
| **dev 快照选择的步数** | **step 0（部署 0 条）** |

**dev 快照（retrieval-only，无 LLM）**：

| feedback | active | R@5 | evidence full@4096 | multi-ev complete |
|---|---|---|---|---|
| 0 | 0 | **.967** | **.933** | **.667** |
| 10–43 | 1 | .933 | .867 | .333 |

学到的技能（被否决）：

| skill_id | 类别 | status | 模板 |
|---|---|---|---|
| `ret_motivation_and_trigger_expansion_for_why` | [1] | **rejected_by_dev** | `{question} reason motivation because wanted to` / `{entities} passionate about love for unique interests blend` / `{entities} lost job unemployment before starting bu…` |

**独立贡献（evaluation）**：

| Exp | R@3 | R@5 | evidence full@4096 | multi-ev complete |
|---|---|---|---|---|
| E0 | .811 | **.932** | **.909** | **.714** |
| E3（静态库） | .614 | .803 | .818 | .429 |
| E4（进化=空库+QDM） | .811 | .886 | .848 | .429 |

**无 sufficiency gate 直接应用的消融**：base R@5 .932 → .879、evidence .909 → .848，捞回 0 题、丢失 2 题。

→ **一个"通过 adaptation 验证但在 dev 上降低证据覆盖"的技能被快照选择正确拦下**——这是本阶段唯一一次"机制按设计工作"的正向证据（负迁移被门控住）。

### 7.11 成本、延迟与稳定性

- 调用记录 **2,582** 条（缓存命中 1,023；端点尝试 2,186，成功率 **.713**，重试行 619；429 = 627）。
- prompt tokens **7,431,528**、completion tokens **98,445**；延迟 P50/P95 = **2.05s / 3.79s**。
- 分阶段（节选）：`verify` 273 次 / 1.23M prompt tok；`suff` 483 次 / 387k；`qdm` 488 次 / 69k；`reflect` 84 次 / 125k。
- 逐题成本：E0 1.00 call / 4,462 prompt tok；E3–E6 3.00 calls / ~5.4k prompt tok；R2 4.26 calls / 10,052 tok。
- 注意：per-question latency 记录的是 **Reader 单次调用**延迟，不累加检索/verifier 的 LLM 延迟。

**重复运行稳定性**（配额只允许 2 次独立运行）：

| 配置 | EM A | EM B | F1 A | F1 B | EM 不稳定题数 |
|---|---|---|---|---|---|
| E0 vs R0 | .303 | .273 | .563 | .562 | 1（q18） |
| E2 vs R3 | .273 | .273 | .508 | .484 | 0（q64 F1 .909→0） |

→ 33 道可答題上 2 点 EM = 约 0.7 题 → **部分"技能收益/损失"落在 HY4 非确定性内**。

### 7.12 阶段 C 结论

1. **机制上可行**：不改权重，生成了 6 个版本快照、接受 3 条技能写入、有版本/合并/淘汰/回滚、有严格三分划分 —— 作为一个**独立策略层**是可实现的（H10 的"机制"部分成立）。
2. **效果上为负**：43 题 evaluation 上没有任何技能层超过固定 P7（E2−E0：EM −3.0pp / F1 −5.6pp / 官方 F1 −3.2pp）；R5 最差。
3. **技能复用 ≠ 泛化**：28/43 题命中技能，仅 1 帮助 / 1 有害 / 26 中性。**"被检索到"不是"被正确使用"**。
4. **修复的是不同故障，不是同一故障**：E2 提升 single-hop（+9.4pp F1）但退化 temporal（−26.0pp）与 multi-hop（−18.6pp）——学到了"时间粒度"规则后**过度应用**到所有时间题。
5. **检索层退化的主因是 QDM 的 dense 合并**，不是技能：E4（0 个技能）已经把 R@5 从 .932 拉到 .886。
6. **天花板由冻结检索决定**：answer 侧技能只能修复"证据已在上下文"的错误；在 P7 的 Oracle 上限约 .60 F1 的前提下，策略层最多回收这部分。

---

## 8. 阶段 D：QASPER Read-Time

**研究问题**：在 QASPER（每篇论文内部问答）上，Read-Time 产生的知识——**Comprehension Triple** 与 **Comprehension Unit**——能否在 Question-Time 提升真实证据检索与最终 QA？并**区分"理解"本身的收益与"普通索引扩充"的收益**。

**为什么换数据集**：LoCoMo 单对话仅 19–32 篇文档、全文 1–2 万 token，检索池太小、天花板太低（R@20 ≈ 100%）。QASPER 的段落池 P50 = 43、P90 = 70，多证据题占 36.8%、跨章节 19.0%，能测"多证据检索"。

### 8.1 数据与样本

- **检索单元 = 原始段落**（一段 = 一个 passage，不做二次切分）；`passage_id = {paper_id}::s{si:02d}::p{pi:03d}`（`si/pi` 为 JSON 原始索引，去重后不连续）；检索**严格限制在题目所属论文内**。
- 文本完全相同的段落去重（`whitespace+lower` 归一），后者记为 alias，不进检索池；表格/图片单独抽为 `FigureTable`，不进检索池。
- **多标注者按"替代证据集"处理**：All-Evidence@k 先在每个标注者集合上算再平均；完全相同的标注者集合去重。
- 冻结样本：seed 20261110，100 篇 / 372 题（**333 道** answerable 且有映射文本证据 / 141 多证据 / 56 跨章节）。
- **防泄漏**：atomic facts / comprehension / verifier / summary **只接收论文标题、段落原文、已抽取事实**；任何 prompt 都不含问题、答案或 gold evidence。

**按论文长度分层的题目难度**：

| 五分段 | 论文 token | papers | questions | 单段落 | 多段落 | 跨章节 | 平均 gold |
|---|---|---|---|---|---|---|---|
| 1 | [796, 2851] | 56 | 158 | 63.3% | 36.7% | 18.4% | 1.89 |
| 2 | [2858, 3752] | 56 | 191 | 64.9% | 35.1% | 17.8% | 1.73 |
| 3 | [3783, 4974] | 56 | 184 | 65.8% | 34.2% | 22.3% | 1.58 |
| 4 | [4985, 5860] | 56 | 191 | 63.9% | 36.1% | 18.3% | 1.71 |
| 5 | [5866, 20902] | 57 | 164 | 57.3% | **42.7%** | 18.3% | **2.23** |

> 多证据比例随论文长度上升，跨章节比例稳定在 18–22% → 支持把多证据/跨章节子集作为主区分条件，而不是只看 Any-Evidence@5。

### 8.2 方法定义（M0–M6）

| 方法 | 图谱 / 检索结构 | 实现要点 |
|---|---|---|
| **M0** | BM25 / Dense / Hybrid | 段落级；BM25 `k1=1.5, b=0.75`，分词 `[A-Za-z0-9]+`（无停用词、无词干），未登录词 idf = 0.1×avg_idf；Dense 用 NV-Embed-v2（文档 instruction 空，查询 instruction "Given a question, retrieve passages that answer the question"）；Hybrid = 等权 RRF（k=60，rank 1-based） |
| **M1** | HippoRAG2 G_raw | 官方 OpenIE（一个段落一个 chunk，chunk 主键 = 文本哈希）→ entity/fact/passage 图 → PPR；`retrieval_top_k=50, linking_top_k=5`；**每篇论文独立建图** |
| **M2** | G_raw + Atomic Facts | 逐段原子事实三元组注入源段落 OpenIE 行后重建 |
| **M3** | M2 + Comprehension Triples | 在 M2 克隆状态上注入跨段理解三元组（**同一批 insight 的三元组表示**） |
| **M4** | M2 + Comprehension Units（独立索引） | unit 文本单独 embedding；检索后**回溯到来源段落**，与 M2 图排序 RRF（等权，k=60） |
| **M5** | M3 + Comprehension Units | 双路：M3 图 + unit → RRF → 回溯真实段落 |
| M4/M5 **gated** | 敏感性 | 只用 top-3 units、unit 路权重 0.5 |
| **M6** | M2 + 单段落摘要 | 每段一句摘要建索引，回溯同段落；另做**与 unit 同数量**的 matched 对照 |

**unit → 段落回溯算法（关键实现）**：对 unit 检索的前 100 个候选，把每个 unit 的 `source_passage_ids` 投影成段落，得分 = `unit_score × 1.0`（第 0 个来源段落）或 `× 0.98`（其余来源段落）；同一段落保留最高分；按分降序取 top-k，并携带 `unit_id/unit_text/unit_rank`。gated 版本只取前 3 个 unit。

**三组最关键对照**：M3（triple）、M4（unit）、M5（两者）**共享同一批通过验证的 insight**，保证"知识数量固定、只变表示"。

### 8.3 Read-time Prompt 约束（生成质量的前提）

| prompt | 关键约束 |
|---|---|
| **Atomic facts** | 一条 fact = 一个可验证 `{s,p,o}`；数字/单位/数据集/指标**原样照抄**；**必须且只能引用一个 passage id**；禁止推断/评价/总结/外部知识；跳过含糊陈述、纯引用、section 标题、致谢；禁止重复；**每段 ≤4 条、每 block ≤24 条**；evidence 为 ≤30 词逐字引用。block 切分：≤6 段且 ≤2500 token；`max_tokens=3072`，两轮重试预算翻倍（上限 16384），截断时按数组做括号级 salvage |
| **Comprehension** | 必须跨**至少两个不同段落**；必须是新的（不是单段陈述、不是某条 fact 的改写）；`source_passage_ids ≥2` 且必须在输入 id 内；`supporting_fact_ids` 必须引用输入 fact id；`concise_text` 一句话 ≤40 词、自带语义；`triple` 用短谓词；`relation_type ∈ 7 类`；`derivation ∈ {Composed, Inferred}`；**禁止把共现/相关写成因果**；**最多 10 条按重要性排序** |
| **Verifier** | 6 项检查：引用的 passage id 是否都存在 / 是否被联合支持 / 是否只是单段复述（→ reject）/ 是否把共现写成因果（→ reject）/ 是否引入源文没有的数字·实体·数据集·结论（→ reject）/ triple 是否与 concise_text 一致。输出 `{insight_id, keep, support, novelty, causality_error, reason}`；**`keep=False` 或「keep=True 但 causality_error=True」都拒绝**；**无裁决（解析失败）视为保留**（fail-open，状态 `missing`） |
| **Summary（M6）** | 每段一句 ≤30 词自足摘要；保留实体/数据集/指标与精确数字；**不得跨段合并、不得添加信息** |

**reading group 组装（section 版，主方案）**：按 `section_index` 升序聚合；单 section > 8192 token 时按段落切分；`cur_tok + sec_tok > 8192` 则封组；加入后 `cur_tok ≥ 4096` 则封组；尾部 <400 token 且能塞进上一组则并入。组 id `{paper}::g{ii}`。**单段落组不送 comprehension（需 ≥2 段落）**。

### 8.4 Read-time 生成质量

| 指标 | 值 |
|---|---|
| 处理论文 | 100 |
| reading groups（section-based，4096–8192 token） | 147 |
| atomic facts | **10,539** |
| candidate insights | 1,313 |
| **verified units** | **733** |
| rejected insights | 580 |
| **verifier 保留率** | **55.8%** |
| units / paper | 7.3 |
| unit 长度（词，mean/P90） | 24.6 / 29.0 |
| 作为 unit 来源的段落比例（每篇平均） | 30.4% |
| **unit 来源跨 section 比例** | **71.2%** |
| unit 来源段落距离（mean/P90） | 9.9 / 25.0 |
| paragraph summaries（M6） | 4,561 |
| read-time tokens | prompt 2,883,843 / completion 1,620,654 |

**关系类型分布与效果（与逐 unit 使用轨迹 join）**：

| relation_type | units | 触及 gold | 进过 unit top-5 | 触及 gold 但未进 top-5 | QA helped | QA harmed |
|---|---|---|---|---|---|---|
| claim_evidence | 150 | 46 | 120 | 4 | 44 | 49 |
| method_result | 135 | 44 | 117 | 5 | 37 | 50 |
| problem_solution | 119 | 35 | 92 | 2 | 25 | 45 |
| setting_outcome | 108 | 44 | 97 | 2 | 36 | 40 |
| combined_fact | 95 | 25 | 75 | 3 | 23 | 36 |
| assumption_limitation | 65 | 11 | 47 | 2 | 16 | 15 |
| entity_resolution | 61 | 24 | 49 | 3 | 16 | 24 |

**Verifier 拒绝原因**：单段复述 207 / 无依据因果 143 / 内容不被支持 85 / 其他 145。

**质量观察**：unit 平均 24.6 词、71.2% 跨 section、来源段落平均相隔约 10 段 → 生成器确实在做跨段组合，不是简单改写；但仍有约 18% 的 unit 来自相邻段落，且验证器拒绝了大量单段复述与虚假因果。

### 8.5 基础检索基线（100 篇 / 333 题）

| 方法 | Any@1 | Any@3 | Any@5 | All@5 | Recall@5 | MRR | nDCG@10 | Cov@4096 |
|---|---|---|---|---|---|---|---|---|
| M0 BM25 | .204 | .362 | .521 | .378 | .435 | .341 | .369 | .716 |
| **M0 Dense** | **.363** | **.628** | **.740** | **.593** | **.658** | **.530** | **.563** | **.889** |
| M0 Hybrid | .312 | .566 | .680 | .517 | .582 | .476 | .501 | .841 |

**按子集（Dense）**：

| subset | n | Any@5 | All@5 | Recall@5 | Cov@4096 |
|---|---|---|---|---|---|
| all | 333 | .740 | .593 | .658 | .889 |
| single | 219 | .733 | .733 | .733 | .970 |
| multi | 141 | .741 | **.394** | .547 | .773 |
| cross | 56 | .750 | **.295** | .497 | .786 |
| multi_annot | 98 | .709 | .444 | .563 | .827 |

→ BM25 远弱于 dense；**等权 hybrid 被 BM25 稀释**；dense 是本轮最强检索器；单证据接近饱和（.733），多证据（.394）与跨章节（.295）仍有空间。

### 8.6 主检索对照 M0–M6（333 题）

| 方法 | Any@1 | Any@3 | Any@5 | All@1 | All@3 | All@5 | Recall@5 | MRR | nDCG@10 | Cov@4096 |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 Dense | .363 | .628 | **.740** | .251 | .470 | **.593** | .658 | .530 | .563 | .889 |
| M1 G_raw | .281 | .553 | .692 | .204 | .443 | .544 | .606 | .461 | .517 | .883 |
| M2 +Atomic | .300 | .571 | .697 | .213 | .441 | .545 | .608 | .477 | .527 | .884 |
| **M3 +Triples** | .321 | .580 | **.707** | .231 | .452 | **.556** | .620 | **.493** | **.540** | .886 |
| M4 +Units | .210 | .441 | .545 | .138 | .336 | .401 | .461 | .375 | .414 | .842 |
| M5 Triples+Units | .216 | .440 | .535 | .146 | .333 | .395 | .454 | .375 | .413 | .836 |
| M4 gated units | .221 | .429 | .565 | .155 | .324 | .411 | .475 | .389 | .456 | .884 |
| M5 gated units | .222 | .425 | .565 | .159 | .318 | .413 | .476 | .390 | .454 | .886 |
| **M6 +Summaries** | **.359** | **.620** | .709 | **.257** | **.482** | **.575** | **.635** | **.521** | **.555** | .884 |
| M6 matched count | .186 | .335 | .518 | .117 | .216 | .360 | .425 | .337 | .395 | .853 |
| M4 units only | .128 | .323 | .399 | .081 | .237 | .291 | .339 | .250 | .281 | .405 |

**成对差值（question-level paired bootstrap，95% CI）**：

| 对比 | ΔAny@5 | 95% CI | ΔAll@5 | 95% CI | ΔRecall@5 |
|---|---|---|---|---|---|
| **m3_atom_comp vs m2_atom** | **+0.011** | **[−0.003, +0.026]** | **+0.011** | **[−0.002, +0.024]** | +0.012 |
| m4_units vs m2_atom | −0.152 | [−0.209, −0.092] | −0.144 | [−0.197, −0.090] | −0.148 |
| m5_units_graph vs m3_atom_comp | −0.173 | [−0.227, −0.116] | −0.161 | [−0.213, −0.107] | −0.166 |
| m6_matched vs m2_atom | −0.179 | [−0.228, −0.129] | −0.185 | [−0.233, −0.135] | −0.183 |
| m5_units_graph vs m2_atom | −0.162 | [−0.218, −0.102] | −0.150 | [−0.203, −0.096] | −0.155 |

- **M3−M2**：+1.1pp，**CI 下界接近 0** → "中性偏正"，**不能说显著**。
- **M2−M1**：原子事实层几乎中性（All@5 +0.1pp）→ 普通事实扩充不改变检索上限。
- **M4/M5−M2**：−14.4pp / −15.0pp，**显著负迁移**；gated 融合只能把损失从 −15pp 收窄到 −13pp，**无法恢复**。

### 8.7 子集与机制（本阶段最关键的证据）

| subset | 方法 | n | Any@5 | All@5 | Recall@5 |
|---|---|---|---|---|---|
| 单段落 | M2 +Atomic | 219 | .674 | .674 | .674 |
| 单段落 | M3 +Triples | 219 | .689 | .689 | .689 |
| 单段落 | M4 +Units | 219 | .530 | .530 | .530 |
| 多证据 | M2 +Atomic | 141 | .716 | .358 | .508 |
| 多证据 | M3 +Triples | 141 | .720 | .362 | .514 |
| 多证据 | M4 +Units | 141 | .557 | .216 | .358 |
| 跨章节 | M2 +Atomic | 56 | .741 | .312 | .499 |
| 跨章节 | M3 +Triples | 56 | .741 | .312 | .499 |
| 跨章节 | M4 +Units | 56 | .571 | .161 | .333 |
| **unit 覆盖 gold** | M2 | 193 | .746 | .567 | .640 |
| **unit 覆盖 gold** | M3 | 193 | .759 | .585 | .659 |
| **unit 覆盖 gold** | **M4** | 193 | **.837** | **.627** | **.716** |
| **unit 未覆盖 gold** | M2 | 140 | .629 | .514 | .564 |
| **unit 未覆盖 gold** | **M4** | 140 | **.143** | **.089** | **.109** |

`unit_covered` 定义：该题 gold 至少被一个 verified unit 的来源段落触及（**排序前的上界**）。

→ **本轮最关键的机制证据**：
- **覆盖子集上 unit 明显有效**（Any@5 .746→.837、All@5 .567→.627）；
- **未覆盖子集上崩塌**（.629→.143、.514→.089）——因为等权 RRF 仍把 unit 来源段落塞进候选，**挤掉正确图证据**。
- 多证据/跨章节子集上 Triple **没有稳定增益**（多证据 All@5 .358→.362、跨章节 .312→.312 不变）。

**RRF 权重敏感性（unit 路权重；graph 权重固定 1.0）**：

| base | unit weight | Any@5 | All@5 | Recall@5 |
|---|---|---|---|---|
| m2_atom | 0.1 | .691 | .529 | .596 |
| m3_atom_comp | 0.1 | .694 | .539 | .605 |
| m2_atom | 0.2 | .596 | .438 | .503 |
| m3_atom_comp | 0.2 | .590 | .432 | .496 |
| m2_atom | 0.5 | .557 | .411 | .471 |
| m3_atom_comp | 0.5 | .545 | .405 | .463 |
| m2_atom | 1.0 | .545 | .401 | .461 |
| m3_atom_comp | 1.0 | .535 | .395 | .454 |

→ **即使 0.1 的固定权重也已低于 graph-only（M2 Any@5 .697）；0.2 起崩溃。固定权重救不回来，必须做相关性门控。**

**语义分组补充（20 篇 / 72 题，探索性）**：section 分组 unit 164 个、gold 覆盖 .722、Any@5 .694 / All@5 .521；semantic 分组 unit 144 个、覆盖 .694、Any@5 .639 / All@5 .465 → **section-based 更好**（semantic 未做 QA）。

### 8.8 Reader QA（372 题，固定 4096-token 预算）

| 方法 | Answer F1 | Evidence F1 | EM | F1 answerable | F1 multi | F1 cross | Gold all in ctx |
|---|---|---|---|---|---|---|---|
| M0 Dense | **.464** | .119 | **.210** | **.436** | **.428** | .389 | .763 |
| M1 G_raw | .455 | .119 | .202 | .427 | .431 | .390 | .758 |
| M2 +Atomic | .457 | .119 | .199 | .429 | .426 | .402 | .761 |
| M3 +Triples | .454 | .119 | .204 | .425 | .422 | **.400** | .763 |
| M4 +Units | .435 | .113 | .191 | .405 | .402 | .369 | .707 |
| M5 Triples+Units | .443 | .112 | .194 | .414 | .411 | .394 | .702 |
| M5 (units shown) | .443 | .113 | .191 | .414 | .402 | .372 | .702 |
| **Full paper** | **.476** | .073 | **.220** | **.448** | **.455** | **.434** | **.890** |
| **Oracle gold** | **.487** | .955 | **.242** | **.455** | **.455** | **.447** | .895 |

**QA 成对差值**：

| 对比 | ΔAnswerF1 | 95% CI |
|---|---|---|
| m3_atom_comp vs m2_atom | −0.003 | [−0.015, +0.008] |
| m4_units vs m2_atom | −0.022 | [−0.039, −0.006] |
| m5_units_graph vs m3_atom_comp | −0.011 | [−0.028, +0.007] |
| m5_units_graph_B vs m5_units_graph | −0.000 | [−0.018, +0.019] |

**按证据类型（Answer F1）**：文本证据 M0 .438 / M2 .431 / M3 .427 / M4 .407 / M5 .415 / Oracle .458；仅图表证据全部 **.369**（方法间零差异，因为文本检索无法影响图表证据）；不可回答全部 .909（Oracle 1.000）。

**unit 覆盖 vs 未覆盖的 QA**：covered：m2 .461 / m3 .467 / m4 .440 / m5 .448 / m5_B .454；uncovered：m2 .400 / m3 .384 / m4 .371 / m5 .381。
→ **即使在覆盖子集，M5 的 QA F1 也没有超过 M2**；mode B 把 unit 文本给 Reader 同样无增益（−0.000）。

**Top-3/Top-5 Reader 敏感性（同批题、M5 排序、不做 token padding）**：k=3 → Answer F1 .311 / Evidence F1 .232；k=5 → .327 / .197。
→ Top-k 下 Answer F1 明显低于 4096 预算；但 **Evidence F1 反而更高**（4096 上下文预测了约 15–20 个段落，官方 paragraph-F1 的 precision 被拉低）→ **证据评估应以检索指标为准，QA Evidence F1 受上下文长度影响。**

### 8.9 机制追踪（逐 unit 使用轨迹）

| 指标 | 值 |
|---|---|
| verified units | 733 |
| 从未成为检索候选 | **0** |
| 从未进 unit top-5 | 136 |
| 至少一次触及 gold | 229 |
| **触及 gold 但从未进 unit top-5** | **21** |
| 所在题 M5 QA 改善的 unit | 197 |
| 所在题 M5 QA 变差的 unit | 259 |
| 至少有一个 gold-touching unit 的题 | 193 |
| **unit 层 gold 覆盖（193/333）** | **58.0%** |
| unit 层覆盖（193/372） | 51.9% |

**机制结论**：
1. **排序不是主要瓶颈**（733 个中 229 个至少一次触及 gold，只有 21 个"触及但从未进 top-5"）；
2. **覆盖是主瓶颈**（只有 58% 的题 gold 被任一 unit 来源触及）；
3. **位移是第二瓶颈**（等权 RRF 在未覆盖题上仍塞入 unit 来源段落）；
4. **Reader 不领情**（覆盖子集上 M5 的 QA 也未超过 M2）。

### 8.10 案例（阶段 D）

**正例 1：unit 覆盖 gold 时提升 All@5**
`50be4a73…` / Q: "What background knowledge do they leverage?" / gold 段落 `s01::p000, s06::p000, s06::p002`。
All@5：M2 .000 → M3 .000 → **M4 .500 → M5 .500**。
top-5 unit 中命中 gold 的是 `1503.00841::u000`：*"The bias problem in prior knowledge motivates three auxiliary regularization terms in the Generalized Expectation Criteria framework to make GE-FL more robust."*（来源 s00::p002/p003/p005 + s01::p000）→ unit 把跨 4 个段落的组合知识压成一句，回溯到 s01::p000。

**正例 2：同上机制的第二例** `02428a8f…` / Q: "How do they define robustness of a model?" → M2 .000 → M4 .500（命中 u000 与 u002）。

**负例 1：unit/关系挤掉正确证据** `6bf5620f…` / Q: "What baseline model is used?" / gold 只有 `1603.01514::s06::p000`。
All@5：M2 1.000 → M3 1.000 → **M4 0.000 / M5 0.000**（Any@5 也从 1.000 掉到 0.000）。
top-5 全被该论文的其它 unit 占据（u011/u000/u009/u010/u005，均 not-gold）→ 等权 RRF 把 unit 侧候选整体前置，单段落 gold 被挤出。

**负例 2** `747b847d…` / Q: "Which parallel corpora are used?" / gold `1603.01514::s08::p000` → 同样 M2 1.000 → M4/M5 0.000。

**过度推断被 verifier 拒绝（3 例，原文理由）**：
1. *"GE-FL is the baseline that leverages labeled features as prior knowledge, and the paper addresses the robustness problem on top of GE-FL."* → 拒绝理由：这是 s01:p000 的复述，且 triple 谓词 "requires" 不被支持（GE-FL 是基线，不是"要求该问题"）。
2. *"…but only 8% of English arguments and 17% of German arguments are aligned."* → 拒绝理由：原文陈述了模型使用对齐与 8%/17% 比例，但**没有支持"这些比例 'limit' 模型使用对齐"这一因果**。
3. *"Because crosslingual parameters are ignored when trained parameters parse monolingual data, information coming from the CLVs acts as a regularizer…"* → 拒绝理由：本质是 p009 的复述，按"单段复述"拒绝。

### 8.11 阶段 D 结论（六个研究问题的回答）

1. **QASPER 是否解决了 LoCoMo 的"池太小"问题？** **部分解决**。池从 19 个 session 提升到 P50=43/P90=70 段落，但 top-5 仍只覆盖池的 14%；Dense Any@5 .740 接近饱和，区分度来自 All-Evidence（.593）与多证据（.394）/跨章节（.295）子集。
2. **Read-Time 理解是否提高 gold 证据召回？** **取决于表示与覆盖**：Triple（M3）整体中性偏正（+1.1pp，CI 跨 0）、覆盖子集为正；Unit 独立索引与联合检索（M4/M5）**整体显著负迁移**。
3. **哪种表示最有效？** **Triple 是唯一整体不伤检索的表示**；Unit 在覆盖子集有显著增益（Any@5 +9.1pp、All@5 +6.0pp）但整体因覆盖不足与位移失败；两者结合（M5）继承 unit 的位移问题，无叠加收益。
4. **有多少收益独立于普通索引扩充？** M6 全覆盖摘要 All@5 **.575**（接近 dense .593），而与 insight 同数量的 M6-matched 仅 **.360** → **摘要层的收益主要来自"覆盖了所有段落"，不是"理解"**；在固定知识数量下 M3 略优于 M4/M6-matched，但没有大效应。
5. **对跨段落/多证据题是否有明显改善？** **没有稳定增益**：多证据 M3 .362 vs M2 .358；跨章节 .312 vs .312。Unit 方法在该子集显著变差（多证据 All@5 .216 vs .358）。现有 7.3 unit/篇的覆盖太稀疏。
6. **下一阶段该研究什么？**（按性价比排序）① **覆盖扩张**（按 section/段落生成 unit，目标覆盖每个章节，而非每篇 ~7 个）；② **相关性门控融合**（用校准分数决定是否启用 unit 通道，而非固定 RRF 权重）；③ **unit → passage reranker**（先检索 unit 再用 unit 重排来源段落，而不是把 unit 排名并入段落排名）；④ Reader 侧不要把 unit 当证据文本塞进上下文（mode B 无增益）；⑤ 同时换更强 Reader，否则检索收益无法体现。

---

## 9. 跨阶段综合与结论

### 9.1 假设判定总表（汇总）

| # | 假设 | 判定 | 量化 |
|---|---|---|---|
| H1 | 无监督分组让相关 session 共置 | 弱成立 | entity_idf 共置 .431 vs .304；端到端无收益 |
| H2 | 联合抽取优于逐 session 抽取 | 成立 | C2 证据 .837 vs .819；test R@5 .875 vs .864 |
| H3 | 理解层产生新可验证语义关联 | 不成立 / 中性偏正 | C6≈C5、C3<C2；QASPER M3 +1.1pp CI 跨 0 |
| H4 | 理解图谱提高检索与 QA | 检索成立、QA 不成立 | test 证据 +0.7~1.8pp；EM/F1 ≤±0.5pp |
| H5 | Question-Time 补全关系链 | 检索弱成立、QA 不成立 | Q2 R@5 +1.7pp(dev)/+0.6pp(test) |
| H6 | 两条路线互补 | 不成立 | R1Q2 检索最好但 QA 不如 R1Q1/R1Q0 |
| H7 | 全文直读已很强、聚类非必要 | 成立 | Full-Context .493 ≈ 最好图检索 .492 |
| H8 | 瓶颈在 Reader | 成立 | Oracle+P0 .387 ≈ 正常 .390；Oracle+P7 .603 |
| H9 | 关系价值取决于是否进 top-5 fact | 成立 | 99 → 5 → 1 漏斗 |
| H10 | 反思生成的技能可泛化 | 不成立 | 28 命中 / 1 help / 1 harm / 26 中性 |
| H11 | Unit 是更好的理解载体 | 整体否、覆盖子集是 | 覆盖 .567→.627；未覆盖 .514→.089 |

### 9.2 三条跨数据集稳定的事实

1. **"检索层改善"可以复现，"答案层改善"几乎不能。**
   - LoCoMo：R@5 .864→.885（+2.1pp）、证据 .831→.852（+2.1pp），而 EM 区间 .236–.244（0.8pp 跨度）、F1 .472–.494（2.2pp 跨度）；
   - QASPER：检索指标差异可达 15pp（M4 负迁移），但 Answer F1 只在 .435–.487 之间（Oracle 也只有 .487）；
   - 两个数据集上的因果链都是"检索 → Reader"这一环被掐断。

2. **"知识数量"不是驱动因素，"覆盖 + 可达性"才是。**
   - LoCoMo：16 条 → 10 条关系，指标不变；27 条（P3）无收益；
   - QASPER：全覆盖摘要（4,561 条）All@5 .575 vs 同数量匹配（733 条）**.360**；unit 侧 733 个中只 21 个"触及 gold 却进不了 top-5"，而 58% 的题**根本没有 unit 覆盖**。
   - 统一表述：**增益 = f(覆盖是否命中 × 是否进入 top-k 竞争)**，与生成量无关。

3. **"更多上下文"通常会伤害性能。**
   - LoCoMo：Full-Context（11.4k token）比 Oracle（761 token）差（P7 .546 vs .603）；
   - LoCoMo Question-Time：Q2/Q3 增加检索结果后证据覆盖上升但 F1 下降（test Q2 .489 尚可、Q3 .472）；
   - QASPER：4096 上下文预测 15–20 段时 Evidence F1 被 precision 拉低；Top-5 时 Answer F1 掉到 .327 但 Evidence F1 升到 .232。
   - 机制解释：Reader 对无关片段敏感；PPR 权重被新增边重新分配会"挤掉"原本正确的证据（q24/q29/q54 三个案例）。

### 9.3 失效模式分类（本项目的核心分析框架）

| 失效模式 | 定义 | 证据 | 归属 |
|---|---|---|---|
| **① 未到达（not reached）** | gold 证据不在上下文 | LoCoMo 12/81 题（14.8%）；QASPER 41% 的题无 unit 覆盖 | 检索 / 覆盖 |
| **② 到达但未被使用（reached-but-unused）** | 证据在上下文，Reader 仍答错 | LoCoMo 53/69；conv-30 P0 Oracle 不升反降 | Reader |
| **③ 到达但结构不对（reached-but-displaced）** | 新知识把原本正确的证据挤出预算或改变 PPR 权重 | q24（s10 被挤出）、q29（s15 掉到第 9）、q54（dense fallback） | 图谱扩充的副作用 |
| **④ 表示层不可检索（representation-unreachable）** | 知识存在但竞争不过 top-k | LoCoMo 99 有机会 → 5 进候选；QASPER unit 排序不是瓶颈但融合会致位移 | 表示 / 融合 |
| **⑤ 策略过度应用（over-application）** | 规则/技能被错误触发 | Retrieval Skill 触发率 92%；进化 answer skill 把确切日期改成 "Last Friday" | Question-Time |

> **这个框架解释了为什么"加知识"和"加技能"都收效甚微**：它们主要作用于 ① 和 ④，而 ② 与 ③ 才是主要损失来源，② 完全在 Reader 侧，③ 是加知识的**负作用**。

### 9.4 两个数据集的同/异结论

| 结论 | LoCoMo | QASPER | 一致性 |
|---|---|---|---|
| 图检索优于 dense | 是（R@5 .864 vs .781） | **否**（Any@5 .692 vs .740） | **不一致** → 图检索优势非普适 |
| 理解层（triple）有正收益 | 检索有（+0.7~1.8pp），QA 无 | 中性偏正（+1.1pp，CI 跨 0） | **一致**（弱正/中性） |
| 理解层（unit/长句）有效 | 未测（LoCoMo 用 triple） | 整体负、覆盖子集正 | 单向证据 |
| 全文直读很强 | 是（.493） | 是（Full paper .476 > 所有检索方法） | **一致** |
| Reader 是上限 | 是（Oracle+P0 .387） | 是（Oracle .487 vs dense .464） | **一致** |
| 池子变大能测出更多 | 否（R@20 饱和） | 部分（All@5 / 多证据子集有区分度） | 设计意图达成度：中 |

### 9.5 方法学教训（供后续项目直接复用）

1. **先做 Oracle 诊断，再改方法。** 本项目最有价值的一次实验是"把 gold evidence 直接给 Reader"——它一次性排除了"检索不够好"这个假设，并解释了前三个阶段的上限。
2. **区分"证据到达"与"证据使用"两类指标。** 只报 R@5/EM 会把两类完全不同的问题混在一起。
3. **对小样本必须报噪声带。** conv-30 上 1 题 = 0.95pp（105 题）/ 2.3pp（43 题）；P0 同 prompt 三次运行 EM 波动 21.4→35.7%。**任何 1–2 题的翻转都不能单独作为方法有效证据。**
4. **"机制可追溯"是硬门槛。** 本项目只承认**同时**通过检索层（gold 名次/Ev）与答案层（EM 翻转）且能**消融定位到单条关系**的案例（q16）。其余"变好"一律归入噪声。
5. **报告"机会数"而不是只报"收益数"。** `90 题有机会 → 1 题兑现` 比 "+0.95pp" 更能说明方法的天花板。
6. **负结果要写清楚"是谁的错"。** q54 的退化不是 Comprehension 的错，是"加知识"这个动作的副作用——如果不做 C2 vs C3 的对照，就会得出错误的归因。
7. **自进化系统的门控比生成更重要。** 阶段 C 中真正按设计发挥作用的只有两处：`gold 未进上下文 → 不生成 answer skill` 的硬门，以及 `dev 快照选择否决了降低证据覆盖的技能`。

---

## 10. 成本与工程账

### 10.1 阶段 A（LoCoMo 全链路）

| 项 | 数值 |
|---|---|
| 记录的 HY4 调用（我们自己的客户端） | **61,818** |
| 其中 429 | **22,159（36%）** |
| 输入 token | **149,371,650** |
| 输出 token | **1,447,612** |
| 缓存命中 | 0（首轮全量，`cache_hits` 字段为 0） |

> 上述只统计 `src/adapters/hy4_client.py` 的调用（Reader、QDM、Retrieval/Answer Skill、抽取、校验）。HippoRAG 官方的 OpenIE（NER+三元组）与 fact rerank 走 `CacheOpenAI`，缓存在各 `save_dir/llm_cache/*.sqlite`，**不在上表**——实际总调用量更高。HY4 目前免费，但延迟与配额是实际瓶颈（夜间有效吞吐低至 0.7 req/s，白天可达 5 QPS）。

### 10.2 阶段 B（conv-30 Prompt 优化）

| log shard | 行数 | ok | errors |
|---|---|---|---|
| `hy4_callspo`（Reader prompt） | 450 | 450 | 0 |
| `hy4_callsor`（Oracle/重复） | 266 | 266 | 0 |
| `hy4_callsjo`（联合） | 237 | 237 | 0 |
| `hy4_callsoa`（官方对抗协议） | 96 | 96 | 0 |
| `hy4_callsco`（Comprehension 抽取） | 11 | 11 | 0 |
| `hy4_callsrp`（重复运行，禁用缓存） | 98 | 84 | **14（429，全部重试成功）** |

> Comprehension 图的官方 LLM fact rerank 调用走 `outputs/raw_graphs/conv-30/llm_cache`，不在上表；无重排消融的检索调用为 **0**。
> **没有把 429/失败/空回答计作拒答。**

### 10.3 阶段 C（Skill 自进化）

| 项 | 数值 |
|---|---|
| 调用记录 | **2,582**（缓存命中 1,023） |
| 端点尝试 / 成功率 | 2,186 / **.713**（429 = 627，重试行 619） |
| prompt / completion tokens | 7,431,528 / 98,445 |
| 延迟 P50 / P95 | 2.05s / 3.79s |
| 最贵的 stage | `verify` 273 次 / 1.23M prompt tok |

逐题成本（evaluation）：E0 1.00 call / 4,462 tok；E3–E6 3.00 calls / ~5.4k tok；R2 4.26 calls / 10,052 tok。
**缓存命中的调用不消耗配额；adaptation 的 reflect/verify 是一次性训练成本，E-matrix 是每评测题成本。**

### 10.4 阶段 D（QASPER）

| 项 | 数值 |
|---|---|
| G_raw 建图累计秒 | **25,465**（100 篇，单卡） |
| M2/M3 变体累计秒 | 1,600 |
| read-time tokens | prompt 2,883,843 / completion 1,620,654 |
| Reader tokens | prompt 9,866,962 / completion 127,279 |
| OpenIE token | 见各 `graphs/raw/*/llm_cache`（官方缓存，未并入本表） |
| 速率 | 跨进程单桶 ~1 QPS（`HY4_RATE_FILE=qasper`） |

> 中途曾尝试多 key 并发，被环境判定为过载共享池，**已回退**；报告所有结果均为单桶 1 QPS 产生。

---

## 11. 局限、噪声与实现风险

### 11.1 设计层面的局限（影响结论强度）

| # | 局限 | 影响 |
|---|---|---|
| 1 | **LoCoMo 结构性天花板**：单对话 1.1–2.2 万 token、19–32 篇文档、R@20 ≈ 100% | Read-Time 的价值被"文档太短、直接读就行"压缩；不宜作为主证据 |
| 2 | **QASPER 池仍然小**：P50 43 段，top-5 占 14%，dense Any@5 .740 | 单证据接近饱和，只有 All@5/多证据子集有区分度 |
| 3 | **conv-30 被前序轮次研究过** | 阶段 B/C 的 evaluation 是**内部探索性测试**，不是独立 benchmark |
| 4 | **小样本**：conv-30 105 题（81 可答）；阶段 C evaluation 仅 43 题（33 可答、multi-hop 5 题） | 1 题 = 0.95pp / 2.3pp；类别级结论只能算方向性 |
| 5 | **QASPER 只跑 100/281 篇** | 扩大是机械性工作，pipeline 已支持；但当前数字是 100 篇样本的估计 |
| 6 | **图表证据（253 条，3.5% 的题仅图表）不进文本检索指标** | 这部分题的检索结论不适用；QA 仍包含它们 |
| 7 | **跨段 insight 用"注入每个来源段落"近似超边** | 不是真正的超图，可能低估跨段证据的联合召回 |
| 8 | **阶段 B 的 P7 是 dev-informed 综合 prompt**（看过 dev 的 P0–P6 结果后加入），P8 是保守变体；val 未参与设计 | P7 的 dev 优势有乐观偏差；val 上的 EM 持平和 test 上的方向一致是可接受的缓解，但**不能宣称 P7 是"最优通用 Reader"** |
| 9 | **Skill Evolution 第一版（E4）不是自动生成**，而是 `(category, suffix)` 网格选择 | 只能作为"检索后缀有效"的弱证据（+0.59pp），不能算 RSI |
| 10 | **阶段 C 只有 2 次独立重复运行**（配额限制） | 无法做 3 次多数票；EM 稳定性只有 1 题级别的证据 |
| 11 | **Answer/Retrieval Skill 的复用计数是派生指标** | `reuse_count` 不落盘，由报告期按 `created_at_feedback` 之后被选中次数回算 |

### 11.2 测量噪声（量化）

| 噪声源 | 量级 | 来源 |
|---|---|---|
| Reader 非确定性（同 prompt/context/T=0） | P0 三次 EM **21.4 / 28.6 / 35.7%**（14 题集）；4 题"连顺序都相同"仍 EM 不同 | 阶段 B §6.5 |
| 图谱间的 EM 翻转 | conv-30 上 1–3 题（C2/C3/C5/C6） | 阶段 A |
| 跨对话波动 | HippoRAG EM .317–.479（**16pp**） | 阶段 A §5.6 |
| 检索非确定性 | G_raw 的 diag 重跑 12/105 题排名不同（C2 完全一致）；同题单独 vs 批量检索结果不同（GPU 批次数值差） | 阶段 B §6.1.3 |
| fact reranker 非确定性 | 每题的 LLM 过滤本身随机 → 同图重跑可改变 top-5 fact 与 PPR | 阶段 B §6.9 |
| 上下文装填敏感性 | 4096 预算下平均 6.1 篇/≈3,940 token；少一篇即可能丢证据 | 阶段 A |

### 11.3 文档与代码不一致（17 项，请审核时以代码为准）

| # | 现象 | 正确表述 | 证据 |
|---|---|---|---|
| 1 | `select_by_budget(all_or_nothing=True)` 的**命名与语义相反** | `True` = 跳过放不下的文档继续装；`False` = 遇到第一个放不下就停止 | `src/evaluation/metrics.py:76-95` |
| 2 | `variants.py` docstring 称"manifest 故意不复制" | 代码 `skip = {"graph.pickle"}`，**manifest 是复制的**（否则身份校验不过） | `src/graph/variants.py:5-7` vs `:57-59` |
| 3 | `verify_candidates` 失败时"全部保留"（**fail-open**）；且 `run_scope` 汇总的 `relations` 用的是**未验证**列表，`relations_verified` 只落在 per-group 记录里 | 若引用验证结果，需显式读 `relations_verified` | `src/comprehension/extract.py:243-252, 278-287` |
| 4 | `src/question_skill/skills.py` 的 `SkillLibrary` / `DEFAULT_SKILLS` 是**死代码**（全仓库无引用） | 真正的冻结机制在 `scripts/run_skill_evolution.py`，产物 `outputs/skills/frozen_skill_library.json` 是 `category→suffix` 映射，与 `Skill` dataclass 结构不同 | `skills.py:37-96` |
| 5 | `answer_skill` 的 `answerable=False` **不会**强制改写为 "No answer" | 拒答完全依赖模型遵循 prompt 第 3 条 | `src/question_skill/pipeline.py:134-136` |
| 6 | Reader context **不含 turn id**（`D10:1`） | turn 级 evidence 覆盖是离线用 `turn_to_session` 计算的，不依赖 reader 是否看到 dia_id；dia_id 只出现在 comprehension 的 `sessions_block` | `reader.py:39-44` + `locomo.py:77-80` |
| 7 | 阶段 C 的快照实际为 **0/10/20/30/40/43**（`%10==0` 命中 40） | 报告文字写"0/10/20/30/43"是笔误 | `answer_evolve.py:477` vs `s7_reports.py:329` |
| 8 | 阶段 C 的技能检索 top-k：dev/adaptation/E-matrix 用 **k=2**，ReAct/重复实验用默认 **k=4** | 两个口径，需在复现时区分 | `skills.py:131` vs `s4_react.py:119` |
| 9 | `common.REFUSAL_MARKERS` 是死代码 | 实际拒答词表在 `src/prompt_optimization/runner.py:29-30` | — |
| 10 | `verify_candidate` 的 `leak_pass` **硬编码 True** | 泄漏约束完全依赖前置 sanitizer | `answer_evolve.py:295` |
| 11 | `MASTER_REPORT.md` §6.5 的 dev 图规模表 **行错位**（"conv-30 505/750" 实为 conv-26；conv-30 真实值 441/600） | 以 `CONV30_READTIME_ANALYSIS.md` §1.1 与 `openie_state.json` 双向核对值为准 | — |
| 12 | `MASTER_REPORT.md` §9 写"C5/C6 在 test 上只跑了 C6" | 与 `metrics.csv`（有 `full_conv_triple,test,1484` 行）和 `FINAL_EXPERIMENT_REPORT.md` §15.5/§15.11 矛盾；**C5 test 已完成** | `metrics.csv` |
| 13 | `FINAL_EXPERIMENT_REPORT.md` §15.8 写 D 类占 95.5% | 按 `comprehension_audit.json` 逐对话求和：D=688/717 = **95.9%** | `COMPREHENSION_AUDIT.md` |
| 14 | QASPER 审计文档写"Any-Evidence@k uses the union" | 代码是**逐标注者集合算 any 再平均**；union 只作次要视图 | `src/qasper/retrieval_metrics.py:41,57-68` |
| 15 | QASPER `experiment_config.md` 写"2,000 paired resamples" | 单方法 CI 是**非配对**单向量重采样；paired 仅用于方法间差值 | `retrieval_metrics.py:118-125` vs `make_reports.py:85-102` |
| 16 | QASPER Evidence F1 的比较层面 | 预测侧是**段落原文**、gold 是**原始 evidence 字符串**，`paragraph_f1_score` 为严格字符串集合运算（无归一化）→ 对空白/裁剪差异敏感，不是 passage-id 级匹配 | `official_eval.py:41-51`；`run_qa.py:126` |
| 17 | QASPER verifier"无裁决视为保留" | fail-open，会略微抬高 unit 数量 | `readtime.py:291-293` |

### 11.4 未做（Not Run）与原因

| 项 | 状态 | 原因 |
|---|---|---|
| Question-Time 的 Q1/Q2 在 test 全量 | **已做**（`metrics.csv` 有 `raw_q1/raw_q2,test`） | — |
| C3v（LLM 验证变体）的 conv-30 端到端 | 未做 | 抽取产物缺失 |
| Skill Evolution 的"自动提出/合并/淘汰"多轮迭代 | 未做 | 现为网格选择；多轮演化在阶段 C 才实现（单对话） |
| 按类别分层的 Reader 改造（时间推理/多跳整合） | 未做 | 被认为是最有价值方向，本轮未实现 |
| 更长语料验证（NarrativeQA / Multi-LoCoMo） | 未做 | LoCoMo 结构性天花板使 Read-Time 价值被压缩 |
| QASPER 全 281 篇 | 未做 | 机械性工作；100 篇已冻结为正式样本 |
| QASPER semantic 分组的 QA | 未做 | 仅 20 篇探索性检索对照 |
| 更强的 Reader（更大模型 / 同模型加长思维链） | 未做 | 本项目限定单一 LLM（HY4 no_think）以保证跨方法可比；这也是"Oracle 只有 .487 F1"的直接原因 |

---

## 12. 复现

### 12.1 环境

```bash
cd /data/workspace/group/licochen/Memory/RSI_GraphRAG
source .venv/bin/activate          # python 3.10, torch 2.5.1, hipporag 2.0.0a5（venv 在本地盘 /root/venvs/rsigraph）
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken
export HF_HUB_OFFLINE=1
export HY4_GLOBAL_QPS=1            # 共享池实测：1 QPS ~90% 成功，2 QPS ~7%
export HY4_RATE_FILE=poolA         # 同池 key 必须同值；独立账号用 poolB
```

### 12.2 阶段 A（LoCoMo）

```bash
python scripts/prepare_data.py
python scripts/run_baselines.py --split dev  --out-prefix baselines
python scripts/run_baselines.py --split test --out-prefix baselines
python scripts/run_grouping.py --split dev  --budgets 4096,8192
python scripts/run_grouping.py --split test --budgets 8192
python scripts/run_read_time_extract.py --split dev  --scopes cluster_hybrid_0.5_b8192,full_conv --modes triple,comp --max-tokens 16384
python scripts/run_read_time_extract.py --split test --scopes cluster_hybrid_0.5_b8192,full_conv --modes triple,comp --max-tokens 16384
python scripts/run_read_time_eval.py --split dev --variants <scope>:<suffix>
python scripts/run_question_time.py --convs conv-30 --strategies q1,q2,q3
python scripts/run_skill_evolution.py                       # dev 学习 + 冻结
python scripts/run_skill_evolution.py --apply-only --library outputs/skills/frozen_skill_library.json
python scripts/aggregate.py && python scripts/build_final_tables.py && python scripts/make_manifest.py
python -m pytest tests -q                                   # 8 项全部通过
```

### 12.3 阶段 B（conv-30 Prompt 优化）

```bash
python scripts/prompt_opt/00_official_metrics.py     # 官方口径 vs 项目口径
python scripts/prompt_opt/01_prompt_audit.py         # prompt 渲染与哈希
python scripts/prompt_opt/02_error_analysis.py       # 105 题错误归因 + 落盘 reader 上下文
python scripts/prompt_opt/03_make_splits.py          # dev40 / val65（seed 42）
HY4_CACHE_SHARD=po python scripts/prompt_opt/04_reader_pilot.py --split dev --pids P1,P2,P3,P4,P5,P6 --max-workers 6
HY4_CACHE_SHARD=po python scripts/prompt_opt/04_reader_pilot.py --split dev --pids P7
HY4_CACHE_SHARD=po python scripts/prompt_opt/04_reader_pilot.py --split dev --pids P8
python scripts/prompt_opt/04b_summarize.py --subdir pilot --pids P1,P2,P3,P4,P5,P6,P7,P8
# 在看到 val 结果之前冻结选择（reader/pilot/selection.json）
HY4_CACHE_SHARD=po python scripts/prompt_opt/04_reader_pilot.py --split val --pids P7
HY4_CACHE_SHARD=po python scripts/prompt_opt/04_reader_pilot.py --split val --pids P8
# Comprehension：P1..P5 生成 → 各建独立图 → 105 题检索 → 质量审计 → 关系数量控制
python scripts/prompt_opt/18_no_llm_rerank.py        # 关闭 LLM fact reranker（top5/top20）
```

### 12.4 阶段 C（Skill 自进化）

```bash
cd outputs/question_evolution_conv30
HY4_CACHE_SHARD=qe HY4_RATE_FILE=qe_main HY4_GLOBAL_QPS=1.2 python -m qecode.s0_split
python -m qecode.s1_baselines
python -m qecode.answer_evolve
python -m qecode.retrieval_evolve
python -m qecode.s3_joint          # E0–E6 + dev 快照选择
python -m qecode.s4_react          # R0–R5
python -m qecode.s5_repeats
python -m qecode.s6_analysis
python -m qecode.s7_reports
```

### 12.5 阶段 D（QASPER）

```bash
export CUDA_VISIBLE_DEVICES=0 HY4_GLOBAL_QPS=1 HY4_RATE_FILE=qasper HY4_OPENIE_WORKERS=24
python -I scripts/qasper/prepare_data.py
python -I scripts/qasper/run_baselines.py --scope sample100 --methods bm25,dense,hybrid
python -I scripts/qasper/build_graphs.py --scope sample100 --stage raw
python -I scripts/qasper/run_readtime.py --scope sample100 --grouping section
python -I scripts/qasper/build_variants.py --scope sample100 --grouping section
python -I scripts/qasper/run_retrieval.py --scope sample100 --only-ready
python -I scripts/qasper/run_qa.py --scope sample100 --grouping section
python -I scripts/qasper/track_units.py --scope sample100 --grouping section
python -I scripts/qasper/make_reports.py --scope sample100 --grouping section
```

---

## 13. 附录

### 13.1 附录 A：QASPER Reader prompt（**要点整理，非逐字**；逐字原文见 `src/qasper/reader.py:15-33`）

```
[System]
Use only the provided excerpts from the paper. Do not use outside knowledge. Do not invent
numbers or entities. If the excerpts do not contain the answer, say so.

[User]
Excerpts from the paper:
{context}

Question: {question}

Rules:
- The final line must start with 'Answer: ' and contain a short answer / short phrase.
- For yes/no questions answer Yes or No.
- Use the exact numbers and units from the excerpts.
- If the excerpts do not answer the question, write 'Answer: Unanswerable'.
- Do not explain after the answer.

Answer:
```

调用参数 `temperature=0.0`、`max_tokens=256`；解析 `Answer:\s*(.*)` 取第一行。
上下文构建（`build_context`）：总 token ≤ budget（默认 4096）；① 若有 unit 且 `unit_budget>0`，按排序依次追加 `[Insight | {unit_id}] {concise_text}`，**装不下就跳过**，直到 unit 预算（默认 1024）用尽；② 再按检索排名贪心追加 `[{passage_id} | {section_name}]\n{text}`；oracle 用 8192（gold 段落），full paper 用 12000（整篇截断，明确非固定预算基线）。

### 13.2 附录 B：数字速查

| 问题 | 数字 |
|---|---|
| LoCoMo HippoRAG2 全量 | EM .244 / F1 .483 / R@5 .863 / 证据 .828 |
| LoCoMo Dense 全量 | EM .221 / F1 .436 / R@5 .791 / 证据 .742 |
| LoCoMo Full-Context 全量 | EM .245 / F1 .491 |
| multi-hop 检索增益 vs QA 增益 | 证据 +16.7pp，F1 +4.0pp |
| temporal | 证据 .875 / F1 .238 |
| test 最好 Read-Time（C5） | EM .245 / F1 .489 / R@5 .874 / 证据 .841 |
| test 最好联合（R1Q1） | EM .244 / F1 .494 |
| 分组随机 8192 gold 覆盖 | .883（entity_idf .905） |
| Skill Evolution（test，1,137 题） | R@5 +0.59pp / 证据 +0.54pp |
| conv-30 G_raw（P0） | EM .210 / F1 .390 / 官方 F1 .426 / 证据 .886 |
| conv-30 P7 | EM .235 / F1 .521 / 官方 F1 .546 |
| Oracle + P0 / + P7 | F1 .387 / .603 |
| 最好 Comprehension（P4） | Ev .8857→.9048（+1 题 q16）、R@5 .9127 |
| P4 + no-rerank top5 + P7 | EM .259 / F1 .539 / 官方 F1 .569 |
| 关系漏斗 | 99 有机会 → 5 进 fact 候选 → 1 兑现 |
| 同输入 3 次 EM 波动（P0） | .214 / .286 / .357 |
| 阶段 C E0 / E2 | EM .303/.273、F1 .563/.508 |
| 阶段 C 技能复用 | 28 命中 / 1 帮助 / 1 有害 / 26 中性 |
| QASPER Dense / M1 / M3 / M4 / M6（All@5） | .593 / .544 / .556 / .401 / .575 |
| QASPER Oracle Answer F1 | .487（dense .464，full paper .476） |
| QASPER unit 覆盖 | 58.0%（193/333） |
| QASPER verifier 保留率 | 55.8%（733/1,313） |

### 13.3 附录 C：产物索引

**阶段 A/B/C（LoCoMo）**

| 内容 | 路径 |
|---|---|
| 总报告 | `outputs/reports/MASTER_REPORT.md` |
| 综合报告 / 验收入口 | `outputs/reports/FINAL_EXPERIMENT_REPORT.md` / `START_HERE.md` |
| 统一指标 CSV（26 行方法×划分组合） | `outputs/reports/metrics.csv` |
| 自动结果表 | `outputs/reports/FINAL_TABLES.md` |
| 数据集审计 / 分组 / Read-Time / Question-Time / 联合 / 审计 | `outputs/reports/{DATASET_AUDIT,GROUPING_REPORT,READ_TIME_REPORT,QUESTION_TIME_REPORT,JOINT_EVALUATION,COMPREHENSION_AUDIT}.md` |
| conv-30 专项分析（含噪声标尺与 10 个案例） | `outputs/reports/CONV30_READTIME_ANALYSIS.md` |
| 运行清单（commit / sha256 / token） | `outputs/reports/run_manifest.json` |
| 每题明细 | `outputs/qa/*.jsonl` |
| 图谱与变体 | `outputs/raw_graphs/<conv>/`、`.../variants/<name>/` |
| 聚类结果 / 抽取结果 | `outputs/groupings/<conv>/*.json`、`outputs/comprehension/<conv>/*.json` |
| 冻结技能库 | `outputs/skills/frozen_skill_library.json` |
| Prompt 优化全套 | `outputs/prompt_optimization_conv30/`（`audit/ reader/ comprehension/ joint/ cases/ reports/`） |
| Prompt 优化完整报告（含全部 prompt 原文、105 题总表、20 个案例、105 题错误归因、完整日志） | `outputs/prompt_optimization_conv30/reports/COMPREHENSIVE_REPORT.md` |
| 自进化代码与产物 | `outputs/question_evolution_conv30/{qecode,artifacts,skill_snapshots,logs}` |
| 自进化完整报告 | `outputs/question_evolution_conv30/COMPREHENSIVE_REPORT.md` |
| HY4 调用日志 | `outputs/logs/hy4_calls*.jsonl` |

**阶段 D（QASPER）**

| 内容 | 路径 |
|---|---|
| 单文件完整报告 | `outputs/qasper_readtime/QASPER_READTIME_FULL_REPORT.md` |
| 自动结果报告 | `outputs/qasper_readtime/FINAL_REPORT.md` |
| 数据审计 / 检索基线 / 生成质量 / 案例 / 配置 | `outputs/qasper_readtime/{dataset_audit,baseline_retrieval,comprehension_quality,cases,experiment_config}.md` |
| 逐题检索 / QA | `retrieval/comparison_sample100.jsonl`（4,662 行）、`qa/predictions_sample100.jsonl`（3,348 行） |
| unit 使用轨迹 / 逐题链路 | `mechanism/unit_usage_sample100.jsonl`、`mechanism/question_chains_sample100.jsonl` |
| 图 / 变体 / read-time 产物 | `graphs/raw/{paper}/`、`graphs/raw/{paper}/variants/{atom,atom_comp}/`、`comprehension/section/{paper}/` |
| 冻结样本 | `dataset/sample.json` |

---

### 13.4 附录 D：术语表

| 术语 | 含义 |
|---|---|
| Read-Time | 索引/阅读阶段（问题到来之前）做的理解与抽取 |
| Question-Time | 问题到来时做的拆解、补检索、校验与修订 |
| G_raw | 官方 HippoRAG2 在原始 per-session OpenIE 上建出的图（冻结，永不被修改） |
| C2 / C3 / C5 / C6 | 见 §5.3（聚类联合抽取 / +理解 / 全文联合抽取 / 全文+理解） |
| COMP-P0…P5 | 阶段 B 的 6 个 comprehension prompt |
| P0…P8 | 阶段 B 的 9 个 reader prompt |
| Q0…Q3 | 阶段 A 的 4 种 question-time 策略 |
| M0…M6 | 阶段 D 的检索方法（BM25/Dense/Hybrid / G_raw / +Atomic / +Triples / +Units / +Units on M3 / +Summaries） |
| unit | QASPER 的 comprehension unit：一句 ≤40 词的自足陈述 + triple + 来源段落集合 |
| insight | verifier 之前的候选理解条目（unit 是验证通过后的版本） |
| dense fallback | rerank 后无 fact 存活 → 整题退化为纯向量检索（top-1 分数被归一化为 1.0） |
| top-5 fact 漏斗 | 新关系 → 进入 top-5 fact 候选 → 改变 PPR → 证据覆盖改善（本项目核心机制指标） |
| E0–E6 / EB / R0–R5 | 阶段 C 的方法矩阵（见 §7.7 / §7.8） |
| Oracle | 把 gold evidence 所在完整文档直接作为 reader 上下文（只用于诊断，非方法） |

### 13.5 附录 E：一句话版结论（用于摘要/答辩）

> 在 LoCoMo 与 QASPER 两个数据集上，Read-Time 的跨文档理解能把**检索端的证据召回**稳定提高 1–2pp（QASPER 上的 unit 表示在有覆盖时 +6pp），但**几乎不改变最终答案质量**；把 gold 证据直接交给 Reader 也不改善原始 Reader 的表现（F1 .387 vs .390），而仅优化 Reader prompt 即可把同一批证据的 F1 从 .390 提到 .603。因此本项目的核心结论是：**当前系统的瓶颈在"如何使用已到达的证据"，而不是"如何找到更多证据"；相应地，继续扩充图谱知识或叠加 Question-Time 技能层，在现有 Reader 下都难以兑现为答案收益。**

---

*文档结束。所有表格数字均可在 §13.3 所列路径下逐题复核；若需针对某一实验做更细的逐题核查，请指明阶段编号与表名。*
