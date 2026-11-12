# QASPER Read-Time Comprehension：完整实验进度与结果报告

> 单文件汇总。所有数字由 `outputs/qasper_readtime/` 下的最终产物直接生成；完整逐题数据、图、unit、日志与缓存见文末产物索引。

**研究问题**：在 QASPER（每篇论文内部问答）上，Read-Time（阅读时）跨段落理解产生的知识——Comprehension Triple 与 Comprehension Unit——能否在 Question-Time 提升真实证据检索与最终 QA？并区分“理解”本身的收益与“普通索引扩充”的收益。

---

## 1. 执行摘要

**状态：首轮 100 篇论文全部完成。** 数据解析、三个无 LLM 基线、每篇 G_raw、M2/M3 图变体、read-time 生成与验证、M0–M6 检索对照、Reader QA（含 Oracle 与 Full-Paper）、逐 unit 机制追踪、案例与报告均已落盘。

核心结论（100 papers / 333 道有 gold 证据题 / 372 道 QA 题）：

1. **QASPER 比 LoCoMo 更难，但没有真正解决“池太小”**：每篇段落池 P50=43、P90=70，top-5 平均只占池的 14.0%。Dense Any@5=74.0%，单证据已接近饱和；All@5=59.3%，多证据子集 All@5 仅 39.4%，区分度主要来自多证据/跨章节。
2. **Comprehension Triple（M3）整体中性偏正**：相对 M2，All@5 +1.1pp（95% CI [−0.2,+2.4]）；在“unit 来源覆盖 gold”的题上更明显（M3 All@5 .585 vs M2 .567）。
3. **Comprehension Unit 的独立索引（M4）与联合检索（M5）整体显著负迁移**：All@5 .401/.395 vs M2 .545。原因不是 unit 排不上（733 个 unit 中只有 21 个“触及 gold 但从未进 unit top-5”），而是**覆盖不足**（只有 58% 的题的 gold 被任一 unit 来源触及）加上**等权 RRF 在未覆盖题上挤掉正确图证据**。
4. **在覆盖子集上 unit 有真实价值**：Any@5 .746→.837、All@5 .567→.627；未覆盖子集则从 .629/.514 崩到 .143/.089。
5. **普通索引扩充对照说明收益主要来自覆盖**：M6 全覆盖段落摘要达到 All@5 .575（接近 dense .593），而与 unit 同数量的 M6-matched 只有 .360。
6. **Reader 是 QA 的硬上限**：Oracle gold context 的 Answer F1 仅 .487，dense .464，Full-Paper .476；检索层的差异很难转化为 QA 收益。Top-3/Top-5 reader F1 只有 .311/.327（4096 预算为 .443）。

---

## 2. 环境与固定配置

| 项目 | 值 |
|---|---|
| 数据集 | QASPER v0.3 validation（`qasper-dev-v0.3.json`，sha256 前缀 `2ae7ee62a65b1c42`） |
| 语料 | 281 papers / 1,005 questions（answerable 945，unanswerable 60） |
| 检索单元 | 原始段落；id = `paper_id::s{section:02d}::p{para:03d}`；检索严格限制在题目所属论文内 |
| LLM | HY4（Tencent TokenLink），`HY4_MODEL=hy4`，temperature=0，`reasoning_effort=no_think` |
| Embedding | `nvidia/NV-Embed-v2`，fp16，dim 4096，本地 `/root/models/NV-Embed-v2` |
| 图谱底座 | OSU-NLP-Group/HippoRAG @ `d5c8329`（v2.0.0a5），官方 index/retrieve 未改 |
| PyTorch / Transformers | 2.5.1+cu124 / 4.45.2 |
| HY4 速率 | 跨进程文件限流器，单共享桶 `HY4_RATE_FILE=qasper`，约 1 QPS |
| 缓存 | 内容寻址 HY4 sqlite + 每个 HippoRAG save_dir 自带 llm_cache；断点可续 |

**冻结的首轮样本**（正式评价前写入 `dataset/sample.json`）：

- seed **20261110**；规则：uniform random sample without replacement over validation papers that have at least one answerable question with mapped gold evidence; ordering by sha1(f'{seed}:{paper_id}'), take the first 100. Rule frozen before any retrieval/QA evaluation.
- 100 papers / 372 questions；其中 333 道 answerable 且有映射文本证据，141 道多证据，56 道跨章节。
- 样本池：mean 48.09 / P10 22.9 / P90 68.2。

**防泄漏规则**：read-time 的 atomic facts、comprehension、verifier、summary 只接收论文标题、段落原文和已抽取事实；**任何 prompt 都不包含问题、答案或 gold evidence**。

---

## 3. 数据审计与 Gold Evidence 映射

### 3.1 语料与候选池

| 指标 | 值 |
|---|---|
| papers | 281 |
| questions | 1005（answerable 945 / unanswerable 60） |
| 有映射 gold 的 answerable 题 | 888 |
| gold 段落数/题/标注者 | mean 1.75，P50 1.0，P90 3.0，max 19 |
| gold 段落数分布 | 1→777, 2→223, 3→61, 4→49, 5→23, 6→9, 7→11, 8→8, 9→7, 10→1, 11→1, 12→1, 14→1, 16→1, 19→1 |
| 单段落 gold | 63.2% |
| 多段落 gold | 36.8% |
| 跨章节 | 19.0% |
| 仅图表证据题 | 3.5% |
| 每篇段落数 | mean 46.95，P10 22.0，P50 43.0，P90 70.0，max 281 |
| 每篇 token | mean 4680.71，P50 4438.0，P90 6636.0 |
| 段落 token | mean 99.7，P50 86.0，P90 193.0 |
| gold 段落 token | mean 120.48，P50 106.0，P90 221.0 |
| top-3/5/10 占池 | 8.4% / 14.0% / 27.8% |

### 3.2 证据映射质量

| 映射方式 | 数量 |
|---|---|
| exact | 2426 |
| float | 253 |
| heading | 129 |

- 文本证据映射率 **100.0%**，未匹配 **0** 条。
- `FLOAT SELECTED`（图表证据）253 条、章节标题证据 129 条：不映射到文本段落，图表证据从文本检索指标中排除、但保留在 QA 中。
- **多标注者按“替代证据集”处理**：All-Evidence@k 先在每个标注者集合上计算再平均，绝不对 union 强行要求同时召回；完全相同的标注者集合去重。
- 段落文本完全相同者做等价合并（74/13,266），保证 chunk 哈希与 gold 映射唯一。

### 3.3 按论文长度分层的题目难度

| 五分段 | 论文 token | papers | questions | 单段落 | 多段落 | 跨章节 | 平均 gold |
|---|---|---|---|---|---|---|---|
| 1 | [796, 2851] | 56 | 158 | 63.3% | 36.7% | 18.4% | 1.89 |
| 2 | [2858, 3752] | 56 | 191 | 64.9% | 35.1% | 17.8% | 1.73 |
| 3 | [3783, 4974] | 56 | 184 | 65.8% | 34.2% | 22.3% | 1.58 |
| 4 | [4985, 5860] | 56 | 191 | 63.9% | 36.1% | 18.3% | 1.71 |
| 5 | [5866, 20902] | 57 | 164 | 57.3% | 42.7% | 18.3% | 2.23 |

> 多证据比例随论文长度上升（最长段 42.7%），跨章节比例基本稳定在 18–22%。这支持把多证据/跨章节子集作为主要区分条件，而不是只看 Any-Evidence@5。

**阶段决策门槛**：段落池 P90=70，Dense Any@5 74.0%，因此没有人为切分窗口来制造 Recall；主实验保持原始段落粒度，困难度用 All-Evidence 与多证据/跨章节子集衡量。

---

## 4. 方法定义与实现

| 方法 | 图谱 / 检索结构 | 实现要点 |
|---|---|---|
| M0 | BM25 / Dense / Hybrid | 段落级；dense 用 NV-Embed-v2；hybrid 为等权 RRF |
| M1 | HippoRAG2 G_raw | 官方 OpenIE（NER+triple）→ 实体/事实/段落图 → PPR；G_raw 冻结 |
| M2 | G_raw + Atomic Facts | 把 LLM 逐段抽取的原子事实三元组注入源段落的 OpenIE 行后重建图 |
| M3 | M2 + Comprehension Triples | 在 M2 克隆状态上注入跨段落理解三元组；同一批 insight 的三元组表示 |
| M4 | M2 + Comprehension Units 独立索引 | unit 文本单独 embedding；检索后回溯到来源段落，与 M2 图排序 RRF |
| M5 | M3 + Comprehension Units | 双路：M3 图检索 + unit 检索 → RRF → 回溯真实段落 |
| M6 | M2 + 单段落摘要 | 每段一句摘要建索引，回溯同段落；另做与 unit 同数量的 matched 对照 |
| M4/M5 gated | 敏感性 | 仅取 top-3 units，unit 路权重 0.5 的 RRF（非主定义） |

实现细节：

- **G_raw 冻结**：先对每篇论文全部段落建图；所有后续版本只作为独立变体目录的增量，通过复制 raw 的 embedding/OpenIE 状态、注入三元组、重跑官方 `index()` 构建，不修改 G_raw。
- **统一 Reader**：所有方法共用同一个 QASPER prompt 和 4096-token 上下文预算；mode B 把 unit 文本放进同一预算（默认 1024 token），不是免费额外文本。
- **QA 模式**：主结果只把回溯后的**原文段落**交给 Reader；mode B 额外显示 unit。
- **三组最关键对照**：M3（triple）、M4（unit）、M5（两者结合）共享同一批通过验证的 insight，保证“知识数量固定、只变表示”。

### 4.1 Read-time Prompt 约束（摘要；全文见 `src/qasper/prompts.py`）

- **Atomic facts**：每段最多 4 条，只引用一个段落，不得跨段组合；数字/单位/数据集名原样保留。
- **Comprehension**：必须引用 ≥2 个不同段落；产出 `concise_text`（≤40 词、自包含）+ `triple` + `relation_type` + `supporting_fact_ids` + `derivation` + `confidence`；禁止复述单段事实、禁止把共现写成因果。
- **Verifier**：检查来源存在性、联合支撑、是否单段复述、是否虚假因果、是否新增未支持的数字/实体；拒绝明显错误但不因措辞不完美而删除。
- **Summary（M6）**：每段一句 ≤30 词，不跨段、不新增信息。
- 示例均为通用占位符，不使用任何评测问题或答案作为 few-shot。

---

## 5. Read-Time 生成质量

| 指标 | 值 |
|---|---|
| 处理论文 | 100 |
| reading groups（section-based，4096–8192 token） | 147 |
| atomic facts | 10,539 |
| candidate insights | 1,313 |
| verified units | 733 |
| rejected insights | 580 |
| verifier 保留率 | 55.8% |
| units / paper | 7.3 |
| unit 长度（词，mean/P90） | 24.6 / 29.0 |
| 作为 unit 来源的段落比例（每篇平均） | 30.4% |
| unit 来源跨 section 比例 | 71.2% |
| unit 来源段落距离（mean/P90） | 9.9 / 25.0 |
| paragraph summaries（M6） | 4,561 |
| read-time prompt tokens | 2,883,843 |
| read-time completion tokens | 1,620,654 |

**关系类型分布**：

| relation_type | 数量 |
|---|---|
| claim_evidence | 150 |
| method_result | 135 |
| problem_solution | 119 |
| setting_outcome | 108 |
| combined_fact | 95 |
| assumption_limitation | 65 |
| entity_resolution | 61 |

**关系类型效果（与逐 unit 使用轨迹 join）**：

| relation_type | units | 触及 gold | 进过 unit top-5 | 触及 gold 但未进 top-5 | QA helped | QA harmed |
|---|---|---|---|---|---|---|
| claim_evidence | 150 | 46 | 120 | 4 | 44 | 49 |
| method_result | 135 | 44 | 117 | 5 | 37 | 50 |
| problem_solution | 119 | 35 | 92 | 2 | 25 | 45 |
| setting_outcome | 108 | 44 | 97 | 2 | 36 | 40 |
| combined_fact | 95 | 25 | 75 | 3 | 23 | 36 |
| assumption_limitation | 65 | 11 | 47 | 2 | 16 | 15 |
| entity_resolution | 61 | 24 | 49 | 3 | 16 | 24 |

**Verifier 拒绝原因（关键词归类）**：

| 类别 | 数量 |
|---|---|
| 单段复述 | 207 |
| 其他 | 145 |
| 无依据因果 | 143 |
| 内容不被支持 | 85 |

**质量观察**：unit 平均 24.6 词、71.2% 跨 section、来源段落平均相隔约 10 段，说明生成器确实在做跨段落组合，而不是简单改写；但仍有约 18% 的 unit 来自相邻段落，且验证器拒绝了大量单段复述与虚假因果。

---

## 6. 基础检索基线（全 100 篇 / 333 题）

| 方法 | n | Any@1 | Any@3 | Any@5 | All@5 | Recall@5 | MRR | nDCG@10 | Cov@4096 |
|---|---|---|---|---|---|---|---|---|---|
| M0 BM25 | 333 | 0.204 | 0.362 | 0.521 | 0.378 | 0.435 | 0.341 | 0.369 | 0.716 |
| M0 Dense | 333 | 0.363 | 0.628 | 0.740 | 0.593 | 0.658 | 0.530 | 0.563 | 0.889 |
| M0 Hybrid | 333 | 0.312 | 0.566 | 0.680 | 0.517 | 0.582 | 0.476 | 0.501 | 0.841 |

**按子集（Dense）**：

| subset | n | Any@5 | All@5 | Recall@5 | Cov@4096 |
|---|---|---|---|---|---|
| all | 333 | 0.740 | 0.593 | 0.658 | 0.889 |
| single | 219 | 0.733 | 0.733 | 0.733 | 0.970 |
| multi | 141 | 0.741 | 0.394 | 0.547 | 0.773 |
| cross | 56 | 0.750 | 0.295 | 0.497 | 0.786 |
| multi_annot | 98 | 0.709 | 0.444 | 0.563 | 0.827 |

解读：BM25 远弱于 dense；等权 hybrid 被 BM25 稀释；**dense 是本轮最强检索器**，单证据接近饱和（single All@5 73.3%），多证据（39.4%）与跨章节（29.5%）仍有明显空间。

---

## 7. 主检索对照 M0–M6（333 题）

| 方法 | n | Any@1 | Any@3 | Any@5 | All@1 | All@3 | All@5 | Recall@5 | MRR | nDCG@10 | Cov@4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 Dense | 333 | 0.363 | 0.628 | 0.740 | 0.251 | 0.470 | 0.593 | 0.658 | 0.530 | 0.563 | 0.889 |
| M1 G_raw | 333 | 0.281 | 0.553 | 0.692 | 0.204 | 0.443 | 0.544 | 0.606 | 0.461 | 0.517 | 0.883 |
| M2 +Atomic | 333 | 0.300 | 0.571 | 0.697 | 0.213 | 0.441 | 0.545 | 0.608 | 0.477 | 0.527 | 0.884 |
| M3 +Triples | 333 | 0.321 | 0.580 | 0.707 | 0.231 | 0.452 | 0.556 | 0.620 | 0.493 | 0.540 | 0.886 |
| M4 +Units | 333 | 0.210 | 0.441 | 0.545 | 0.138 | 0.336 | 0.401 | 0.461 | 0.375 | 0.414 | 0.842 |
| M5 Triples+Units | 333 | 0.216 | 0.440 | 0.535 | 0.146 | 0.333 | 0.395 | 0.454 | 0.375 | 0.413 | 0.836 |
| M4 gated units | 333 | 0.221 | 0.429 | 0.565 | 0.155 | 0.324 | 0.411 | 0.475 | 0.389 | 0.456 | 0.884 |
| M5 gated units | 333 | 0.222 | 0.425 | 0.565 | 0.159 | 0.318 | 0.413 | 0.476 | 0.390 | 0.454 | 0.886 |
| M6 +Summaries | 333 | 0.359 | 0.620 | 0.709 | 0.257 | 0.482 | 0.575 | 0.635 | 0.521 | 0.555 | 0.884 |
| M6 matched count | 333 | 0.186 | 0.335 | 0.518 | 0.117 | 0.216 | 0.360 | 0.425 | 0.337 | 0.395 | 0.853 |
| M4 units only | 333 | 0.128 | 0.323 | 0.399 | 0.081 | 0.237 | 0.291 | 0.339 | 0.250 | 0.281 | 0.405 |

### 7.1 成对差值（question-level paired bootstrap，95% CI）

| 对比 | ΔAny@5 | 95% CI | ΔAll@5 | 95% CI | ΔRecall@5 |
|---|---|---|---|---|---|
| m3_atom_comp vs m2_atom | +0.011 | [-0.003,+0.026] | +0.011 | [-0.002,+0.024] | +0.012 |
| m4_units vs m2_atom | -0.152 | [-0.209,-0.092] | -0.144 | [-0.197,-0.090] | -0.148 |
| m5_units_graph vs m3_atom_comp | -0.173 | [-0.227,-0.116] | -0.161 | [-0.213,-0.107] | -0.166 |
| m6_matched vs m2_atom | -0.179 | [-0.228,-0.129] | -0.185 | [-0.233,-0.135] | -0.183 |
| m5_units_graph vs m2_atom | -0.162 | [-0.218,-0.102] | -0.150 | [-0.203,-0.096] | -0.155 |

- **M3−M2**：Any@5 +1.1pp、All@5 +1.1pp，CI 下界接近 0，属于“中性偏正”，不能说显著。
- **M2−M1**：原子事实层几乎中性（All@5 +0.1pp），说明普通事实扩充本身不改变检索上限。
- **M4/M5−M2**：All@5 −14.4pp / −15.0pp，显著负迁移。
- **gated 融合**只能把损失从约 −15pp 收窄到 −13pp，无法恢复。

### 7.2 子集：单/多证据、跨章节、unit 覆盖

| subset | 方法 | n | Any@5 | All@5 | Recall@5 |
|---|---|---|---|---|---|
| 单段落 | M1 G_raw | 219 | 0.687 | 0.687 | 0.687 |
| 单段落 | M2 +Atomic | 219 | 0.674 | 0.674 | 0.674 |
| 单段落 | M3 +Triples | 219 | 0.689 | 0.689 | 0.689 |
| 单段落 | M4 +Units | 219 | 0.530 | 0.530 | 0.530 |
| 单段落 | M5 Triples+Units | 219 | 0.523 | 0.523 | 0.523 |
| 单段落 | M5 gated units | 219 | 0.537 | 0.537 | 0.537 |
| 单段落 | M6 matched count | 219 | 0.468 | 0.468 | 0.468 |
| 多证据 | M1 G_raw | 141 | 0.691 | 0.340 | 0.488 |
| 多证据 | M2 +Atomic | 141 | 0.716 | 0.358 | 0.508 |
| 多证据 | M3 +Triples | 141 | 0.720 | 0.362 | 0.514 |
| 多证据 | M4 +Units | 141 | 0.557 | 0.216 | 0.358 |
| 多证据 | M5 Triples+Units | 141 | 0.539 | 0.209 | 0.348 |
| 多证据 | M5 gated units | 141 | 0.589 | 0.230 | 0.379 |
| 多证据 | M6 matched count | 141 | 0.571 | 0.199 | 0.352 |
| 跨章节 | M1 G_raw | 56 | 0.679 | 0.250 | 0.436 |
| 跨章节 | M2 +Atomic | 56 | 0.741 | 0.312 | 0.499 |
| 跨章节 | M3 +Triples | 56 | 0.741 | 0.312 | 0.499 |
| 跨章节 | M4 +Units | 56 | 0.571 | 0.161 | 0.333 |
| 跨章节 | M5 Triples+Units | 56 | 0.554 | 0.152 | 0.320 |
| 跨章节 | M5 gated units | 56 | 0.580 | 0.188 | 0.354 |
| 跨章节 | M6 matched count | 56 | 0.554 | 0.161 | 0.324 |
| unit 覆盖 gold | M1 G_raw | 193 | 0.741 | 0.562 | 0.639 |
| unit 覆盖 gold | M2 +Atomic | 193 | 0.746 | 0.567 | 0.640 |
| unit 覆盖 gold | M3 +Triples | 193 | 0.759 | 0.585 | 0.659 |
| unit 覆盖 gold | M4 +Units | 193 | 0.837 | 0.627 | 0.716 |
| unit 覆盖 gold | M5 Triples+Units | 193 | 0.829 | 0.619 | 0.709 |
| unit 覆盖 gold | M5 gated units | 193 | 0.798 | 0.583 | 0.675 |
| unit 覆盖 gold | M6 matched count | 193 | 0.492 | 0.324 | 0.396 |
| unit 未覆盖 gold | M1 G_raw | 140 | 0.625 | 0.518 | 0.560 |
| unit 未覆盖 gold | M2 +Atomic | 140 | 0.629 | 0.514 | 0.564 |
| unit 未覆盖 gold | M3 +Triples | 140 | 0.636 | 0.514 | 0.566 |
| unit 未覆盖 gold | M4 +Units | 140 | 0.143 | 0.089 | 0.109 |
| unit 未覆盖 gold | M5 Triples+Units | 140 | 0.129 | 0.086 | 0.102 |
| unit 未覆盖 gold | M5 gated units | 140 | 0.243 | 0.179 | 0.201 |
| unit 未覆盖 gold | M6 matched count | 140 | 0.554 | 0.411 | 0.466 |

> `unit_covered`：该题 gold 至少被一个 verified unit 的来源段落触及（排序前的上界）；`unit_uncovered` 是其余题。覆盖子集上 M4/M5 明显优于 M2；未覆盖子集上崩塌——这是本轮最关键的机制证据。

### 7.3 RRF 权重敏感性（unit 路权重；graph 权重=1.0）

| base | unit weight | n | Any@5 | All@5 | Recall@5 |
|---|---|---|---|---|---|
| m2_atom | 0.1 | 333 | 0.691 | 0.529 | 0.596 |
| m3_atom_comp | 0.1 | 333 | 0.694 | 0.539 | 0.605 |
| m2_atom | 0.2 | 333 | 0.596 | 0.438 | 0.503 |
| m3_atom_comp | 0.2 | 333 | 0.590 | 0.432 | 0.496 |
| m2_atom | 0.3 | 333 | 0.559 | 0.413 | 0.472 |
| m3_atom_comp | 0.3 | 333 | 0.553 | 0.411 | 0.470 |
| m2_atom | 0.5 | 333 | 0.557 | 0.411 | 0.471 |
| m3_atom_comp | 0.5 | 333 | 0.545 | 0.405 | 0.463 |
| m2_atom | 0.75 | 333 | 0.557 | 0.413 | 0.472 |
| m3_atom_comp | 0.75 | 333 | 0.544 | 0.404 | 0.462 |
| m2_atom | 1.0 | 333 | 0.545 | 0.401 | 0.461 |
| m3_atom_comp | 1.0 | 333 | 0.535 | 0.395 | 0.454 |

即使 0.1 的固定权重也已低于 graph-only（M2 Any@5 .697）；0.2 起崩溃。固定权重无法恢复 unit 信号，必须做**相关性门控**，而不是调一个全局权重。

### 7.4 语义分组补充实验（20 papers / 72 题，探索性）

| grouping | papers | units | unit gold coverage | Any@5 | All@5 | Recall@5 |
|---|---|---|---|---|---|---|
| section | 20 | 164 | 0.722 | 0.694 | 0.521 | 0.596 |
| semantic | 20 | 144 | 0.694 | 0.639 | 0.465 | 0.531 |

Section-based 分组在该子集上更好；语义分组未做 QA，仅作为两种分组方式的可行性对照。

---

## 8. Reader QA（372 题，固定 4096-token 预算）

| 方法 | n | Answer F1 | Evidence F1 | EM | F1 answerable | F1 multi | F1 cross | Gold all in ctx |
|---|---|---|---|---|---|---|---|---|
| M0 Dense | 372 | 0.464 | 0.119 | 0.210 | 0.436 | 0.428 | 0.389 | 0.763 |
| M1 G_raw | 372 | 0.455 | 0.119 | 0.202 | 0.427 | 0.431 | 0.390 | 0.758 |
| M2 +Atomic | 372 | 0.457 | 0.119 | 0.199 | 0.429 | 0.426 | 0.402 | 0.761 |
| M3 +Triples | 372 | 0.454 | 0.119 | 0.204 | 0.425 | 0.422 | 0.400 | 0.763 |
| M4 +Units | 372 | 0.435 | 0.113 | 0.191 | 0.405 | 0.402 | 0.369 | 0.707 |
| M5 Triples+Units | 372 | 0.443 | 0.112 | 0.194 | 0.414 | 0.411 | 0.394 | 0.702 |
| M5 (units shown) | 372 | 0.443 | 0.113 | 0.191 | 0.414 | 0.402 | 0.372 | 0.702 |
| Full paper | 372 | 0.476 | 0.073 | 0.220 | 0.448 | 0.455 | 0.434 | 0.890 |
| Oracle gold | 372 | 0.487 | 0.955 | 0.242 | 0.455 | 0.455 | 0.447 | 0.895 |

**QA 成对差值**：

| 对比 | ΔAnswerF1 | 95% CI | ΔEvidenceF1 |
|---|---|---|---|
| m3_atom_comp vs m2_atom | -0.003 | [-0.015,+0.008] | +0.001 |
| m4_units vs m2_atom | -0.022 | [-0.039,-0.006] | -0.006 |
| m5_units_graph vs m3_atom_comp | -0.011 | [-0.028,+0.007] | -0.007 |
| m5_units_graph vs m2_atom | -0.014 | [-0.031,+0.002] | -0.006 |
| m5_units_graph_B vs m5_units_graph | -0.000 | [-0.018,+0.019] | +0.001 |

**按证据类型（Answer F1）**：

| 证据类型 | 方法 | n | Answer F1 |
|---|---|---|---|
| 文本证据 | M0 Dense | 339 | 0.438 |
| 文本证据 | M2 +Atomic | 339 | 0.431 |
| 文本证据 | M3 +Triples | 339 | 0.427 |
| 文本证据 | M4 +Units | 339 | 0.407 |
| 文本证据 | M5 Triples+Units | 339 | 0.415 |
| 文本证据 | Oracle gold | 339 | 0.458 |
| 仅图表证据 | M0 Dense | 11 | 0.369 |
| 仅图表证据 | M2 +Atomic | 11 | 0.369 |
| 仅图表证据 | M3 +Triples | 11 | 0.369 |
| 仅图表证据 | M4 +Units | 11 | 0.369 |
| 仅图表证据 | M5 Triples+Units | 11 | 0.369 |
| 仅图表证据 | Oracle gold | 11 | 0.364 |
| 不可回答 | M0 Dense | 22 | 0.909 |
| 不可回答 | M2 +Atomic | 22 | 0.909 |
| 不可回答 | M3 +Triples | 22 | 0.909 |
| 不可回答 | M4 +Units | 22 | 0.909 |
| 不可回答 | M5 Triples+Units | 22 | 0.909 |
| 不可回答 | Oracle gold | 22 | 1.000 |

**unit 覆盖 vs 未覆盖题的 QA**：

| subset | 方法 | n | Answer F1 |
|---|---|---|---|
| covered | m2_atom | 193 | 0.461 |
| covered | m3_atom_comp | 193 | 0.467 |
| covered | m4_units | 193 | 0.440 |
| covered | m5_units_graph | 193 | 0.448 |
| covered | m5_units_graph_B | 193 | 0.454 |
| uncovered | m2_atom | 140 | 0.400 |
| uncovered | m3_atom_comp | 140 | 0.384 |
| uncovered | m4_units | 140 | 0.371 |
| uncovered | m5_units_graph | 140 | 0.381 |
| uncovered | m5_units_graph_B | 140 | 0.372 |

**Top-3 / Top-5 Reader 敏感性（同一批题，M5 排序，不做 token padding）**：

| k | n | Answer F1 | Evidence F1 | EM |
|---|---|---|---|---|
| 3 | 333 | 0.311 | 0.232 | 0.132 |
| 5 | 333 | 0.327 | 0.197 | 0.129 |

Top-k 下 Answer F1 明显低于 4096 预算；但 Evidence F1 反而更高，因为 4096 上下文预测了约 15–20 个段落，官方 paragraph-F1 的 precision 被拉低。**证据评估应以检索指标为准，QA Evidence F1 受上下文长度影响。**

---

## 9. 作用机制追踪

| 指标 | 值 |
|---|---|
| verified units | 733 |
| 从未成为检索候选 | 0 |
| 从未进 unit top-5 | 136 |
| 至少一次触及 gold | 229 |
| 触及 gold 但从未进 unit top-5 | 21 |
| 所在题 M5 QA 改善的 unit | 197 |
| 所在题 M5 QA 变差的 unit | 259 |
| 题目数 | 372 |
| 至少有一个 gold-touching unit 的题 | 193 |
| unit 层 gold 覆盖（有 gold 的检索题，193/333） | 58.0% |
| unit 层覆盖（全部 372 道 QA 题，193/372） | 51.9% |

机制结论：

- Unit **排序不是主要瓶颈**：733 个 unit 中 229 个至少一次触及 gold，只有 21 个“触及 gold 但从未进 unit top-5”。
- **覆盖是主瓶颈**：只有 58% 的题的 gold 被任一 unit 来源触及；剩余 42% 的题无法从 unit 层受益。
- **位移是第二瓶颈**：等权 RRF 在未覆盖题上仍把 unit 来源段落塞进候选，挤掉正确图证据，使 M4/M5 在未覆盖子集上从 .629/.514 掉到 .143/.089。
- **Reader 不领情**：即使在覆盖子集，M5 的 QA F1 也没有超过 M2；mode B 把 unit 文本给 Reader 同样无增益。

---

## 10. 案例

### 10.1 正例：unit 覆盖 gold 时提升证据召回

#### 50be4a737dc0951b35d139f51075011095d77f2a — What background knowledge do they leverage?

- gold passages: 1503.00841::s01::p000, 1503.00841::s06::p000, 1503.00841::s06::p002
- All@5: M2=0.000 → M3=0.000 → M4=0.500 → M5=0.500
- units in top-5:
  - `1503.00841::u000` [GOLD] The bias problem in prior knowledge motivates three auxiliary regularization terms in the Generalized Expectation Criteria framework to make GE-FL more robust. (src: 1503.00841::s00::p002, 1503.00841::s00::p003, 1503.00841::s00::p005, 1503.00841::s01::p000)
  - `1503.00841::u010` [not-gold] Unlike the maximum entropy regularization term, KL divergence handles unbalanced corpora better by using the reference class distribution instead of assuming uniformly distributed categories. (src: 1503.00841::s13::p003, 1503.00841::s13::p004)
  - `1503.00841::u009` [not-gold] The maximum entropy regularization term controls unbalance without extra knowledge, but its uniform-category assumption creates degraded performance when violated. (src: 1503.00841::s13::p002, 1503.00841::s13::p003)
  - `1503.00841::u002` [not-gold] With unbalanced labeled features on balanced and unbalanced movie datasets, incorporating the KL divergence of class distribution performs much better than GE-FL. (src: 1503.00841::s07::p001, 1503.00841::s07::p002, 1503.00841::s07::p005)
  - `1503.00841::u007` [not-gold] The top 10 frequent words after removing stop words are used as neutral features in all experiments, implementing the proposed approach without manual annotation. (src: 1503.00841::s00::p004, 1503.00841::s04::p004, 1503.00841::s06::p004)

#### 02428a8fec9788f6dc3a86b5d5f3aa679935678d — How do they define robustness of a model?

- gold passages: 1503.00841::s00::p002, 1503.00841::s00::p003, 1503.00841::s04::p000, 1503.00841::s07::p001, 1503.00841::s09::p000, 1503.00841::s09::p002
- All@5: M2=0.000 → M3=0.000 → M4=0.500 → M5=0.500
- units in top-5:
  - `1503.00841::u000` [GOLD] The bias problem in prior knowledge motivates three auxiliary regularization terms in the Generalized Expectation Criteria framework to make GE-FL more robust. (src: 1503.00841::s00::p002, 1503.00841::s00::p003, 1503.00841::s00::p005, 1503.00841::s01::p000)
  - `1503.00841::u006` [not-gold] LDA selects predictive labeled features without considering class balance, and GE-FL does not control this issue, so auxiliary regularization terms promote the model significantly. (src: 1503.00841::s11::p001, 1503.00841::s11::p002)
  - `1503.00841::u009` [not-gold] The maximum entropy regularization term controls unbalance without extra knowledge, but its uniform-category assumption creates degraded performance when violated. (src: 1503.00841::s13::p002, 1503.00841::s13::p003)
  - `1503.00841::u007` [not-gold] The top 10 frequent words after removing stop words are used as neutral features in all experiments, implementing the proposed approach without manual annotation. (src: 1503.00841::s00::p004, 1503.00841::s04::p004, 1503.00841::s06::p004)
  - `1503.00841::u002` [GOLD] With unbalanced labeled features on balanced and unbalanced movie datasets, incorporating the KL divergence of class distribution performs much better than GE-FL. (src: 1503.00841::s07::p001, 1503.00841::s07::p002, 1503.00841::s07::p005)

### 10.2 负例：unit/关系挤掉正确证据

#### 6bf5620f295b5243230bc97b340fae6e92304595 — What baseline model is used?

- gold passages: 1603.01514::s06::p000
- All@5: M2=1.000 → M3=1.000 → M4=0.000 → M5=0.000; Any@5: M2=1.000 → M5=0.000
  - `1603.01514::u011` [not-gold] The paper uses garg2012unsupervised's Bayesian model as the monolingual basis regularized by crosslingual latent variables in its joint Bayesian model of multilingual SRI. (src: 1603.01514::s12::p001, 1603.01514::s13::p000)
  - `1603.01514::u000` [not-gold] The proposed joint Bayesian model implements the expected benefit of word alignment information in parallel text by using crosslingual latent variables to incorporate soft role agreement between aligned constituents. (src: 1603.01514::s00::p003, 1603.01514::s00::p006)
  - `1603.01514::u009` [not-gold] The multilingual model is deficient because aligned roles are generated twice, an approximation used because an additional conditioning variable breaks Dirichlet-multinomial conjugacy. (src: 1603.01514::s03::p000, 1603.01514::s03::p002)
  - `1603.01514::u010` [not-gold] The paper extends the superlingual latent variables idea proposed for POS tagging to SRI through crosslingual latent variables that capture role alignments in parallel corpora. (src: 1603.01514::s12::p000, 1603.01514::s13::p000)
  - `1603.01514::u005` [not-gold] The small percentage of aligned roles probably limits the impact of cross-lingual information in the labeled-source transfer setting, consistent with only 8% of English arguments and 17% of German arguments being aligned (src: 1603.01514::s10::p000, 1603.01514::s09::p005)

#### 747b847d687f703cc20a87877c5b138f26ff137d — Which parallel corpora are used?

- gold passages: 1603.01514::s08::p000
- All@5: M2=1.000 → M3=1.000 → M4=0.000 → M5=0.000; Any@5: M2=1.000 → M5=0.000
  - `1603.01514::u010` [not-gold] The paper extends the superlingual latent variables idea proposed for POS tagging to SRI through crosslingual latent variables that capture role alignments in parallel corpora. (src: 1603.01514::s12::p000, 1603.01514::s13::p000)
  - `1603.01514::u000` [not-gold] The proposed joint Bayesian model implements the expected benefit of word alignment information in parallel text by using crosslingual latent variables to incorporate soft role agreement between aligned constituents. (src: 1603.01514::s00::p003, 1603.01514::s00::p006)
  - `1603.01514::u011` [not-gold] The paper uses garg2012unsupervised's Bayesian model as the monolingual basis regularized by crosslingual latent variables in its joint Bayesian model of multilingual SRI. (src: 1603.01514::s12::p001, 1603.01514::s13::p000)
  - `1603.01514::u009` [not-gold] The multilingual model is deficient because aligned roles are generated twice, an approximation used because an additional conditioning variable breaks Dirichlet-multinomial conjugacy. (src: 1603.01514::s03::p000, 1603.01514::s03::p002)
  - `1603.01514::u005` [not-gold] The small percentage of aligned roles probably limits the impact of cross-lingual information in the labeled-source transfer setting, consistent with only 8% of English arguments and 17% of German arguments being aligned (src: 1603.01514::s10::p000, 1603.01514::s09::p005)

### 10.3 过度推断被 verifier 拒绝

Overreach rejected by the verifier

- GE-FL is the baseline that leverages labeled features as prior knowledge, and the paper addresses the robustness problem on top of GE-FL.  
  reason: The concise_text is a restatement of s01:p000, and the triple's predicate 'requires' is not supported: GE-FL is the baseline/method on top of which the robustness problem is addressed, not something requiring that problem.
- The multilingual model uses word alignments between sentences in a parallel corpus to couple monolingual models, but only 8% of English arguments and 17% of German arguments are aligned.  
  reason: Passages state the model uses alignments and that only 8%/17% of arguments are aligned, but do not support that these percentages 'limit' the model's use of alignments as a causal relation; this is inferred.
- Because crosslingual parameters are ignored when trained parameters parse monolingual data, information coming from the CLVs acts as a regularizer for the monolingual models.  
  reason: This is essentially a restatement of p009: crosslingual parameters are ignored at parsing and thus CLV information acts as a regularizer. Reject as single-paragraph paraphrase.

---

## 11. 对六个研究问题的回答

**1. QASPER 是否真正解决了 LoCoMo 检索池太小的问题？**

部分解决。池从 LoCoMo 的 19 个 session 提升到 P50=43、P90=70 个段落，但仍是小池：top-5 平均覆盖 14% 的候选。Dense Any@5 74.0% 接近饱和，区分度来自 All-Evidence（59.3%）与多证据（39.4%）、跨章节（29.5%）子集。

**2. Read-Time 理解是否提高了 Gold Evidence 召回？**

Triple 表示（M3）整体中性偏正（All@5 +1.1pp，CI 跨 0），在覆盖子集上为正；Unit 独立索引与联合检索（M4/M5）整体显著负迁移。**答案取决于表示与覆盖**，不能简单说“理解有用”或“没用”。

**3. 哪种表示最有效：Triple、Unit，还是二者结合？**

Triple（M3）是唯一整体不伤检索的表示；Unit 在覆盖子集上有显著增益（Any@5 +9.1pp、All@5 +6.0pp），但整体因覆盖不足与位移而失败；二者结合（M5）继承 unit 的位移问题，没有叠加收益。

**4. 有多少收益独立于普通索引扩充？**

M6 全覆盖摘要 All@5 .575（接近 dense .593），而与 insight 同数量的 M6-matched 仅 .360。这说明“摘要层”的收益主要来自覆盖了所有段落，而不是理解；在固定知识数量下，M3 的 triple 略优于 M4/M6-matched，但没有大效应。

**5. 对跨段落、多证据问题是否有明显改善？**

多证据子集：M3 All@5 .362 vs M2 .358（+0.4pp，基本无改善）；跨章节：M3 .312 vs M2 .312（无改善）。Unit 方法在该子集显著变差（多证据 All@5 .216 vs .358）。**没有看到针对跨段落问题的稳定增益**——现有 7.3 个 unit/篇的覆盖太稀疏，且融合方式会挤掉原有证据。

**6. 下一阶段值得研究哪种 Comprehension 机制？**

优先级建议：

1. **覆盖扩张**：按 section/段落生成 unit，目标覆盖每个章节，而不是每篇 ~7 个；当前 58% 的题有 gold 覆盖，未覆盖题无法受益。
2. **相关性门控融合**：用校准分数决定是否启用 unit 通道，而不是固定 RRF 权重；固定权重 0.1 已损失，0.2+ 崩溃。
3. **unit → passage 的 reranker**：先检索 unit，再用 unit 对来源段落做重排，而不是把 unit 排名直接并入段落排名。
4. **Reader 侧**：不要把 unit 当证据文本塞进上下文（mode B 无增益）；unit 只用于路由到原文。
5. **Reader 上限**：Oracle 仅比 dense 高 2.3pp F1，下一阶段若继续做检索，需要同时换更强的 Reader，否则检索收益无法体现。

---

## 12. 局限与诚实说明

- 段落池小，Any-Evidence@5 有天花板；All-Evidence 与多证据子集才是主要信号。
- 图表证据（`FLOAT SELECTED`，253 条，3.5% 的题仅图表）不进入文本检索指标；QA 仍包含这些题。
- 跨段落 insight 用三元组注入每个来源段落，是对超边的近似，不是真正的超图。
- 语义分组只在 20 篇上做了探索性对照，没有 semantic QA。
- Top-3/Top-5 Reader 敏感性不是固定预算比较（没有 token padding）。
- HY4 是共享池，所有正式运行使用单桶 ~1 QPS 跨进程限流；生成完整 100 篇耗时数小时，延迟波动大。中途曾尝试多 key 并发被环境判定为过载共享池并已回退，报告结果均为单桶 1 QPS 产生。
- 所有结论基于 100 篇随机样本；扩大到完整 281 篇 validation 是机械性工作，pipeline 已支持。

---

## 13. 成本与运行

| 项目 | 值 |
|---|---|
| G_raw 建图累计秒 | 25,465 |
| M2/M3 variant 累计秒 | 1,600 |
| read-time prompt tokens | 2,883,843 |
| read-time completion tokens | 1,620,654 |
| Reader prompt tokens | 9,866,962 |
| Reader completion tokens | 127,279 |
| OpenIE token | 见每个 `graphs/raw/*/llm_cache`（官方缓存，未并入本表） |

所有 HY4 调用（adapter 侧）逐条记录在 `logs/hy4_calls*.jsonl`（状态、延迟、token、缓存命中；不含密钥）。

---

## 14. 复现命令

```bash
cd /group/40150/licochen/Memory/RSI_GraphRAG
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken
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

语义分组补充：`run_readtime.py --grouping semantic --papers-file <20篇列表>`，再跑 `semantic_probe.py`。

---

## 15. 产物索引（`outputs/qasper_readtime/`）

| 路径 | 内容 |
|---|---|
| `QASPER_READTIME_FULL_REPORT.md` | 本文件（单文件汇总） |
| `FINAL_REPORT.md` | 自动生成的正式结果报告 |
| `dataset_audit.md` | 数据解析、映射、池统计 |
| `baseline_retrieval.md` | BM25/Dense/Hybrid 全量基线 |
| `comprehension_quality.md` | 生成量、验证保留率、关系类型、来源距离 |
| `retrieval_comparison.csv` | M0–M6 检索指标（7 个子集 × 14 方法） |
| `qa_comparison.csv` | Reader QA 指标（9 方法） |
| `cases.md` | 正例/负例/无效果/过度推断案例 |
| `experiment_config.md` | 冻结配置与模型版本 |
| `source_provenance.jsonl` | 733 个 unit 的源段落/章节/摘录映射 |
| `dataset/sample.json` | 冻结 100 篇样本（seed 20261110，规则） |
| `dataset/papers/{paper}/paragraphs.jsonl` | 稳定 passage_id、section、原文 |
| `dataset/papers/{paper}/qa.jsonl` | 逐标注者答案、证据、映射结果 |
| `graphs/raw/{paper}/` | G_raw 与 `qasper_graph_info.json` |
| `graphs/raw/{paper}/variants/atom/` | M2 图 |
| `graphs/raw/{paper}/variants/atom_comp/` | M3 图 |
| `comprehension/section/{paper}/` | groups、atomic facts、raw/verified insights、units、summaries、逐段落三元组 |
| `comprehension/semantic/{paper}/` | 20 篇语义分组补充运行 |
| `retrieval/comparison_sample100.jsonl` | 逐题 14 方法排序与指标（4,662 行） |
| `retrieval/baselines_sample100.jsonl` | 全 100 篇 M0 逐题结果 |
| `retrieval/fusion_sweep_sample100.json` | 融合权重扫描 |
| `retrieval/grouping_comparison.json` | section vs semantic 对照 |
| `qa/predictions_sample100.jsonl` | 逐题 9 方法答案、上下文、评分（3,348 行） |
| `qa/topk_sensitivity_sample100.json` | Top-3/5 reader 敏感性 |
| `mechanism/unit_usage_sample100.jsonl` | 733 个 unit 的完整使用轨迹 |
| `mechanism/question_chains_sample100.jsonl` | 逐题检索→unit→QA 链路 |
| `logs/` | HY4 调用日志、各阶段运行日志 |
| `cache/` | 内容寻址响应缓存（可续跑） |

---

*生成时间：2026-11-10；所有数字对应最终 100-paper run。*