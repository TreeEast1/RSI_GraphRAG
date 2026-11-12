# 方法细节（METHOD）

> 本文把顶层 README §2 的 pipeline 拆到模块级别：每个模块怎么实现、有哪些硬约束、代码在哪、各数据集上有什么差异。
> 数字结果不在这里，在各数据集目录（[experiments/](../experiments/README.md)）。最完整的原始描述见 [主报告](RSI_GraphRAG_FULL_METHOD_AND_EXPERIMENTS.md) §4–§8。
> **文档与代码不一致时以代码为准**，已知的 17 处不一致见主报告 §11.3。

| # | 模块 | 阶段 |
|---|---|---|
| 1 | [统一环境与底座](#1-统一环境与底座) | — |
| 2 | [图变体注入](#2-图变体注入) | Read-Time |
| 3 | [阅读分组](#3-阅读分组) | Read-Time |
| 4 | [Comprehension（理解层）](#4-comprehension理解层) | Read-Time |
| 5 | [检索与 Question-Time](#5-检索与-question-time) | Question-Time |
| 6 | [RSI Skill Memory 闭环](#6-rsi-skill-memory-闭环) | RSI |
| 7 | [指标口径](#7-指标口径) | 评测 |
| 8 | [工程与可复现性](#8-工程与可复现性) | — |

---

## 1. 统一环境与底座

| 项 | 设定 |
|---|---|
| 图检索底座 | 官方 HippoRAG2（`third_party/HippoRAG` @ `d5c8329`）。`index()` / `retrieve()` / PPR **没有改动**，所有变化都通过修改输入（`openie_state.json`）或在外层复现中间态实现 |
| LLM | HY4（混元，`reasoning_effort=no_think`，`temperature=0`）。MuSiQue 阶段用过 gpt-5.6-luna 和本地 Qwen3.5-9B |
| Embedding | NV-Embed-v2（本地权重，4096 维）；文档 instruction 为空，查询 instruction 为 "Given a question, retrieve passages that answer the question" |
| Reader 预算 | 4096 token（cl100k 估算），按检索名次贪心装填 |
| G_raw | 官方 OpenIE（NER → 三元组）+ 同义边；建完后**冻结只读**，所有变体在副本上生成 |

---

## 2. 图变体注入

**目的**：把 Read-Time 产生的新三元组 / 关系加进 HippoRAG2 图，但不改官方代码，也不重新调用 OpenIE。代码在 `src/graph/variants.py`。

```mermaid
flowchart LR
  RAW["raw 工作目录"] -->|复制除 graph.pickle 外的一切<br/>llm_cache 用符号链接| VAR["variants/&lt;name&gt;"]
  NEW["新三元组 / 关系"] -->|compute_mdhash_id(text) 定位 chunk| OS["openie_state.json<br/>extracted_triples += …<br/>extracted_entities += …"]
  VAR --> OS --> IDX["open_hipporag(force_index_from_scratch=True)<br/>→ 官方 index(chunks)"]
  IDX --> GV[("G_variant")]
```

1. 新建 `{conv}/variants/{name}`，复制 raw 工作目录除 `graph.pickle` 外的全部文件（embedding parquet、`chunk_metadata.json`、`openie_state.json`、index manifest）。
2. `llm_cache` 用符号链接指向 raw，避免重复 OpenIE 调用。
3. 用 `compute_mdhash_id(text, "chunk-")` 按内容定位 chunk，把新三元组追加到该行的 `extracted_triples`，subject/object 补进 `extracted_entities`（去重）。
4. 重跑官方 `index()`。OpenIE state 已覆盖所有 chunk，所以**不会触发任何 LLM 抽取**，只给新实体和新 fact 算 embedding。
5. 在 `payload["read_time_injection"]` 写溯源，文件用 `.tmp` + `os.replace` 原子写入。

**挂载位置**：跨文档关系挂到它引用的文档上。DetectiveQA 需要防止提前泄漏，所以关系统一**挂到最晚的证据块**（`available_at`）。QASPER 的跨段 insight 用"注入每个来源段落"来近似超边。

---

## 3. 阅读分组

### 3.1 LoCoMo：9 种无监督方法（`src/grouping/`）

实体集合来自 G_raw 的官方 NER 输出，不额外调用 LLM。文档向量来自 NV-Embed-v2。

| 方法 | 实现 |
|---|---|
| `random`（5 seeds） | 打乱后按预算贪心打包，组内再按原顺序排 |
| `temporal` | 按 session 时间顺序连续打包 |
| `entity_jaccard` | 共享实体 Jaccard + 受限层次聚类 |
| `entity_idf` | `idf(e)=ln((n+1)/(df(e)+1))+1`；`sim = Σ_{e∈A∩B} idf(e) / (‖A‖_idf · ‖B‖_idf)` |
| `embedding` | L2 归一化后的 cosine |
| `hybrid_{0.25,0.5,0.75}` | `w·sim_entity + (1−w)·sim_emb` |
| `community` | doc–entity 二部图 + Louvain（resolution=1.0）；跳过 `idf<1.0` 的 hub 实体，边权 `1/sqrt(1+idf)` |

**打包规则**：贪心顺序装箱，`used + t > budget` 时封箱；单个超预算文档自成一组；受限层次聚类只合并 token 和不超预算的两簇；超预算的 Louvain 社区用 `split_oversized` 切分。预算为 4096 和 8192。

### 3.2 其它数据集

| 数据集 | 分组 | 代码 |
|---|---|---|
| MuSiQue | 实体二部图连通分量、Jaccard 硬组、top-k 邻居、向量 / 实体 / 融合的容量贪心、Leiden + 递归切分 + 1-hop overlap | `legacy/algernon_musique/{entity_doc_bipartite_audit, musique_zero_api_grouping, rsi_graphrag_v4}` |
| QASPER | **section 组**：按 `section_index` 顺序聚合；单 section >8192 时按段落切；`cur + sec > 8192` 封组，加入后 `≥4096` 封组；尾部 <400 token 并入上一组；单段落组不送理解层 | `src/qasper/grouping.py` |
| DetectiveQA | **实体时间分布包**：选反复出现但非 hub 的实体，在故事早 / 中 / 晚期各取若干 chunk，组成 4–6 chunk 的包；不看题目 / 选项 / clue | `src/detectiveqa/readtime.py` |

---

## 4. Comprehension（理解层）

### 4.1 各数据集共享的约束

| 约束 | 目的 |
|---|---|
| 至少引用 **2 个不同文档**（session / 段落 / chunk） | 保证是"跨文档"知识，否则丢弃（`needs_two_sessions` / `needs_two_chunks`） |
| 每个被引文档一句**逐字证据**（≤25–30 词） | 可验证；`quote_not_found` 即拒 |
| `derivation ∈ {Composed, Inferred}` | 区分"原文组合可得"和"需要常识推理" |
| 不得是已有原子事实或其同义改写 | 新颖性 |
| 禁止把共现 / 相关写成因果 | 防虚假因果 |
| 只进图 / 索引，不进 Reader prompt | 只能通过检索生效，没有文本泄漏 |

### 4.2 各数据集的 schema

| 数据集 | 输出 | 关系类型 | 代码 |
|---|---|---|---|
| LoCoMo（C3/C6） | `subject/predicate/object/relation_type/source_dia_ids/supporting_fact_ids/evidence/derivation/confidence` | causal、enabling_link、state_change、goal_outcome、event_dependency、identity_unification、temporal_ordering | `src/comprehension/` |
| LoCoMo COMP-P4（面向检索） | 同上，另加：subject/object ≤6 词并优先对齐已有实体；predicate 1–4 词下划线；禁止 `related_to`；temporal_ordering 仅当顺序本身是新信息时使用 | 同上 | `src/prompt_optimization/comprehension_prompts.py` |
| QASPER | insight：`concise_text`（≤40 词，自带语义）+ `triple` + `source_passage_ids≥2` + `supporting_fact_ids`；每组 ≤10 条。同一批 insight 同时生成 triple（M3）和 unit（M4）两种表示 | 7 类 | `src/qasper/readtime.py` |
| DetectiveQA | 关系 + `status ∈ {supported, claimed, suspected}`（区分事实、证词、怀疑）+ `available_at` | alias_or_identity、alibi_or_opportunity、motive_or_interest、method_or_capability、contradiction_or_deception、timeline_or_causal、state_change | `src/detectiveqa/readtime.py` |

### 4.3 校验（规则 + LLM verifier）

| 层 | LoCoMo | QASPER | DetectiveQA |
|---|---|---|---|
| 规则 | `needs_two_sessions` | id 存在、≥2 段落、fact id 合法 | `needs_two_chunks` / `quote_not_found` / `too_long` / `generic_predicate` / `duplicate_fact` |
| LLM verifier | C3v / COMP-P5：带被引 turn 原文逐条审 | 6 项：id 存在 / 联合支持 / 非单段复述 / 非虚假因果 / 无外来数字和实体 / triple 与文本一致 | `keep / fixed_status / semantic_support / cross_chunk_necessary / specificity` |
| 已知风险 | `verify_candidates` 失败时 fail-open | 无裁决视为保留（fail-open） | 人工审计 kept precision 66.7% |

MuSiQue 阶段的分层验证器 V1–V5（引文连续片段 → 专名绑定 → 日期代码比较 → 对照全部 OpenIE 的新颖性 → 语义蕴含）是上述校验的来源，见 [01_musique_wiki §2.2](../experiments/01_musique_wiki/README.md#22-理解怎么做v4--v5)。

### 4.4 知识的三种表示（QASPER 的对照设计）

| 表示 | 进入检索的方式 |
|---|---|
| Triple（M3） | 注入 HippoRAG 图，参与 fact 匹配和 PPR |
| Unit（M4） | `concise_text` 单独建 embedding 索引；取前 100 个 unit，把 `source_passage_ids` 投影回段落（第一个来源 ×1.0，其余 ×0.98，同段落取最高分），再和图排序做 RRF |
| Triple + Unit（M5） | 双路 RRF；gated 变体只用 top-3 unit，unit 路权重 0.5 |

M3 / M4 / M5 共享同一批通过验证的 insight，保证"知识数量固定，只变表示"。M6（单段落摘要，含同数量 matched 对照）用来区分"理解"和"普通索引扩充"。

---

## 5. 检索与 Question-Time

### 5.1 官方 HippoRAG2 `retrieve()` 的真实链条

```text
query embedding
  → get_fact_scores()：与全部 fact embedding 点积，取 linking_top_k = 5 个候选
  → rerank_facts()：每题一次 LLM 调用（DSPy filter prompt）
  → 若过滤后为空 → 整题退化成 dense_passage_retrieval（top-1 分数被 min-max 归一化为 1.0）
  → 否则以保留 fact 的 subject/object 为种子 + dense passage 种子 → PPR → top-k passage
```

推论：**新增关系必须先进入 top-5 fact 候选并通过 rerank，才可能影响 PPR。** `src/prompt_optimization/retrieval_diag.py` 在外层复现了完整中间态（`facts_before_rerank` / `facts_after_rerank` / `fallback_dense`）。

### 5.2 检索变体

| 变体 | 定义 | 使用 |
|---|---|---|
| dense | NV-Embed-v2 cosine | 所有数据集 |
| BM25 | `k1=1.5, b=0.75`，分词 `[A-Za-z0-9]+` | QASPER |
| graph（官方） | fact rerank + PPR | LoCoMo 阶段 A |
| graph no-rerank top5 | `use_llm_rerank=False`：直接用 embedding top-5 fact 做 PPR 种子，0 次 LLM 调用，不存在 dense 回退 | LoCoMo 阶段 B/C（`COMP-P4_top5`）、DetectiveQA 默认 |
| RRF | `score(d) = Σ_list 1/(60 + rank)`（LoCoMo 代码里 rank 从 0 开始并 +1，QASPER 从 1 开始，等价） | 子问题合并、unit/关系通道融合 |
| **prefix-safe PPR** | 一部小说一张图；每题删掉 cutoff 之后的 passage、只有未来来源的 fact 和只在未来出现的实体，边权只用可见来源重算，再在可见子图上跑 PPR。配套 6 项防泄漏检查 | DetectiveQA（`src/detectiveqa/graph.py`） |

### 5.3 Question-Time 模块（`src/question_skill/`）

| 模块 | 输入 → 输出 | 关键参数 |
|---|---|---|
| **QDM** | 问题 → 是否拆解 + ≤4 个子问题；每个子问题独立检索后 RRF | 解析失败时不拆（fail-closed） |
| **Retrieval Skill** | 问题 + top-5 证据摘要（每条 ≤280 字符）→ 缺失的关系链 + ≤2 个补充查询 | `max_new=2` |
| **Sufficiency gate**（阶段 C 加入） | 问题 + 检索结果 → `{missing: [...], sufficient: bool}`；仅当不充分时才调用技能 | 评测时从不使用 gold |
| **Answer Skill** | 问题 + 上下文 + 草稿 → `{answer, answerable, support}` | `answerable=False` 不强制改写拒答 |
| **Grounded Verifier**（R1–R5） | 问题 + 同一份 4096 上下文 + 候选答案 → `supported / missing_parts / date_error / wrong_target / over_enumeration / should_refuse / verdict ∈ {ok, revise, refuse}` | 修订至多 1 次 |

### 5.4 Reader

| Reader | 要点 | 代码 |
|---|---|---|
| LoCoMo P0 | 短 span + 严格拒答；`max_tokens=256` | `src/retrieval/reader.py` |
| **LoCoMo P7** | 短 span 或逗号列表 + 列举完整 + when 只给日期（相对时间锚定到 session 日期）+ how long 由起止日期计算 + 可跨 session 组合 + 拒答校准 + 禁止编造 | `src/prompt_optimization/reader_prompts.py` |
| QASPER | 支持 extractive / abstractive / yes-no / unanswerable | `src/qasper/reader.py` |
| DetectiveQA | 四选一，按故事顺序渲染上下文，输出 `{answer, evidence_ids}` | `src/detectiveqa/reader.py` |

**装填规则**：`select_by_budget(all_or_nothing=True)` 的含义是**跳过放不下的文档、继续装后面更短的**，所以上下文可能是排名的非连续子集。命名和语义相反。

---

## 6. RSI Skill Memory 闭环

代码：`outputs/question_evolution_conv30/qecode/`（`answer_evolve.py`、`retrieval_evolve.py`、`skills.py`、`s0_split.py` … `s7_reports.py`）。

### 6.1 数据隔离

- 在学习任何技能之前，用固定 seed 按类别分层三分：adaptation / development / evaluation。
- 两个进化循环只遍历 adaptation；快照选择只读 dev；evaluation 只在最终阶段读一次。
- adaptation 的学习顺序单独再打乱一次。

### 6.2 Answer Skill 单步状态机

```text
1  select_skills(active, question, cat, k=2)
2  answer()                        有技能 → 带技能作答；无 → 复用冻结 P7 缓存
3  判分 + answer_support
4  record_marginal()               只在 E0错→本轮对 / E0对→本轮错 时计分
5  wrong = em==0 or (cat!=5 and f1<0.5)
6  反思：error_axis / error_type（11 选 1）/ diagnosis / generalizable_pattern / candidate_skill
7  硬门：gold evidence 不在上下文 → 丢弃候选（"没检出来"不产生 answer skill）
8  Sanitizer
9  verify_candidate：4 个 gate
10 被拒 → 用 previous + failures 自修复一次
11 写 memory：命中 update_skill_id → 拼接 action、version+1、confidence=max(0.5, c*0.9)；否则 add（confidence=0.55）
12 每 10 步或末步写快照 + 跑 dev；dev F1 < best−0.05 → 从 best 快照回滚
13 收尾：merge_duplicates(threshold=0.90)
```

**error_type**：`evidence_not_retrieved / evidence_not_used / temporal_arithmetic / temporal_granularity / multi_fact_omission / over_verbose / format_mismatch / wrong_refusal / unsupported_inference / wrong_target_or_premise / other`。

### 6.3 四个验证 gate

| gate | 通过条件 |
|---|---|
| `fix_pass` | 源题 F1 ≥ baseline + 0.10，或候选 EM=1 且基线 EM=0 |
| `similar_pass` | ≤3 个同类相似题（embedding cosine）中没有"基线对 → 候选错" |
| `dissimilar_pass` | 其它 3 个可答类别各取 1 题，不得 EM 1→0 |
| `adversarial_pass` | ≤2 个 cat-5 题，不得"基线拒答 → 候选不拒答" |

### 6.4 Sanitizer（防泄漏）

- **Answer 技能**：从 `question + gold + adversarial_answer` 抽 token，扣掉约 200 个通用词白名单。命中以下任一情况即拒绝：
  - 长度 >2 的纯数字。
  - 专有名词（`jon` / `gina` 豁免）。
  - 长度 >4 的实词。
  - 显式年份 `(19|20)\d{2}`。
  - gold 整串作为子串出现。
- **Retrieval 技能**（更宽松）：
  - 只禁大写开头词、年份、≥3 位数字、gold 整串。
  - 模板必须含 `{question}` 或 `{entities}` 占位符。
  - category 必须包含源题类别。

### 6.5 技能检索与信誉

- 打分：`2.0`（类别匹配）/ `0.8` + `1.2·pattern_overlap` + `0.6·Jaccard` + `1.5·max(0, cos)` + `0.3·confidence`，`score ≥ 1.0` 才入选。
- `failure ≥ 3` 且 `failure ≥ success + 2` 时自动 disable。
- 渲染默认只给 action；`QE_INCLUDE_TRIGGER=1` 时同时给出触发条件。

### 6.6 Retrieval Skill 进化

- base 排序始终是冻结的 `COMP-P4_top5`，技能只追加查询（每个技能最多取 3 条模板）。
- pipeline 模式为 `base | qdm | qdm_suff | full`。
- 需要学习的条件：可答题，且 evidence_full≠1、R@5≠1 或多证据不完整。
- 验证只做本地 dense replay，不调 LLM：
  - `fix_pass`（三选一）：evidence_full 0→1 / partial +0.25 / R@5 +0.5。
  - 负迁移检查：在 3 道最近的同类题上。

---

## 7. 指标口径

### 7.1 检索

| 指标 | 定义 | 数据集 |
|---|---|---|
| Recall@k | top-k 与 gold 的交集大小 ÷ gold 数 | 全部 |
| MRR | 首个命中的 1-based 名次倒数，没命中记 0 | 全部 |
| nDCG@10 | 二值增益，IDCG 按 `min(n_gold, 10)` | 全部 |
| **evidence_full@B** | turn 级：所有 gold turn 所在的 session 都装进了 B token 上下文 | LoCoMo 主证据指标 |
| Any@k / All@k | QASPER：**逐标注者**算"至少一条 / 全部 gold 在 top-k"，再平均 | QASPER |
| Cov@4096 | 同 evidence_full@4096，逐标注者算 | QASPER |
| All@20 | 全部 gold chunk 都在 top-20 | DetectiveQA |
| fallback_dense | rerank 后无 fact 存活，整题退化成 dense | LoCoMo 机制诊断 |
| 新关系进入 top-5 fact | 新关系文本与每题最终 top-5 fact 规范化匹配 | H9 漏斗 |

### 7.2 QA

| 口径 | 定义 |
|---|---|
| **项目 EM/F1**（SQuAD 风格） | 小写 → 删标点 → 删 a/an/the → 空白归一；EM 字符串相等；F1 token 多重集，多 gold 取 max；**只统计 category 1–4**，cat-5 单独算拒答率 |
| 官方 LoCoMo F1 | 先删逗号，再删 a/an/the/and；Porter 词干；cat-1 按逗号拆子答案平均；cat-3 只取 `;` 前部分。通常比项目 F1 高 2.9–4.1pp（`src/prompt_optimization/official_metrics.py`） |
| `metrics.csv` 里的 em/f1 | 全题口径（cat-5 按拒答率计入），**不等于**报告里的 EM（可答） |
| QASPER Answer F1 | 每个标注者算 token-F1 取 max，再按题平均 |
| QASPER Evidence F1 | 预测段落原文集合 vs 原始 evidence 字符串集合，严格字符串匹配 |
| DetectiveQA Acc | 四选一准确率；两个选项顺序 seed 分别报告 |

### 7.3 统计

| 数据集 | 方法 |
|---|---|
| QASPER | 单方法 CI 做 2,000 次非配对 bootstrap；方法间差值按题配对后 bootstrap 2,000 次 |
| DetectiveQA | 按小说聚类 bootstrap（5,000 次）+ McNemar |
| LoCoMo | 重复运行估噪声：同 prompt、同 context、T=0 下，P0 三次 EM 在 .214–.357 之间波动 |

---

## 8. 工程与可复现性

| 项 | 实现 |
|---|---|
| 限流 | 跨进程文件锁令牌桶（HY4 约 1 QPS 共享配额），429 指数退避（`src/adapters/ratelimit.py`、`hy4_client.py`）。429/失败/空回答从不计作拒答 |
| 缓存 | 内容寻址 sqlite 缓存（LLM 和 embedding）；HippoRAG 官方调用走各自的 `llm_cache/*.sqlite` |
| 原子写 | `.tmp` + `os.replace`；断点续跑写 `*.partial.jsonl`（不入库） |
| 身份校验 | index manifest、数据 sha256、`run_manifest.json` |
| 防泄漏 | LoCoMo 人工摘要字段不进索引；DetectiveQA 在 cutoff 处强制切块 + prefix-safe 检索；技能 sanitizer；划分先于学习 |
| 测试 | `tests/`（core / detectiveqa / round2），`python -m unittest discover -s tests -v` |
