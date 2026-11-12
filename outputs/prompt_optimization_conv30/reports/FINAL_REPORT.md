# LoCoMo conv-30：Read-Time Comprehension Prompt 与 Reader/QA Prompt 优化实验报告

> 范围：仅 LoCoMo `conv-30`（19 sessions / 105 questions / 81 answerable / 24 adversarial）。
> 基座：官方 HippoRAG2（`third_party/HippoRAG` @ d5c8329）、HY4（`reasoning_effort=no_think`、temperature=0）、NV-Embed-v2。
> 所有原始产物（G_raw / C2 / C3 / C5 / C6 图谱与 `outputs/qa/`）**只读**；本轮所有新实验写入
> `outputs/prompt_optimization_conv30/`，图谱写入 `outputs/raw_graphs/conv-30/variants/cmpo_*`，未覆盖任何历史目录。
> 日期：2026-11-09。

---

## 0. 执行摘要

1. **现有 Reader Prompt 的核心问题不是 `max_tokens`，也不是缺少 Session 日期。** 105 题中
   completion tokens 最大 35（`max_tokens=256` 从未触发截断），Session 日期、说话人、Session 编号都在上下文中。
   真正的问题是「短答案 + 保守拒答」的组合：81 道可回答题中有 **15 题被拒答**，其中 **13 题的金标证据已经完整进入 4096 token 上下文**。
2. **评测口径确实有差异，但不是低分主因。** 用官方 `task_eval/evaluation.py` 重算（不重跑 LLM）：
   G_raw 项目 F1 39.0% → 官方 F1 42.6%（+3.6pp），差异主要来自 Porter 词干化与逗号/多跳分项；
   官方 EM（集合相等）与项目 EM 恰好都约 21%。**官方 cat-5 协议影响很大**：按官方「(a) Not mentioned / (b) 诱导答案」多选协议，官方 prompt 是 21/24=87.5%，我们的 P0 prompt 是 20/24=83.3%，P7 是 21/24=87.5%；若直接用官方关键词（"not mentioned"）判我们输出的 "No answer"，会得到 0%——这是协议/措辞错配，不是模型失败。
3. **错误归因（G_raw，81 道可回答题）：** 正确 17，错误 64。主因分布：
   表达/措辞 **23**、过度拒答 **13**、检索缺失 **11**、时间推理 **9**、多事实整合 **3**、证据不完整 **1**、错误推断 **4**；
   另有 **16** 题带「评测器/语义等价」次要标签。**没有一题因长度截断出错。**
4. **Oracle 诊断给出了最关键的结论：** 把 Gold Evidence 所在的完整 Session 直接给 Reader，
   **原始 Prompt 的 F1 仍是 38.7%（EM 18.5%），与正常检索的 39.0%/21.0% 基本不变**；
   而优化后的 P7 在同样 Oracle 上下文上达到 **F1 60.3% / EM 27.2%**。
   说明：**在 conv-30 上，Reader Prompt/推理方式才是主要瓶颈；检索不是原始方法的唯一瓶颈，但优化 Prompt 后检索仍有约 4–8pp 的可兑现空间。**
5. **Reader Prompt：** dev 40 题上 P7（短答案 + 精确用词 + 时间规则 + 列表完整 + 拒答校准）显著最优
   （EM 29.4% vs P0 23.5%，F1 60.0% vs 39.4%，官方 F1 61.8% vs 42.3%）。
   val 65 题上 P7 的 EM 与 P0 持平（19.1%），F1 +7.8pp、官方 F1 +6.6pp；收益集中在 Temporal（EM 21.4%→42.9%、F1 32.9%→82.9%），
   代价是 Single-hop 变长导致 EM −9.1pp。全 105 题：**EM 21.0%→23.5%（+2.5pp），F1 39.0%→52.1%（+13.1pp），官方 F1 42.6%→54.6%**；
   对抗拒答 95.8%→91.7%（官方多选协议下 P0 83.3% / P7 87.5%，P7 不差于 P0）。
6. **输出波动是真实存在的：** 完全相同的 prompt/context/temperature=0 重复 3 次，
   P0 在 14 题代表集上的 EM 为 21.4% / 28.6% / 35.7%（词面不稳定率 14%，Jaccard 0.976）；
   P7 词面不稳定率 29%，但得分不稳定率 0%。**105 题上 1–2 题的差异不能当作方法收益。**
7. **Comprehension Prompt：** 原 Prompt 的缺陷被证实——26 条关系里 17 条（65%）是 `temporal_ordering`，
   10 条谓词 ≥5 词（平均 5.2，最长 12），只有 7/26（27%）的关系两端是图中已有实体。
   新 Prompt 把 temporal_ordering 压到 0–2 条、长谓词清零，但**检索收益仍很小**：
   最好的 COMP-P4（16 条关系）把 Evidence@4096 从 89.5% 提到 **90.5%（+1 题：q16）**、R@5 90.3%→91.3%、MRR +0.52pp，
   且**没有负迁移**（93 道原本成功的题零下降）。
   **但 q16 是单 session 时间题；11 道真正跨 session 的多证据题 Evidence 完全没变（.455→.455）。**
   即：更好的 Comprehension Prompt 救回的是一道时间题，而不是它本来要解决的跨 session 多跳题。
8. **机制解释（本轮最重要的新发现）：** HippoRAG2 的 fact reranking 是一个**逐题 LLM 调用**：
   先按 embedding 取 top-5 fact 候选，再由 LLM 过滤；过滤为空则整题退化为 dense 检索。
   因此新增关系只有先进入「top-5 fact 候选」才可能影响 PPR。对 COMP-P4，新增关系在 **5/105 道题**的最终 top-5 fact 中真正出现；
   其余 100 题里，关系生成了、注入了、但没有参与检索。**关系生成本身不是瓶颈，fact 候选窗口和 LLM reranker 才是。**
9. **联合实验：** 见 §5。E0=G_raw+P0，E1=G_raw+P7，E2=COMP-P4+P0，E3=COMP-P4+P7。
   Prompt 的收益（E1−E0：EM +2.5pp / F1 +13.1pp）远大于 Comprehension 的收益（E2−E0：EM 0 / F1 −0.3pp，
   但 q16 由错变对）；两者基本可加：E3 相对 E0 为 EM +3.7pp / F1 +13.4pp，相对 E1 为 EM +1.2pp。
   P4 在 q16 上把证据覆盖从 0 提到 1，P0 和 P7 都能据此答对——两个因素解决的是不同故障。
10. **本轮是 conv-30 内部的探索性实验**，不是跨 Conversation 泛化证明；所有「提升」都应视为小样本、带噪声的估计。
11. **追加消融：关闭 LLM fact reranker（§4.7）。** 用 embedding top-5 直接作为 PPR 种子、完全不调用 LLM：
    R@5/Evidence 基本保持甚至略升（P4 Ev .9048→.9143，+2 题 q16/q57），但 MRR 下降 3–4pp（.764→.722），
    且所有 dense fallback 消失。最佳组合变为 **COMP-P4 + no-rerank top5 + P7：EM 25.9% / F1 53.9% / 官方 F1 56.9%**，
    相对正常 E3 为 +1.2pp EM / +1.5pp F1 / +3.0pp 官方 F1；但 EM 仍是 1 题量级。
    **把候选窗口开大（top20）会显著变差**（P4 R@5 .913→.806），不要用。

---

## 1. 原始 Prompt 审计

### 1.1 Reader Prompt（`src/retrieval/reader.py`）

实际发送给 HY4 的完整内容如下（system + user；`{context}` 由 `format_context` 填充）。

**System**

```text
You are a careful reading-comprehension assistant. You answer questions about a long conversation between two speakers using only the retrieved context. Use the dates and session numbers to reason about time when needed; resolve relative expressions such as 'yesterday' or 'last week' against the session date. If the context truly does not contain the answer, reply exactly: No answer
```

**User template**

```text
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

调用参数：`max_completion_tokens=256`、`temperature=0.0`、`reasoning_effort=no_think`。
`_parse_answer` 用正则 `Answer:\s*(.*)` 取第一行；找不到时退化为原文第一行。

**真实渲染样例**（q16，G_raw，4096 token 预算）已保存到
`outputs/prompt_optimization_conv30/audit/reader_prompt_q016.txt`。关键结构：

- 每篇 session 的 header 是 `[Session 5 | 9:32 am on 8 February, 2023]`，**出现两次**：
  `format_context` 先加一次 header，`Session.text` 自身又带一次。多花约 12 token/篇，不影响语义。
- 6 篇 session 被选中，共 4085 token；正文保留 `Gina: ...` / `Jon: ...` 说话人；
  **turn id（D10:1）不进入 Reader 上下文**（只有 Session 编号和日期）。
- 文档按检索名次排列，**不是时间顺序**。
- `select_by_budget` 的 `all_or_nothing=True`：装不下的文档跳过、继续尝试后续文档，因此 context 可能是非连续的（例如 q16 选中的是 s02,s05,s09,s11,s15,s18）。
- 上下文里日期信息完整，系统提示也明确要求用 Session 日期解析相对时间。

**审计结论（有代码/产物依据）**

| 问题 | 结论 | 依据 |
|---|---|---|
| 是否强制短答案 | 是。"contain only the answer span itself" + "No explanation, no full sentence" | 105 题平均答案 4.5 词；completion 平均 8.1 token |
| 是否 `max_tokens=256` 截断 | **否** | 105 题 completion 最大 35，finish_reason 均为 stop |
| HY4 `no_think` 是否生效 | 是 | 每次调用参数含 `reasoning_effort=no_think`；usage 中 reasoning_tokens 全为 null |
| Session 日期是否进入上下文 | 是（且重复一次） | 见渲染样例与所有 context 文件 |
| 说话人 / Session ID / Turn ID | 说话人、Session ID 保留；**Turn ID 不保留** | `Session.text` 只渲染 speaker+text |
| 超预算时截断还是跳过整篇 | 跳过整篇、继续尝试后续文档 | `select_by_budget(all_or_nothing=True)` |
| 答案解析 | 取 `Answer:` 后第一行 | `_parse_answer` |
| 是否明确要求回答全部事实 | 否（恰恰相反：要求 span、no full sentence） | 模板 |
| 是否明确处理多事件列表/相对日期/时间跨度 | 只在 System 里一句"resolve relative expressions"；没有列表/跨度规则 | System + 模板 |
| 何时要求 No answer | "If the context truly does not contain the answer" | System + 模板 |

**待验证假设 vs 已证实问题：**「短答案要求导致多事实遗漏」（q4/q24 类）在本轮被部分证实（§3）；
但「短答案要求是 EM 低的主因」不成立——EM 的 23/64 错误属于表达/措辞（§2.3），
真正可被 Prompt 直接修复的是 13 题过度拒答与 9 题时间推理。

### 1.2 Comprehension Prompt（`src/comprehension/prompts.py`）

**System**

```text
You are a careful analyst building a cross-document knowledge graph from a long conversation. You only state relations that the cited dialogue turns actually support. You output only valid JSON.
```

**Instructions（原文，C3/C6 共用）**

```text
You are given several conversation sessions that were grouped together, and the atomic
facts already extracted from them. Your job is to find relations that the atomic facts do NOT
explicitly state, but which are jointly supported by facts from *different* sessions.

What counts as a useful new relation:
* Two facts about the same entity from different sessions that jointly establish a link
  (e.g. Alice starts a new job at Company X in S5, and meets Bob at Company X in S10
  -> "Alice" "colleague_of" "Bob" IS NOT allowed unless the text states they are colleagues).
* A change of state over time (moved_to city, then changed_job, then lives_in ...).
* Temporal ordering of events described in different sessions.
* Identity unification: two mentions across sessions that clearly refer to the same event or object.
* Causal or enabling links asserted or clearly implied by the dialogue text.

Hard requirements:
1. Every relation MUST cite at least two source_dia_ids from DIFFERENT sessions, or at least one
   dialogue id plus at least one supporting fact id from another session.
2. evidence must contain verbatim quotes (max 25 words each) from each cited session.
3. derivation must be "Composed" when combining explicit facts needs no world knowledge, and
   "Inferred" when common-sense or semantic inference is needed.
4. confidence is your calibrated probability (0-1) that the relation is true given the evidence.
5. Do NOT restate facts that are already in the atomic fact list. Do NOT produce synonyms or
   paraphrases of existing triples. Do NOT produce vague relations such as "related_to".
6. If you cannot find any cross-session relation that satisfies rule 1, return {"relations": []}.

Return JSON of exactly this shape:
{"relations": [
  {"subject": "Alice", "predicate": "moved_to", "object": "New York",
   "relation_type": "state_change", "source_dia_ids": ["D1:4", "D5:2"],
   "supporting_fact_ids": ["f12"],
   "evidence": ["I'm moving to New York next month", "Since I moved here I started at Acme"],
   "derivation": "Composed", "confidence": 0.9}
]}
```

**真实调用链**（`scripts/run_read_time_extract.py` → `src/comprehension/extract.py::run_scope` → `comprehend_group`）：

1. 读入 C2 的阅读分组（conv-30 为 2 组：15 篇 + 4 篇，8192 token 预算）。
2. Session 块用 `=== Session 10 | 24 April, 2023 ===` + `[D10:1] Jon: ...` 渲染（**turn id 在这里是保留的**）。
3. facts 块是 `f0: subject | predicate | object`，来自该组已抽取的 C2 三元组（去重后 92 / 139 条）。
4. 输出 JSON，`comprehend_group` 做后处理：非空字段、`source_dia_ids` 必须命中本组 turn、
   至少跨 2 个 session、不与 C2 三元组完全重复、组内去重。
5. `Triple` 记录保留 `source_sessions`、`relation_type`、`confidence`、`supporting_fact_ids`。
6. `src/graph/variants.py::inject_triples` 把三元组/关系按 `source_sessions` 挂到**对应 session 的 chunk** 的 OpenIE 行，
   再重跑官方 `index()`；关系的 fact 节点因此连到 subject/object 实体节点和该 chunk 的 passage 节点。

实际 prompt 已完整保存：`audit/comp_prompt_group0.txt`、`audit/comp_prompt_group1.txt`
（group0：10821 token，sha256 `485dee1c…`；group1：5910 token，sha256 `2aed6028…`）。

**审计结论：现有 Comprehension Prompt 的结构性缺陷（全部有数据支持）**

| # | 缺陷 | 证据 |
|---|---|---|
| 1 | 把 `temporal_ordering` 与其它类型并列，模型自然产出最多 | C3 26 条中 17 条（65%）是 temporal_ordering |
| 2 | 没有约束 subject/object 必须是短名词短语，也没有约束 predicate 必须是短动词 | 10/26 谓词 ≥5 词，平均 5.2，最长 12；C6 有 4 个 object >6 词 |
| 3 | 没有要求关系两端对齐图中已有实体 | 只有 7/26（C3）、1/18（C6）的关系两端都是 C2 已有实体 |
| 4 | 只按「是否跨 session」判合法性，不判「组合是否真的带来新知识」 | 26 条全部通过 `needs_two_sessions` 检查，但多数只是把两条事实改写成事件短语 |
| 5 | 没有区分「关系内容」与「是否可被未来问题检索到」 | 新关系在最终 top-5 fact 中只出现于 5–11/105 题（§4.5） |
| 6 | 关系表面对冲突/时间错误没有系统校验 | 例如 C3 的 `Jon --lost_job_before--> started dream business` 与 `lost_job_enabled` 两条语义近乎重复 |
| 7 | 已有的 verify prompt 默认不启用（C3 是 `comp` 而不是 `comp_ver`），且不含被引用的对话原文 | `verify_candidates` 只把候选的 subject/predicate/object/evidence 给模型，模型无法核对证据 |

### 1.3 检索路径审计：真正决定结果的是「LLM fact reranker」

官方 `HippoRAG.retrieve()` 的流程是：

1. `get_fact_scores(query)`：query embedding 与全部 fact embedding 点积，`linking_top_k=5`。
2. `rerank_facts()`：取 top-5 fact 候选，调用 **LLM（DSPy filter prompt）** 过滤，`len_after_rerank=5`。
3. 若过滤结果为空 → **整题退化为 `dense_passage_retrieval`**（top-1 分数被 min-max 归一化成 1.0）。
4. 否则用保留的 fact 的 subject/object 作为 PPR 种子，做 Personalized PageRank。

这意味着：

- 新增 Comprehension 关系只有先进入 **top-5 fact 候选**、并通过 LLM filter，才可能改变 PPR；
- 「有没有进 PPR」往往比「关系内容好不好」更决定结果（q16 在 C2 退化为 dense，在 C3/P4 重新进 PPR；q54 相反）；
- 每道题每次检索都会产生一次 LLM rerank 调用，检索本身也带非确定性。

本轮在 `src/prompt_optimization/retrieval_diag.py` 中复现了官方 retrieve 的完整中间状态
（`facts_before_rerank` / `facts_after_rerank` / `fallback_dense` / top facts），未修改官方代码。
验证：用 105 道题一次性批量检索，G_raw 的 ranked_docs 与历史记录逐题一致的比例为 93/105，
C2 的 R@3/R@5/MRR/Evidence 与历史 C2 **完全一致**（0.8413/0.9032/0.7586/0.8952）。
剩余差异来自 GPU/embedding 的批次级数值非确定性（同一题单独检索与批量检索结果会不同）。

---

## 2. QA 低分原因分析

### 2.1 原始指标（只读复算，未重跑 LLM）

| 方法 | R@3 | R@5 | Ev@4096 | EM(1–4) | F1(1–4) | 拒答率(cat5) |
|---|---:|---:|---:|---:|---:|---:|
| G_raw | 81.0% | 90.3% | 88.6% | 21.0% | 39.0% | 95.8% |
| C2 | 84.1% | 90.3% | 89.5% | 22.2% | 40.4% | 91.7% |
| C3 | 83.2% | 90.8% | 90.5% | 24.7% | 41.5% | 91.7% |
| C5 | 85.4% | 89.8% | 88.6% | 21.0% | 40.5% | 95.8% |
| C6 | 86.4% | 90.6% | 89.5% | 19.8% | 40.1% | 95.8% |

（与 `CONV30_READTIME_ANALYSIS.md` 完全一致；R@5/Ev 是全 105 题均值，EM/F1 是 81 道可回答题均值。）

### 2.2 评测口径差异（官方 `task_eval/evaluation.py`，无 LLM）

官方实现（2026-11-09 从 `snap-research/locomo` 拉取，逐行重写于
`src/prompt_optimization/official_metrics.py`）与项目实现的关键差异：

1. `normalize_answer` 先**删除逗号**（不补空格），再小写、去标点、去 `a/an/the/and`；
2. 官方 EM 是 **token 集合相等**（顺序无关），项目 EM 是字符串相等；
3. 官方 F1 对每个 token 做 **Porter 词干化**；
4. 官方 multi-hop（cat1）用 `f1()`：把预测和金标都按逗号切开，对每个金标子答案取与任一预测子答案的最大 F1，再平均；
5. cat3 只取金标分号前部分；
6. cat5 官方是二选一多选（"(a) Not mentioned in the conversation / (b) 诱导答案"），并按是否选择 "Not mentioned" 计分。

**重算结果（conv-30，预测不变）**

| 方法 | 项目 EM | 项目 F1 | 官方 F1(1–4) | 官方 EM(1–4) | 项目拒答率 | 官方严格拒答率 | 官方协议多选准确率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| G_raw (P0 prompt) | 21.0% | 39.0% | **42.6%** | 21.0% | 95.8% | 0.0% | **83.3% (20/24)** |
| C2 | 22.2% | 40.4% | 43.3% | 22.2% | 91.7% | 0.0% | — |
| C3 | 24.7% | 41.5% | **44.5%** | 24.7% | 91.7% | 0.0% | — |
| C5 | 21.0% | 40.5% | 44.2% | 21.0% | 95.8% | 0.0% | — |
| C6 | 19.8% | 40.1% | 43.7% | 19.8% | 95.8% | 0.0% | — |

分题型（官方 F1 / 项目 F1）：

| 方法 | cat1 multi-hop | cat2 temporal | cat4 single-hop |
|---|---:|---:|---:|
| G_raw | .487 / .489 | .292 / .292 | **.490 / .423** |
| C3 | .489 / .496 | .369 / .369 | .478 / .422 |
| C5 | .425 / .458 | .262 / .262 | **.552 / .476** |

**结论：**

- 官方口径把 F1 抬高约 **2.9–4.1pp**，主要来自 cat4 的词干化（`Hoodies→hoodie`、`dance→dancing`、`excitement→excited`）。
  它不是「低分」的主要解释，但会让同一系统的分数看起来不同。
- 官方 cat-5 的 0% 严格拒答率是**协议错配**：我们的 Prompt 输出 "No answer"，官方关键词只认 "no information available"/"not mentioned"。
  按官方真实多选协议重跑：官方 prompt 21/24（87.5%）、P0 prompt 20/24（83.3%）、P7 prompt 21/24（87.5%）。
- 项目 EM 与官方 EM 在本数据上几乎相同（21.0% vs 21.0%），说明**换 EM 口径不会救回这些题**。

**cat-5 三种协议口径（同一批 24 道对抗题，G_raw 4096 上下文）：**

| 输入 Prompt | 官方多选协议准确率 |
|---|---:|
| 官方 `QA_PROMPT_CAT_5` | 87.5% (21/24) |
| 项目 P0 | 83.3% (20/24) |
| 项目 P7 | 87.5% (21/24) |

P7 在官方协议下不差于 P0，也不差于官方 prompt；P0 略低是因为它的 `Answer:` 短 span 约束偶尔让模型只输出选项文本而不是 "(a)/(b)"（解析仍能处理 8 次 "not_mentioned" 的自由文本）。
**因此「P7 导致对抗拒答下降」只在项目关键词口径下成立（95.8%→91.7%），在官方协议下不成立。**

**至少 10 个真实差异案例**（官方 F1 − 项目 F1，节选；完整表在 `audit/official_metric_diff_cases.csv`）：

| qid | cat | gold | pred | 项目F1 | 官方F1 | Δ |
|---|---|---|---:|---:|---:|---:|
| q77 | 4 | Hoodies | hoodie | 0.00 | 1.00 | +1.00 |
| q2 | 4 | by dancing | dance | 0.00 | 0.67 | +0.67 |
| q76 | 4 | excitement | excited and positive | 0.00 | 0.67 | +0.67 |
| q61 | 4 | a trophy | one of my trophies | 0.00 | 0.40 | +0.40 |
| q55 | 4 | Freedom and expressing herself through dance | freedom - dancing without worrying what people think | 0.15 | 0.33 | +0.18 |
| q81 | 4 | Helping with making content and managing his social media accounts | help with making content or even managing your accounts | 0.53 | 0.67 | +0.14 |
| q53 | 4 | making them want to come back | Just creating an experience that'll make 'em wanna come back | 0.27 | 0.40 | +0.13 |
| q49 | 4 | cozy and comfortable | cozy and inviting | 0.67 | 0.50 | −0.17 |
| q25 | 1 | one-on-one mentoring and training to dancers, workshops and classes | dance classes and workshops, one-on-one mentoring and training | 0.55 | 0.47 | −0.07 |
| q23 | 1 | worked with an artist to make unique fashion pieces, made limited-edition… | launched an ad campaign, new offers and promotions, … | 0.35 | 0.28 | −0.07 |
| q4 | 4 | He lost his job and decided to start his own business to share his passion. | he's passionate about dancing and it'd be great to share it with others | 0.21 | 0.31 | +0.09 |

（q49/q25/q23 是官方口径反而更低的例子，说明它不是单纯「更宽松」。）

**列表顺序、连接词与日期格式的影响（单独说明，避免和「答错」混淆）：**

- q29：金标 `Paris, Rome`，预测 `Paris and Rome`。项目 EM=0（字符串不等，且项目 normalize 不去 "and"）；
  官方 EM=1（token 集合相等）；官方 cat1 F1=0.667（按逗号分项后 "and" 已去除，但每个金标子答案只匹配到预测的一部分）。
  也就是说，同一答案在三种口径下是 0 / 1 / 0.667，**列表顺序/连接词本身就是分数差异来源**。
- q14：金标 `April, 2023`，预测 `3 April, 2023`；项目 F1=0.8、官方 F1=0.8，但两者 EM 都是 0。
- q77：`Hoodies` vs `hoodie`；项目 F1=0，官方 F1=1.0（Porter 词干化）。
- 结论：**项目 EM 对「顺序、and、词形、日期粒度」都非常脆**；官方口径缓解了词形和顺序，但不改变知识层错误。
  这也是 §6.8 把「措辞差异」单列为 H，而不算作模型答错的原因。

### 2.3 逐题错误归因（G_raw，105 题，规则化 + 人工核对）

逐题文件：`audit/error_analysis_conv30.jsonl`；每题的**实际 Reader 上下文**：`audit/graw_contexts/qNNN.txt`。

| 指标 | 数值 |
|---|---:|
| 可回答题 | 81 |
| 正确（EM=1） | 17（21.0%） |
| 错误 | 64 |
| 证据完整进入 4096 | 69/81（85.2%） |
| 证据完全缺失 | 12/81（14.8%） |
| 答案截断（completion≥250） | 0 |
| 可回答题被拒答 | 15（其中 13 题证据完整） |
| 对抗题正确拒答 | 23/24（95.8%） |

主错误分布（同一题可带次要标签）：

| 标签 | 含义 | 题数 | 占错误 |
|---|---|---:|---:|
| E | 答案表达问题（语义接近但过短/措辞不同/部分） | 23 | 35.9% |
| F | 过度拒答（证据完整却 No answer） | 13 | 20.3% |
| A | 检索错误（金标 session 不在上下文） | 11 | 17.2% |
| C | 时间推理错误（证据在，日期/跨度算错） | 9 | 14.1% |
| G | 错误推断/幻觉 | 4 | 6.2% |
| D | 多事实整合错误 | 3 | 4.7% |
| B | 证据不完整（部分金标证据缺失） | 1 | 1.6% |
| H | 评测器/语义等价（次要标签） | 16 | 25.0% |

按类别：

| 类别 | n | 正确 | 主错误分布 |
|---|---:|---:|---|
| cat1 multi-hop | 11 | 4 | A:4, D:1, F:1, B:1 |
| cat2 temporal | 26 | 6 | C:9, F:9, A:2 |
| cat3 open-domain | 0 | — | — |
| cat4 single-hop | 44 | 7 | E:23, A:5, G:4, F:3, D:2 |

**逐项回答任务中的怀疑点：**

| 怀疑点 | 结论 | 数据 |
|---|---|---|
| 评测口径差异 | 有，约 +3.6pp F1；不是主因 | §2.2 |
| 证据缺失 | 12/81 题完全缺失，其中仅 1 题仍答对 | 14.8% 错误可从检索解释 |
| 有完整证据仍答错 | 69 题证据完整，其中 53 题错 | 这是主要矛盾：80% 的错误发生在证据已进入上下文时 |
| 时间问题 | cat2 仅 6/26 正确；9 题日期算错、9 题证据完整却拒答、2 题证据缺失 | cat2 F1 29.2% |
| 多事实问题 | cat1 4/11；3 题主因 D，4 题主因 A | cat1 F1 48.9% |
| 过度拒答 | 13 题（证据完整） | 若全部修复，EM 最多 +13 题 ≈ +16pp |
| 幻觉 | 可回答题 G=4；对抗题 1/24 幻觉 | 幻觉不是主要问题 |
| 答案截断 | 0 题 | max completion 35 |
| 输出波动 | P0 同 prompt 3 次 EM 21.4/28.6/35.7%（14 题集） | §2.6 |

### 2.4 证据缺失不等于「Reader 没看到答案内容」

在 12 道证据缺失题中，有多题的答案内容其实散落在被召回的其他 session 里
（`gold_recall_in_context ≥ 0.75`，例如 q4/q17/q68）。这说明：

- `evidence_full@4096` 是**严格的金标 session 覆盖**指标，不等于「答案不可得」；
- 但「答案内容在别处」时，Reader 需要跨 session 组合，而这正是它目前最弱的能力。

### 2.5 Oracle / Full-Context 诊断（R1/R2/R3）

| 上下文 | Prompt | n | EM | F1 | 官方F1 | 过拒率 | 平均 context token |
|---|---|---:|---:|---:|---:|---:|---:|
| R1 正常 G_raw 检索 | P0 | 81 | 21.0% | 39.0% | 42.6% | 18.5% | 3.9k（4096 预算） |
| R1 正常 G_raw 检索 | P7 | 81 | 23.5% | 52.1% | 54.6% | 8.6% | 3.9k |
| R2 Gold Evidence Oracle | P0 | 81 | **18.5%** | **38.7%** | 40.2% | 17.3% | 761 |
| R2 Gold Evidence Oracle | P7 | 81 | **27.2%** | **60.3%** | 62.7% | 6.2% | 761 |
| R3 Full Conversation | P0 | 105 | 19.8% | 36.0% | 40.0% | 17.3% | 11,435 |
| R3 Full Conversation | P7 | 105 | 25.9% | 54.6% | 57.7% | 4.9% | 11,796 |

**解读（这是本轮最重要的对照）：**

1. **P0 在 Oracle 上没有变好，甚至略差（39.0→38.7 F1）。** 把正确证据直接给它，它仍然答不对。
   问题不在检索，而在 Reader 的「短 span / 拒答」策略与推理方式。
2. **P7 在 Oracle 上显著变好（52.1→60.3 F1，23.5→27.2 EM）。** 说明优化后的 Prompt 能利用更干净的证据，
   但正常检索下只能兑现一部分 → 检索仍有约 4–8pp 的可兑现空间。
3. **Full Conversation 反而比 Oracle 差**（P7：54.6 vs 60.3；P0：36.0 vs 38.7）：更多上下文带来干扰，
   「把整段对话塞进去」不是解法。
4. R2 平均只有 761 token（1–4 个 session），R3 有 11.4k token，两者不是等预算比较，不能直接说「Oracle 比 Full 强多少」。

### 2.6 HY4 输出波动（相同 prompt/context，temperature=0，3 次真实调用，禁用缓存）

代表题 14 道（q4,9,16,17,24,29,31,33,39,47,54,63,74,85）：

| Prompt | 词面不稳定率 | 得分（EM）不稳定率 | 平均两两 Jaccard | 每次运行 EM | 多数票 EM |
|---|---:|---:|---:|---|---:|
| P0 | 14.3% | 14.3% | 0.976 | 21.4% / 28.6% / 35.7% | 28.6% |
| P7 | 28.6% | **0.0%** | 0.886 | 14.3% / 14.3% / 14.3% | 14.3% |

**结论：**

- P0 的单题 EM 可以在 3 次运行间翻转（2 题），**105 题上的 1–2 题差异属于噪声**；
- P7 的措辞会变，但**得分稳定**（语义稳定），这是它相对 P0 的一个实际优势；
- 之前审计中 q33/q74/q85 的「同 context 不同答案」在这里被直接复现，不能再把这类翻转归因于图谱。

### 2.7 `max_tokens=256` 的截断敏感性

- P0：105 题 **0 题** `finish_reason=length`（最大 completion 35 token），256 不构成限制。
- P7：在长上下文设置下出现 1–2 题截断（主要是模型把上下文里的句子无限枚举）：
  `R3_P7` 的 q50/q68、`C2_P7`/`E3` 的 q3。
- 把这些行用 `max_tokens=512` 重跑（`reader/truncation_sensitivity.json`）：
  **4 行的 EM 全部保持 0→0**；F1 仅轻微上升（q3 从 .032 到 .171/.250，q50/q68 从 .04 到 .05）。
  对 105 题总 F1 的影响 ≤0.2pp。
- **结论：256 的截断不是 P7 收益/损失的主要来源；但 P7 的「完整枚举」规则确实会诱发 runaway enumeration，
  这是下一版 Prompt 需要修的真实缺陷（例如加「最多 10 个词/最多 5 个列表项」的硬约束）。**

---

## 3. Reader Prompt 实验结果（Task B）

### 3.1 设计

固定 G_raw 的 4096-token 检索上下文（`reader/contexts/graw/qNNN.json`，逐题校验 token 数与历史记录一致）。
P0 直接复用历史 G_raw 答案（同 prompt、同 context，零调用）；P1–P8 每道题一次真实调用，
统一 `max_tokens=256`、`temperature=0`、`no_think`。Prompt 原文与哈希在 `reader/prompts/`。
开发集 40 题（含任务指定的 12 个案例），内部验证集 65 题，划分在 `splits/split_dev40_val65.json`；
**选择文件 `reader/pilot/selection.json` 在跑 val 之前就已写入**。

**官方 LoCoMo QA Prompt（`task_eval/gpt_utils.py`，原文）：**

```text
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
```

官方 cat-5 使用另一段：

```text
Based on the above context, answer the following question.

Question: {} Short answer:
```

并且在 cat-2 的 question 后面追加 `" Use DATE of CONVERSATION to answer with an approximate date."`；
cat-5 则把问题改写成 "(a) Not mentioned in the conversation (b) <诱导答案>" 的二选一。
我们的 P6 复现了「短语句 + 精确用词」的主体，但**没有使用 Gold Category 来决定是否追加日期提示**
（这属于禁止的隐藏标签输入），而是用通用句式；cat-5 的多选协议单独在 §2.2/§3 的官方协议实验里跑。

### 3.2 开发集（40 题：34 可答 + 6 对抗）

| Prompt | EM | F1 | 官方F1 | 过拒率 | 对抗拒答 | 平均词数 |
|---|---:|---:|---:|---:|---:|---:|
| P0 Baseline | 23.5% | 39.4% | 42.3% | 26.5% | 100.0% | 3.7 |
| P1 简洁但完整 | 0.0% | 17.8% | 19.3% | 20.6% | 83.3% | 25.3 |
| P2 先整合再回答 | 2.9% | 22.0% | 23.9% | 23.5% | 83.3% | 21.1 |
| P3 时间敏感 | 20.6% | 44.4% | 45.0% | 2.9% | **0.0%** | 21.2 |
| P4 拒答校准 | 14.7% | 36.4% | 40.3% | 20.6% | 83.3% | 12.8 |
| P5 按问题类型 | 8.8% | 34.5% | 34.9% | 20.6% | 83.3% | 13.0 |
| P6 官方短语句风格 | 14.7% | 52.3% | **58.4%** | 11.8% | 83.3% | 5.3 |
| **P7 综合（短span+时间+拒答校准）** | **29.4%** | **60.0%** | **61.8%** | 11.8% | 83.3% | 8.4 |
| P8 综合+前提检查 | 20.6% | 50.4% | 54.4% | 20.6% | **100.0%** | 8.9 |

关键现象：

- **P1/P2/P5 全部失败**，原因不是内容错，而是答案变成了 20–25 词的长句/复述；项目 EM 对长答案几乎必然判 0。
  「要求完整」如果不与「只输出 span / 枚举项用逗号」绑定，会直接毁掉分数。
- **P3 把 cat2 从 EM 25%/F1 25% 提到 42%/55%**，过拒率 50%→0%，但对所有对抗题都不再拒答（0%）。这验证了「时间题被过度拒答」的怀疑，也暴露了「去掉拒答约束」的副作用。
- **P6（官方风格）在 P0–P6 中 F1 最高（52.3%）但 EM 不升**：答案短、精确用词，部分分高，但日期常带 "around"、列表用 "and" 而非逗号，EM 仍低。
- **P7** 是唯一同时把 EM、F1、官方 F1 都明显抬高的方案；它的收益来自三处：时间规则（q13/q19）、拒答校准（q7 类）、列表完整（q24 类）。
- **P8 的前提检查**恢复了对抗拒答，但把 EM 从 29.4% 压到 20.6%，是「过严拒答」的代价；最终未选它。

### 3.3 内部验证集（65 题：47 可答 + 18 对抗）

| Prompt | EM | F1 | 官方F1 | 过拒率 | 对抗拒答 | 平均词数 |
|---|---:|---:|---:|---:|---:|---:|
| P0 Baseline | 19.1% | 38.7% | 42.8% | 12.8% | 94.4% | 4.9 |
| **P7** | **19.1%** | **46.5%** | **49.4%** | 6.4% | 94.4% | 14.6 |
| P8 | 19.1% | 46.4% | 48.2% | 8.5% | 94.4% | 13.1 |

分题型（val）：

| 类别 | P0 EM / F1 / 官方F1 | P7 EM / F1 / 官方F1 |
|---|---|---|
| cat2 temporal (14) | 21.4% / 32.9% / 32.9% | **42.9% / 82.9% / 82.9%** |
| cat4 single-hop (33) | 18.2% / 41.2% / 47.1% | 9.1% / 31.0% / 35.3% |

**验证集暴露的真实代价：** P7 在 Temporal 上翻倍，但在 Single-hop 上因为答案变长（q71/q72/q78 等把 span 写成短句）损失 EM。
全量 105 题合并后净 EM 仍为正（+2.5pp），但不能宣称「所有题型都变好」。

### 3.4 全 105 题（dev+val）

| Prompt | EM | F1 | 官方F1 | 过拒率 | 对抗拒答 | 词数 |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 21.0% | 39.0% | 42.6% | 18.5% | 95.8% | 4.5 |
| **P7** | **23.5%** | **52.1%** | **54.6%** | 8.6% | 91.7% | 12.2 |
| P8 | 19.8% | 48.1% | 50.8% | 13.6% | 95.8% | 11.5 |

**按任务要求格式的完整表（标明数据划分）：**

| 数据 | Prompt | EM | F1 | 官方F1 | Multi-hop F1 | Temporal F1 | 拒答率 |
|---|---|---:|---:|---:|---:|---:|---:|
| dev 40 | P0 | 0.235 | 0.394 | 0.423 | 0.489 | 0.250 | 0.265 |
| dev 40 | P1 | 0.000 | 0.178 | 0.193 | 0.252 | 0.104 | 0.206 |
| dev 40 | P2 | 0.029 | 0.220 | 0.239 | 0.254 | 0.133 | 0.235 |
| dev 40 | P3 | 0.206 | 0.444 | 0.450 | 0.360 | 0.554 | 0.029 |
| dev 40 | P4 | 0.147 | 0.364 | 0.403 | 0.268 | 0.490 | 0.206 |
| dev 40 | P5 | 0.088 | 0.345 | 0.349 | 0.273 | 0.358 | 0.206 |
| dev 40 | P6 | 0.147 | 0.523 | 0.584 | 0.382 | 0.627 | 0.118 |
| dev 40 | **P7** | **0.294** | **0.600** | **0.618** | **0.508** | **0.806** | 0.118 |
| dev 40 | P8 | 0.206 | 0.504 | 0.544 | 0.403 | 0.639 | 0.206 |
| val 65 | P0 | 0.191 | 0.387 | 0.428 | — | 0.329 | 0.128 |
| val 65 | **P7** | 0.191 | **0.465** | **0.494** | — | **0.829** | 0.064 |
| val 65 | P8 | 0.191 | 0.464 | 0.482 | — | 0.829 | 0.085 |
| all 105 | P0 | 0.210 | 0.390 | 0.426 | 0.489 | 0.292 | 0.185 |
| all 105 | **P7** | **0.235** | **0.521** | **0.546** | **0.508** | **0.818** | 0.086 |
| all 105 | P8 | 0.198 | 0.481 | 0.508 | 0.403 | 0.741 | 0.136 |

（val 65 题按设计不含 cat1，因此 Multi-hop F1 为空；dev 40 含全部 11 道 cat1，因此 dev 的 Multi-hop F1 就是全量 cat1 的结果。）

分题型（全 105）：

| 类别 | n | P0 EM | P7 EM | P0 F1 | P7 F1 |
|---|---:|---:|---:|---:|---:|
| cat1 multi-hop | 11 | 36.4% | 27.3% | 48.9% | 50.8% |
| cat2 temporal | 26 | 23.1% | **42.3%** | 29.2% | **81.8%** |
| cat4 single-hop | 44 | 15.9% | 11.4% | 42.3% | 34.9% |
| cat5 adversarial | 24 | — | — | 拒答 95.8% | 拒答 91.7% |

### 3.5 最佳方案与真实案例

**选择：P7 为主方案，P8 为保守备选。** 选择依据：dev 上 EM/F1/官方F1 三项均最优；
val 上 EM 持平、F1/官方F1 明显提升；对抗拒答基本保持（val 相同，全量 −1 题）；
官方多选协议下 P7 为 21/24（87.5%），略高于 P0 的 20/24（83.3%）。

真实的修复案例：

| qid | 现象 | P0 | P7 | 原因 |
|---|---|---|---|---|
| q7 | 日期在 session header（29 Jan 2023） | `No answer` | `29 January, 2023` | P7 明确允许用 session date |
| q13 | `16 March, 2023` 在上下文 | `No answer` | `16 March, 2023` | 同上 |
| q19 | 事件说 "yesterday" | `yesterday` | `10 May, 2023` | 相对时间锚定 |
| q30 | 事件说 "tomorrow" | `tomorrow` | `20 June, 2023` | 相对时间锚定 |
| q38 | "Last Friday" | `Last Friday` | `21 July, 2023` | 相对时间锚定 |
| q39 | 金标 `Contemporary` | `contemporary dance` | `contemporary` | 短 span 约束 |
| q33 | 金标 20 June | `No answer` | `21 June, 2023`（错 1 天） | 相对时间算到 session 日期而非前一天 |
| q24 | 3 个事件 | 正确 | 漏 `dance competition`，多一项 | 列表枚举仍不稳定 |
| q4 | 两个原因 | 只覆盖"lost job" | 逗号拼接了 5 条重复理由 | 过度枚举，F1 反而下降 |
| q71/72/78 | 单跳短答案 | 正确 | 写成短句，EM 0 | P7 的格式约束未被稳定遵守 |
| q85 | 对抗题 | `No answer` | 回答了 Gina 的兴奋 | 前提检查缺失（P8 可修复） |

这些案例说明：**P7 的收益是「把拒答的时间题变成可算的日期」，代价是「单跳/多跳的答案变长」。**
它不是全面更好的 Reader，而是一个更适合 Temporal、需要配合答案长度约束的 Reader。

---

## 4. Comprehension Prompt 实验结果（Task A）

### 4.1 原关系审计（26 条 C3 / 18 条 C6）

| 指标 | C3 (COMP-P0) | C6 |
|---|---:|---:|
| 关系数 | 26 | 18 |
| temporal_ordering | 17 (65%) | 7 (39%) |
| Composed / Inferred | 20 / 6 | 9 / 9 |
| 谓词平均词数 / 最长 | 5.2 / 12 | 2.1 / 4 |
| 谓词 ≥5 词 | 10 | 0 |
| 两端都是已有实体 | 7 (27%) | 1 (6%) |
| 含 before/after/precede 语义 | 21 | 10 |
| 与 C2 事实近重复（Jaccard≥0.8） | 0 | 0 |
| 平均置信度 | 0.883 | 0.842 |

真实例子（全部来自 `outputs/comprehension/conv-30/cluster_hybrid_0.5_b8192_comp.json`）：

- `Jon --attended_fair_before--> worked_on_opening dance studio`（Composed, 0.88, s10+s15）——q16 唯一有效关系；
- `Jon's dance studio --expanded_from_startup_to_social_media_and_competition_before_financial_trouble--> tenuous grounds and investor search`（12 词谓词）；
- `Jon --moved_from_banker_to_dance_studio_owner_before_temp_job--> temp job`（把时间顺序塞进谓词）；
- `Gina --lost_job_before--> opened online clothing store`（与 `Gina --lost_job_enabled--> ...` 语义重复）。

**四条原始判断的核实结果：**

1. **过度强调时间关系：成立。** 65% 是 temporal_ordering，而 cat2 的 R@5 已 96.2%、Ev 96.2%。
2. **缺少真正需要跨 session 联合理解的关系：部分成立。** 所有关系都跨 2 个 session，但多数是把两条独立事实改写成事件短语，
   并不需要「联合理解」；真正需要 3 个 session 的只有 4 条。
3. **关系无法有效参与检索：成立。** 只有 7/26 两端是已有实体；新增关系在最终 top-5 fact 中只出现于 5–11/105 题（§4.5）。
4. **关系数量不是关键：成立。** P3 生成 27 条、P5 只有 7 条，两者的 Evidence@4096 都与 C2 相同（89.5%）。

### 4.2 候选 Prompt

- **COMP-P0**：原 Prompt（复用 C3 结果，不重跑）。
- **COMP-P1**：Evidence-Grounded Compositional——只保留必须组合两个 session 才能得到的关系；短实体、短谓词；temporal_ordering 降为最低优先级。
- **COMP-P2**：Cross-Session Bridge——把对话读成连续历史，优先 plan→outcome、event→consequence、state before/after、identity unification。
- **COMP-P3**：Event and State Tracking——先建时间线，再输出状态转移（state_transition/status_change/goal_change）。
- **COMP-P4**：Retrieval-Oriented——面向未来未知问题构建可检索关系：短实体、1–4 词具体谓词、两端尽量用已有实体、避免把答案塞进谓词。
- **COMP-P5**：Generate → Verify——用 P4 风格生成最多 12 条候选，再用带「被引用 turn 原文」的 verifier 逐条审查；
  verifier 只拒绝不支持/重复/误导的关系，不要求完美措辞。

所有 prompt 都保留原 JSON schema（`subject/predicate/object/relation_type/source_dia_ids/supporting_fact_ids/evidence/derivation/confidence`）。

### 4.3 生成质量

| Prompt | 原始候选 | 格式合法 | 最终关系 | temporal_ordering | 谓词≥5词 | 两端已有实体 | 平均置信度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| COMP-P0 (C3) | — | — | 26 | 17 | 10 | 7 (27%) | 0.883 |
| COMP-P1 | 19 | 14 | 14 | 0 | 0 | 1 (7%) | 0.811 |
| COMP-P2 | 23 | 19 | 19 | 0 | 0 | 1 (5%) | 0.794 |
| COMP-P3 | 30 | 27 | 27 | 0 | 0 | 1 (4%) | 0.834 |
| COMP-P4 | 20 | 16 | 16 | 2 | 0 | 1 (6%) | 0.861 |
| COMP-P5 | 24 | 18 | **7** | 1 | 0 | 0 | 0.879 |

- 所有被拒候选的唯一原因都是 `needs_two_sessions`（模型有时只引用一个 session）；没有格式错误。
- **P5 的 verifier 从 18 条中只保留 7 条**，拒绝了大量「因果关系过度声称」的候选（例如 "search led to opening"），
  说明 verifier 确实在起过滤作用，但也可能误杀（例如它把 "Jon's fair attendance generated possible leads" 判为 cross_session=false 却仍保留）。
- 新 Prompt 的**关系更短、更具体**，但**两端仍然很少命中已有实体**——这是下一个要解决的问题。

真实关系样例：

- P4：`Jon's fair attendance --generated_leads_for--> Jon's dance studio`（q16 的有效关系）
- P4：`Gina's fashion internship --followed--> Gina's design internship interview`
- P4：`Gina's marketing advice --led_to--> Jon's dance videos`
- P3：`Gina's fashion internship --changed_from--> interview to accepted`（把转移塞进 object，仍不理想）
- P5：`Jon losing job --pushed_start_of--> dance studio business`（保留的 7 条之一）

### 4.4 检索结果（105 题，固定 C2 三元组，只加关系）

| 图 | 新增关系 | R@1 | R@3 | R@5 | MRR | nDCG@10 | Ev@4096 | 负迁移(丢失证据) | 新增证据 | 丢失R@5 | 新增R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G_raw | 0 | .6198 | .8238 | .8984 | .7619 | .8067 | .8857 | — | — | — | — |
| C2（无理解） | 0 | .5944 | .8413 | .9032 | .7586 | .8026 | .8952 | — | — | — | — |
| COMP-P0（原版） | 26 | .5944 | .8317 | .9079 | .7575 | .8041 | .9048 | 0 | 1 | 1 | 2 |
| COMP-P1 | 14 | .5921 | .8532 | .9103 | .7547 | .8034 | .8952 | 1 | 1 | 1 | 2 |
| COMP-P2 | 19 | .5944 | .8317 | .9079 | .7598 | .8064 | .8952 | 0 | 0 | 0 | 1 |
| COMP-P3 | 27 | .5944 | .8460 | .9079 | .7579 | .8043 | .8952 | 1 | 1 | 0 | 1 |
| **COMP-P4** | 16 | .5944 | **.8508** | **.9127** | **.7638** | **.8100** | **.9048** | **0** | **1** | **0** | **1** |
| COMP-P5 | 7 | .5944 | .8413 | .9032 | .7582 | .8022 | .8952 | 0 | 0 | 0 | 0 |

（COMP-P0 行采用历史 C3 记录；其余为同一 diag 管线在 105 题上重跑的结果。C2 行的 diag 数值与历史完全一致；
G_raw 行是 diag 重跑值，与历史记录相比 R@5 为 .8984 vs .9032，差异来自 12/105 题的 GPU 数值级排名抖动，见 §1.3。）

**解读：**

- **最好的 COMP-P4**：R@3 +0.95pp、R@5 +0.95pp、MRR +0.52pp、nDCG +0.74pp、Ev +0.95pp（**恰好 +1 题 q16**），且没有任何题丢失证据。
  它用 16 条关系达到了原版 26 条关系的 Evidence 水平，并在排序上略好。
- **P1/P3** 都是 q31 进、q24 出（一进一出，Ev 不变）；P2/P5 完全没有改变 Evidence。
- **没有出现系统性排序改善**：所有 MRR 变化都在 ±0.5pp 内，和 C3 当年的结论一致。
- **P5 的 strict verification 没有帮助**：7 条关系太少，最终没有任何题的变化。

**分层分析（按 G_raw 的检索状态划分，P4 vs C2，105 题）：**

| 分层 | n | C2 Ev@4096 | P4 Ev@4096 | C2 R@5 | P4 R@5 | 证据增加题 | 证据丢失题 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始检索已成功（G_raw Ev=1） | 93 | .989 | .989 | .975 | .975 | 0 | **0** |
| 原始完全失败（Ev=0 且 partial=0） | 7 | .286 | **.429** | .286 | **.429** | 1 | 0 |
| 原始部分证据（0<partial<1） | 5 | .000 | .000 | .433 | .433 | 0 | 0 |
| 证据已在上下文但答案错误 | 54 | .981 | .981 | .963 | .963 | 0 | 0 |
| 真正跨 session 多证据问题（cat1） | 11 | .455 | **.455** | .621 | .621 | 0 | 0 |
| 金标 session 数 ≥2 | 11 | .455 | .455 | .621 | .621 | 0 | 0 |

**这张表比总分更能说明问题：**

1. **P4 没有负迁移**：93 道原始检索成功的题，证据覆盖没有一题下降（这是 P4 相对 P1/P3 的关键优势）。
2. **P4 的全部收益来自「原始完全失败」的 7 道题中的 1 道**（q16）；其余 6 道完全失败题没有变化。
3. **11 道真正跨 session 的多证据题，Evidence 完全没有改善（.455 → .455）**。
   这意味着：**面向「跨 session 理解」设计的 Prompt，最终救回的却是一道单 session 的时间题（q16），而不是多跳题。**
   这是本轮对 Comprehension 研究假设最重要的负面证据。
4. **54 道「证据已在上下文但答错」的题，检索层完全没有变化**——它们只能靠 Reader 解决，符合 §2.5 的 Oracle 结论。

### 4.5 机制追踪：新关系到底有没有进入检索？

用 `scripts/prompt_opt/06b_trace_facts.py` 把每题的最终 top-5 fact 与新增关系做规范化匹配：

| 图 | 关系数 | 新关系出现在 top-5 fact 的题数 | 命中次数 | 被用到的关系数 |
|---|---:|---:|---:|---:|
| COMP-P1 | 14 | 8 / 105 | 8 | 4 |
| COMP-P2 | 19 | 7 / 105 | 7 | 5 |
| COMP-P3 | 27 | 11 / 105 | 12 | 11 |
| **COMP-P4** | 16 | **5 / 105** | 5 | 4 |
| COMP-P5 | 7 | 3 / 105 | 3 | 1 |

COMP-P4 的 q16 命中：

- 关系：`Jon's fair attendance --generated_leads_for--> Jon's dance studio`（s10+s14，conf .86）
- 结果：q16 gold s10 从 C2 的第 8 名（dense fallback）升到 **第 1 名**，Ev 0→1。
- 这是本轮唯一一条「新关系 → top-5 fact → PPR → 证据覆盖改善」的完整链路。

与之对照，P3 的 27 条关系里有 11 条在 top-5 中出现过（次数更多），但**都没有把 gold session 拉进 4096 预算**；
命中次数多 ≠ 命中正确题目。**关系的「检索价值」取决于它是否与某道题的 top-5 fact 候选竞争成功，而不是数量。**

### 4.6 关系数量控制

为了把「关系更好」与「图谱更大」分开，本轮额外做了一个**标签无关的数量控制图**：
把 COMP-P4 的 16 条关系按 `confidence` 降序保留前 10 条（`COMP-P4-c10`，脚本
`scripts/prompt_opt/16_make_count_control.py`），其余设置完全相同，重新建图并检索 105 题。
（排序只用了模型自报 confidence，没有使用任何 QA 标签；前 10 条**包含** q16 的那条 fair→studio 关系。）

| 图 | 关系数 | R@3 | R@5 | MRR | nDCG@10 | Ev@4096 | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| C2 | 0 | .8413 | .9032 | .7586 | .8026 | .8952 | 16 |
| COMP-P4 | 16 | .8508 | .9127 | .7638 | .8100 | .9048 | 15 |
| **COMP-P4-c10** | **10** | **.8508** | **.9127** | .7665 | .8101 | **.9048** | 15 |

**结论：去掉 6 条低置信关系后，所有主要检索指标（R@3/R@5/Ev）完全不变；q16 的 s10 仍然是第 1 名、Ev=1。**
因此 P4 的收益来自**少数关键关系**（q16 的 fair→studio），而不是「关系数量更多」。
加上三个自然对照：

1. **P3（27 条）vs P4（16 条）**：P3 的 Evidence 与 C2 相同（89.5%），P4 提升到 90.5%——更多关系没有更好。
2. **P5（7 条）vs C2（0 条）**：P5 的 Evidence 完全不变——更少但更精的关系也没有效果。
3. **P1（14）vs P3（27）**：两者的 Evidence 都是 89.5%，且都是 q31 进/q24 出——数量翻倍没有改变结果。

**结论：在 conv-30 上，关系数量与检索收益没有单调关系；关系「是否进入 top-5 fact 候选并指向正确的 passage」才是关键。**
（数量控制仍有一个残余混杂：被删掉的 6 条关系也可能影响 LLM reranker 的输入候选；本轮只报告观测到的结果，不声称完全因果隔离。）

### 4.7 追加消融：关闭 LLM fact reranker

用户追加问题：**能不能不做 LLM 重排，看下效果？** 可以。官方 `rerank_facts` 是逐题 LLM 调用
（top-5 fact 候选 → LLM 过滤 → PPR；过滤为空则 dense fallback）。我们在
`src/prompt_optimization/retrieval_diag.py` 里加了一个 `use_llm_rerank=False` 分支：
**直接用 embedding 分数取 top-k fact 作为 PPR 种子，不调用任何 LLM**。
本实验不在官方代码里改任何东西，也不产生 LLM 调用。

两种设置：
- `top5`：k=5，与官方 reranker 的最终 fact 数一致（最直接的替换）；
- `top20`：k=20，观察「把候选窗口开大」是否更好。

**检索结果（105 题；无 LLM 调用）**

| 图 / 设置 | R@3 | R@5 | MRR | nDCG@10 | Ev@4096 | dense fallback |
|---|---:|---:|---:|---:|---:|---:|
| G_raw（官方 LLM rerank） | .8238 | .8984 | **.7619** | **.8067** | .8857 | 14 |
| G_raw / no-rerank top5 | .8111 | .8865 | .7173 | .7769 | .8762 | **0** |
| G_raw / no-rerank top20 | .6087 | .7437 | .5431 | .6211 | .7619 | 0 |
| C2（官方 LLM rerank） | .8413 | .9032 | **.7586** | **.8026** | .8952 | 16 |
| C2 / no-rerank top5 | .8405 | .9103 | .7164 | .7732 | .8952 | **0** |
| C2 / no-rerank top20 | .7190 | .8127 | .6225 | .6913 | .8095 | 0 |
| COMP-P4（官方 LLM rerank） | .8508 | .9127 | **.7638** | **.8100** | .9048 | 15 |
| **COMP-P4 / no-rerank top5** | **.8500** | **.9294** | .7224 | .7797 | **.9143** | **0** |
| COMP-P4 / no-rerank top20 | .7214 | .8056 | .6303 | .6991 | .7905 | 0 |

**结论 1：top5 无重排不是灾难，但也不是免费午餐。**
- R@3/R@5/Ev 基本保持，C2/P4 的 R@5 和 Ev 甚至略升（P4 R@5 +1.7pp、Ev +0.95pp）；
- **MRR/nDCG 稳定下降 3–4pp**——LLM reranker 的价值主要在「把最正确的那条 fact 排到第一」，
  而不是「多召回」；去掉它会让 top-1 变差。
- **dense fallback 从 14–16 题降到 0**：因为不再有「LLM 过滤为空」的路径。q54 依然失败
  （s4 仍不在 top-5），说明 q54 的根因不是 fallback 本身，而是图/查询不匹配。
- **top20 是明确更差的**：把更多 embedding-only fact 作为 PPR 种子会让主题漂移（G_raw R@5 .898→.744）。
  「多给种子」不是解法。

**结论 2：关闭 reranker 反而让 Comprehension 的收益翻倍。**
- 正常重排：C2 Ev .8952 → P4 Ev .9048（+1 题 q16）；
- 无重排 top5：C2 Ev .8952 → P4 Ev .9143（**+2 题 q16、q57**），R@5 .9103→.9294（**+1.9pp**）。
- 逐题追踪：无重排把 **q57** 的 s07 从 top-5 外拉到第 1（C2/P4 都是），把 **q44** 的 s09 拉到第 1；
  代价是 **q23** 的 R@5 .5→.25。也就是说，LLM reranker 在 q57/q44 上**压制**了本可用的证据。
- 这解释了为什么「P4 的收益只有 q16」：其中一部分被 reranker 吃掉了；无重排时 P4 的检索优势更明显。

**QA 结果（同一 105 题、同一 4096 预算、P0/P7 两个 Reader）**

| 图 | Reader | EM | F1 | 官方F1 | 过拒率 | 对抗拒答 |
|---|---|---:|---:|---:|---:|---:|
| G_raw + LLM rerank | P0 | .210 | .390 | .426 | .185 | .958 |
| G_raw + LLM rerank | P7 | .235 | .521 | .546 | .086 | .917 |
| G_raw + no-rerank top5 | P0 | .185 | .388 | .418 | .222 | .917 |
| G_raw + no-rerank top5 | P7 | .235 | .531 | **.569** | .062 | .875 |
| P4 + LLM rerank | P0 | .210 | .387 | .425 | .198 | .958 |
| P4 + LLM rerank | P7 | .247 | .524 | .539 | .086 | .917 |
| **P4 + no-rerank top5** | **P0** | **.222** | **.417** | **.445** | .173 | .958 |
| **P4 + no-rerank top5** | **P7** | **.259** | **.539** | **.569** | .062 | .917 |

**结论 3：效果高度依赖 Reader 和图。**
- **P0 在 G_raw 上被无重排拖累**（EM .210→.185、F1 .390→.388、对抗 .958→.917）：
  少了 reranker 的 top-1 锐化，短答案 P0 更容易被次优 passage 带偏。
- **P7 在 G_raw 上基本持平/略好**（EM 相同、官方 F1 +2.3pp），在 **P4 上全面变好**
  （EM .247→.259、F1 .524→.539、官方 F1 .539→.569，对抗不变）。
- 最佳组合：**P4 + no-rerank top5 + P7 = EM 25.9% / F1 53.9% / 官方 F1 56.9%**，
  相对正常 E3（24.7/52.4/53.9）为 +1.2pp EM / +1.5pp F1 / +3.0pp 官方 F1；
  相对 E1（G_raw+P7，23.5/52.1/54.6）为 +2.4pp EM / +1.8pp F1 / +2.3pp 官方 F1。
- 分题型：cat2 Temporal EM .500 / F1 .869（正常 E3 为 .462/.831），cat1 与 cat4 的 EM 不变。
  也就是说，无重排的收益仍然集中在 Temporal；Single-hop 没有变好。
- 与正常 E3 的逐题 EM 差异只有 q11（0→1），其余是 F1/官方 F1 的部分分变化。**EM 上 +1.2pp 仍是 1 题量级，
  在 §2.6 的噪声带内；官方 F1 +3pp 更值得关注，但也不能单独作为强结论。**

**结论 4：工程上值得关，但要看你优化的是 R@5/Evidence 还是 MRR/top-1。**
- 无重排每个图只花约 50 秒、**0 次 LLM 调用**；有重排每图要 105 次 LLM 调用（数分钟）且受共享配额影响。
- 如果目标是「让正确证据进入 4096 上下文」（Recall/Evidence），无重排 + P4 + P7 是当前最好的组合；
  如果目标是「top-1 命中」，官方 LLM reranker 仍然更好（MRR .764 vs .722）。
- **top20 不要用**；候选窗口开大只会让 PPR 主题漂移。

产物：`comprehension/retrieval_norerank/{graph}_top{k}/`、`joint/{G_raw_top5,COMP-P4_top5}_{P0,P7}.jsonl`、
`scripts/prompt_opt/18_no_llm_rerank.py`。

---

## 5. 联合实验（Graph × Reader Prompt）

四种必需组合，固定在相同的 105 题、相同 4096 预算、相同检索参数；图谱侧只用 C2 固定三元组 + 各自的关系，
Reader 侧用 P0 或 P7。为使「图」与「Reader」两个因素可分离，另加 C2+P7 一行。

| 实验 | 图谱 | Reader Prompt | EM | F1 | 官方F1 | 过拒率 | 对抗拒答 | 平均词数 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| E0 | G_raw | P0 | 21.0% | 39.0% | 42.6% | 18.5% | 95.8% | 4.5 |
| E1 | G_raw | P7 | 23.5% | 52.1% | 54.6% | 8.6% | 91.7% | 12.2 |
| E2 | COMP-P4 | P0 | 21.0% | 38.7% | 42.5% | 19.8% | 95.8% | 4.6 |
| **E3** | **COMP-P4** | **P7** | **24.7%** | **52.4%** | 53.9% | 8.6% | 91.7% | 13.0 |
| 附加 | C2 | P7 | 23.5% | 51.5% | 53.0% | 9.9% | 91.7% | 12.4 |

> 追加消融（关闭 LLM fact reranker）见 §4.7。最佳无重排组合为
> **COMP-P4 + no-rerank top5 + P7：EM 25.9% / F1 53.9% / 官方 F1 56.9%**；
> 与上表 E3 的差异为 +1.2pp EM / +1.5pp F1 / +3.0pp 官方 F1。

分题型：

| 实验 | cat1 EM / F1 | cat2 EM / F1 | cat4 EM / F1 |
|---|---|---|---|
| E0 | 36.4% / 48.9% | 23.1% / 29.2% | 15.9% / 42.3% |
| E1 | 27.3% / 50.8% | 42.3% / 81.8% | 11.4% / 34.9% |
| E2 | 36.4% / 56.8% | 19.2% / 25.4% | 18.2% / 42.1% |
| E3 | 27.3% / 52.2% | **46.2% / 83.1%** | 11.4% / 34.4% |
| C2+P7 | 27.3% / 52.2% | 42.3% / 79.2% | 11.4% / 35.0% |

**三个独立问题：**

**（1）QA Prompt 本身带来多少提升？**
E1 − E0（图与检索上下文完全不变）= **EM +2.5pp、F1 +13.1pp、官方 F1 +12.0pp**，
Temporal 子集 EM +19.2pp、F1 +52.6pp；代价是 Single-hop EM −4.5pp、F1 −7.4pp。
这是本轮最大、也最稳定的单项收益，但它不是「全面提升」，而是「把时间题从拒答/相对词变成确切日期」。

**（2）新的 Comprehension Prompt 带来多少提升？**
E2 − E0（Reader 不变）= EM 0.0pp、F1 −0.3pp；单看总数似乎没有收益。
但逐题看，E2 把 **q16 从 `No answer` 变成正确的 `24 April, 2023`**（证据从 Ev=0 变为 Ev=1，s10 第 1）；
同时发生了 4 个与图无关的 Reader 翻转（q0/q11/q26/q78，见 §2.6 的噪声带），把总数抹平了。
用同一 Reader 比较 P4 与 C2（E3 vs C2+P7）更干净：**EM 24.7% vs 23.5%（+1.2pp）、F1 52.4% vs 51.5%（+0.9pp）**。
结论：Comprehension 的独立收益是 **1 道题（q16）**，在 105 题上是 +1.2pp EM；
它真实存在、机制可追踪，但量级远小于 Reader Prompt。

**（3）两者是否互补？**
E3 − E1 = EM +1.2pp、F1 +0.3pp；E3 − E2 = EM +3.7pp、F1 +13.7pp。
即：Reader 与 Comprehension 的收益**基本可加**，且 Reader 贡献占绝大多数；
但 E3 的官方 F1（53.9%）略低于 E1（54.6%），说明在部分题上 P4 的图变化会让 P7 的措辞变差（q11 等），
因此不能宣称「E3 在所有指标上都优于 E1」。

**（4）最重要的联合案例 q16：**
- G_raw + P0：s10 第 9，Ev=0，`No answer`（错）。
- COMP-P4 + P0（E2）：s10 第 1，Ev=1，`24 April, 2023`（对）。
- COMP-P4 + P7（E3）：s10 第 1，Ev=1，`24 April, 2023`（对）。
这是唯一一条「Comprehension 改变检索 → 证据覆盖从 0 到 1 → 答案从错到对」的完整链路；
它同时说明：**当检索真的把证据送进来时，P0 也能答对这一题；P7 并不是必要条件。
Comprehension 的价值在「让证据到达」，Reader 的价值在「让到达的证据变成答案」，两者解决的是不同故障。**

---

## 6. 最有价值的正负案例

完整 20 题案例见 `cases/cases.md`（含每题 G_raw/C2/COMP-P4 的 top-5、gold rank、Ev、fallback、P0/P7 答案、触及金标 session 的所有新关系）。
以下按机制挑选。

### 6.1 正例：Comprehension 独立增益（q16）

- Q: `When did Jon go to a fair to get more exposure for his dance studio?`；Gold: `24 April, 2023`；Gold evidence: s10 / D10:1。
- G_raw: top-5 `s05,s09,s11,s02,s18`，s10 第 9，Ev=0，P0=`No answer`。
- C2: top-5 `s18,s05,s01,s02,s17`，**dense fallback**（top-1=1.0），s10 第 8，Ev=0，P0=`No answer`。
- COMP-P4: top-5 `s10,s14,s18,s05,s01`，**s10 第 1**，Ev=1，无 fallback。
- 起作用的关系：`Jon's fair attendance --generated_leads_for--> Jon's dance studio`（s10+s14，conf .86），
  在 q16 的最终 top-5 fact 中被追踪到（§4.5）。
- 在 P4 图上，**P0 与 P7 都回答 `24 April, 2023`（正确）**；而在 G_raw 上 P0/P7 都是 `No answer`
  （P7 在 G_raw 的 q16 上下文里仍然拒答）。
  所以这是一条「Comprehension 改变检索 → 证据覆盖 0→1 → 答案对」的完整链路，
  也是本轮唯一一条可同时通过检索层与答案层验证的 Comprehension 正例。

### 6.2 正例：Reader Prompt 修复时间拒答（q7 / q13 / q19 / q30 / q38）

| qid | Gold | P0 | P7 | 机制 |
|---|---|---|---|---|
| q7 | 29 January, 2023 | No answer | 29 January, 2023 | session header 就是答案 |
| q13 | 16 March, 2023 | No answer | 16 March, 2023 | 同上 |
| q19 | 10 May, 2023 | yesterday | 10 May, 2023 | "yesterday" 锚定 session date |
| q30 | 20 June, 2023 | tomorrow | 20 June, 2023 | "tomorrow" 锚定 session date |
| q38 | 21 July 2023 | Last Friday | 21 July, 2023 | 相对时间锚定 |

这 5 题在 G_raw 的检索里证据都已完整（Ev=1），P0 的失败完全是 Prompt/推理问题。

### 6.3 负例：Comprehension 把 q24 的证据挤出预算（P1/P3）

- Q: `Which events has Jon participated in to promote his business venture?`；Gold `fair, networking events, dance competition`；Gold sessions s8/s10/s16。
- C2: `s16,s18,s08,s04,s14`，Ev=1，P0 答案正确。
- COMP-P1/P3: s10 掉到第 7 名，4096 预算只装下 `s16,s18,s08,s05,s14,s04`，**s10 被挤出**，Ev 0。
- 结果：P0 答案仍正确（s16/s18/s08 已覆盖大部分），但这题在 P7 下会答错（见 6.4）。
- 说明：图扩充的负迁移不是「关系内容错」，而是「多出来的实体/边改变了 fact 候选与 PPR 权重」。

### 6.4 负例：Reader 列表枚举不稳定（q24 / q4）

- q24：P0 `a fair, networking events, a dance competition`（EM 1）；P7 `networking events, a fair, the last networking event`（EM 0，漏 `dance competition`，多一个错误项）。
- q4：P0 `Losing my job gave me the push to finally start my dream business: my own dance studio`（F1 .32）；
  P7 把 5 条同义证据用逗号拼成一句（F1 .20）。
- 说明：P7 的「完整」规则会被模型理解为「把所有相关句子都列出来」，反而伤害 EM/F1；
  需要在下一版加硬长度约束/去重规则（本轮未做，以免用 val 反调 Prompt）。

### 6.5 负例：P4 也没有修复 q54（dense fallback）

- Q: `How is Gina's store doing?`；Gold `The store is doing great.`；Evidence s4。
- G_raw: s4 第 1，Ev=1，P0 `doing great`（F1 .67）。
- C2/COMP-P4: s4 掉到第 9，**dense fallback**，Ev=0，P0 长句复述，F1=0。
- 原因：q54 是「短、泛、无实体」的问题；C2/P4 的图扩充改变了 fact 候选池，recognition memory 后没有 fact 存活 → 纯向量检索。
- 这不是 Comprehension 的错（C2 与 P4 相同），而是「往图里加知识」这一动作的副作用；P4 没有新增任何触及 s4 的关系。

### 6.6 负例：对抗题的前提误判（q85）

- Q: `What is Gina's attitude towards participating in the dance festival?`；Gold None。
- G_raw/P0: `No answer`（正确）；COMP-P4 graph 的 top-1 仍是 s11/s01 一带，P7 回答 Gina 很兴奋（错误）。
- P8 的前提检查可以修复，但代价是 dev EM −8.8pp；官方多选协议下 P0 是 20/24、P7 是 21/24，说明这更多是「回答格式」问题，而不是知识缺失。

### 6.7 负例：日期的 off-by-one（q33）

- Q: `When did Jon visit networking events for his store?`；Gold `20 June, 2023`；证据 D16:6 说 "yesterday"，session 16 日期是 21 June。
- P7 回答 `21 June, 2023`（用了 session 日期，没减 1 天）；P0 直接 `No answer`。
- 这是「相对时间锚定」规则没有覆盖「事件日期 ≠ session 日期」的细节；q19/q30 的 "yesterday/tomorrow" 恰好需要加/减一天，P7 有时对有时错。

### 6.8 正例：官方口径下的措辞差异（q77 / q2 / q76）

| qid | Gold | Pred | 项目 EM | 项目 F1 | 官方 F1 |
|---|---|---|---:|---:|---:|
| q77 | Hoodies | hoodie | 0 | 0.00 | **1.00** |
| q2 | by dancing | dance | 0 | 0.00 | **0.67** |
| q76 | excitement | excited and positive | 0 | 0.00 | **0.67** |

这些题不是「模型答错」，而是「词形/词干/多词表达」被项目 SQuAD 口径判 0；官方口径会给出部分分。
这支持「报告双口径」的必要性，但不支持「换口径就能大幅提高」。

### 6.9 正例：同一输入下的真实波动（q33 / q74 / q85）

之前的审计发现 q33/q74/q85 在「相同 context、相同 prompt」下答案不同。本轮重复实验（§2.6）直接复现：

- P0 在 14 题代表集上 3 次运行 EM = 21.4% / 28.6% / 35.7%；
- q33/q74/q85 都出现过同 context 不同答案。

因此，**任何 ≤2 题的 EM 翻转都不能单独作为方法有效的证据**；本报告只把有检索层独立佐证的 q16 记为 Comprehension 的硬正例。

---

## 7. 最终科研判断（Q1–Q10）

**Q1：现有 Comprehension Prompt 到底存在哪些问题？**
（1）把 temporal_ordering 与其它类型并列，导致 65% 产出集中在已经饱和的时间题；
（2）不约束谓词/实体长度，10/26 谓词 ≥5 词，最长 12 词，关系不可被自然语言问题命中；
（3）不要求关系两端对齐已有实体，只有 27% 两端是图中已有实体；
（4）只检查「跨 2 个 session」，不检查「组合是否带来新知识」；
（5）不检查关系是否进入 top-5 fact 候选，生成与检索脱节；
（6）verify 默认不启用，且看不到被引用的对话原文。
这些均有代码与产物依据；「关系内容本身错误」不是主要问题（与 C2 事实的近重复为 0）。

**Q2：哪种新的理解 Prompt 最有效？为什么？**
COMP-P4（Retrieval-Oriented）最好：它把关系写成「短实体 + 1–4 词具体谓词 + 指向来源 session」，
在 16 条关系下取得 R@3 .8508 / R@5 .9127 / MRR .7638 / Ev .9048，**优于原版 26 条**且零负迁移。
原因不是它生成了更多关系，而是它生成的关系更容易进入 top-5 fact 候选（q16 的 fair→studio 关系），
并且没有引入会挤掉已有证据的无关连接。COMP-P2 次之（排序略好但无证据增益）。

**Q3：它是否真正产生了普通联合抽取无法提供的知识？**
是，但很少，而且方向出乎预期。P4 的 16 条关系中有 1 条（fair attendance → dance studio）在 q16 上完成了
「原图 top-5 之外 → 新关系 → top-5 fact → PPR → gold session 第 1」的完整链路；
其余 15 条没有改变任何题的证据覆盖。原版 C3 也只有 q16 一条可归因。
**关键限定：q16 是单 session 的时间题（gold evidence 只在 s10），不是 11 道真正跨 session 的多证据题之一；
那 11 道题的 Evidence 在 P4 下完全没变（.455 → .455）。**
所以「能产生新知识」成立，「能稳定产生有用的、尤其是跨 session 的新知识」不成立。

**Q4：这些新知识是否改善了检索，而不是仅仅增加图谱规模？**
改善存在但非常小且高度局部：P4 的 Evidence +0.95pp（+1 题）、R@5 +0.95pp、MRR +0.52pp；
P3 的 27 条关系没有带来 Evidence 增益，P5 的 7 条也没有；P4-c10（删掉 6 条）指标完全不变。
**规模不是驱动因素。** 分层结果进一步说明：93 道原始成功题零负迁移，
7 道原始完全失败题里只有 1 道（q16）被救回，11 道跨 session 题零改善。
更准确的表述是：少数「恰好与某题 top-5 fact 竞争成功」的关系有价值，大多数关系不参与检索。

**Q5：当前 EM/F1 低的主要原因究竟是什么？**
按贡献排序：
1. **Reader 的短答案/拒答策略**：13 题证据完整却拒答（EM 的上限损失约 16pp）；
2. **时间推理**：cat2 另有 9 题日期算错（不含已计入上面的 9 题时间性拒答），cat2 F1 仅 29%；
3. **答案表达/部分正确**：23 题语义接近但过短/措辞不同（EM 判 0，F1 有部分分）；
4. **检索缺失**：12/81 题完全缺证据，只解释约 15% 的错误；
5. **评测口径**：官方 F1 比项目 F1 高约 3–4pp，但不是主因；
6. **输出波动**：1–2 题的翻转在噪声带内。
一句话：**「找到了证据但没答对」比「没找到证据」更重要。**

**Q6：仅修改 QA Prompt，能提高多少？**
全 105 题：EM 21.0%→23.5%（+2.5pp），F1 39.0%→52.1%（+13.1pp），官方 F1 42.6%→54.6%（+12.0pp）；
Temporal 子集 EM +19.2pp、F1 +52.6pp；Single-hop EM −4.5pp、F1 −7.4pp；
对抗拒答 95.8%→91.7%（官方多选协议下 P0 83.3% → P7 87.5%，反而略升）。
dev 上更高（EM +5.9pp、F1 +20.6pp），val 上 EM 持平、F1 +7.8pp——**收益不稳定、题型依赖强**，
不能宣称「一个 Prompt 全面解决」。

**Q7：使用 Oracle Evidence 后，Reader 的表现达到什么水平？**
原始 P0：Oracle EM 18.5% / F1 38.7%，**与正常检索基本一样**——说明 P0 的瓶颈不是检索。
优化 P7：Oracle EM 27.2% / F1 60.3%，比正常检索的 23.5%/52.1% 高 3.7pp EM / 8.2pp F1——
说明优化 Prompt 后检索仍有可兑现空间，但 Oracle 也不是「全对」（27% EM），Reader/评测仍是限制。
Full Conversation 更差（P7 25.9%/54.6%），不是解法。

**Q8：在优化 Reader 之后，Comprehension 是否更容易体现价值？**
是，但只在 q16 上得到验证。P4 把 q16 的 s10 从第 8 名（dense fallback）拉到第 1 名、Ev 0→1；
在 P4 图上，**P0 和 P7 都答出了 `24 April, 2023`**（E2 和 E3 都正确）。
所以更准确的说法是：**Comprehension 修的是「证据到达」，Reader 修的是「证据→答案」；
q16 只需要前者，P7 不是必要条件。** 在 P7 已经能处理大部分时间题之后，P4 额外带来的收益只有 +1.2pp EM（1 题），
并没有出现「Reader 变好 → Comprehension 收益被放大」的系统性现象。
原因仍是 §4.5 的 top-5 fact 窗口：P4 只有 5/105 题的新关系进入最终 fact 候选，
Reader 再好也无法让未进入检索的关系发挥作用。

**Q9：是否找到了可以进一步形成论文创新点的新机制？**
有两个候选，但都还需要更多数据：
1. **「关系是否进入 top-5 fact 候选」作为 Comprehension 的质量指标**：它比「关系是否跨 session / 是否事实正确」更接近检索收益；
   本轮 P4 有 99/105 题至少挂了 1 条触及金标 session 的关系（其中 18 题有 ≥2 个词面重叠），
   但只有 5/105 题的新关系进入最终 top-5 fact，最终只有 1 题兑现证据覆盖。
   **「99 题有机会 → 5 题进入候选 → 1 题兑现」是比单纯看关系数量更有解释力的漏斗。**
2. **把 Comprehension 从「生成更多关系」改成「生成能被 query-time fact reranker 选中的桥」**：
   即优化目标从「知识正确」转向「在 top-k fact 竞争中可被检索到」。这是一个可写成方法论的视角，
   但需要在更多 Conversation 上验证，并解决 LLM reranker 的非确定性。

**Q10：下一步更应该继续研究 Comprehension，还是优先改进检索与 QA？**
优先 **QA/Reader**，其次 **检索的 fact 候选与 reranker**，最后才是 Comprehension 的规模化。
理由：Oracle 实验显示 P0 即使拿到金标证据也不提升；P7 把 Oracle F1 提到 60.3%，
说明「Reader 会不会用证据」比「图谱多不多一条关系」更能解释当前分数。
追加的无重排消融（§4.7）进一步指出：**当前 LLM fact reranker 不是纯收益**——
它提高 top-1（MRR +3–4pp），但会在 q57/q44 上压制正确证据，并让 P4 的 Ev 收益从 +2 题降到 +1 题。
因此「检索改进」的第一步不一定是换更强的模型，而是决定优化目标：
- 若论文指标是 **EM/F1/Evidence**，**关掉 LLM reranker、用 embedding top5 + P4 + P7** 是当前最好的组合
  （EM 25.9% / F1 53.9% / 官方 F1 56.9%），且省掉每图 105 次 LLM 调用；
- 若指标是 **top-1/MRR**，保留 reranker，并研究如何让它不删掉跨 session 桥接事实；
- 无论哪种，**top20 不要用**（主题漂移）。
Comprehension 的下一步不应继续堆关系，而应研究「如何让关系进入 top-5 fact 候选」以及
「如何在 4096 token 内同时装下多条证据」；否则在 conv-30 上它的上限仍然很小。

---

## 8. 复现信息与限制

### 8.1 产物结构

```text
outputs/prompt_optimization_conv30/
  audit/                    # prompt 审计、官方口径、错误归因、105 题 Reader 上下文
  splits/                   # dev40 / val65 固定划分（seed 42）
  reader/
    prompts/                # P0–P8 原文 + system/template 哈希
    contexts/graw/          # 105 题冻结的 Reader 输入（token 数逐题对齐历史）
    pilot/                  # dev 结果 + selection.json（先于 val 写入）
    final/                  # val 结果
    oracle/                 # R2 / R3
    repeats/                # 同输入 ×3 稳定性
    official_adversarial/   # 官方 cat-5 多选协议
  comprehension/
    prompts/                # COMP-P1..P5 实际 prompt（含哈希）
    extractions/            # 关系、拒绝原因、token 用量
    retrieval/              # 每图的逐题记录 + diagnostics + summary
    retrieval_norerank/     # 关闭 LLM fact reranker 的消融（top5 / top20）
    quality_report.{md,json}
    fact_trace_summary.json
  joint/                    # E2/E3/C2+P7 的 Reader 结果
  cases/                    # 20 题案例
  reports/                  # 本报告 + auto_tables.md + all_metrics.json
```

### 8.2 实验记录

- 代码：`scripts/prompt_opt/00–17_*.py`（含 04b/06b/06c/08b 辅助脚本）、`src/prompt_optimization/*`（本轮新增，未修改官方 HippoRAG 与历史脚本）。
- 模型：HY4，`reasoning_effort=no_think`，`temperature=0.0`，Reader `max_tokens=256`，Comprehension 8192。
- Embedding：NV-Embed-v2，fp16，dim 4096。
- 检索：top-20，`linking_top_k=5`，`passage_node_weight=0.05`，PPR damping 0.5（官方默认）。
- 缓存：每个实验使用独立 `HY4_CACHE_SHARD`（po/co/or/oa/rp/jo），调用日志在 `outputs/logs/hy4_calls*.jsonl`；
  Reader 重复实验显式 `use_cache=False`。
- API 错误单独统计：`hy4_callsrp.jsonl` 中 14 次 `RateLimitError`（429）均被重试成功，
  **没有把 429、调用失败或空回答计作拒答**；其余 shard（po/co/or/oa/jo）在本轮新增调用中 0 错误。
- 所有历史 `outputs/qa/`、`outputs/raw_graphs/conv-30/` 根目录与 `variants/cluster_*`、`variants/full_conv_*` 均未被修改。

### 8.3 明确限制

1. **单 Conversation、探索性**：所有结论只在 conv-30 上成立，不能宣称跨 Conversation 泛化。
2. **小样本**：105 题、81 可答；1 题 = 0.95pp（全量）/ 1.23pp（81 题），EM 的 1–2 题差异在噪声带内。
3. **Reader 非确定性**：temperature=0 仍会变（§2.6）；单次运行的 EM/F1 差异不能作为强结论。
4. **检索非确定性**：官方 fact reranker 本身是 LLM 调用；同一图重跑也可能改变 top-5 fact 与 PPR。
   本轮 C2 的 diag 与历史完全一致，但 G_raw 的 diag 有 12/105 题排名不同（数值级差异）。
5. **关系数量控制有一个残余混杂**：P4-c10 只按 confidence 截断，未做「同数量随机关系」对照；
   被删关系也可能影响 LLM reranker 的候选输入。结论（数量不是主因）由 P4 vs P4-c10 + 三个自然对照共同支持。
6. **Oracle 不是等预算比较**：R2 平均 761 token，R3 11,435 token；不能把差异简单归因于「证据质量」。
7. **P7 是 dev-informed 综合 Prompt**：在看过 dev 的 P0–P6 结果后加入；val 未参与设计。P8 是 P7 的保守变体。
8. **官方评分器**：从 `snap-research/locomo` `main` 分支拉取（2026-11-09）；如果官方后续修改，需要重新对齐。
9. **潜在数据泄漏边界**：Read-Time 生成只使用 Session 原文与 C2 事实，未使用任何 Gold QA；Oracle 仅用 Gold Evidence 选 session，
   不把 Gold Answer 放入 prompt；这些均在代码中可核查。
10. **`max_tokens=256`**：P0 无截断；P7 在长上下文下有 1–2 题截断，512 重跑后 EM 不变、F1 仅 +0.2pp 以内（§2.7），
    不改变主结论，但说明 P7 的枚举规则有 runaway 风险。

---

## 附录 A：本案的核心数字速查

| 问题 | 数字 |
|---|---|
| G_raw 原始 EM / F1 | 21.0% / 39.0% |
| G_raw 官方 F1 | 42.6% |
| 可回答题被拒答 | 15（13 题证据完整） |
| 证据完整仍错 | 53/69 |
| Oracle + P0 | EM 18.5% / F1 38.7% |
| Oracle + P7 | EM 27.2% / F1 60.3% |
| P7 全量 | EM 23.5% / F1 52.1% / 官方 F1 54.6% |
| 最好 Comprehension（P4） | Ev 89.5%→90.5%（q16），R@5 90.3%→91.3% |
| P4 关系进入 top-5 fact | 5/105 题 |
| 同输入 3 次 EM 波动（P0） | 21.4% / 28.6% / 35.7% |

## 附录 B：关键文件索引

| 用途 | 路径 |
|---|---|
| Reader Prompt 原文 | `src/retrieval/reader.py`；本轮渲染 `audit/reader_prompt_qNNN.txt` |
| Comprehension Prompt 原文 | `src/comprehension/prompts.py`；本轮渲染 `audit/comp_prompt_group{0,1}.txt` |
| 官方口径重算 | `audit/official_metric_comparison.{md,json,csv}` |
| 105 题错误归因 | `audit/error_analysis_conv30.jsonl` / `.csv` / `_summary.md` |
| 105 题 Reader 上下文 | `audit/graw_contexts/qNNN.txt` |
| Reader 结果 | `reader/pilot/`、`reader/final/`、`reader/oracle/`、`reader/repeats/` |
| Comprehension 结果 | `comprehension/extractions/`、`comprehension/retrieval/`、`comprehension/quality_report.md` |
| 机制追踪 | `comprehension/fact_trace_summary.json` |
| 截断敏感性 | `reader/truncation_sensitivity.json` |
| 案例 | `cases/cases.md` |
| 自动表 | `reports/auto_tables.md`、`reports/all_metrics.json` |
