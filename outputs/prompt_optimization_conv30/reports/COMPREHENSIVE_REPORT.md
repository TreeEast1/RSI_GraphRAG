# conv-30 Prompt 优化实验：完整结果与全部细节

> 本文档把本轮所有实验（Reader Prompt P0–P8、Comprehension Prompt COMP-P0–P5、关系数量控制、Oracle/Full-Context 诊断、重复运行、官方评分口径、官方 cat-5 协议、以及关闭 LLM fact reranker 的追加消融）整合到一个文件里。
> 范围：LoCoMo conv-30，19 sessions / 105 questions（81 可答、24 对抗）。基座：官方 HippoRAG2 + HY4（no_think, temperature=0）+ NV-Embed-v2。
> 所有历史产物只读；新产物在 `outputs/prompt_optimization_conv30/`。

## 目录


**Part I — 叙述报告与结论**（原 `FINAL_REPORT.md` 全文）

- §0 执行摘要
- §1 原始 Prompt 审计（Reader / Comprehension / 检索路径）
- §2 QA 低分原因分析（原始指标、官方口径、错误归因、Oracle、波动、截断）
- §3 Reader Prompt 实验结果（dev / val / 全量 / 最佳方案）
- §4 Comprehension Prompt 实验结果（关系质量、检索、分层、机制、数量控制、**关闭 LLM reranker 消融**）
- §5 Graph × Reader 联合实验（E0–E3、C2+P7）
- §6 正负案例（20 题）
- §7 Q1–Q10
- §8 复现信息与限制

**Part II — 数据附录**

- 附录 0：全部指标总览（Reader / Comprehension / 无重排 / 联合 / Oracle / 重复 / 官方 cat-5）
- 附录 A：所有 Prompt 原文（Reader P0–P8、Comprehension COMP-P0–P5）
- 附录 B：105 题总表（问题、金标、各系统答案与 EM/F1）
- 附录 C：105 题检索状态总表（C2 / P4 / P4-c10 / P4-no-rerank 的 Evidence、gold rank、fallback、R@5）
- 附录 D：所有 EM 翻转逐题清单（P7 vs P0、E2 vs E0、E3 vs E1、no-rerank vs 正常）
- 附录 E：所有 Comprehension 关系清单（C3 / C6 / COMP-P1–P5 / P4-c10）
- 附录 F：20 题完整案例（含每图 top-5、gold rank、Ev、fallback、top facts、四种 Reader 答案）
- 附录 G：105 题完整错误归因（A–H 标签与理由）
- 附录 H：Oracle、重复运行、官方 cat-5、截断敏感性、HY4 调用统计
- 附录 I：完整实验日志（命令、shard、产物）

---

# Part I — 叙述报告与结论

## LoCoMo conv-30：Read-Time Comprehension Prompt 与 Reader/QA Prompt 优化实验报告

> 范围：仅 LoCoMo `conv-30`（19 sessions / 105 questions / 81 answerable / 24 adversarial）。
> 基座：官方 HippoRAG2（`third_party/HippoRAG` @ d5c8329）、HY4（`reasoning_effort=no_think`、temperature=0）、NV-Embed-v2。
> 所有原始产物（G_raw / C2 / C3 / C5 / C6 图谱与 `outputs/qa/`）**只读**；本轮所有新实验写入
> `outputs/prompt_optimization_conv30/`，图谱写入 `outputs/raw_graphs/conv-30/variants/cmpo_*`，未覆盖任何历史目录。
> 日期：2026-11-09。

---

### 0. 执行摘要

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

### 1. 原始 Prompt 审计

#### 1.1 Reader Prompt（`src/retrieval/reader.py`）

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

#### 1.2 Comprehension Prompt（`src/comprehension/prompts.py`）

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

#### 1.3 检索路径审计：真正决定结果的是「LLM fact reranker」

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

### 2. QA 低分原因分析

#### 2.1 原始指标（只读复算，未重跑 LLM）

| 方法 | R@3 | R@5 | Ev@4096 | EM(1–4) | F1(1–4) | 拒答率(cat5) |
|---|---:|---:|---:|---:|---:|---:|
| G_raw | 81.0% | 90.3% | 88.6% | 21.0% | 39.0% | 95.8% |
| C2 | 84.1% | 90.3% | 89.5% | 22.2% | 40.4% | 91.7% |
| C3 | 83.2% | 90.8% | 90.5% | 24.7% | 41.5% | 91.7% |
| C5 | 85.4% | 89.8% | 88.6% | 21.0% | 40.5% | 95.8% |
| C6 | 86.4% | 90.6% | 89.5% | 19.8% | 40.1% | 95.8% |

（与 `CONV30_READTIME_ANALYSIS.md` 完全一致；R@5/Ev 是全 105 题均值，EM/F1 是 81 道可回答题均值。）

#### 2.2 评测口径差异（官方 `task_eval/evaluation.py`，无 LLM）

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

#### 2.3 逐题错误归因（G_raw，105 题，规则化 + 人工核对）

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

#### 2.4 证据缺失不等于「Reader 没看到答案内容」

在 12 道证据缺失题中，有多题的答案内容其实散落在被召回的其他 session 里
（`gold_recall_in_context ≥ 0.75`，例如 q4/q17/q68）。这说明：

- `evidence_full@4096` 是**严格的金标 session 覆盖**指标，不等于「答案不可得」；
- 但「答案内容在别处」时，Reader 需要跨 session 组合，而这正是它目前最弱的能力。

#### 2.5 Oracle / Full-Context 诊断（R1/R2/R3）

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

#### 2.6 HY4 输出波动（相同 prompt/context，temperature=0，3 次真实调用，禁用缓存）

代表题 14 道（q4,9,16,17,24,29,31,33,39,47,54,63,74,85）：

| Prompt | 词面不稳定率 | 得分（EM）不稳定率 | 平均两两 Jaccard | 每次运行 EM | 多数票 EM |
|---|---:|---:|---:|---|---:|
| P0 | 14.3% | 14.3% | 0.976 | 21.4% / 28.6% / 35.7% | 28.6% |
| P7 | 28.6% | **0.0%** | 0.886 | 14.3% / 14.3% / 14.3% | 14.3% |

**结论：**

- P0 的单题 EM 可以在 3 次运行间翻转（2 题），**105 题上的 1–2 题差异属于噪声**；
- P7 的措辞会变，但**得分稳定**（语义稳定），这是它相对 P0 的一个实际优势；
- 之前审计中 q33/q74/q85 的「同 context 不同答案」在这里被直接复现，不能再把这类翻转归因于图谱。

#### 2.7 `max_tokens=256` 的截断敏感性

- P0：105 题 **0 题** `finish_reason=length`（最大 completion 35 token），256 不构成限制。
- P7：在长上下文设置下出现 1–2 题截断（主要是模型把上下文里的句子无限枚举）：
  `R3_P7` 的 q50/q68、`C2_P7`/`E3` 的 q3。
- 把这些行用 `max_tokens=512` 重跑（`reader/truncation_sensitivity.json`）：
  **4 行的 EM 全部保持 0→0**；F1 仅轻微上升（q3 从 .032 到 .171/.250，q50/q68 从 .04 到 .05）。
  对 105 题总 F1 的影响 ≤0.2pp。
- **结论：256 的截断不是 P7 收益/损失的主要来源；但 P7 的「完整枚举」规则确实会诱发 runaway enumeration，
  这是下一版 Prompt 需要修的真实缺陷（例如加「最多 10 个词/最多 5 个列表项」的硬约束）。**

---

### 3. Reader Prompt 实验结果（Task B）

#### 3.1 设计

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

#### 3.2 开发集（40 题：34 可答 + 6 对抗）

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

#### 3.3 内部验证集（65 题：47 可答 + 18 对抗）

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

#### 3.4 全 105 题（dev+val）

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

#### 3.5 最佳方案与真实案例

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

### 4. Comprehension Prompt 实验结果（Task A）

#### 4.1 原关系审计（26 条 C3 / 18 条 C6）

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

#### 4.2 候选 Prompt

- **COMP-P0**：原 Prompt（复用 C3 结果，不重跑）。
- **COMP-P1**：Evidence-Grounded Compositional——只保留必须组合两个 session 才能得到的关系；短实体、短谓词；temporal_ordering 降为最低优先级。
- **COMP-P2**：Cross-Session Bridge——把对话读成连续历史，优先 plan→outcome、event→consequence、state before/after、identity unification。
- **COMP-P3**：Event and State Tracking——先建时间线，再输出状态转移（state_transition/status_change/goal_change）。
- **COMP-P4**：Retrieval-Oriented——面向未来未知问题构建可检索关系：短实体、1–4 词具体谓词、两端尽量用已有实体、避免把答案塞进谓词。
- **COMP-P5**：Generate → Verify——用 P4 风格生成最多 12 条候选，再用带「被引用 turn 原文」的 verifier 逐条审查；
  verifier 只拒绝不支持/重复/误导的关系，不要求完美措辞。

所有 prompt 都保留原 JSON schema（`subject/predicate/object/relation_type/source_dia_ids/supporting_fact_ids/evidence/derivation/confidence`）。

#### 4.3 生成质量

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

#### 4.4 检索结果（105 题，固定 C2 三元组，只加关系）

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

#### 4.5 机制追踪：新关系到底有没有进入检索？

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

#### 4.6 关系数量控制

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

#### 4.7 追加消融：关闭 LLM fact reranker

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

### 5. 联合实验（Graph × Reader Prompt）

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

### 6. 最有价值的正负案例

完整 20 题案例见 `cases/cases.md`（含每题 G_raw/C2/COMP-P4 的 top-5、gold rank、Ev、fallback、P0/P7 答案、触及金标 session 的所有新关系）。
以下按机制挑选。

#### 6.1 正例：Comprehension 独立增益（q16）

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

#### 6.2 正例：Reader Prompt 修复时间拒答（q7 / q13 / q19 / q30 / q38）

| qid | Gold | P0 | P7 | 机制 |
|---|---|---|---|---|
| q7 | 29 January, 2023 | No answer | 29 January, 2023 | session header 就是答案 |
| q13 | 16 March, 2023 | No answer | 16 March, 2023 | 同上 |
| q19 | 10 May, 2023 | yesterday | 10 May, 2023 | "yesterday" 锚定 session date |
| q30 | 20 June, 2023 | tomorrow | 20 June, 2023 | "tomorrow" 锚定 session date |
| q38 | 21 July 2023 | Last Friday | 21 July, 2023 | 相对时间锚定 |

这 5 题在 G_raw 的检索里证据都已完整（Ev=1），P0 的失败完全是 Prompt/推理问题。

#### 6.3 负例：Comprehension 把 q24 的证据挤出预算（P1/P3）

- Q: `Which events has Jon participated in to promote his business venture?`；Gold `fair, networking events, dance competition`；Gold sessions s8/s10/s16。
- C2: `s16,s18,s08,s04,s14`，Ev=1，P0 答案正确。
- COMP-P1/P3: s10 掉到第 7 名，4096 预算只装下 `s16,s18,s08,s05,s14,s04`，**s10 被挤出**，Ev 0。
- 结果：P0 答案仍正确（s16/s18/s08 已覆盖大部分），但这题在 P7 下会答错（见 6.4）。
- 说明：图扩充的负迁移不是「关系内容错」，而是「多出来的实体/边改变了 fact 候选与 PPR 权重」。

#### 6.4 负例：Reader 列表枚举不稳定（q24 / q4）

- q24：P0 `a fair, networking events, a dance competition`（EM 1）；P7 `networking events, a fair, the last networking event`（EM 0，漏 `dance competition`，多一个错误项）。
- q4：P0 `Losing my job gave me the push to finally start my dream business: my own dance studio`（F1 .32）；
  P7 把 5 条同义证据用逗号拼成一句（F1 .20）。
- 说明：P7 的「完整」规则会被模型理解为「把所有相关句子都列出来」，反而伤害 EM/F1；
  需要在下一版加硬长度约束/去重规则（本轮未做，以免用 val 反调 Prompt）。

#### 6.5 负例：P4 也没有修复 q54（dense fallback）

- Q: `How is Gina's store doing?`；Gold `The store is doing great.`；Evidence s4。
- G_raw: s4 第 1，Ev=1，P0 `doing great`（F1 .67）。
- C2/COMP-P4: s4 掉到第 9，**dense fallback**，Ev=0，P0 长句复述，F1=0。
- 原因：q54 是「短、泛、无实体」的问题；C2/P4 的图扩充改变了 fact 候选池，recognition memory 后没有 fact 存活 → 纯向量检索。
- 这不是 Comprehension 的错（C2 与 P4 相同），而是「往图里加知识」这一动作的副作用；P4 没有新增任何触及 s4 的关系。

#### 6.6 负例：对抗题的前提误判（q85）

- Q: `What is Gina's attitude towards participating in the dance festival?`；Gold None。
- G_raw/P0: `No answer`（正确）；COMP-P4 graph 的 top-1 仍是 s11/s01 一带，P7 回答 Gina 很兴奋（错误）。
- P8 的前提检查可以修复，但代价是 dev EM −8.8pp；官方多选协议下 P0 是 20/24、P7 是 21/24，说明这更多是「回答格式」问题，而不是知识缺失。

#### 6.7 负例：日期的 off-by-one（q33）

- Q: `When did Jon visit networking events for his store?`；Gold `20 June, 2023`；证据 D16:6 说 "yesterday"，session 16 日期是 21 June。
- P7 回答 `21 June, 2023`（用了 session 日期，没减 1 天）；P0 直接 `No answer`。
- 这是「相对时间锚定」规则没有覆盖「事件日期 ≠ session 日期」的细节；q19/q30 的 "yesterday/tomorrow" 恰好需要加/减一天，P7 有时对有时错。

#### 6.8 正例：官方口径下的措辞差异（q77 / q2 / q76）

| qid | Gold | Pred | 项目 EM | 项目 F1 | 官方 F1 |
|---|---|---|---:|---:|---:|
| q77 | Hoodies | hoodie | 0 | 0.00 | **1.00** |
| q2 | by dancing | dance | 0 | 0.00 | **0.67** |
| q76 | excitement | excited and positive | 0 | 0.00 | **0.67** |

这些题不是「模型答错」，而是「词形/词干/多词表达」被项目 SQuAD 口径判 0；官方口径会给出部分分。
这支持「报告双口径」的必要性，但不支持「换口径就能大幅提高」。

#### 6.9 正例：同一输入下的真实波动（q33 / q74 / q85）

之前的审计发现 q33/q74/q85 在「相同 context、相同 prompt」下答案不同。本轮重复实验（§2.6）直接复现：

- P0 在 14 题代表集上 3 次运行 EM = 21.4% / 28.6% / 35.7%；
- q33/q74/q85 都出现过同 context 不同答案。

因此，**任何 ≤2 题的 EM 翻转都不能单独作为方法有效的证据**；本报告只把有检索层独立佐证的 q16 记为 Comprehension 的硬正例。

---

### 7. 最终科研判断（Q1–Q10）

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

### 8. 复现信息与限制

#### 8.1 产物结构

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

#### 8.2 实验记录

- 代码：`scripts/prompt_opt/00–17_*.py`（含 04b/06b/06c/08b 辅助脚本）、`src/prompt_optimization/*`（本轮新增，未修改官方 HippoRAG 与历史脚本）。
- 模型：HY4，`reasoning_effort=no_think`，`temperature=0.0`，Reader `max_tokens=256`，Comprehension 8192。
- Embedding：NV-Embed-v2，fp16，dim 4096。
- 检索：top-20，`linking_top_k=5`，`passage_node_weight=0.05`，PPR damping 0.5（官方默认）。
- 缓存：每个实验使用独立 `HY4_CACHE_SHARD`（po/co/or/oa/rp/jo），调用日志在 `outputs/logs/hy4_calls*.jsonl`；
  Reader 重复实验显式 `use_cache=False`。
- API 错误单独统计：`hy4_callsrp.jsonl` 中 14 次 `RateLimitError`（429）均被重试成功，
  **没有把 429、调用失败或空回答计作拒答**；其余 shard（po/co/or/oa/jo）在本轮新增调用中 0 错误。
- 所有历史 `outputs/qa/`、`outputs/raw_graphs/conv-30/` 根目录与 `variants/cluster_*`、`variants/full_conv_*` 均未被修改。

#### 8.3 明确限制

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

### 原报告附录 A：核心数字速查

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

### 原报告附录 B：关键文件索引

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


---

# Part II — 数据附录

## 附录 0：全部指标总览

### 0.1 Reader Prompt（P0–P8）

dev=40 题（含全部 11 道 cat1），val=65 题，all=105 题。P1–P6 只在 dev 上跑；P7/P8 在 dev+val 上跑。

| pid | split | n | EM | F1 | offF1 | refusal | adv refusal | words | cat1 F1 | cat2 F1 | cat4 F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | dev | 40 | 0.235 | 0.394 | 0.423 | 0.265 | 1.000 | 3.7 | 0.489 | 0.250 | 0.457 |
| P0 | val | 65 | 0.191 | 0.387 | 0.428 | 0.128 | 0.944 | 4.9 | — | 0.329 | 0.412 |
| P0 | all | 105 | 0.210 | 0.390 | 0.426 | 0.185 | 0.958 | 4.5 | 0.489 | 0.292 | 0.423 |
| P1 | dev | 40 | 0.000 | 0.178 | 0.193 | 0.206 | 0.833 | 25.3 | 0.252 | 0.104 | 0.186 |
| P2 | dev | 40 | 0.029 | 0.220 | 0.239 | 0.235 | 0.833 | 21.1 | 0.254 | 0.133 | 0.280 |
| P3 | dev | 40 | 0.206 | 0.444 | 0.450 | 0.029 | 0.000 | 21.2 | 0.360 | 0.554 | 0.406 |
| P4 | dev | 40 | 0.147 | 0.364 | 0.403 | 0.206 | 0.833 | 12.8 | 0.268 | 0.490 | 0.323 |
| P5 | dev | 40 | 0.088 | 0.345 | 0.349 | 0.206 | 0.833 | 13.0 | 0.273 | 0.358 | 0.403 |
| P6 | dev | 40 | 0.147 | 0.523 | 0.584 | 0.118 | 0.833 | 5.3 | 0.382 | 0.627 | 0.550 |
| P7 | dev | 40 | 0.294 | 0.600 | 0.618 | 0.118 | 0.833 | 8.4 | 0.508 | 0.806 | 0.466 |
| P7 | val | 65 | 0.191 | 0.465 | 0.494 | 0.064 | 0.944 | 14.6 | — | 0.829 | 0.310 |
| P7 | all | 105 | 0.235 | 0.521 | 0.546 | 0.086 | 0.917 | 12.2 | 0.508 | 0.818 | 0.349 |
| P8 | dev | 40 | 0.206 | 0.504 | 0.544 | 0.206 | 1.000 | 8.9 | 0.403 | 0.639 | 0.457 |
| P8 | val | 65 | 0.191 | 0.464 | 0.482 | 0.085 | 0.944 | 13.1 | — | 0.829 | 0.310 |
| P8 | all | 105 | 0.198 | 0.481 | 0.508 | 0.136 | 0.958 | 11.5 | 0.403 | 0.741 | 0.347 |

### 0.2 Comprehension 检索（105 题）

| graph | relations | R@1 | R@3 | R@5 | MRR | nDCG@10 | Ev@4096 | fallback |
|---|---|---|---|---|---|---|---|---|
| C2 | 0 | 0.5944 | 0.8413 | 0.9032 | 0.7586 | 0.8026 | 0.8952 | 16 |
| G_raw | 0 | 0.6198 | 0.8238 | 0.8984 | 0.7619 | 0.8067 | 0.8857 | 14 |
| cmpo_compp1 | 14 | 0.5921 | 0.8532 | 0.9103 | 0.7547 | 0.8034 | 0.8952 | 15 |
| cmpo_compp2 | 19 | 0.5944 | 0.8317 | 0.9079 | 0.7598 | 0.8064 | 0.8952 | 16 |
| cmpo_compp3 | 27 | 0.5944 | 0.8460 | 0.9079 | 0.7579 | 0.8043 | 0.8952 | 15 |
| cmpo_compp4 | 16 | 0.5944 | 0.8508 | 0.9127 | 0.7638 | 0.8100 | 0.9048 | 15 |
| cmpo_compp4c10 | 10 | 0.6040 | 0.8508 | 0.9127 | 0.7665 | 0.8101 | 0.9048 | 15 |
| cmpo_compp5 | 7 | 0.5944 | 0.8413 | 0.9032 | 0.7582 | 0.8022 | 0.8952 | 16 |

### 0.3 无 LLM rerank 消融（105 题）

| graph | R@3 | R@5 | MRR | nDCG@10 | Ev@4096 | fallback |
|---|---|---|---|---|---|---|
| G_raw_top5 | 0.8111 | 0.8865 | 0.7173 | 0.7769 | 0.8762 | 0 |
| G_raw_top20 | 0.6087 | 0.7437 | 0.5431 | 0.6211 | 0.7619 | 0 |
| C2_top5 | 0.8405 | 0.9103 | 0.7164 | 0.7732 | 0.8952 | 0 |
| C2_top20 | 0.7190 | 0.8127 | 0.6225 | 0.6913 | 0.8095 | 0 |
| COMP-P4_top5 | 0.8500 | 0.9294 | 0.7224 | 0.7797 | 0.9143 | 0 |
| COMP-P4_top20 | 0.7214 | 0.8056 | 0.6303 | 0.6991 | 0.7905 | 0 |

### 0.4 联合 QA（105 题）

| graph\|reader | n | EM | F1 | offF1 | refusal | adv |
|---|---|---|---|---|---|---|
| cmpo_compp4 \| P0 | 105 | 0.210 | 0.387 | 0.425 | 0.198 | 0.958 |
| cmpo_compp4 \| P7 | 105 | 0.247 | 0.524 | 0.539 | 0.086 | 0.917 |
| C2 \| P7 | 105 | 0.235 | 0.515 | 0.530 | 0.099 | 0.917 |
| G_raw_top5 \| P0 | 105 | 0.185 | 0.388 | 0.418 | 0.222 | 0.917 |
| G_raw_top5 \| P7 | 105 | 0.235 | 0.531 | 0.569 | 0.062 | 0.875 |
| COMP-P4_top5 \| P0 | 105 | 0.222 | 0.417 | 0.445 | 0.173 | 0.958 |
| COMP-P4_top5 \| P7 | 105 | 0.259 | 0.539 | 0.569 | 0.062 | 0.917 |

### 0.5 Oracle / 重复 / 官方 cat-5（汇总数字）

| condition | EM | F1 | offF1 | refusal | ctx tokens |
|---|---|---|---|---|---|
| Oracle R2_P0 | 0.185 | 0.387 | 0.402 | 0.173 | 761 |
| Oracle R2_P7 | 0.272 | 0.603 | 0.627 | 0.062 | 761 |
| Oracle R3_P7 | 0.259 | 0.546 | 0.577 | 0.049 | 11796 |
| Oracle R3_P0 | 0.198 | 0.360 | 0.400 | 0.173 | 11435 |

| run | wording unstable | scored unstable | Jaccard | EM/run | majority EM |
|---|---|---|---|---|---|
| repeat P0 | 0.143 | 0.143 | 0.976 | [0.214, 0.286, 0.357] | 0.286 |
| repeat P7 | 0.286 | 0.000 | 0.886 | [0.143, 0.143, 0.143] | 0.143 |

| condition | accuracy | n |
|---|---|---|
| official cat5 P0 | 0.833 | 24 |
| official cat5 P7 | 0.875 | 24 |
| official cat5 OFFICIAL | 0.875 | 24 |

## 附录 A：所有 Prompt 原文

### A.1 Reader Prompt P0–P8

#### P0 — Baseline (current project prompt)

*notes*: exact copy of src/retrieval/reader.py; saved baseline outputs are reused, no re-call

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

#### P1 — Concise but complete

*notes*: length constraint is softened: complete before concise

**System**

```text
You answer questions about a long conversation between two people. Use only the provided conversation evidence.
```

**User template**

```text
Conversation evidence:
{context}

Question: {question}

Answer the question accurately and completely.

Rules:
1. Identify exactly what the question asks.
2. Include all necessary facts supported by the evidence.
3. Be concise, but never omit required information just to make the answer shorter.
4. For list questions, include every supported item needed to answer.
5. For why questions, include the relevant causes or motivations.
6. For date questions, give the precise date when it is supported.
7. Preserve names, dates, places, and other factual details.
8. Do not invent or assume missing information.
9. If the evidence is insufficient, answer "No answer".

Write exactly one line of the form:
Answer: <answer>

```

#### P2 — Integrate evidence before answering

*notes*: forces cross-session aggregation internally, no visible chain of thought

**System**

```text
You are a careful reading-comprehension assistant. You answer from retrieved conversation evidence and never invent facts.
```

**User template**

```text
Retrieved conversation sessions:
{context}

Question: {question}

Before answering, silently do the following:
1. Identify the people, events, and facts relevant to the question.
2. Collect evidence across all relevant sessions, not only the first one that looks related.
3. Resolve references to the same person, event, or object across sessions.
4. Combine multiple facts when the question requires more than one.
5. Check that the evidence supports the complete answer.

Answer requirements:
- Include every necessary part of the answer.
- Do not omit supporting causes, events, or entities.
- Prefer exact information from the conversation.
- Do not use unsupported assumptions.
- If the information is insufficient, answer "No answer".

Do not output your reasoning. Write exactly one line:
Answer: <answer>

```

#### P3 — Temporal and event-state sensitive

*notes*: explicit session-date anchoring; duration arithmetic

**System**

```text
You answer questions about conversations that take place across multiple dates and sessions. Each session header gives the date of that conversation. Use only the provided evidence and never fabricate a date or duration.
```

**User template**

```text
Conversation evidence (each session starts with its date):
{context}

Question: {question}

Rules:
1. Distinguish the date of a conversation from the date of an event mentioned in it.
2. Resolve relative expressions such as yesterday, last week, next month, or six months
   later only when a valid temporal anchor is available.
3. For duration questions, identify the start and end events before computing the interval.
4. For event-order questions, distinguish planned, ongoing, completed, and cancelled events.
5. If several dates appear, choose the one that corresponds to the event asked about.
6. Never fabricate a date or duration.
7. For non-temporal questions, answer normally and completely.

Write exactly one line:
Answer: <answer>

```

#### P4 — Evidence-sufficiency and refusal calibration

*notes*: lowers over-refusal; adds explicit permission for supported cross-session derivation

**System**

```text
You answer questions using retrieved conversation evidence. You may combine explicitly supported facts across sessions, including event order and simple date arithmetic, but you never invent facts.
```

**User template**

```text
Conversation evidence:
{context}

Question: {question}

First decide silently whether the available evidence is sufficient.

- You may combine explicitly supported facts across sessions, including event order and
  simple date calculations.
- Do not invent facts, dates, motivations, or events that the evidence does not support.
- If the necessary evidence is present, answer the question instead of refusing.
- If a required fact is missing and cannot be reliably derived, answer "No answer".

Keep the answer concise but complete. Write exactly one line:
Answer: <answer>

```

#### P5 — Adapt answer form to the question

*notes*: question-type-conditioned completeness; question text only, no gold category

**System**

```text
You answer questions based on retrieved long-term conversation records. You use only the provided evidence.
```

**User template**

```text
Conversation evidence:
{context}

Question: {question}

Decide what kind of information the question requires, then give the shortest complete answer.

- Who: the correct person or people.
- When: the exact supported date or time.
- How long: the duration computed from supported dates.
- Where: the relevant location or locations.
- Which events/items: all relevant items, enumerated.
- Why: all necessary causes or motivations.
- How: the relevant outcome or process.
- Other: the requested fact.

Use evidence from multiple sessions when necessary.
Preserve distinctions between different people, dates, and events.
Do not omit required facts just to shorten the answer.
Do not add unnecessary explanation.
If the information is insufficient, answer "No answer".

Write exactly one line:
Answer: <answer>

```

#### P6 — Official-LoCoMo-style short phrase

*notes*: mirrors official QA_PROMPT text; generic date hint instead of gold category

**System**

```text
You answer questions based on the provided conversation context. Answer with exact words from the context whenever possible.
```

**User template**

```text
{context}

Based on the above context, write an answer in the form of a short phrase for the following question.

Question: {question}

If the question asks for a date, use the date of the conversation to answer with an approximate date.
If the context does not contain the answer, write "Not mentioned in the conversation".

Short answer:
```

#### P7 — Short-span complete (synthesis of P0/P3/P6)

*notes*: dev-informed synthesis; P7 was added after seeing P0-P6 on the 40 dev questions

**System**

```text
You are a careful reading-comprehension assistant answering questions about a long conversation between two speakers. Use only the retrieved context and answer with exact words from the context whenever possible. Each session header gives the date of that conversation. If the context truly does not contain the answer, reply exactly: No answer
```

**User template**

```text
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

#### P8 — Short-span complete + presupposition check

*notes*: P7 plus a generic presupposition check for adversarial questions

**System**

```text
You are a careful reading-comprehension assistant answering questions about a long conversation between two speakers. Use only the retrieved context and answer with exact words from the context whenever possible. Each session header gives the date of that conversation. If the context truly does not contain the answer, reply exactly: No answer
```

**User template**

```text
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
7. Check the premise of the question. If the question assumes that a person did, attended,
   participated in, or possesses something that the context does not state, write "No answer"
   instead of answering a different question.
8. Never invent facts or dates.

Answer:
```

### A.2 Comprehension Prompt COMP-P0–P5

#### COMP-P0 — Original comprehension prompt

**System**

```text
You are a careful analyst building a cross-document knowledge graph from a long conversation. You only state relations that the cited dialogue turns actually support. You output only valid JSON.
```

**Instructions**

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

#### COMP-P1 — Evidence-grounded composition

**System**

```text
You are a careful analyst building a cross-document knowledge graph from a long conversation. You only state relations that the cited dialogue turns actually support. You output only valid JSON.
```

**Instructions**

```text
You are reading multiple conversation sessions to construct new, evidence-grounded
knowledge relations. Your task is NOT to repeat individual facts.

Identify relations that emerge only when evidence from at least two different sessions is
combined. A relation is valuable only if a reader of a single session could not state it
without the other session.

Prioritise:
- goal -> action -> outcome connections
- causal relations that the dialogue explicitly supports
- changes of state (before state vs after state)
- enabling links between two documented events
- identity resolution of one specific event or object mentioned twice
- event dependencies that are not already implied by the session dates

For every proposed relation:
1. Identify the supporting facts (use the f-ids from the atomic fact list when possible).
2. Cite dialogue ids from at least two different sessions.
3. Make sure the relation is not already expressed by one atomic fact.
4. Reject unsupported or redundant relations.

Do not invent motivations, causality, dates, or identity links. If no useful new relation
exists, return an empty list.

Hard requirements:
1. At least two source_dia_ids from DIFFERENT sessions.
2. evidence: one short verbatim quote (max 25 words) per cited session.
3. subject and object are short canonical noun phrases (<= 6 words): a person, place,
   organisation, event, object or concrete state. Do NOT put a whole clause in the subject
   or object.
4. predicate is a short verb or verb phrase (1-4 words, e.g. "led_to", "enabled",
   "opened_after", "became"). Do NOT use long compound predicates and do NOT encode
   "before"/"after" into the predicate unless you are explicitly reporting event order.
5. relation_type is one of: causal, enabling_link, state_change, goal_outcome,
   event_dependency, identity_unification, temporal_ordering. Use temporal_ordering only
   when the two events are separately described and their order is NOT already obvious
   from the session dates; it is the least valuable relation type here.
6. derivation is "Composed" when combining explicit facts needs no world knowledge and
   "Inferred" when common-sense inference is needed.
7. confidence is your calibrated probability (0-1).
8. Do NOT restate, paraphrase, or synonymise an existing atomic fact.
9. If nothing qualifies, return {"relations": []}.

Return JSON of exactly this shape:
{"relations": [
  {"subject": "Jon", "predicate": "attended_fair_to_promote", "object": "Jon's dance studio",
   "relation_type": "enabling_link", "source_dia_ids": ["D10:1", "D15:2"],
   "supporting_fact_ids": ["f12", "f40"],
   "evidence": ["Yesterday, I went to a fair to show off my studio",
                "Still working on opening a dance studio."],
   "derivation": "Composed", "confidence": 0.9}
]}

```

#### COMP-P2 — Cross-session bridge discovery

**System**

```text
You are a careful analyst building a cross-document knowledge graph from a long conversation. You only state relations that the cited dialogue turns actually support. You output only valid JSON.
```

**Instructions**

```text
Read the given conversation sessions as a connected history, not as independent
documents.

First identify the important entities, events, and states across the sessions. Then look for
relations that connect information distributed across DIFFERENT sessions. A relation is
worth adding only when the connection itself is not written in any single session.

Prefer bridges of these kinds:
- a plan stated in one session and its outcome in a later session
- an event in one session and its documented consequence in another
- the state of a business, project, or relationship before and after a major change
- two mentions that clearly refer to the same specific event or object
- evidence in one session that explains why a later action happened
- a sequence of separate facts that only together describe what happened

Create a relation only when the source evidence supports the connection itself. A shared
person name is not a relation. Do not repeat existing facts. Use specific, unambiguous
names for people, events, and objects. Return no relation when the evidence is insufficient.

Hard requirements:
1. Cite dialogue ids from at least two different sessions; every cited id must appear in
   the excerpts.
2. evidence: short verbatim quotes from each cited session (<= 25 words each).
3. subject/object: named people, organisations, places, events or concrete states (<= 6
   words). The predicate is a short verb phrase (1-4 words).
4. relation_type is one of: plan_outcome, event_consequence, state_change, identity_unification,
   causal, enabling_link, event_sequence. Do NOT use temporal_ordering as a fallback when you
   cannot find a real bridge.
5. Do NOT generate a relation merely because two statements mention the same person.
6. derivation is "Composed" or "Inferred"; confidence is 0-1.
7. Return {"relations": []} if no cross-session bridge is supported.

Return JSON of exactly this shape:
{"relations": [
  {"subject": "Jon", "predicate": "attended_fair_to_promote", "object": "Jon's dance studio",
   "relation_type": "enabling_link", "source_dia_ids": ["D10:1", "D15:2"],
   "supporting_fact_ids": ["f12", "f40"],
   "evidence": ["Yesterday, I went to a fair to show off my studio",
                "Still working on opening a dance studio."],
   "derivation": "Composed", "confidence": 0.9}
]}

```

#### COMP-P3 — Event and state tracking

**System**

```text
You are a careful analyst building a cross-document knowledge graph from a long conversation. You only state relations that the cited dialogue turns actually support. You output only valid JSON.
```

**Instructions**

```text
Analyse the conversation as a timeline of entities, events, goals, and changing states.

First identify the important people, businesses, projects, events, and goals. Then track
across sessions:
- initial goals
- planned actions
- ongoing activities
- completed outcomes
- changes in status
- changes in relationships
- changes in plans

For each important state transition:
1. Identify the earlier state.
2. Identify the later state.
3. Find the exact supporting sessions.
4. Decide whether a meaningful new relation can be derived from the change.
5. Reject relations that rely on unsupported assumptions about cause or intention.

Generate concise, evidence-grounded relations that explicitly describe the change. Prefer a
state transition over a vague ordering. Preserve event dates when they are stated.

Hard requirements:
1. Cite dialogue ids from at least two different sessions.
2. evidence: verbatim quotes from the supporting sessions (<= 25 words each).
3. subject/object are short canonical noun phrases (<= 6 words) naming the entity, project,
   event, or state. The predicate is a short verb phrase (1-4 words) describing the
   transition (e.g. "expanded_to", "changed_from", "became", "led_to", "recovered_from").
4. relation_type is one of: state_transition, goal_change, plan_outcome, status_change,
   relationship_change, causal. Prefer these over temporal_ordering.
5. If two events happened in a certain order but there is no evidence that one caused or
   enabled the other, report only the documented state change, not a causal claim.
6. Do not restate existing facts. confidence is 0-1.
7. Return {"relations": []} if there is no supported state change.

Return JSON of exactly this shape:
{"relations": [
  {"subject": "Jon", "predicate": "attended_fair_to_promote", "object": "Jon's dance studio",
   "relation_type": "enabling_link", "source_dia_ids": ["D10:1", "D15:2"],
   "supporting_fact_ids": ["f12", "f40"],
   "evidence": ["Yesterday, I went to a fair to show off my studio",
                "Still working on opening a dance studio."],
   "derivation": "Composed", "confidence": 0.9}
]}

```

#### COMP-P4 — Retrieval-oriented comprehension

**System**

```text
You are a careful analyst building a cross-document knowledge graph from a long conversation. You only state relations that the cited dialogue turns actually support. You output only valid JSON.
```

**Instructions**

```text
You are constructing a reusable comprehension graph for future questions that are not
known in advance. Read the provided conversation sessions and identify important relations
that are missing from independent fact extraction.

A useful relation should:
1. represent a distinct, evidence-grounded fact or cross-session inference;
2. connect specific entities, events, or states;
3. preserve enough semantic detail to be found by a future natural-language query;
4. point back to the original supporting sessions;
5. avoid vague predicates and generic links;
6. avoid repetition of existing facts.

Think about how a future question would be phrased. The subject and object should be names
that a question could plausibly mention (a person, business, event, place, project, or
state), and the predicate should state a specific relation between them. Do NOT hide the
whole answer inside a long predicate.

Do not invent relations to increase graph size. Do not use any question-answer pairs or gold
labels (none are provided). If the context does not support a valuable new relation, return
an empty list.

Hard requirements:
1. At least two source_dia_ids from DIFFERENT sessions.
2. subject/object: short, specific, human-readable noun phrases (<= 6 words). Prefer names
   that already occur in the atomic fact list, so the relation attaches to existing graph
   entities.
3. predicate: a short, specific verb phrase (1-4 words), lower-case with underscores
   (e.g. "promoted", "enabled_opening_of", "expanded_into", "resulted_in"). Avoid
   "related_to", "associated_with", and any predicate longer than 4 words.
4. evidence: short verbatim quote (<= 25 words) per cited session.
5. relation_type is one of: causal, enabling_link, state_change, goal_outcome,
   event_dependency, identity_unification, temporal_ordering. Use temporal_ordering only
   when order is the actual new information and is not implied by session dates.
6. derivation is "Composed" or "Inferred"; confidence is 0-1.
7. Do NOT output a relation whose text is already an atomic fact, and do not output two
   relations that say the same thing with different words.
8. Return {"relations": []} if nothing qualifies.

Return JSON of exactly this shape:
{"relations": [
  {"subject": "Jon", "predicate": "attended_fair_to_promote", "object": "Jon's dance studio",
   "relation_type": "enabling_link", "source_dia_ids": ["D10:1", "D15:2"],
   "supporting_fact_ids": ["f12", "f40"],
   "evidence": ["Yesterday, I went to a fair to show off my studio",
                "Still working on opening a dance studio."],
   "derivation": "Composed", "confidence": 0.9}
]}

```

#### COMP-P5 — Generate then verify

**System**

```text
You are a careful analyst building a cross-document knowledge graph from a long conversation. You only state relations that the cited dialogue turns actually support. You output only valid JSON.
```

**Instructions**

```text
You are constructing a reusable comprehension graph for future questions that are not
known in advance. Read the provided conversation sessions and propose candidate relations
that are missing from independent fact extraction.

Generate up to 12 candidates per reading group. Candidates are cheap; a separate verification
step will reject weak ones. Still, every candidate must be grounded in the supplied sessions.

A useful candidate:
1. combines evidence from at least two different sessions;
2. connects specific entities, events, projects, or states;
3. uses a short subject/object noun phrase and a short specific verb predicate;
4. is not already stated by an atomic fact.

Prefer causal links, enabling links, goal -> outcome chains, state changes, and identity
unification. Use temporal_ordering only when the order is the actual new information.

Do not invent facts. Do not use question-answer pairs or gold labels.

Hard requirements:
1. At least two source_dia_ids from DIFFERENT sessions.
2. subject/object: short specific noun phrases (<= 6 words), preferably names already in the
   atomic fact list.
3. predicate: short specific verb phrase (1-4 words), lower-case with underscores.
4. evidence: short verbatim quote per cited session.
5. relation_type: causal, enabling_link, state_change, goal_outcome, event_dependency,
   identity_unification, or temporal_ordering.
6. derivation is "Composed" or "Inferred"; confidence is 0-1.
7. Return {"relations": []} if nothing qualifies.

Return JSON of exactly this shape:
{"relations": [
  {"subject": "Jon", "predicate": "attended_fair_to_promote", "object": "Jon's dance studio",
   "relation_type": "enabling_link", "source_dia_ids": ["D10:1", "D15:2"],
   "supporting_fact_ids": ["f12", "f40"],
   "evidence": ["Yesterday, I went to a fair to show off my studio",
                "Still working on opening a dance studio."],
   "derivation": "Composed", "confidence": 0.9}
]}

```

### A.3 官方 LoCoMo QA Prompt（原文对照）

```text
# 普通问题（cat 1/2/3/4）
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:

# cat-5 对抗题
Based on the above context, answer the following question.

Question: {} Short answer:
```

官方在 cat-2 的 question 后追加：`Use DATE of CONVERSATION to answer with an approximate date.`；
cat-5 则改写成 `(a) Not mentioned in the conversation (b) <诱导答案>` 的二选一。

## 附录 B：105 题总表（问题、金标、答案与 EM/F1）

列说明：E0=G_raw+P0，E1=G_raw+P7，E2=P4+P0，E3=P4+P7；no-rerank 两列是 COMP-P4+no-rerank top5 配 P0/P7。答案过长时截断到 90 字符。

| qid | cat | question | gold | E0 ans | E0 EM/F1 | E1 ans | E1 EM/F1 | E2 ans | E2 EM/F1 | E3 ans | E3 EM/F1 | P4nr+P0 ans | EM/F1 | P4nr+P7 ans | EM/F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q0 | 2 | When Jon has lost his job as a banker? | 19 January, 2023 | 19 January, 2023 | 1/1.00 | 19 January, 2023 | 1/1.00 | yesterday | 0/0.00 | 19 January, 2023 | 1/1.00 | 19 January, 2023 | 1/1.00 | 19 January, 2023 | 1/1.00 |
| q1 | 2 | When Gina has lost her job at Door Dash? | January, 2023 | this month | 0/0.00 | 20 January, 2023 | 0/0.80 | this month | 0/0.00 | 20 January, 2023 | 0/0.80 | this month | 0/0.00 | 20 January, 2023 | 0/0.80 |
| q2 | 4 | How do Jon and Gina both like to destress? | by dancing | dance | 0/0.00 | Dance | 0/0.00 | dance | 0/0.00 | Dance, Dancing | 0/0.50 | dancing | 0/0.67 | Dance, Dancing | 0/0.50 |
| q3 | 1 | What do Jon and Gina both have in common? | They lost their jobs and decided to start their own businesses. | both face the same challenges | 0/0.00 | running their own biz, losing my job, passionate about dance, chasing our dreams, support… | 0/0.13 | both lost their jobs and started their own businesses | 0/0.70 | starting a business, running their own biz, losing my job, chasing our dreams, supporting… | 0/0.03 | both lost their jobs and started their own businesses | 0/0.70 | starting a business, running their own biz, losing my job, chasing our dreams, … | 0/0.03 |
| q4 | 4 | Why did Jon decide to start his dance studio? | He lost his job and decided to start his own business to share his pa… | Losing my job gave me the push to finally start my dream business: my own dance studio | 0/0.32 | Losing my job gave me the push to finally start my dream business: my own dance studio, L… | 0/0.20 | he's passionate about dancing and it'd be great to share it with others | 0/0.21 | 'cause I'm passionate about dancing and it'd be great to share it with others, I've been … | 0/0.15 | because he is passionate about dancing and wanted to share it with others | 0/0.29 | 'cause I'm passionate about dancing and it'd be great to share it with others, … | 0/0.15 |
| q5 | 1 | What Jon thinks the ideal dance studio should look like? | By the water, with natural light and Marley flooring | by the water | 0/0.40 | by the water, downtown, great natural light, good dance floor with enough bounce, Marley … | 0/0.56 | by the water | 0/0.40 | by the water, downtown, great natural light, good dance floor with enough bounce, Marley … | 0/0.64 | by the water | 0/0.40 | by the water, downtown, great natural light, good dance floor with enough bounc… | 0/0.56 |
| q6 | 2 | When is Jon's group performing at a festival? | February, 2023 | next month | 0/0.00 | 20 February, 2023 | 0/0.80 | next month | 0/0.00 | 20 February, 2023 | 0/0.80 | next month | 0/0.00 | 20 February, 2023 | 0/0.80 |
| q7 | 2 | When did Gina launch an ad campaign for her store? | 29 January, 2023 | No answer | 0/0.00 | 29 January, 2023 | 1/1.00 | No answer | 0/0.00 | 29 January, 2023 | 1/1.00 | No answer | 0/0.00 | 29 January, 2023 | 1/1.00 |
| q8 | 2 | When was Jon in Paris? | 28 January 2023 | 28 January, 2023 | 1/1.00 | 28 January, 2023 | 1/1.00 | 28 January, 2023 | 1/1.00 | 28 January, 2023 | 1/1.00 | 28 January, 2023 | 1/1.00 | 28 January, 2023 | 1/1.00 |
| q9 | 1 | Which city have both Jean and John visited? | Rome | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 |
| q10 | 2 | When did Gina team up with a local artist for some cool designs? | February, 2023 | No answer | 0/0.00 | 8 February, 2023 | 0/0.80 | 8 February, 2023 | 0/0.80 | 8 February, 2023 | 0/0.80 | 8 February, 2023 | 0/0.80 | 8 February, 2023 | 0/0.80 |
| q11 | 2 | When did Gina get her tattoo? | A few years ago | a few years ago | 1/1.00 | a few years ago | 1/1.00 | No answer | 0/0.00 | No answer | 0/0.00 | a few years ago | 1/1.00 | a few years ago | 1/1.00 |
| q12 | 2 | When did Jon start to go to the gym? | March, 2023 | last week | 0/0.00 | 9 March, 2023 | 0/0.80 | last week | 0/0.00 | 9 March, 2023 | 0/0.80 | last week | 0/0.00 | 9 March, 2023 | 0/0.80 |
| q13 | 2 | When did Gina open her online clothing store? | 16 March, 2023 | No answer | 0/0.00 | 16 March, 2023 | 1/1.00 | No answer | 0/0.00 | 16 March, 2023 | 1/1.00 | No answer | 0/0.00 | 16 March, 2023 | 1/1.00 |
| q14 | 2 | When did Jon start expanding his studio's social media presence? | April, 2023 | 3 April, 2023 | 0/0.80 | 3 April, 2023 | 0/0.80 | No answer | 0/0.00 | 3 April, 2023 | 0/0.80 | No answer | 0/0.00 | 3 April, 2023 | 0/0.80 |
| q15 | 2 | When did Jon host a dance competition? | May, 2023 | No answer | 0/0.00 | 3 May, 2023 | 0/0.80 | No answer | 0/0.00 | 3 May, 2023 | 0/0.80 | No answer | 0/0.00 | 3 May, 2023 | 0/0.80 |
| q16 | 2 | When did Jon go to a fair to get more exposure for his dance studio? | 24 April, 2023 | No answer | 0/0.00 | No answer | 0/0.00 | 24 April, 2023 | 1/1.00 | 24 April, 2023 | 1/1.00 | 24 April, 2023 | 1/1.00 | 24 April, 2023 | 1/1.00 |
| q17 | 1 | Why did Gina decide to start her own clothing store? | She always loved fashion trends and finding unique pieces and she los… | After losing my job, I wanted to take control of my own destiny | 0/0.16 | After losing my job, I wanted to take control of my own destiny, I've been one 'cause I l… | 0/0.18 | After losing her job, she wanted to take control of her own destiny | 0/0.32 | After losing my job, I wanted to take control of my own destiny, I'm passionate about fas… | 0/0.36 | After losing her job, she wanted to take control of her own destiny | 0/0.32 | After losing my job, I wanted to take control of my own destiny, I'm passionate… | 0/0.36 |
| q18 | 1 | Do Jon and Gina start businesses out of what they love? | Yes | Yes | 1/1.00 | Yes | 1/1.00 | Yes | 1/1.00 | Yes | 1/1.00 | Yes | 1/1.00 | Yes | 1/1.00 |
| q19 | 2 | When did Gina interview for a design internship? | 10 May, 2023 | yesterday | 0/0.00 | 10 May, 2023 | 1/1.00 | yesterday | 0/0.00 | 10 May, 2023 | 1/1.00 | 10 May, 2023 | 1/1.00 | 10 May, 2023 | 1/1.00 |
| q20 | 2 | When did Gina get accepted for the design internship? | 27 May, 2023 | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 |
| q21 | 2 | When did Jon start reading "The Lean Startup"? | May, 2023 | No answer | 0/0.00 | 27 May, 2023 | 0/0.80 | No answer | 0/0.00 | 27 May, 2023 | 0/0.80 | No answer | 0/0.00 | 27 May, 2023 | 0/0.80 |
| q22 | 2 | When did Gina develop a video presentation to teach how to style her fashion pieces? | June, 2023 | No answer | 0/0.00 | 13 June, 2023 | 0/0.80 | No answer | 0/0.00 | 13 June, 2023 | 0/0.80 | No answer | 0/0.00 | 13 June, 2023 | 0/0.80 |
| q23 | 1 | How did Gina promote her clothes store? | worked with an artist to make unique fashion pieces, made limited-edi… | launched an ad campaign, new offers and promotions, working with fashion bloggers and inf… | 0/0.27 | launched an ad campaign for my clothing store, new offers and promotions going on my onli… | 0/0.34 | launched an ad campaign, new offers and promotions, working with fashion bloggers and inf… | 0/0.27 | launched an ad campaign for my clothing store, new offers and promotions going on my onli… | 0/0.34 | new offers and promotions, working with fashion bloggers and influencers, and m… | 0/0.29 | new offers and promotions, working with some fashion bloggers and influencers, … | 0/0.29 |
| q24 | 1 | Which events has Jon participated in to promote his business venture? | fair, networking events, dance competition | a fair, networking events, a dance competition | 1/1.00 | networking events, a fair, the last networking event | 0/0.55 | a fair, networking events, a dance competition | 1/1.00 | networking events, a fair, the last networking event | 0/0.55 | a fair, networking events | 0/0.75 | networking events, a fair, the last networking event | 0/0.55 |
| q25 | 1 | What does Jon's dance studio offer? | one-on-one metoring and training to dancers,  workshops and classes t… | dance classes and workshops, one-on-one mentoring and training | 0/0.55 | dance classes and workshops, one-on-one mentoring and training | 0/0.55 | dance classes and workshops, one-on-one mentoring and training | 0/0.55 | dance classes and workshops, one-on-one mentoring and training | 0/0.55 | dance classes and workshops, one-on-one mentoring and training | 0/0.55 | dance classes and workshops, one-on-one mentoring and training | 0/0.55 |
| q26 | 2 | When did Jon receive mentorship to promote his venture? | 15 June, 2023 | 15 June, 2023 | 1/1.00 | 15 June, 2023 | 1/1.00 | yesterday | 0/0.00 | 15 June, 2023 | 1/1.00 | yesterday | 0/0.00 | 15 June, 2023 | 1/1.00 |
| q27 | 1 | Did Jon and Gina both participate in dance competitions? | Yes | Yes | 1/1.00 | Yes | 1/1.00 | Yes | 1/1.00 | Yes | 1/1.00 | Yes | 1/1.00 | Yes | 1/1.00 |
| q28 | 2 | When was Jon in Rome? | June 2023 | last week | 0/0.00 | 12 June, 2023 | 0/0.80 | last week | 0/0.00 | 12 June, 2023 | 0/0.80 | last week | 0/0.00 | 12 June, 2023 | 0/0.80 |
| q29 | 1 | Which cities has Jon visited? | Paris, Rome | Paris, Rome | 1/1.00 | Paris, Rome | 1/1.00 | Paris, Rome | 1/1.00 | Paris, Rome | 1/1.00 | Paris, Rome | 1/1.00 | Paris, Rome | 1/1.00 |
| q30 | 2 | When Jon is planning to open his dance studio? | 20 June, 2023 | tomorrow | 0/0.00 | 20 June, 2023 | 1/1.00 | 20 June, 2023 | 1/1.00 | 20 June, 2023 | 1/1.00 | tomorrow | 0/0.00 | 20 June, 2023 | 1/1.00 |
| q31 | 1 | How long did it take for Jon to open his studio? | six months | No answer | 0/0.00 | 4 months and 18 days | 0/0.29 | No answer | 0/0.00 | 2 months and 10 days | 0/0.29 | No answer | 0/0.00 | 2 months and 10 days | 0/0.29 |
| q32 | 2 | When did Gina design a limited collection of hoodies? | June 2023 | last week | 0/0.00 | 14 June, 2023 | 0/0.80 | last week | 0/0.00 | 14 June, 2023 | 0/0.80 | last week | 0/0.00 | 14 June, 2023 | 0/0.80 |
| q33 | 2 | When did Jon visit networking events for his store? | 20 June, 2023 | No answer | 0/0.00 | 21 June, 2023 | 0/0.67 | No answer | 0/0.00 | 20 June, 2023 | 1/1.00 | No answer | 0/0.00 | 20 June, 2023 | 1/1.00 |
| q34 | 2 | When did Gina start being recognized by fashion editors? | July 2023 | No answer | 0/0.00 | 2 July, 2023 | 0/0.80 | No answer | 0/0.00 | 9 July, 2023 | 0/0.80 | last week | 0/0.00 | 9 July, 2023 | 0/0.80 |
| q35 | 2 | When did Jon start learning marketing and analytics tools? | July, 2023 | 9 July, 2023 | 0/0.80 | 9 July, 2023 | 0/0.80 | 9 July, 2023 | 0/0.80 | 9 July, 2023 | 0/0.80 | 9 July, 2023 | 0/0.80 | 9 July, 2023 | 0/0.80 |
| q36 | 2 | When did Jon and Gina decide to collaborate to create dance content? | 21 July 2023 | 21 July, 2023 | 1/1.00 | 21 July, 2023 | 1/1.00 | 21 July, 2023 | 1/1.00 | 21 July, 2023 | 1/1.00 | 21 July, 2023 | 1/1.00 | 21 July, 2023 | 1/1.00 |
| q37 | 2 | When did Gina mention Shia Labeouf? | 23 July, 2023 | 23 July, 2023 | 1/1.00 | 23 July, 2023 | 1/1.00 | 23 July, 2023 | 1/1.00 | 23 July, 2023 | 1/1.00 | 23 July, 2023 | 1/1.00 | 23 July, 2023 | 1/1.00 |
| q38 | 2 | When did Gina go to a dance class with a group of friends? | 21 July 2023 | Last Friday | 0/0.00 | 21 July, 2023 | 1/1.00 | Last Friday | 0/0.00 | 21 July, 2023 | 1/1.00 | Last Friday | 0/0.00 | 21 July, 2023 | 1/1.00 |
| q39 | 4 | What is Gina's favorite style of dance? | Contemporary | contemporary dance | 0/0.67 | contemporary | 1/1.00 | contemporary | 1/1.00 | contemporary | 1/1.00 | contemporary | 1/1.00 | contemporary | 1/1.00 |
| q40 | 4 | What is Jon's favorite style of dance? | Contemporary | contemporary | 1/1.00 | contemporary | 1/1.00 | contemporary | 1/1.00 | contemporary | 1/1.00 | contemporary | 1/1.00 | contemporary | 1/1.00 |
| q41 | 4 | What was Gina's favorite dancing memory? | Winning first place at a regionals dance competition | when my team won first place at a regionals at age fifteen | 0/0.44 | when my team won first place at a regionals at age fifteen | 0/0.44 | when my team won first place at a regionals at age fifteen | 0/0.44 | when my team won first place at a regionals at age fifteen | 0/0.44 | when my team won first place at a regionals at age fifteen | 0/0.44 | when my team won first place at a regionals at age fifteen | 0/0.44 |
| q42 | 4 | What kind of dance piece did Gina's team perform to win first place? | "Finding Freedom" | a contemporary piece called "Finding Freedom" | 0/0.57 | a contemporary piece called "Finding Freedom." | 0/0.57 | a contemporary piece called "Finding Freedom" | 0/0.57 | a contemporary piece called "Finding Freedom." | 0/0.57 | a contemporary piece called "Finding Freedom" | 0/0.57 | a contemporary piece called "Finding Freedom." | 0/0.57 |
| q43 | 4 | What do the dancers in the photo represent? | They are performing at the festival | the ones performing at the festival | 0/0.67 | the ones performing at the festival | 0/0.67 | the ones performing at the festival | 0/0.67 | the ones performing at the festival | 0/0.67 | the ones performing at the festival | 0/0.67 | the ones performing at the festival | 0/0.67 |
| q44 | 4 | What does Gina say about the dancers in the photo? | They look graceful | they're so graceful | 0/0.33 | Wow, they look great! Can't wait to see them rock the festival. Gonna be awesome! | 0/0.24 | You guys look great and passionate about dancing | 0/0.18 | Wow, they look great! Can't wait to see them rock the festival. Gonna be awesome! | 0/0.24 | they're so graceful | 0/0.33 | Wow, they look great! Can't wait to see them rock the festival. Gonna be awesom… | 0/0.24 |
| q45 | 4 | What is Jon's attitude towards being part of the dance festival? | Glad | Glad to be part of it | 0/0.29 | Glad to be part of it | 0/0.29 | Glad to be part of it | 0/0.29 | Glad to be part of it | 0/0.29 | excited | 0/0.00 | Glad to be part of it | 0/0.29 |
| q46 | 4 | What kind of flooring is Jon looking for in his dance studio? | Marley flooring | Marley flooring | 1/1.00 | Marley flooring | 1/1.00 | Marley flooring | 1/1.00 | Marley flooring | 1/1.00 | Marley flooring | 1/1.00 | Marley flooring | 1/1.00 |
| q47 | 4 | What did Gina find for her clothing store on 1 February, 2023? | The perfect spot for her store | a perfect spot for your store | 0/0.80 | the perfect spot for your store | 0/0.80 | the perfect spot for your store | 0/0.80 | the perfect spot for your store | 0/0.80 | the perfect spot for your store | 0/0.80 | the perfect spot for your store | 0/0.80 |
| q48 | 4 | What did Gina design for her store? | the space, furniture, and decor | a limited edition line | 0/0.00 | the space, a limited edition line, cool designs | 0/0.20 | the space | 0/0.40 | the space, cool designs, a limited edition line | 0/0.20 | the space | 0/0.40 | the space, cool designs, a limited edition line | 0/0.20 |
| q49 | 4 | What did Gina want her customers to feel in her store? | cozy and comfortable | cozy and inviting | 0/0.67 | cozy, comfy, like they're in a cool oasis, welcome | 0/0.18 | cozy and inviting | 0/0.67 | cozy, comfy, like they're in a cool oasis, welcome | 0/0.18 | cozy and inviting | 0/0.67 | cozy, comfy, like they're in a cool oasis, welcome | 0/0.18 |
| q50 | 4 | What did Jon say about Gina's progress with her store? | hard work's paying off | Looks like your store is growing | 0/0.00 | Congrats on the new fashion piece! Looks like your store is growing, Your commitment and … | 0/0.07 | Looks like your store is growing | 0/0.00 | Congrats on the new fashion piece! Looks like your store is growing, It looks awesome. Yo… | 0/0.10 | Your commitment and creativity in your business really stands out | 0/0.00 | Congrats on the new fashion piece! Looks like your store is growing, It looks a… | 0/0.09 |
| q51 | 4 | What made Gina choose the furniture and decor for her store? | personal style and customer comfort | she wanted to make the place look like her own style and make her customers feel cozy | 0/0.19 | I wanted to make the place look like my own style and make my customers feel cozy, I chos… | 0/0.10 | she wanted to make the place look like her own style and make her customers feel cozy | 0/0.19 | to make the place look like my own style, make my customers feel cozy, furniture that loo… | 0/0.11 | she wanted to make the place look like her own style and make her customers fee… | 0/0.19 | I wanted to make the place look like my own style and make my customers feel co… | 0/0.10 |
| q52 | 4 | What did Jon say about creating a special experience for customers? | It's the key to making them feel welcome and coming back | Creating a special experience for customers is the key to making them feel welcome and co… | 0/0.72 | That's a great goal! Creating a special experience for customers is the key to making the… | 0/0.49 | Creating a special experience for customers is the key to making them feel welcome and co… | 0/0.72 | Creating a special experience for customers is the key to making them feel welcome and co… | 0/0.72 | Creating a special experience for customers is the key to making them feel welc… | 0/0.72 | That's a great goal! Creating a special experience for customers is the key to … | 0/0.49 |
| q53 | 4 | What did Gina say about creating an experience for her customers? | making them want to come back | Just creating an experience that'll make 'em wanna come back | 0/0.27 | Making my spot comfortable and inviting for my customers is key, I want 'em to feel like … | 0/0.19 | Just creating an experience that'll make 'em wanna come back | 0/0.27 | Making my spot comfortable and inviting for my customers is key, I want 'em to feel like … | 0/0.19 | Just creating an experience that'll make 'em wanna come back | 0/0.27 | Making my spot comfortable and inviting for my customers is key, I want 'em to … | 0/0.19 |
| q54 | 4 | How is Gina's store doing? | The store is doing great. | doing great | 0/0.67 | The store's doing great! | 0/0.57 | so far, so good - customers love the new offers and promotions, which means I'm seeing mo… | 0/0.00 | So far, so good - customers love the new offers and promotions, which means I'm seeing mo… | 0/0.06 | so far, so good - customers love the new offers and promotions, which means I'm… | 0/0.00 | So far, so good - customers love the new offers and promotions, which means I'm… | 0/0.06 |
| q55 | 4 | What does Gina's tattoo symbolize? | Freedom and expressing herself through dance | freedom - dancing without worrying what people think | 0/0.15 | freedom - dancing without worrying what people think, A reminder to follow my passions an… | 0/0.19 | freedom - dancing without worrying what people think | 0/0.15 | freedom - dancing without worrying what people think, follow my passions and express myse… | 0/0.21 | freedom - dancing without worrying what people think | 0/0.15 | freedom - dancing without worrying what people think, follow my passions and ex… | 0/0.21 |
| q56 | 4 | What did Jon and Gina compare their entrepreneurial journeys to? | dancing together and supporting each other | a roller coaster | 0/0.00 | a roller coaster, a rollercoaster, a crazy ride, an amazing ride, a balancing act, a path… | 0/0.00 | a rollercoaster | 0/0.00 | a rollercoaster, an amazing ride, a wild ride, a balancing act, a tough road, a journey t… | 0/0.00 | a rollercoaster | 0/0.00 | a roller coaster, a rollercoaster, a balancing act, a partner to dance with | 0/0.00 |
| q57 | 4 | What advice does Gina give to Jon about running a successful business? | build relationships with customers, create a strong brand image, stay… | Don't be scared to reach out to people in your field for help and contacts | 0/0.00 | Don't be scared to reach out to people in your field for help and contacts, Marketing is … | 0/0.02 | Don't be scared to reach out to people in your field for help and contacts | 0/0.00 | Don't be scared to reach out to people in your field for help and contacts, stay passiona… | 0/0.05 | stay passionate, focused and resilient. Challenges will come, but believe in yo… | 0/0.06 | brand identity is key, Make sure yours stands out, build relationships with you… | 0/0.13 |
| q58 | 4 | Why did Jon shut down his bank account? | for his business | for his biz | 0/0.67 | I needed to do it for my biz, I thought it'd help my biz grow | 0/0.11 | for his biz | 0/0.67 | for my biz, I thought it'd help my biz grow | 0/0.15 | for his biz | 0/0.67 | for my biz, I thought it'd help my biz grow | 0/0.15 |
| q59 | 4 | Why did Gina combine her clothing business with dance? | she is passionate about dance and fashion | she's passionate about dance and fashion so combining them lets her show her creativity a… | 0/0.33 | I'm passionate about dance and fashion so combining them lets me show my creativity and s… | 0/0.28 | she is passionate about dance and fashion so combining them lets her show her creativity … | 0/0.38 | I'm passionate about dance and fashion so combining them lets me show my creativity and s… | 0/0.20 | she is passionate about dance and fashion so combining them lets her show her c… | 0/0.38 | I'm passionate about dance and fashion so combining them lets me show my creati… | 0/0.20 |
| q60 | 4 | What does Jon's dance make him? | happy | so happy | 0/0.67 | so happy, happy, most alive | 0/0.33 | No answer | 0/0.00 | most alive | 0/0.00 | No answer | 0/0.00 | most alive | 0/0.00 |
| q61 | 4 | What did Gina receive from a dance contest? | a trophy | one of my trophies | 0/0.00 | one of my trophies | 0/0.00 | one of my trophies | 0/0.00 | one of my trophies | 0/0.00 | one of my trophies | 0/0.00 | one of my trophies | 0/0.00 |
| q62 | 4 | How does Gina stay confident in her business? | By reminding herself of her successes and progress, having a support … | by reminding myself of my successes and progress, having a good support system, and focus… | 0/0.68 | reminding myself of my successes and progress, have a good support system, focus on why y… | 0/0.51 | reminding myself of my successes and progress, having a good support system, and focusing… | 0/0.65 | reminding myself of my successes and progress, have a good support system, focus on why y… | 0/0.51 | reminding myself of my successes and progress, having a good support system, an… | 0/0.65 | reminding myself of my successes and progress, have a good support system, focu… | 0/0.51 |
| q63 | 4 | What kind of professional experience did Gina get accepted for on May 23, 2023? | fashion internship | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 |
| q64 | 4 | Where is Gina's fashion internship? | fashion department of an international company | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 |
| q65 | 4 | What book is Jon currently reading? | The Lean Startup | The Lean Startup | 1/1.00 | "The Lean Startup" | 1/1.00 | The Lean Startup | 1/1.00 | "The Lean Startup" | 1/1.00 | The Lean Startup | 1/1.00 | "The Lean Startup" | 1/1.00 |
| q66 | 4 | What is Jon offering to the dancers at his dance studio? | One-on-one mentoring and training | dance classes and workshops, one-on-one mentoring and training | 0/0.67 | dance classes and workshops, one-on-one mentoring and training | 0/0.67 | dance classes and workshops, one-on-one mentoring and training | 0/0.67 | dance classes and workshops, one-on-one mentoring and training | 0/0.67 | dance classes and workshops, one-on-one mentoring and training | 0/0.67 | dance classes and workshops, one-on-one mentoring and training | 0/0.67 |
| q67 | 4 | How does Jon use the clipboard with a notepad attached to it? | To set goals, track achievements, and find areas for improvement | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 | No answer | 0/0.00 |
| q68 | 4 | What does Jon tell Gina he won't do? | quit | give up | 0/0.00 | give up, quit, let anything get me down, let you down, let anything or anyone stop me, st… | 0/0.08 | give up | 0/0.00 | give up, quit, let anything get me down, let you down, let anything or anyone stop me, st… | 0/0.10 | give up | 0/0.00 | give up, quit, let anything get me down, let you down, let anything or anyone s… | 0/0.10 |
| q69 | 4 | What did Jon take a trip to Rome for? | To clear his mind | to clear my mind a little | 0/0.67 | to clear my mind a little | 0/0.67 | to clear my mind a little | 0/0.67 | to clear my mind a little | 0/0.67 | to clear my mind a little | 0/0.67 | to clear my mind a little | 0/0.67 |
| q70 | 4 | What is Jon working on opening? | a dance studio | a dance studio | 1/1.00 | a dance studio | 1/1.00 | a dance studio | 1/1.00 | a dance studio | 1/1.00 | a dance studio | 1/1.00 | a dance studio | 1/1.00 |
| q71 | 4 | How does Gina describe the studio that Jon has opened? | amazing | amazing | 1/1.00 | The studio looks amazing | 0/0.50 | amazing | 1/1.00 | The studio looks amazing | 0/0.50 | amazing | 1/1.00 | The studio looks amazing | 0/0.50 |
| q72 | 4 | How does Jon feel about the opening night of his dance studio? | excited | excited | 1/1.00 | I'm excited!, feeling good and ready to give it my best, Tomorrow's gonna be an awesome n… | 0/0.10 | excited | 1/1.00 | I'm excited!, feeling good and ready to give it my best, Tomorrow's gonna be an awesome n… | 0/0.06 | excited | 1/1.00 | I'm excited!, feeling good and ready to give it my best, Tomorrow's gonna be an… | 0/0.06 |
| q73 | 4 | How does Gina describe the feeling that dance brings? | magical | ain't nothing like the feeling it gives us | 0/0.00 | ain't nothing like the feeling it gives us | 0/0.00 | ain't nothing like the feeling it gives us | 0/0.00 | ain't nothing like the feeling it gives us | 0/0.00 | ain't nothing like the feeling it gives us | 0/0.00 | ain't nothing like the feeling it gives us | 0/0.00 |
| q74 | 4 | What does Jon plan to do at the grand opening of his dance studio? | savor all the good vibes | make some awesome memories | 0/0.00 | savor all the good vibes, make some awesome memories tomorrow at the grand opening, make … | 0/0.42 | savor all the good vibes | 1/1.00 | savor all the good vibes, make some awesome memories tomorrow at the grand opening, make … | 0/0.42 | savor all the good vibes | 1/1.00 | savor all the good vibes, make some awesome memories tomorrow at the grand open… | 0/0.42 |
| q75 | 4 | What does Gina say to Jon about the grand opening? | Let's live it up and make some great memories | Can't wait for tomorrow's grand opening! | 0/0.00 | Congrats, Jon! The studio looks amazing. You've put a lot of work into this and I'm so pu… | 0/0.12 | Can't wait for tomorrow's grand opening! | 0/0.00 | Congrats, Jon! The studio looks amazing. You've put a lot of work into this and I'm so pu… | 0/0.12 | Can't wait for tomorrow's grand opening! | 0/0.00 | Congrats, Jon! The studio looks amazing. You've put a lot of work into this and… | 0/0.12 |
| q76 | 4 | What is the general sentiment about the upcoming grand opening? | excitement | excited and positive | 0/0.00 | excited, pumped, awesome, blast, unforgettable, so much fun, Can't wait, great memories, … | 0/0.00 | excited and positive | 0/0.00 | excited, so pumped, awesome, so much fun, unforgettable, Can't wait, blast, good vibes, g… | 0/0.00 | excited and positive | 0/0.00 | excited, so pumped, awesome, so much fun, unforgettable, Can't wait, blast, goo… | 0/0.00 |
| q77 | 4 | What did Gina make a limited edition line of? | Hoodies | hoodie | 0/0.00 | No answer | 0/0.00 | hoodie | 0/0.00 | No answer | 0/0.00 | hoodie | 0/0.00 | hoodie | 0/0.00 |
| q78 | 4 | According to Gina, what makes Jon a perfect mentor and guide? | His positivity and determination | His positivity and determination | 1/1.00 | Loving what you do and bringing joy to others, Your positivity and determination | 0/0.35 | Your positivity and determination | 0/0.75 | Loving what you do and bringing joy to others is so rewarding, Your positivity and determ… | 0/0.23 | Your positivity and determination | 0/0.75 | Your positivity and determination | 0/0.75 |
| q79 | 5 | What temporary job did Jon take to cover expenses? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q80 | 4 | What plans does Jon have after receiving advice at the networking event? | Sprucing up his business plan, tweaking his pitch to investors, and w… | sprucing up my biz plan and tweaking my pitch to investors, and working on an online plat… | 0/0.65 | sprucing up my biz plan, tweaking my pitch to investors, working on an online platform to… | 0/0.63 | sprucing up my biz plan and tweaking my pitch to investors, and working on an online plat… | 0/0.65 | sprucing up my biz plan, tweaking my pitch to investors, working on an online platform to… | 0/0.63 | sprucing up my biz plan and tweaking my pitch to investors, and working on an o… | 0/0.65 | sprucing up my biz plan, tweaking my pitch to investors, working on an online p… | 0/0.63 |
| q81 | 4 | What offer does Gina make to Jon regarding social media? | Helping with making content and managing his social media accounts. | help with making content or even managing your accounts | 0/0.53 | I could help you with making content or even managing your accounts if you want | 0/0.40 | help with making content or even managing your accounts | 0/0.53 | I could help you with making content or even managing your accounts if you want | 0/0.40 | help with making content or even managing your accounts | 0/0.53 | I could help you with making content or even managing your accounts if you want | 0/0.40 |
| q82 | 5 | What is Jon's favorite style of painting? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q83 | 5 | What was Jon's favorite dancing memory? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q84 | 5 | What kind of dance piece did Jon's team perform to win first place? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q85 | 5 | What is Gina's attitude towards participating in the dance festival? |  | No answer | 1/1.00 | Wow, they look great! Can't wait to see them rock the festival. Gonna be awesome! | 0/0.00 | No answer | 1/1.00 | Can't wait to see them rock the festival. Gonna be awesome! | 0/0.00 | No answer | 1/1.00 | Can't wait to see them rock the festival. Gonna be awesome! | 0/0.00 |
| q86 | 5 | What kind of flooring is Gina looking for in her dance studio? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q87 | 5 | What did Jon find for his clothing store on 1 February, 2023? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q88 | 5 | What did Jon design for his store? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q89 | 5 | What did Jon want his customers to feel in her store? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q90 | 5 | What made Jon choose the furniture and decor for his store? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q91 | 5 | How is Jon's store doing? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q92 | 5 | What does Jon's tattoo symbolize? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q93 | 5 | Why did Gina shut down her bank account? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q94 | 5 | Why did Jon combine his clothing business with dance? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q95 | 5 | What did Gina receive from a dance contest? |  | one of my trophies | 0/0.00 | one of my trophies | 0/0.00 | one of my trophies | 0/0.00 | one of my trophies | 0/0.00 | one of my trophies | 0/0.00 | one of my trophies | 0/0.00 |
| q96 | 5 | What kind of professional experience did Jon get accepted for on May 23, 2023? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q97 | 5 | Where is Gina's HR internship? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q98 | 5 | Where is Jon's fashion internship? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q99 | 5 | What book is Gina currently reading? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q100 | 5 | How does Gina use the clipboard with a notepad attached to it? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q101 | 5 | What did Jon take a trip to Barcelona for? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q102 | 5 | What did Jon make a limited edition line of? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q103 | 5 | What temporary job did Gina take to cover expenses? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |
| q104 | 5 | What plans does Gina have after receiving advice at the networking event? |  | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 | No answer | 1/1.00 |


## 附录 C：105 题检索状态总表

Ev=Evidence@4096（1 表示全部金标 turn 所在 session 都进了 4096 上下文）；rank 是每个金标 session 在 ranked_docs 中的名次；fb=dense fallback；R@5 是 session 级 recall。

| qid | cat | C2 Ev | P4 Ev | P4c10 Ev | P4noRR Ev | C2noRR Ev | C2 gold rank | P4 gold rank | P4noRR gold rank | C2 R@5 | P4 R@5 | P4noRR R@5 | C2 fb | P4 fb | P4noRR fb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q0 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q1 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q2 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [5] | [5] | [5] | 1.0 | 1.0 | 1.0 | False | False | False |
| q3 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [9, 11] | [9, 11] | [9, 11] | 0.0 | 0.0 | 0.0 | False | False | False |
| q4 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [4] | [4] | [5] | 1.0 | 1.0 | 1.0 | False | False | False |
| q5 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [3, 2] | [3, 2] | [4, 3] | 1.0 | 1.0 | 1.0 | False | False | False |
| q6 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q7 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q8 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q9 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1, 4] | [1, 4] | [1, 2] | 1.0 | 1.0 | 1.0 | True | True | False |
| q10 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [2] | [2] | [2] | 1.0 | 1.0 | 1.0 | False | False | False |
| q11 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q12 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q13 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [2] | [2] | [2] | 1.0 | 1.0 | 1.0 | False | False | False |
| q14 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [2] | [2] | 1.0 | 1.0 | 1.0 | False | False | False |
| q15 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [3] | [3] | [2] | 1.0 | 1.0 | 1.0 | False | False | False |
| q16 | 2 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | [8] | [1] | [2] | 0.0 | 1.0 | 1.0 | True | False | False |
| q17 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [11, 6] | [11, 6] | [11, 6] | 0.0 | 0.0 | 0.0 | False | False | False |
| q18 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [5, 3] | [5, 3] | [5, 3] | 1.0 | 1.0 | 1.0 | False | False | False |
| q19 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q20 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [2] | [2] | [2] | 1.0 | 1.0 | 1.0 | False | False | False |
| q21 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q22 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q23 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [3, 1, 16, 7] | [3, 1, 16, 7] | [8, 2, 16, 10] | 0.5 | 0.5 | 0.25 | True | True | False |
| q24 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [3, 7, 1] | [3, 7, 1] | [4, 7, 1] | 0.667 | 0.667 | 0.667 | False | False | False |
| q25 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [12, 1] | [12, 1] | [12, 1] | 0.5 | 0.5 | 0.5 | False | False | False |
| q26 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [2] | [2] | [3] | 1.0 | 1.0 | 1.0 | False | False | False |
| q27 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [1, 2, 12] | [1, 2, 8] | [1, 2, 8] | 0.667 | 0.667 | 0.667 | False | False | False |
| q28 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q29 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1, 3] | [1, 3] | [2, 4] | 1.0 | 1.0 | 1.0 | False | False | False |
| q30 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [2] | [2] | [3] | 1.0 | 1.0 | 1.0 | False | False | False |
| q31 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [11, 4] | [9, 4] | [9, 4] | 0.5 | 0.5 | 0.5 | False | False | False |
| q32 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q33 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [2] | [2] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q34 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q35 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [2] | 1.0 | 1.0 | 1.0 | False | False | False |
| q36 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q37 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [2] | 1.0 | 1.0 | 1.0 | True | True | False |
| q38 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q39 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q40 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [2] | 1.0 | 1.0 | 1.0 | False | False | False |
| q41 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q42 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q43 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q44 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [6] | [6] | [3] | 0.0 | 0.0 | 1.0 | False | False | False |
| q45 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [2] | [2] | [2] | 1.0 | 1.0 | 1.0 | False | False | False |
| q46 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q47 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q48 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q49 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q50 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [2] | [2] | [3] | 1.0 | 1.0 | 1.0 | False | False | False |
| q51 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q52 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q53 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q54 | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [9] | [9] | [11] | 0.0 | 0.0 | 0.0 | True | True | False |
| q55 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q56 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [5] | 1.0 | 1.0 | 1.0 | True | True | False |
| q57 | 4 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | [7] | [7] | [1] | 0.0 | 0.0 | 1.0 | False | False | False |
| q58 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q59 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q60 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [2] | [2] | [2] | 1.0 | 1.0 | 1.0 | False | False | False |
| q61 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [2] | 1.0 | 1.0 | 1.0 | False | False | False |
| q62 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q63 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q64 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q65 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q66 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q67 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [2] | 1.0 | 1.0 | 1.0 | True | True | False |
| q68 | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [8] | [8] | [8] | 0.0 | 0.0 | 0.0 | False | False | False |
| q69 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q70 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [2] | [2] | [3] | 1.0 | 1.0 | 1.0 | False | False | False |
| q71 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [3] | [3] | [3] | 1.0 | 1.0 | 1.0 | False | False | False |
| q72 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [3] | [3] | [3] | 1.0 | 1.0 | 1.0 | False | False | False |
| q73 | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [7] | [7] | [7] | 0.0 | 0.0 | 0.0 | False | False | False |
| q74 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [3] | [3] | [3] | 1.0 | 1.0 | 1.0 | False | False | False |
| q75 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q76 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q77 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q78 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [4] | 1.0 | 1.0 | 1.0 | False | False | False |
| q79 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q80 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [2] | [2] | [2] | 1.0 | 1.0 | 1.0 | False | False | False |
| q81 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q82 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | True | True | False |
| q83 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [2] | [2] | [1] | 1.0 | 1.0 | 1.0 | True | True | False |
| q84 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q85 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [2] | 1.0 | 1.0 | 1.0 | False | False | False |
| q86 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q87 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [3] | [2] | [2] | 1.0 | 1.0 | 1.0 | False | False | False |
| q88 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [2] | [2] | [1] | 1.0 | 1.0 | 1.0 | True | True | False |
| q89 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q90 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [2] | [2] | [3] | 1.0 | 1.0 | 1.0 | False | False | False |
| q91 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [2] | [2] | [4] | 1.0 | 1.0 | 1.0 | False | False | False |
| q92 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q93 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [5] | [5] | [1] | 1.0 | 1.0 | 1.0 | True | True | False |
| q94 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [3] | [3] | [3] | 1.0 | 1.0 | 1.0 | False | False | False |
| q95 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [2] | 1.0 | 1.0 | 1.0 | False | False | False |
| q96 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [4] | [4] | [3] | 1.0 | 1.0 | 1.0 | False | False | False |
| q97 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | True | True | False |
| q98 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q99 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | True | True | False |
| q100 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | [4] | [4] | [5] | 1.0 | 1.0 | 1.0 | True | True | False |
| q101 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [3] | 1.0 | 1.0 | 1.0 | True | True | False |
| q102 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [2] | [2] | [1] | 1.0 | 1.0 | 1.0 | True | True | False |
| q103 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1] | [1] | [1] | 1.0 | 1.0 | 1.0 | False | False | False |
| q104 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [3] | [3] | [2] | 1.0 | 1.0 | 1.0 | False | False | False |


## 附录 D：所有 EM 翻转逐题清单

### D.1 P7 vs P0（G_raw，同 context，全 105 题）

| qid | cat | gold | P0 | P0 EM/F1 | P7 | P7 EM/F1 |
|---|---|---|---|---|---|---|
| q7 | 2 | 29 January, 2023 | No answer | 0 / 0.00 | 29 January, 2023 | 1 / 1.00 |
| q13 | 2 | 16 March, 2023 | No answer | 0 / 0.00 | 16 March, 2023 | 1 / 1.00 |
| q19 | 2 | 10 May, 2023 | yesterday | 0 / 0.00 | 10 May, 2023 | 1 / 1.00 |
| q24 | 1 | fair, networking events, dance competition | a fair, networking events, a dance competition | 1 / 1.00 | networking events, a fair, the last networking event | 0 / 0.55 |
| q30 | 2 | 20 June, 2023 | tomorrow | 0 / 0.00 | 20 June, 2023 | 1 / 1.00 |
| q38 | 2 | 21 July 2023 | Last Friday | 0 / 0.00 | 21 July, 2023 | 1 / 1.00 |
| q39 | 4 | Contemporary | contemporary dance | 0 / 0.67 | contemporary | 1 / 1.00 |
| q71 | 4 | amazing | amazing | 1 / 1.00 | The studio looks amazing | 0 / 0.50 |
| q72 | 4 | excited | excited | 1 / 1.00 | I'm excited!, feeling good and ready to give it my best, To… | 0 / 0.10 |
| q78 | 4 | His positivity and determination | His positivity and determination | 1 / 1.00 | Loving what you do and bringing joy to others, Your positiv… | 0 / 0.35 |
| q85 | 5 |  | No answer | 1 / 1.00 | Wow, they look great! Can't wait to see them rock the festi… | 0 / 0.00 |

共 11 题 EM 发生变化。

### D.2 E2 vs E0（P4 图 + P0 vs G_raw + P0）

| qid | cat | gold | E0 | E0 EM/F1 | E2 | E2 EM/F1 |
|---|---|---|---|---|---|---|
| q0 | 2 | 19 January, 2023 | 19 January, 2023 | 1 / 1.00 | yesterday | 0 / 0.00 |
| q11 | 2 | A few years ago | a few years ago | 1 / 1.00 | No answer | 0 / 0.00 |
| q16 | 2 | 24 April, 2023 | No answer | 0 / 0.00 | 24 April, 2023 | 1 / 1.00 |
| q26 | 2 | 15 June, 2023 | 15 June, 2023 | 1 / 1.00 | yesterday | 0 / 0.00 |
| q30 | 2 | 20 June, 2023 | tomorrow | 0 / 0.00 | 20 June, 2023 | 1 / 1.00 |
| q39 | 4 | Contemporary | contemporary dance | 0 / 0.67 | contemporary | 1 / 1.00 |
| q74 | 4 | savor all the good vibes | make some awesome memories | 0 / 0.00 | savor all the good vibes | 1 / 1.00 |
| q78 | 4 | His positivity and determination | His positivity and determination | 1 / 1.00 | Your positivity and determination | 0 / 0.75 |

共 8 题 EM 发生变化。

### D.3 E3 vs E1（P4 图 + P7 vs G_raw + P7）

| qid | cat | gold | E1 | E1 EM/F1 | E3 | E3 EM/F1 |
|---|---|---|---|---|---|---|
| q11 | 2 | A few years ago | a few years ago | 1 / 1.00 | No answer | 0 / 0.00 |
| q16 | 2 | 24 April, 2023 | No answer | 0 / 0.00 | 24 April, 2023 | 1 / 1.00 |
| q33 | 2 | 20 June, 2023 | 21 June, 2023 | 0 / 0.67 | 20 June, 2023 | 1 / 1.00 |

共 3 题 EM 发生变化。

### D.4 P4+no-rerank+P7 vs E3（正常重排）

| qid | cat | gold | E3 | E3 EM/F1 | E3noRR | E3noRR EM/F1 |
|---|---|---|---|---|---|---|
| q11 | 2 | A few years ago | No answer | 0 / 0.00 | a few years ago | 1 / 1.00 |

共 1 题 EM 发生变化。

### D.5 P4+no-rerank+P0 vs E2（正常重排）

| qid | cat | gold | E2 | E2 EM/F1 | E2noRR | E2noRR EM/F1 |
|---|---|---|---|---|---|---|
| q0 | 2 | 19 January, 2023 | yesterday | 0 / 0.00 | 19 January, 2023 | 1 / 1.00 |
| q11 | 2 | A few years ago | No answer | 0 / 0.00 | a few years ago | 1 / 1.00 |
| q19 | 2 | 10 May, 2023 | yesterday | 0 / 0.00 | 10 May, 2023 | 1 / 1.00 |
| q24 | 1 | fair, networking events, dance competition | a fair, networking events, a dance competition | 1 / 1.00 | a fair, networking events | 0 / 0.75 |
| q30 | 2 | 20 June, 2023 | 20 June, 2023 | 1 / 1.00 | tomorrow | 0 / 0.00 |

共 5 题 EM 发生变化。

### D.6 G_raw+no-rerank+P0 vs E0

| qid | cat | gold | E0 | E0 EM/F1 | E0noRR | E0noRR EM/F1 |
|---|---|---|---|---|---|---|
| q0 | 2 | 19 January, 2023 | 19 January, 2023 | 1 / 1.00 | No answer | 0 / 0.00 |
| q11 | 2 | A few years ago | a few years ago | 1 / 1.00 | No answer | 0 / 0.00 |
| q19 | 2 | 10 May, 2023 | yesterday | 0 / 0.00 | 10 May, 2023 | 1 / 1.00 |
| q24 | 1 | fair, networking events, dance competition | a fair, networking events, a dance competition | 1 / 1.00 | a fair, networking events | 0 / 0.75 |
| q39 | 4 | Contemporary | contemporary dance | 0 / 0.67 | contemporary | 1 / 1.00 |
| q78 | 4 | His positivity and determination | His positivity and determination | 1 / 1.00 | Your positivity and determination | 0 / 0.75 |
| q83 | 5 |  | No answer | 1 / 1.00 | when his dance crew took home first in a local comp last ye… | 0 / 0.00 |

共 7 题 EM 发生变化。

### D.7 G_raw+no-rerank+P7 vs E1

| qid | cat | gold | E1 | E1 EM/F1 | E1noRR | E1noRR EM/F1 |
|---|---|---|---|---|---|---|
| q0 | 2 | 19 January, 2023 | 19 January, 2023 | 1 / 1.00 | 8 February, 2023 | 0 / 0.33 |
| q18 | 1 | Yes | Yes | 1 / 1.00 | Yes, "I'm starting a dance studio 'cause I'm passionate abo… | 0 / 0.06 |
| q20 | 2 | 27 May, 2023 | No answer | 0 / 0.00 | 27 May, 2023 | 1 / 1.00 |
| q64 | 4 | fashion department of an international company | No answer | 0 / 0.00 | the fashion department of an international company | 1 / 1.00 |
| q83 | 5 |  | No answer | 1 / 1.00 | when my dance crew took home first in a local comp last year | 0 / 0.00 |

共 5 题 EM 发生变化。

## 附录 E：所有 Comprehension 关系清单

### E.1 COMP-P0 / C3（原版，26 条）

| # | subject | predicate | object | type | deriv | sessions | conf | dia_ids | evidence |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Jon | lost_job_before | started dream business | temporal_ordering | Composed | [6, 11] | 0.95 | ['D6:11', 'D11:3'] | Losing my job was hard, but I'm livin' my dreams now. Startin' my biz has been tough \|\|… |
| 1 | Gina | had_interview_before | accepted_for fashion internship | temporal_ordering | Inferred | [11, 12] | 0.86 | ['D11:14', 'D12:1'] | I had an interview for a design internship yesterday! It was so cool. \|\| A lot's happen… |
| 2 | Jon | searched_for_location_before | official opening night | temporal_ordering | Composed | [3, 15] | 0.94 | ['D3:1', 'D15:5'] | I'm still searching for a place to open my dance studio. \|\| The official opening night … |
| 3 | Jon | worked_on_business_plan_before | worked_on_opening dance studio | temporal_ordering | Composed | [12, 15] | 0.88 | ['D12:12', 'D15:3'] | I'm wrapping up the business plan and looking for investors. \|\| Still working on openin… |
| 4 | Jon | looked_for_investors_before | official opening night | temporal_ordering | Composed | [12, 15] | 0.9 | ['D12:12', 'D15:5'] | I'm wrapping up the business plan and looking for investors. \|\| The official opening ni… |
| 5 | Jon | attended_networking_events_before | went_full-time_with biz idea | temporal_ordering | Composed | [16, 17] | 0.82 | ['D16:6', 'D17:2'] | Yesterday I chose to go to networking events to make things happen. \|\| I'm gonna stay d… |
| 6 | Jon | posted_dance_videos_before | started_learning marketing and analytics tools | temporal_ordering | Composed | [14, 17] | 0.87 | ['D14:5', 'D17:4'] | I already started doing what you said about social media and posted some of my dance vide… |
| 7 | Gina | lost_job_before | opened online clothing store | temporal_ordering | Composed | [6, 14] | 0.93 | ['D6:4', 'D14:8'] | Since I lost my job at Door Dash, things have been tough. \|\| I've been one 'cause I los… |
| 8 | Jon | lost_job_enabled | started dream business | causal | Composed | [9, 11] | 0.97 | ['D9:3', 'D11:3'] | Losing my job was tough but it gave me the push I needed to do what I love. \|\| Losing m… |
| 9 | Gina | lost_job_enabled | started own online clothing store | causal | Inferred | [10, 14] | 0.9 | ['D10:4', 'D14:8'] | After losing my job, I wanted to take control of my own destiny and this seemed like the … |
| 10 | Jon | used_whiteboard_before | color-coded achievements | temporal_ordering | Inferred | [12, 13] | 0.78 | ['D12:10', 'D13:15'] | It helps me keep track of ideas and milestones. Gives me a visual of my progress \|\| Yes… |
| 11 | Jon | offered_one-on-one_mentoring_before | mentored aspiring dancers | temporal_ordering | Inferred | [13, 17] | 0.8 | ['D13:7', 'D17:6'] | Besides the dance classes and workshops, I'm offering one-on-one mentoring and training \… |
| 12 | Jon | visited_Paris_before | took_trip_to Rome | temporal_ordering | Composed | [2, 15] | 0.92 | ['D2:4', 'D15:1'] | Oh, I've been to Paris yesterday! It was sooo cool. \|\| Took a short trip last week to R… |
| 13 | Gina | launched_ad_campaign_before | emailed wholesalers | temporal_ordering | Composed | [2, 3] | 0.86 | ['D2:1', 'D3:2'] | I just launched an ad campaign for my clothing store in hopes of growing the business. \|… |
| 14 | Gina | opened_online_clothes_store_before | acquired new unique pieces | temporal_ordering | Composed | [6, 16] | 0.89 | ['D6:6', 'D16:1'] | My online clothes store is open! I've been dreaming of this for a while now \|\| I have a… |
| 15 | Jon | attended_fair_before | worked_on_opening dance studio | temporal_ordering | Composed | [10, 15] | 0.88 | ['D10:1', 'D15:3'] | Yesterday, I went to a fair to show off my studio, it was both stressful and great! \|\| … |
| 16 | Jon | got_possible_leads_before | official opening night | temporal_ordering | Composed | [10, 15] | 0.86 | ['D10:1', 'D15:5'] | I got some possible leads, so that was awesome. \|\| The official opening night is tomorr… |
| 17 | Jon's dance studio | expanded_from_startup_to_social_media_and_competition_before_financial_trouble | tenuous grounds and investor search | state_change | Composed | [1, 8, 18] | 0.92 | ['D1:4', 'D8:13', 'D18:2'] | I'm starting a dance studio 'cause I'm passionate about dancing \|\| I'm expanding my dan… |
| 18 | Gina's clothing store | developed_from_online_store_to_customer_order_website | new website for customers to make orders | state_change | Composed | [5, 8, 18] | 0.9 | ['D5:5', 'D8:4', 'D18:1'] | I've been working hard on my online store \|\| It's been a wild ride starting my business… |
| 19 | nearby festival performance | identity_unified_with | Jon's performance that was awesome | identity_unification | Composed | [1, 5] | 0.95 | ['D1:24', 'D5:2'] | Finishing up choreography to perform at a nearby festival next month \|\| Remember the fe… |
| 20 | Jon's dance studio | later_needed_investors_after_hosting_competition | investors | temporal_ordering | Composed | [8, 18] | 0.88 | ['D8:13', 'D18:2', 'D18:8'] | I'm also hosting a dance competition next month \|\| while I look for investors \|\| I me… |
| 21 | Gina | later_helped_Jon_with_social_media_after_Jon_expanded_social_media_presence | Jon's dance studio social media | enabling_link | Composed | [8, 18] | 0.9 | ['D8:13', 'D18:13', 'D18:14', 'D18:15'] | I'm expanding my dance studio's social media presence \|\| I could help you with making c… |
| 22 | Jon | continued_business_after_shutting_down_bank_account | dance studio | state_change | Inferred | [8, 18] | 0.82 | ['D8:1', 'D8:3', 'D18:2', 'D18:4'] | I had to shut down my bank account \|\| I needed to do it for my biz \|\| The dance studi… |
| 23 | Gina's planned influencer work | preceded_Jon's_request_for_marketing_help | marketing strategies | temporal_ordering | Composed | [8, 18] | 0.78 | ['D8:12', 'D18:12', 'D18:13'] | working with some fashion bloggers and influencers in the next few months \|\| Need help … |
| 24 | Jon | moved_from_banker_to_dance_studio_owner_before_temp_job | temp job | state_change | Composed | [1, 5, 18] | 0.94 | ['D1:2', 'D5:4', 'D18:2'] | Lost my job as a banker yesterday \|\| I love running my own studio \|\| I got a temp job… |
| 25 | Gina | moved_from_Door_Dash_job_loss_to_clothing_store_business | clothing store | state_change | Inferred | [1, 8, 18] | 0.86 | ['D1:3', 'D8:4', 'D18:2'] | I also lost my job at Door Dash this month \|\| It's been a wild ride starting my busines… |

### E.2 C6（全文原版，18 条）

| # | subject | predicate | object | type | deriv | sessions | conf | dia_ids | evidence |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Jon's dance studio | opened_after | Jon searched for ideal spot | temporal_ordering | Composed | [2, 15] | 0.93 | ['D2:4', 'D15:5'] | I'm on the hunt for the ideal spot for my dance studio \|\| The official opening night is… |
| 1 | Gina's online clothing store | opened_after | Gina launched ad campaign | temporal_ordering | Composed | [2, 6] | 0.9 | ['D2:1', 'D6:6'] | I just launched an ad campaign for my clothing store \|\| My online clothes store is open! |
| 2 | Jon's festival performance | same_event_as | nearby festival performance by Jon's dancers | identity_unification | Composed | [1, 5] | 0.95 | ['D1:24', 'D5:2'] | Finishing up choreography to perform at a nearby festival next month \|\| Remenber the fe… |
| 3 | Gina's fashion internship | resulted_from | Gina's design internship interview | causal_link | Inferred | [11, 12] | 0.82 | ['D11:14', 'D12:1'] | I had an interview for a design internship yesterday! \|\| I just got accepted for a fash… |
| 4 | Jon's dance studio grand opening | occurred_after | Jon was working on opening a dance studio | temporal_ordering | Composed | [15, 17] | 0.88 | ['D15:3', 'D15:5', 'D17:1'] | Still working on opening a dance studio. \|\| The official opening night is tomorrow. \|\… |
| 5 | Jon's networking events | enabled | Jon met investors | causal_link | Inferred | [16, 18] | 0.86 | ['D16:6', 'D18:8'] | Yesterday I chose to go to networking events to make things happen \|\| I met some invest… |
| 6 | Jon's business plan | revised_after | Jon wrapped up business plan | temporal_ordering | Composed | [12, 18] | 0.91 | ['D12:12', 'D18:10'] | I'm wrapping up the business plan and looking for investors \|\| I'm sprucing up my biz p… |
| 7 | Jon's search for investors | continued_after | Jon got temp job | temporal_ordering | Composed | [12, 18] | 0.9 | ['D12:14', 'D18:2'] | Searching for investors has been tough, but I'm staying hopeful \|\| I got a temp job to … |
| 8 | Gina's online clothing store | expanded_after | Gina opened online clothes store | state_change | Inferred | [6, 18] | 0.84 | ['D6:6', 'D18:1'] | My online clothes store is open! \|\| Last week, I built a new website for customers to m… |
| 9 | Jon's dance studio social media presence | expanded_after | Gina advised social media channels and influencers | causal_link | Inferred | [8, 14] | 0.88 | ['D8:13', 'D14:4', 'D14:5'] | I'm expanding my dance studio's social media presence \|\| Use social media channels and … |
| 10 | Jon's dance studio | became_financially_tenuous_after | official opening night | state_change | Inferred | [15, 18] | 0.8 | ['D15:5', 'D18:2'] | The official opening night is tomorrow \|\| The dance studio is on tenuous grounds right … |
| 11 | Jon's lost job as banker | enabled | Jon started dream business | causal_link | Composed | [1, 11] | 0.96 | ['D1:2', 'D11:3'] | Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business \… |
| 12 | Gina's lost job at Door Dash | enabled | Gina started own online clothing store | causal_link | Composed | [1, 10, 14] | 0.94 | ['D1:3', 'D10:4', 'D14:8'] | I also lost my job at Door Dash this month \|\| After losing my job, I wanted to take con… |
| 13 | Jon's dance studio | progressed_from | searching for location to running own studio | state_change | Composed | [3, 5] | 0.9 | ['D3:1', 'D5:4'] | I'm still searching for a place to open my dance studio \|\| I love running my own studio |
| 14 | Gina's clothing store | progressed_from | ad campaign to online store opening | state_change | Inferred | [2, 6] | 0.83 | ['D2:1', 'D6:6'] | I just launched an ad campaign for my clothing store \|\| My online clothes store is open! |
| 15 | Jon's dance competition hosting | occurred_after | Jon was getting ready for a dance competition | temporal_ordering | Inferred | [4, 8] | 0.62 | ['D4:13', 'D8:13'] | I'm getting ready for a dance comp near me next month \|\| I'm also hosting a dance compe… |
| 16 | Jon's whiteboard use | continued_after | Jon read The Lean Startup | temporal_ordering | Inferred | [12, 13] | 0.68 | ['D12:6', 'D13:13'] | I'm currently reading "The Lean Startup" \|\| I also use whiteboard to stay on track |
| 17 | Jon's dance studio | was_promoted_at | fair | identity_unification | Inferred | [10, 12] | 0.55 | ['D10:1', 'D12:12'] | I went to a fair to show off my studio \|\| I'm wrapping up the business plan and looking… |

### E.3 COMP-P1 — Evidence-grounded composition（14 条）

原始候选/合法/最终：[(0, 10, 8, 2), (1, 9, 6, 3)]

| # | subject | predicate | object | type | deriv | sessions | conf | dia_ids | evidence |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Jon's job loss | pushed | Jon's dance studio business | causal | Composed | [9, 11] | 0.96 | ['D9:3', 'D11:3'] | Losing my job was tough but it gave me the push I needed \|\| Losing my job gave me the p… |
| 1 | Jon's location search | enabled | Jon's studio opening | enabling_link | Inferred | [2, 15] | 0.82 | ['D2:4', 'D15:5'] | I'm on the hunt for the ideal spot for my dance studio \|\| The official opening night is… |
| 2 | Jon's business plan | preceded | Jon's investor search | event_dependency | Inferred | [12, 16] | 0.63 | ['D12:12', 'D16:6'] | I'm wrapping up the business plan and looking for investors \|\| Yesterday I chose to go … |
| 3 | Gina's design internship interview | led_to | Gina's fashion internship | identity_unification | Inferred | [11, 12] | 0.72 | ['D11:14', 'D12:1'] | I had an interview for a design internship yesterday! It was so cool \|\| I just got acce… |
| 4 | Gina's online clothes store | became | Gina's own online clothing store | identity_unification | Composed | [6, 10] | 0.88 | ['D6:6', 'D10:2'] | My online clothes store is open! \|\| I started my own online clothing store not so long … |
| 5 | Jon's fair attendance | produced | Jon's possible leads | goal_outcome | Composed | [10, 14] | 0.91 | ['D10:1', 'D14:3'] | I went to a fair to show off my studio \|\| Been doing some promotion for my business. |
| 6 | Jon's grand opening | enabled | Jon's full-time biz idea | state_change | Inferred | [15, 17] | 0.78 | ['D15:5', 'D17:2'] | The official opening night is tomorrow. \|\| I'm gonna stay determined and make my own pa… |
| 7 | Jon's whiteboard | supported | Jon's business plan | enabling_link | Inferred | [12, 13] | 0.69 | ['D12:9', 'D13:13'] | that whiteboard's got a bunch of good ideas! \|\| I also use whiteboard to stay on track,… |
| 8 | nearby festival performance | became | Jon's awesome performance | identity_unification | Composed | [1, 5] | 0.93 | ['D1:24', 'D5:2'] | Finishing up choreography to perform at a nearby festival next month. \|\| Remember the f… |
| 9 | Jon's dance studio plan | became | running own studio | state_change | Composed | [1, 5] | 0.88 | ['D1:4', 'D5:4'] | I'm starting a dance studio 'cause I'm passionate about dancing \|\| I love running my ow… |
| 10 | Jon's dance studio | became | tenuous grounds | state_change | Composed | [8, 18] | 0.86 | ['D8:13', 'D18:2'] | I'm expanding my dance studio's social media presence and offering workshops and classes … |
| 11 | Gina's online store | enabled | new website | enabling_link | Inferred | [8, 18] | 0.72 | ['D8:4', 'D18:1'] | I got some new offers and promotions going on my online store \|\| Last week, I built a n… |
| 12 | Jon's social media expansion | enabled | Gina's social media help | enabling_link | Inferred | [8, 18] | 0.68 | ['D8:13', 'D18:14'] | I'm expanding my dance studio's social media presence \|\| I'd really appreciate your hel… |
| 13 | Jon's banker job loss | led_to | dance studio startup | causal | Composed | [1, 5] | 0.89 | ['D1:2', 'D5:10'] | Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business. … |

### E.4 COMP-P2 — Cross-session bridge discovery（19 条）

原始候选/合法/最终：[(0, 14, 13, 1), (1, 9, 6, 3)]

| # | subject | predicate | object | type | deriv | sessions | conf | dia_ids | evidence |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Jon's dance studio search | led_to | official opening night | plan_outcome | Composed | [3, 15] | 0.86 | ['D3:1', 'D15:5'] | I'm still searching for a place to open my dance studio. \|\| The official opening night … |
| 1 | Jon's business plan | preceded | official opening night | plan_outcome | Composed | [12, 15] | 0.82 | ['D12:12', 'D15:5'] | I'm wrapping up the business plan and looking for investors. \|\| The official opening ni… |
| 2 | Jon's fair promotion | generated | possible leads | event_consequence | Inferred | [10, 16] | 0.62 | ['D10:1', 'D16:6'] | Yesterday, I went to a fair to show off my studio \|\| Yesterday I chose to go to network… |
| 3 | Gina's design internship interview | resulted_in | fashion internship acceptance | plan_outcome | Inferred | [11, 12] | 0.88 | ['D11:14', 'D12:1'] | I had an interview for a design internship yesterday! \|\| I just got accepted for a fash… |
| 4 | Gina's online clothes store | expanded_into | new unique pieces | state_change | Composed | [6, 16] | 0.78 | ['D6:6', 'D16:1'] | My online clothes store is open! \|\| I have acquired some new unique pieces for my store. |
| 5 | Jon's lost job | pushed | dream business | causal | Composed | [9, 11] | 0.95 | ['D9:3', 'D11:3'] | Losing my job was tough but it gave me the push I needed \|\| Losing my job gave me the p… |
| 6 | Jon's lost job | enabled | full-time biz idea | causal | Composed | [16, 17] | 0.9 | ['D16:8', 'D17:4'] | It's been tough going since I lost my job \|\| Losing my job was a bummer, but it pushed … |
| 7 | Gina's lost Door Dash job | led_to | online clothing store | causal | Composed | [6, 14] | 0.9 | ['D6:4', 'D14:8'] | Since I lost my job at Door Dash, things have been tough. \|\| I've been one 'cause I los… |
| 8 | Jon's whiteboard | became | color-coded achievement system | state_change | Composed | [12, 13] | 0.83 | ['D12:9', 'D13:15'] | that whiteboard's got a bunch of good ideas! \|\| I color-code achievements so I can easi… |
| 9 | Jon's grand opening | became | welcoming dancers of all ages | state_change | Inferred | [15, 19] | 0.74 | ['D15:5', 'D19:7'] | The official opening night is tomorrow. \|\| Can't wait 'til my studio starts welcoming d… |
| 10 | Jon's one-on-one mentoring | continued_as | mentored aspiring dancers | event_sequence | Composed | [13, 17] | 0.81 | ['D13:7', 'D17:6'] | I'm offering one-on-one mentoring and training to help dancers reach their full potential… |
| 11 | Jon's dance routine video | preceded | posted dance videos | event_sequence | Inferred | [11, 14] | 0.58 | ['D11:21', 'D14:5'] | I'll get the video to you soon! \|\| posted some of my dance videos. It's creating a bit … |
| 12 | Gina's fashion internship acceptance | preceded | fashion editors reach-out | state_change | Inferred | [12, 17] | 0.61 | ['D12:1', 'D17:1'] | I just got accepted for a fashion internship! \|\| Last week was wild, I got noticed by f… |
| 13 | nearby festival performance | was_completed_as | awesome performance | plan_outcome | Composed | [1, 5] | 0.93 | ['D1:24', 'D5:2'] | "Finishing up choreography to perform at a nearby festival next month." \|\| "Had that pe… |
| 14 | Jon's dance studio | changed_from | tenuous grounds | state_change | Composed | [1, 8, 18] | 0.86 | ['D1:4', 'D8:13', 'D18:2'] | "I'm starting a dance studio" \|\| "I'm expanding my dance studio's social media presence… |
| 15 | Gina's online store | developed_into | clothing store | state_change | Composed | [8, 18] | 0.82 | ['D8:4', 'D18:1', 'D18:2'] | "new offers and promotions going on my online store" \|\| "Last week, I built a new websi… |
| 16 | Jon's banker job loss | led_to | dance studio startup | causal | Composed | [1, 5] | 0.9 | ['D1:2', 'D1:4', 'D5:10'] | "Lost my job as a banker yesterday, so I'm gonna take a shot" \|\| "I'm starting a dance … |
| 17 | Gina's Door Dash job loss | preceded | Gina's online store | event_sequence | Inferred | [1, 8] | 0.68 | ['D1:3', 'D8:4', 'D8:10'] | "I also lost my job at Door Dash this month." \|\| "It's been a wild ride starting my bus… |
| 18 | Gina's planned influencer work | aligned_with | social media collaboration | plan_outcome | Inferred | [8, 18] | 0.61 | ['D8:12', 'D18:13', 'D18:15'] | "working with some fashion bloggers and influencers" \|\| "collaborate with local influen… |

### E.5 COMP-P3 — Event and state tracking（27 条）

原始候选/合法/最终：[(0, 16, 16, 0), (1, 14, 11, 3)]

| # | subject | predicate | object | type | deriv | sessions | conf | dia_ids | evidence |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Jon's dance studio | changed_from | location search | state_transition | Composed | [3, 9, 15] | 0.86 | ['D3:1', 'D9:1', 'D15:5'] | I'm still searching for a place to open my dance studio. \|\| I'm turning my loves of dan… |
| 1 | Jon's business plan | changed_from | wrapping up | status_change | Composed | [12, 16, 19] | 0.72 | ['D12:12', 'D16:10', 'D19:1'] | I'm wrapping up the business plan and looking for investors. \|\| I am writing all my pla… |
| 2 | Jon's biz idea | became | full-time venture | status_change | Composed | [16, 17] | 0.88 | ['D16:8', 'D17:2'] | I'm sure investing my time in my business will pay off eventually. \|\| I'm gonna stay de… |
| 3 | Jon's grand opening | changed_from | planned to upcoming | status_change | Composed | [15, 19] | 0.68 | ['D15:5', 'D19:7'] | The official opening night is tomorrow. \|\| Can't wait 'til my studio starts welcoming d… |
| 4 | Gina's clothing store | expanded_to | physical space | state_transition | Composed | [2, 3] | 0.83 | ['D2:1', 'D3:2', 'D3:4'] | I just launched an ad campaign for my clothing store in hopes of growing the business. \|… |
| 5 | Gina's online clothes store | changed_from | dream to open | state_transition | Composed | [6, 7, 16] | 0.9 | ['D6:6', 'D7:2', 'D16:1'] | My online clothes store is open! I've been dreaming of this for a while now \|\| my onlin… |
| 6 | Gina's fashion internship | changed_from | interview to accepted | status_change | Composed | [11, 12] | 0.94 | ['D11:14', 'D12:1'] | I had an interview for a design internship yesterday! \|\| I just got accepted for a fash… |
| 7 | Gina's fashion recognition | changed_from | store growth to editor notice | status_change | Composed | [14, 17] | 0.81 | ['D14:2', 'D17:1'] | I'm working on my online store, growing the customer base. \|\| Last week was wild, I got… |
| 8 | Jon's promotion | expanded_from | fair to social media | plan_outcome | Composed | [10, 14] | 0.79 | ['D10:1', 'D14:5'] | Yesterday, I went to a fair to show off my studio \|\| I already started doing what you s… |
| 9 | Jon's business prep | expanded_from | plans to investors | plan_outcome | Composed | [12, 16] | 0.7 | ['D12:12', 'D16:6'] | I'm wrapping up the business plan and looking for investors. \|\| Yesterday I chose to go… |
| 10 | Jon's mentoring | expanded_from | young dancers to aspiring dancers | state_transition | Composed | [13, 17] | 0.76 | ['D13:5', 'D17:6'] | It's been so inspiring to work with our young dancers \|\| I'm also excited to guide and … |
| 11 | Jon's support system | expanded_to | business mentor | relationship_change | Composed | [13, 14] | 0.62 | ['D13:8', 'D14:1'] | I had a mentor too when I was learning how to dance. \|\| I got mentored by this amazing … |
| 12 | Jon's business skills | expanded_to | marketing and analytics tools | state_transition | Composed | [12, 17] | 0.78 | ['D12:6', 'D17:4'] | I'm currently reading "The Lean Startup" and hoping it'll give me tips for my biz. \|\| S… |
| 13 | Jon's confidence | recovered_from | low confidence | state_transition | Composed | [10, 16] | 0.87 | ['D10:7', 'D16:16'] | I've been feeling kinda low on confidence lately. \|\| I'm feeling confident and won't gi… |
| 14 | Jon's job loss | led_to | dream business start | causal | Explicit | [9, 11] | 0.95 | ['D9:3', 'D11:3'] | Losing my job was tough but it gave me the push I needed to do what I love. \|\| Losing m… |
| 15 | Gina's job loss | led_to | online clothing store | causal | Explicit | [10, 14] | 0.91 | ['D10:4', 'D14:8'] | After losing my job, I wanted to take control of my own destiny \|\| I've been one 'cause… |
| 16 | Jon | changed_from | banker to business starter | state_transition | Composed | [1, 5] | 0.95 | ['D1:2', 'D5:10'] | Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business. … |
| 17 | Jon's dance studio | became | running own studio | status_change | Composed | [1, 5] | 0.9 | ['D1:4', 'D5:4'] | I'm starting a dance studio 'cause I'm passionate about dancing \|\| I love running my ow… |
| 18 | nearby festival performance | became | completed awesome performance | plan_outcome | Composed | [1, 5] | 0.95 | ['D1:24', 'D5:2'] | Finishing up choreography to perform at a nearby festival next month. \|\| Had that perfo… |
| 19 | Gina's online store | expanded_to | new website for orders | status_change | Composed | [5, 18] | 0.86 | ['D5:5', 'D18:1'] | I've been working hard on my online store and just teamed up with a local artist \|\| Las… |
| 20 | Jon's dance studio | expanded_to | social media and workshops | status_change | Composed | [1, 8] | 0.88 | ['D1:4', 'D8:13'] | I'm starting a dance studio 'cause I'm passionate about dancing \|\| I'm expanding my dan… |
| 21 | Jon's dance studio | became | on tenuous grounds | status_change | Composed | [8, 18] | 0.9 | ['D8:13', 'D18:2'] | All the work's paying off - I'm seeing progress and the dancers are so excited. \|\| The … |
| 22 | Jon | added | temp job and investor search | plan_outcome | Composed | [1, 18] | 0.84 | ['D1:2', 'D18:2'] | I'm gonna take a shot at starting my own business. \|\| I got a temp job to help cover ex… |
| 23 | Jon and Gina | became | marketing collaborators | relationship_change | Composed | [8, 18] | 0.86 | ['D8:23', 'D18:14', 'D18:15'] | Sure, I am always there for you! \|\| I'd really appreciate your help with making content… |
| 24 | Jon's marketing plan | expanded_to | online platform and social media | goal_change | Composed | [8, 18] | 0.82 | ['D8:13', 'D18:10', 'D18:14'] | I'm expanding my dance studio's social media presence \|\| I'm also working on an online … |
| 25 | Gina's business focus | changed_from | customer base to website orders | goal_change | Composed | [8, 18] | 0.72 | ['D8:10', 'D18:1'] | Growing my customer base is the main focus right now. \|\| Last week, I built a new websi… |
| 26 | Jon's bank account | became | shut down for biz | status_change | Composed | [1, 8] | 0.83 | ['D1:2', 'D8:1'] | Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business. … |

### E.6 COMP-P4 — Retrieval-oriented comprehension（16 条）

原始候选/合法/最终：[(0, 12, 10, 2), (1, 8, 6, 2)]

| # | subject | predicate | object | type | deriv | sessions | conf | dia_ids | evidence |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Jon's job loss | pushed_start_of | Jon's dance studio | causal | Composed | [9, 11, 17] | 0.96 | ['D9:3', 'D11:3', 'D17:4'] | Losing my job was tough but it gave me the push I needed to do what I love. \|\| Losing m… |
| 1 | Jon's dance studio | opened_after | Jon's location search | temporal_ordering | Composed | [3, 15, 19] | 0.9 | ['D3:1', 'D15:5', 'D19:7'] | I'm still searching for a place to open my dance studio. \|\| The official opening night … |
| 2 | Gina's support | sustained_motivation_for | Jon's dance studio | enabling_link | Composed | [7, 16, 19] | 0.88 | ['D7:15', 'D16:16', 'D19:9'] | I won't quit - your words motivate me to keep going! \|\| Your support means a ton to me.… |
| 3 | Jon's fair attendance | generated_leads_for | Jon's dance studio | goal_outcome | Composed | [10, 14] | 0.86 | ['D10:1', 'D14:3', 'D14:5'] | Yesterday, I went to a fair to show off my studio \|\| I got some possible leads, so that… |
| 4 | Gina's marketing advice | led_to | Jon's dance videos | causal | Composed | [14, 17] | 0.9 | ['D14:4', 'D14:5', 'D17:4'] | Use social media channels and work with influencers for bigger reach. \|\| I already star… |
| 5 | Gina's fashion internship | followed | Gina's design internship interview | temporal_ordering | Inferred | [11, 12] | 0.82 | ['D11:14', 'D12:1', 'D12:3'] | I had an interview for a design internship yesterday! \|\| I just got accepted for a fash… |
| 6 | Gina's online clothing store | blended | dance and fashion | identity_unification | Composed | [6, 9, 14] | 0.84 | ['D6:8', 'D9:9', 'D14:14'] | I wanted to blend my love for dance and fashion, so it was a perfect match. \|\| I'm happ… |
| 7 | Jon's whiteboard | supported_tracking_of | Jon's business milestones | enabling_link | Composed | [12, 13, 16] | 0.91 | ['D12:10', 'D13:13', 'D16:10'] | It helps me keep track of ideas and milestones. \|\| I also use whiteboard to stay on tra… |
| 8 | Jon's dance studio | became_space_for | dancer self-expression | state_change | Composed | [11, 13, 19] | 0.89 | ['D11:11', 'D13:5', 'D19:6'] | I want to create a place for people to dance and express themselves \|\| I want it to be … |
| 9 | Jon's dance studio | expanded_into | one-on-one mentoring | state_change | Composed | [13, 17, 19] | 0.87 | ['D13:7', 'D17:6', 'D19:7'] | I'm offering one-on-one mentoring and training to help dancers reach their full potential… |
| 10 | Jon's banker job loss | enabled_start_of | Jon's dance studio | causal | Composed | [1, 5] | 0.86 | ['D1:2', 'D5:10'] | Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business. … |
| 11 | nearby festival performance | reaffirmed | Jon's dancing passion | causal | Composed | [1, 5] | 0.91 | ['D1:24', 'D5:2'] | Finishing up choreography to perform at a nearby festival next month. \|\| Had that perfo… |
| 12 | Jon's dance studio | shifted_from_progress_to | tenuous grounds | state_change | Composed | [8, 18] | 0.89 | ['D8:13', 'D18:2'] | All the work's paying off - I'm seeing progress and the dancers are so excited. \|\| The … |
| 13 | Gina's clothing store | expanded_into | new website | state_change | Composed | [8, 18] | 0.84 | ['D8:4', 'D18:1'] | I got some new offers and promotions going on my online store \|\| Last week, I built a n… |
| 14 | Gina's social media help | extended | dance studio's social media presence | enabling_link | Composed | [8, 18] | 0.82 | ['D8:13', 'D18:14'] | I'm expanding my dance studio's social media presence \|\| I'd really appreciate your hel… |
| 15 | Gina's fashion bloggers plan | anticipated | Instagram and TikTok strategy | goal_outcome | Inferred | [8, 18] | 0.63 | ['D8:12', 'D18:13'] | I'm thinking of working with some fashion bloggers and influencers \|\| Instagram and Tik… |

### E.7 COMP-P5 — Generate then verify（7 条）

原始候选/合法/最终：[(0, 12, 12, 0), (1, 12, 6, 6)]

| # | subject | predicate | object | type | deriv | sessions | conf | dia_ids | evidence |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Jon losing job | pushed_start_of | dance studio business | causal | Composed | [9, 11, 17] | 0.95 | ['D9:3', 'D11:3', 'D17:4'] | Losing my job was tough but it gave me the push I needed to do what I love. \|\| Losing m… |
| 1 | whiteboard | enabled_tracking_of | ideas and milestones | enabling_link | Composed | [12, 13] | 0.93 | ['D12:9', 'D12:10', 'D13:13'] | that whiteboard's got a bunch of good ideas! \|\| It helps me keep track of ideas and mil… |
| 2 | written goals | kept_motivated | Jon | causal | Composed | [13, 16] | 0.88 | ['D16:10', 'D16:12', 'D13:17'] | I am writing all my plans down. \|\| Seeing my goals written down on paper really helps k… |
| 3 | Jon's fair attendance | generated | possible leads | goal_outcome | Composed | [10, 14] | 0.84 | ['D10:1', 'D14:3', 'D14:5'] | Yesterday, I went to a fair to show off my studio, it was both stressful and great! I got… |
| 4 | Gina's design internship interview | preceded | fashion internship acceptance | temporal_ordering | Inferred | [11, 12] | 0.78 | ['D11:14', 'D12:1', 'D12:3'] | I had an interview for a design internship yesterday! \|\| I just got accepted for a fash… |
| 5 | Jon's one-on-one mentoring | supports_goal_of | dancers' full potential | goal_outcome | Composed | [13, 17] | 0.91 | ['D13:7', 'D13:8', 'D17:6'] | I'm offering one-on-one mentoring and training to help dancers reach their full potential… |
| 6 | Jon's banker job loss | enabled_starting | dance studio | causal | Composed | [1, 5] | 0.86 | ['D1:2', 'D5:10'] | Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business. … |

### E.8 COMP-P4-c10（数量控制，10 条）

| # | subject | predicate | object | type | deriv | sessions | conf | dia_ids | evidence |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Jon's job loss | pushed_start_of | Jon's dance studio | causal | Composed | [9, 11, 17] | 0.96 | ['D9:3', 'D11:3', 'D17:4'] | Losing my job was tough but it gave me the push I needed to do what I love. \|\| Losing m… |
| 1 | Jon's whiteboard | supported_tracking_of | Jon's business milestones | enabling_link | Composed | [12, 13, 16] | 0.91 | ['D12:10', 'D13:13', 'D16:10'] | It helps me keep track of ideas and milestones. \|\| I also use whiteboard to stay on tra… |
| 2 | nearby festival performance | reaffirmed | Jon's dancing passion | causal | Composed | [1, 5] | 0.91 | ['D1:24', 'D5:2'] | Finishing up choreography to perform at a nearby festival next month. \|\| Had that perfo… |
| 3 | Jon's dance studio | opened_after | Jon's location search | temporal_ordering | Composed | [3, 15, 19] | 0.9 | ['D3:1', 'D15:5', 'D19:7'] | I'm still searching for a place to open my dance studio. \|\| The official opening night … |
| 4 | Gina's marketing advice | led_to | Jon's dance videos | causal | Composed | [14, 17] | 0.9 | ['D14:4', 'D14:5', 'D17:4'] | Use social media channels and work with influencers for bigger reach. \|\| I already star… |
| 5 | Jon's dance studio | became_space_for | dancer self-expression | state_change | Composed | [11, 13, 19] | 0.89 | ['D11:11', 'D13:5', 'D19:6'] | I want to create a place for people to dance and express themselves \|\| I want it to be … |
| 6 | Jon's dance studio | shifted_from_progress_to | tenuous grounds | state_change | Composed | [8, 18] | 0.89 | ['D8:13', 'D18:2'] | All the work's paying off - I'm seeing progress and the dancers are so excited. \|\| The … |
| 7 | Gina's support | sustained_motivation_for | Jon's dance studio | enabling_link | Composed | [7, 16, 19] | 0.88 | ['D7:15', 'D16:16', 'D19:9'] | I won't quit - your words motivate me to keep going! \|\| Your support means a ton to me.… |
| 8 | Jon's dance studio | expanded_into | one-on-one mentoring | state_change | Composed | [13, 17, 19] | 0.87 | ['D13:7', 'D17:6', 'D19:7'] | I'm offering one-on-one mentoring and training to help dancers reach their full potential… |
| 9 | Jon's fair attendance | generated_leads_for | Jon's dance studio | goal_outcome | Composed | [10, 14] | 0.86 | ['D10:1', 'D14:3', 'D14:5'] | Yesterday, I went to a fair to show off my studio \|\| I got some possible leads, so that… |


## 附录 F：20 题完整案例

## Case studies

### q16 (cat2)

**Q**: When did Jon go to a fair to get more exposure for his dance studio?  
**Gold**: 24 April, 2023  
**Gold sessions**: [10] turns ['D10:1']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s05, s09, s11, s02, s18 | [9] | 0.0 | False | 5 |
| C2 | s18, s05, s01, s02, s17 | [8] | 0.0 | True | 0 |
| cmpo_compp4 | s10, s14, s18, s05, s01 | [1] | 1.0 | False | 1 |

**P0 on G_raw (E0)**: 'No answer' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: 'No answer' (EM 0, F1 0.00)  
**P0 on this graph (E2)**: '24 April, 2023'  
**P7 on this graph (E3)**: '24 April, 2023'

Relations touching gold sessions:
- `Jon's fair attendance --generated_leads_for--> Jon's dance studio` [goal_outcome] sessions=[10, 14] conf=0.86

### q17 (cat1)

**Q**: Why did Gina decide to start her own clothing store?  
**Gold**: She always loved fashion trends and finding unique pieces and she lost her job so decided it was time to start her own business.  
**Gold sessions**: [1, 6] turns ['D6:8', 'D1:3']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s08, s05, s06, s03, s18 | [12, 3] | 0.0 | True | 0 |
| C2 | s10, s02, s03, s08, s05 | [11, 6] | 0.0 | False | 3 |
| cmpo_compp4 | s10, s02, s03, s08, s05 | [11, 6] | 0.0 | False | 3 |

**P0 on G_raw (E0)**: 'After losing my job, I wanted to take control of my own destiny' (EM 0, F1 0.16)  
**P7 on G_raw (E1)**: "After losing my job, I wanted to take control of my own destiny, I've been one 'cause I lost my job" (EM 0, F1 0.18)  
**P0 on this graph (E2)**: 'After losing her job, she wanted to take control of her own destiny'  
**P7 on this graph (E3)**: "After losing my job, I wanted to take control of my own destiny, I'm passionate about fashion trends and finding unique pieces, I wanted to blend my love for dance and fashion"

Relations touching gold sessions:
- `Gina's online clothing store --blended--> dance and fashion` [identity_unification] sessions=[6, 9, 14] conf=0.84
- `Jon's banker job loss --enabled_start_of--> Jon's dance studio` [causal] sessions=[1, 5] conf=0.86
- `nearby festival performance --reaffirmed--> Jon's dancing passion` [causal] sessions=[1, 5] conf=0.91

### q24 (cat1)

**Q**: Which events has Jon participated in to promote his business venture?  
**Gold**: fair, networking events, dance competition  
**Gold sessions**: [8, 10, 16] turns ['D10:1', 'D16:6', 'D8:4']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s16, s18, s08, s04, s14 | [3, 7, 1] | 1.0 | False | 3 |
| C2 | s16, s18, s08, s04, s14 | [3, 7, 1] | 1.0 | False | 3 |
| cmpo_compp4 | s16, s18, s08, s14, s04 | [3, 7, 1] | 1.0 | False | 3 |

**P0 on G_raw (E0)**: 'a fair, networking events, a dance competition' (EM 1, F1 1.00)  
**P7 on G_raw (E1)**: 'networking events, a fair, the last networking event' (EM 0, F1 0.55)  
**P0 on this graph (E2)**: 'a fair, networking events, a dance competition'  
**P7 on this graph (E3)**: 'networking events, a fair, the last networking event'

Relations touching gold sessions:
- `Gina's support --sustained_motivation_for--> Jon's dance studio` [enabling_link] sessions=[7, 16, 19] conf=0.88
- `Jon's fair attendance --generated_leads_for--> Jon's dance studio` [goal_outcome] sessions=[10, 14] conf=0.86
- `Jon's whiteboard --supported_tracking_of--> Jon's business milestones` [enabling_link] sessions=[12, 13, 16] conf=0.91
- `Jon's dance studio --shifted_from_progress_to--> tenuous grounds` [state_change] sessions=[8, 18] conf=0.89
- `Gina's clothing store --expanded_into--> new website` [state_change] sessions=[8, 18] conf=0.84
- `Gina's social media help --extended--> dance studio's social media presence` [enabling_link] sessions=[8, 18] conf=0.82
- `Gina's fashion bloggers plan --anticipated--> Instagram and TikTok strategy` [goal_outcome] sessions=[8, 18] conf=0.63

### q29 (cat1)

**Q**: Which cities has Jon visited?  
**Gold**: Paris, Rome  
**Gold sessions**: [2, 15] turns ['D2:4', 'D15:1']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s02, s18, s15, s17, s04 | [1, 3] | 1.0 | False | 2 |
| C2 | s02, s18, s15, s17, s04 | [1, 3] | 1.0 | False | 2 |
| cmpo_compp4 | s02, s18, s15, s17, s04 | [1, 3] | 1.0 | False | 2 |

**P0 on G_raw (E0)**: 'Paris, Rome' (EM 1, F1 1.00)  
**P7 on G_raw (E1)**: 'Paris, Rome' (EM 1, F1 1.00)  
**P0 on this graph (E2)**: 'Paris, Rome'  
**P7 on this graph (E3)**: 'Paris, Rome'

Relations touching gold sessions:
- `Jon's dance studio --opened_after--> Jon's location search` [temporal_ordering] sessions=[3, 15, 19] conf=0.9

### q31 (cat1)

**Q**: How long did it take for Jon to open his studio?  
**Gold**: six months  
**Gold sessions**: [1, 15] turns ['D1:2', 'D15:13']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s05, s10, s11, s15, s09 | [11, 4] | 0.0 | False | 5 |
| C2 | s05, s10, s11, s15, s09 | [11, 4] | 0.0 | False | 5 |
| cmpo_compp4 | s05, s10, s11, s15, s09 | [9, 4] | 0.0 | False | 5 |

**P0 on G_raw (E0)**: 'No answer' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: '4 months and 18 days' (EM 0, F1 0.29)  
**P0 on this graph (E2)**: 'No answer'  
**P7 on this graph (E3)**: '2 months and 10 days'

Relations touching gold sessions:
- `Jon's dance studio --opened_after--> Jon's location search` [temporal_ordering] sessions=[3, 15, 19] conf=0.9
- `Jon's banker job loss --enabled_start_of--> Jon's dance studio` [causal] sessions=[1, 5] conf=0.86
- `nearby festival performance --reaffirmed--> Jon's dancing passion` [causal] sessions=[1, 5] conf=0.91

### q33 (cat2)

**Q**: When did Jon visit networking events for his store?  
**Gold**: 20 June, 2023  
**Gold sessions**: [16] turns ['D16:6']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s16, s18, s08, s10, s01 | [1] | 1.0 | False | 4 |
| C2 | s18, s16, s08, s10, s01 | [2] | 1.0 | False | 1 |
| cmpo_compp4 | s18, s16, s08, s10, s01 | [2] | 1.0 | False | 1 |

**P0 on G_raw (E0)**: 'No answer' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: '21 June, 2023' (EM 0, F1 0.67)  
**P0 on this graph (E2)**: 'No answer'  
**P7 on this graph (E3)**: '20 June, 2023'

Relations touching gold sessions:
- `Gina's support --sustained_motivation_for--> Jon's dance studio` [enabling_link] sessions=[7, 16, 19] conf=0.88
- `Jon's whiteboard --supported_tracking_of--> Jon's business milestones` [enabling_link] sessions=[12, 13, 16] conf=0.91

### q54 (cat4)

**Q**: How is Gina's store doing?  
**Gold**: The store is doing great.  
**Gold sessions**: [4] turns ['D4:2']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s04, s08, s05, s03, s06 | [1] | 1.0 | False | 1 |
| C2 | s08, s05, s06, s03, s02 | [9] | 0.0 | True | 0 |
| cmpo_compp4 | s08, s05, s06, s03, s02 | [9] | 0.0 | True | 0 |

**P0 on G_raw (E0)**: 'doing great' (EM 0, F1 0.67)  
**P7 on G_raw (E1)**: "The store's doing great!" (EM 0, F1 0.57)  
**P0 on this graph (E2)**: "so far, so good - customers love the new offers and promotions, which means I'm seeing more sales"  
**P7 on this graph (E3)**: "So far, so good - customers love the new offers and promotions, which means I'm seeing more sales, People seem to really like my designs, Growing my customer base is the main focus right now"

### q4 (cat4)

**Q**: Why did Jon decide to start his dance studio?  
**Gold**: He lost his job and decided to start his own business to share his passion.  
**Gold sessions**: [1] turns ['D1:2', 'D1:4']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s13, s09, s11, s05, s02 | [7] | 0.0 | False | 5 |
| C2 | s13, s11, s05, s01, s09 | [4] | 1.0 | False | 4 |
| cmpo_compp4 | s13, s11, s05, s01, s09 | [4] | 1.0 | False | 4 |

**P0 on G_raw (E0)**: 'Losing my job gave me the push to finally start my dream business: my own dance studio' (EM 0, F1 0.32)  
**P7 on G_raw (E1)**: "Losing my job gave me the push to finally start my dream business: my own dance studio, Losing my job was tough but it gave me the push I needed to do what I love, I want to create a place for people to dance and express themselves - it's been a dream of mine, I'm turning my loves of dance into a business, I'm aiming to turn my dancing passion into a business" (EM 0, F1 0.20)  
**P0 on this graph (E2)**: "he's passionate about dancing and it'd be great to share it with others"  
**P7 on this graph (E3)**: "'cause I'm passionate about dancing and it'd be great to share it with others, I've been into dancing since I was a kid and it's been my passion and escape, so I can teach others the joy that dancing brings me, Losing my job gave me the push to finally start my dream business: my own dance studio!, Losing my job was tough but it gave me the push I needed to do what I love, I want to create a place for people to dance and express themselves - it's been a dream of mine, I want it to be a place of support and encouragement for all our dancers"

Relations touching gold sessions:
- `Jon's banker job loss --enabled_start_of--> Jon's dance studio` [causal] sessions=[1, 5] conf=0.86
- `nearby festival performance --reaffirmed--> Jon's dancing passion` [causal] sessions=[1, 5] conf=0.91

### q7 (cat2)

**Q**: When did Gina launch an ad campaign for her store?  
**Gold**: 29 January, 2023  
**Gold sessions**: [2] turns ['D2:1']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s02, s08, s05, s16, s18 | [1] | 1.0 | False | 3 |
| C2 | s02, s08, s05, s18, s03 | [1] | 1.0 | False | 3 |
| cmpo_compp4 | s02, s08, s05, s18, s03 | [1] | 1.0 | False | 3 |

**P0 on G_raw (E0)**: 'No answer' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: '29 January, 2023' (EM 1, F1 1.00)  
**P0 on this graph (E2)**: 'No answer'  
**P7 on this graph (E3)**: '29 January, 2023'

### q13 (cat2)

**Q**: When did Gina open her online clothing store?  
**Gold**: 16 March, 2023  
**Gold sessions**: [6] turns ['D6:6']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s06, s10, s14, s07, s08 | [1] | 1.0 | False | 5 |
| C2 | s10, s06, s07, s14, s08 | [2] | 1.0 | False | 5 |
| cmpo_compp4 | s10, s06, s14, s07, s08 | [2] | 1.0 | False | 5 |

**P0 on G_raw (E0)**: 'No answer' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: '16 March, 2023' (EM 1, F1 1.00)  
**P0 on this graph (E2)**: 'No answer'  
**P7 on this graph (E3)**: '16 March, 2023'

Relations touching gold sessions:
- `Gina's online clothing store --blended--> dance and fashion` [identity_unification] sessions=[6, 9, 14] conf=0.84

### q19 (cat2)

**Q**: When did Gina interview for a design internship?  
**Gold**: 10 May, 2023  
**Gold sessions**: [11] turns ['D11:14']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s11, s12, s16, s05, s03 | [1] | 1.0 | False | 1 |
| C2 | s11, s12, s16, s05, s03 | [1] | 1.0 | False | 2 |
| cmpo_compp4 | s11, s12, s16, s05, s03 | [1] | 1.0 | False | 3 |

**P0 on G_raw (E0)**: 'yesterday' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: '10 May, 2023' (EM 1, F1 1.00)  
**P0 on this graph (E2)**: 'yesterday'  
**P7 on this graph (E3)**: '10 May, 2023'

Relations touching gold sessions:
- `Jon's job loss --pushed_start_of--> Jon's dance studio` [causal] sessions=[9, 11, 17] conf=0.96
- `Gina's fashion internship --followed--> Gina's design internship interview` [temporal_ordering] sessions=[11, 12] conf=0.82
- `Jon's dance studio --became_space_for--> dancer self-expression` [state_change] sessions=[11, 13, 19] conf=0.89

### q30 (cat2)

**Q**: When Jon is planning to open his dance studio?  
**Gold**: 20 June, 2023  
**Gold sessions**: [15] turns ['D15:5']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s11, s15, s03, s01, s02 | [2] | 1.0 | False | 4 |
| C2 | s11, s15, s01, s03, s02 | [2] | 1.0 | False | 4 |
| cmpo_compp4 | s11, s15, s01, s02, s05 | [2] | 1.0 | False | 4 |

**P0 on G_raw (E0)**: 'tomorrow' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: '20 June, 2023' (EM 1, F1 1.00)  
**P0 on this graph (E2)**: '20 June, 2023'  
**P7 on this graph (E3)**: '20 June, 2023'

Relations touching gold sessions:
- `Jon's dance studio --opened_after--> Jon's location search` [temporal_ordering] sessions=[3, 15, 19] conf=0.9

### q38 (cat2)

**Q**: When did Gina go to a dance class with a group of friends?  
**Gold**: 21 July 2023  
**Gold sessions**: [19] turns ['D19:6']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s19, s01, s05, s17, s02 | [1] | 1.0 | False | 1 |
| C2 | s19, s01, s05, s17, s02 | [1] | 1.0 | False | 2 |
| cmpo_compp4 | s19, s01, s05, s17, s02 | [1] | 1.0 | False | 2 |

**P0 on G_raw (E0)**: 'Last Friday' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: '21 July, 2023' (EM 1, F1 1.00)  
**P0 on this graph (E2)**: 'Last Friday'  
**P7 on this graph (E3)**: '21 July, 2023'

Relations touching gold sessions:
- `Jon's dance studio --opened_after--> Jon's location search` [temporal_ordering] sessions=[3, 15, 19] conf=0.9
- `Gina's support --sustained_motivation_for--> Jon's dance studio` [enabling_link] sessions=[7, 16, 19] conf=0.88
- `Jon's dance studio --became_space_for--> dancer self-expression` [state_change] sessions=[11, 13, 19] conf=0.89
- `Jon's dance studio --expanded_into--> one-on-one mentoring` [state_change] sessions=[13, 17, 19] conf=0.87

### q39 (cat4)

**Q**: What is Gina's favorite style of dance?  
**Gold**: Contemporary  
**Gold sessions**: [1] turns ['D1:9']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s01, s05, s02, s17, s04 | [1] | 1.0 | False | 1 |
| C2 | s01, s05, s02, s17, s08 | [1] | 1.0 | False | 1 |
| cmpo_compp4 | s01, s05, s02, s17, s08 | [1] | 1.0 | False | 1 |

**P0 on G_raw (E0)**: 'contemporary dance' (EM 0, F1 0.67)  
**P7 on G_raw (E1)**: 'contemporary' (EM 1, F1 1.00)  
**P0 on this graph (E2)**: 'contemporary'  
**P7 on this graph (E3)**: 'contemporary'

Relations touching gold sessions:
- `Jon's banker job loss --enabled_start_of--> Jon's dance studio` [causal] sessions=[1, 5] conf=0.86
- `nearby festival performance --reaffirmed--> Jon's dancing passion` [causal] sessions=[1, 5] conf=0.91

### q71 (cat4)

**Q**: How does Gina describe the studio that Jon has opened?  
**Gold**: amazing  
**Gold sessions**: [15] turns ['D15:6']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s11, s19, s15, s03, s05 | [3] | 1.0 | False | 4 |
| C2 | s11, s19, s15, s03, s05 | [3] | 1.0 | False | 4 |
| cmpo_compp4 | s11, s19, s15, s03, s05 | [3] | 1.0 | False | 4 |

**P0 on G_raw (E0)**: 'amazing' (EM 1, F1 1.00)  
**P7 on G_raw (E1)**: 'The studio looks amazing' (EM 0, F1 0.50)  
**P0 on this graph (E2)**: 'amazing'  
**P7 on this graph (E3)**: 'The studio looks amazing'

Relations touching gold sessions:
- `Jon's dance studio --opened_after--> Jon's location search` [temporal_ordering] sessions=[3, 15, 19] conf=0.9

### q72 (cat4)

**Q**: How does Jon feel about the opening night of his dance studio?  
**Gold**: excited  
**Gold sessions**: [15] turns ['D15:7']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s05, s11, s15, s09, s03 | [3] | 1.0 | False | 4 |
| C2 | s05, s11, s15, s09, s01 | [3] | 1.0 | False | 4 |
| cmpo_compp4 | s05, s11, s15, s09, s01 | [3] | 1.0 | False | 4 |

**P0 on G_raw (E0)**: 'excited' (EM 1, F1 1.00)  
**P7 on G_raw (E1)**: "I'm excited!, feeling good and ready to give it my best, Tomorrow's gonna be an awesome night, Can't wait for it" (EM 0, F1 0.10)  
**P0 on this graph (E2)**: 'excited'  
**P7 on this graph (E3)**: "I'm excited!, feeling good and ready to give it my best, Tomorrow's gonna be an awesome night, Woohoo! Tomorrow's opening will be so much fun. Can't wait for it - and for you to be there!"

Relations touching gold sessions:
- `Jon's dance studio --opened_after--> Jon's location search` [temporal_ordering] sessions=[3, 15, 19] conf=0.9

### q78 (cat4)

**Q**: According to Gina, what makes Jon a perfect mentor and guide?  
**Gold**: His positivity and determination  
**Gold sessions**: [17] turns ['D17:7']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s07, s17, s14, s13, s05 | [2] | 1.0 | False | 4 |
| C2 | s17, s14, s05, s13, s04 | [1] | 1.0 | False | 2 |
| cmpo_compp4 | s17, s14, s05, s13, s04 | [1] | 1.0 | False | 2 |

**P0 on G_raw (E0)**: 'His positivity and determination' (EM 1, F1 1.00)  
**P7 on G_raw (E1)**: 'Loving what you do and bringing joy to others, Your positivity and determination' (EM 0, F1 0.35)  
**P0 on this graph (E2)**: 'Your positivity and determination'  
**P7 on this graph (E3)**: 'Loving what you do and bringing joy to others is so rewarding, Your positivity and determination will make your dance studio a hit'

Relations touching gold sessions:
- `Jon's job loss --pushed_start_of--> Jon's dance studio` [causal] sessions=[9, 11, 17] conf=0.96
- `Gina's marketing advice --led_to--> Jon's dance videos` [causal] sessions=[14, 17] conf=0.9
- `Jon's dance studio --expanded_into--> one-on-one mentoring` [state_change] sessions=[13, 17, 19] conf=0.87

### q85 (cat5)

**Q**: What is Gina's attitude towards participating in the dance festival?  
**Gold**: None  
**Gold sessions**: [1] turns ['D1:28']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s11, s01, s05, s08, s04 | [2] | 1.0 | False | 4 |
| C2 | s01, s11, s04, s05, s08 | [1] | 1.0 | False | 4 |
| cmpo_compp4 | s01, s11, s04, s05, s08 | [1] | 1.0 | False | 4 |

**P0 on G_raw (E0)**: 'No answer' (EM 1, F1 1.00)  
**P7 on G_raw (E1)**: "Wow, they look great! Can't wait to see them rock the festival. Gonna be awesome!" (EM 0, F1 0.00)  
**P0 on this graph (E2)**: 'No answer'  
**P7 on this graph (E3)**: "Can't wait to see them rock the festival. Gonna be awesome!"

Relations touching gold sessions:
- `Jon's banker job loss --enabled_start_of--> Jon's dance studio` [causal] sessions=[1, 5] conf=0.86
- `nearby festival performance --reaffirmed--> Jon's dancing passion` [causal] sessions=[1, 5] conf=0.91

### q63 (cat4)

**Q**: What kind of professional experience did Gina get accepted for on May 23, 2023?  
**Gold**: fashion internship  
**Gold sessions**: [12] turns ['D12:1']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s12, s16, s02, s03, s17 | [1] | 1.0 | False | 3 |
| C2 | s12, s02, s16, s03, s17 | [1] | 1.0 | False | 3 |
| cmpo_compp4 | s12, s11, s02, s16, s17 | [1] | 1.0 | False | 3 |

**P0 on G_raw (E0)**: 'No answer' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: 'No answer' (EM 0, F1 0.00)  
**P0 on this graph (E2)**: 'No answer'  
**P7 on this graph (E3)**: 'No answer'

Relations touching gold sessions:
- `Gina's fashion internship --followed--> Gina's design internship interview` [temporal_ordering] sessions=[11, 12] conf=0.82
- `Jon's whiteboard --supported_tracking_of--> Jon's business milestones` [enabling_link] sessions=[12, 13, 16] conf=0.91

### q67 (cat4)

**Q**: How does Jon use the clipboard with a notepad attached to it?  
**Gold**: To set goals, track achievements, and find areas for improvement  
**Gold sessions**: [13] turns ['D13:11']

| graph | top-5 | gold rank | Ev@4096 | fallback | facts after |
|---|---|---|---:|---|---:|
| G_raw | s13, s12, s17, s04, s19 | [1] | 1.0 | True | 0 |
| C2 | s13, s12, s17, s04, s19 | [1] | 1.0 | True | 0 |
| cmpo_compp4 | s13, s12, s17, s04, s19 | [1] | 1.0 | True | 0 |

**P0 on G_raw (E0)**: 'No answer' (EM 0, F1 0.00)  
**P7 on G_raw (E1)**: 'No answer' (EM 0, F1 0.00)  
**P0 on this graph (E2)**: 'No answer'  
**P7 on this graph (E3)**: 'No answer'

Relations touching gold sessions:
- `Jon's whiteboard --supported_tracking_of--> Jon's business milestones` [enabling_link] sessions=[12, 13, 16] conf=0.91
- `Jon's dance studio --became_space_for--> dancer self-expression` [state_change] sessions=[11, 13, 19] conf=0.89
- `Jon's dance studio --expanded_into--> one-on-one mentoring` [state_change] sessions=[13, 17, 19] conf=0.87


## 附录 G：105 题完整错误归因（G_raw / P0）

| qid | cat | question | gold | pred | EM | F1 | offF1 | Ev full/partial | main | secondary | rationale |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q0 | 2 | When Jon has lost his job as a banker? | 19 January, 2023 | 19 January, 2023 | 1 | 1.00 | 1.00 | 1/1.00 | - | - | correct |
| q1 | 2 | When Gina has lost her job at Door Dash? | January, 2023 | this month | 0 | 0.00 | 0.00 | 1/1.00 | C | - | date evidence present but temporal reasoning failed |
| q2 | 4 | How do Jon and Gina both like to destress? | by dancing | dance | 0 | 0.00 | 0.67 | 0/0.00 | A | B,H | gold session missing but answer content largely present |
| q3 | 1 | What do Jon and Gina both have in common? | They lost their jobs and decided to start their own busines… | both face the same challenges | 0 | 0.00 | 0.00 | 0/0.00 | A | - | gold evidence missing from context |
| q4 | 4 | Why did Jon decide to start his dance studio? | He lost his job and decided to start his own business to sh… | Losing my job gave me the push to finally start my dream business: my… | 0 | 0.32 | 0.33 | 0/0.00 | A | B | gold session missing but answer content largely present |
| q5 | 1 | What Jon thinks the ideal dance studio should look like? | By the water, with natural light and Marley flooring | by the water | 0 | 0.40 | 0.50 | 1/1.00 | D | H | multi-hop answer missing part(s) of the gold facts |
| q6 | 2 | When is Jon's group performing at a festival? | February, 2023 | next month | 0 | 0.00 | 0.00 | 1/1.00 | C | - | date evidence present but temporal reasoning failed |
| q7 | 2 | When did Gina launch an ad campaign for her store? | 29 January, 2023 | No answer | 0 | 0.00 | 0.00 | 1/1.00 | F | C | gold evidence in context but model refused |
| q8 | 2 | When was Jon in Paris? | 28 January 2023 | 28 January, 2023 | 1 | 1.00 | 1.00 | 1/1.00 | - | - | correct |
| q9 | 1 | Which city have both Jean and John visited? | Rome | No answer | 0 | 0.00 | 0.00 | 1/1.00 | F | - | gold evidence in context but model refused |
| q10 | 2 | When did Gina team up with a local artist for some cool designs? | February, 2023 | No answer | 0 | 0.00 | 0.00 | 1/1.00 | F | C | gold evidence in context but model refused |
| q11 | 2 | When did Gina get her tattoo? | A few years ago | a few years ago | 1 | 1.00 | 1.00 | 1/1.00 | - | - | correct |
| q12 | 2 | When did Jon start to go to the gym? | March, 2023 | last week | 0 | 0.00 | 0.00 | 1/1.00 | C | - | date evidence present but temporal reasoning failed |
| q13 | 2 | When did Gina open her online clothing store? | 16 March, 2023 | No answer | 0 | 0.00 | 0.00 | 1/1.00 | F | C | gold evidence in context but model refused |
| q14 | 2 | When did Jon start expanding his studio's social media presence? | April, 2023 | 3 April, 2023 | 0 | 0.80 | 0.80 | 1/1.00 | C | H | date evidence present but temporal reasoning failed |
| q15 | 2 | When did Jon host a dance competition? | May, 2023 | No answer | 0 | 0.00 | 0.00 | 1/1.00 | F | C | gold evidence in context but model refused |
| q16 | 2 | When did Jon go to a fair to get more exposure for his dance studio? | 24 April, 2023 | No answer | 0 | 0.00 | 0.00 | 0/0.00 | A | - | no gold evidence in context; model refused |
| q17 | 1 | Why did Gina decide to start her own clothing store? | She always loved fashion trends and finding unique pieces a… | After losing my job, I wanted to take control of my own destiny | 0 | 0.16 | 0.13 | 0/0.00 | A | B | gold session missing but answer content largely present |
| q18 | 1 | Do Jon and Gina start businesses out of what they love? | Yes | Yes | 1 | 1.00 | 1.00 | 1/1.00 | - | - | correct |
| q19 | 2 | When did Gina interview for a design internship? | 10 May, 2023 | yesterday | 0 | 0.00 | 0.00 | 1/1.00 | A | C | gold session present but the exact date is not in context |
| q20 | 2 | When did Gina get accepted for the design internship? | 27 May, 2023 | No answer | 0 | 0.00 | 0.00 | 1/1.00 | F | C | gold evidence in context but model refused |
| q21 | 2 | When did Jon start reading "The Lean Startup"? | May, 2023 | No answer | 0 | 0.00 | 0.00 | 1/1.00 | F | C | gold evidence in context but model refused |
| q22 | 2 | When did Gina develop a video presentation to teach how to style her fashion pi… | June, 2023 | No answer | 0 | 0.00 | 0.00 | 1/1.00 | F | C | gold evidence in context but model refused |
| q23 | 1 | How did Gina promote her clothes store? | worked with an artist to make unique fashion pieces, made l… | launched an ad campaign, new offers and promotions, working with fash… | 0 | 0.27 | 0.25 | 0/0.50 | A | B | gold evidence missing from context |
| q24 | 1 | Which events has Jon participated in to promote his business venture? | fair, networking events, dance competition | a fair, networking events, a dance competition | 1 | 1.00 | 1.00 | 1/1.00 | - | - | correct |
| q25 | 1 | What does Jon's dance studio offer? | one-on-one metoring and training to dancers,  workshops and… | dance classes and workshops, one-on-one mentoring and training | 0 | 0.55 | 0.47 | 0/0.50 | A | B | gold evidence missing from context |
| q26 | 2 | When did Jon receive mentorship to promote his venture? | 15 June, 2023 | 15 June, 2023 | 1 | 1.00 | 1.00 | 1/1.00 | - | - | correct |
| q27 | 1 | Did Jon and Gina both participate in dance competitions? | Yes | Yes | 1 | 1.00 | 1.00 | 0/0.80 | - | - | correct |
| q28 | 2 | When was Jon in Rome? | June 2023 | last week | 0 | 0.00 | 0.00 | 1/1.00 | C | - | date evidence present but temporal reasoning failed |
| q29 | 1 | Which cities has Jon visited? | Paris, Rome | Paris, Rome | 1 | 1.00 | 1.00 | 1/1.00 | - | - | correct |
| q30 | 2 | When Jon is planning to open his dance studio? | 20 June, 2023 | tomorrow | 0 | 0.00 | 0.00 | 1/1.00 | C | - | date evidence present but temporal reasoning failed |
| q31 | 1 | How long did it take for Jon to open his studio? | six months | No answer | 0 | 0.00 | 0.00 | 0/0.50 | B | F | only part of gold evidence in context; model refused |
| q32 | 2 | When did Gina design a limited collection of hoodies? | June 2023 | last week | 0 | 0.00 | 0.00 | 1/1.00 | C | - | date evidence present but temporal reasoning failed |
| q33 | 2 | When did Jon visit networking events for his store? | 20 June, 2023 | No answer | 0 | 0.00 | 0.00 | 1/1.00 | F | C | gold evidence in context but model refused |
| q34 | 2 | When did Gina start being recognized by fashion editors? | July 2023 | No answer | 0 | 0.00 | 0.00 | 1/1.00 | F | C | gold evidence in context but model refused |
| q35 | 2 | When did Jon start learning marketing and analytics tools? | July, 2023 | 9 July, 2023 | 0 | 0.80 | 0.80 | 1/1.00 | C | H | date evidence present but temporal reasoning failed |
| q36 | 2 | When did Jon and Gina decide to collaborate to create dance content? | 21 July 2023 | 21 July, 2023 | 1 | 1.00 | 1.00 | 1/1.00 | - | - | correct |
| q37 | 2 | When did Gina mention Shia Labeouf? | 23 July, 2023 | 23 July, 2023 | 1 | 1.00 | 1.00 | 1/1.00 | - | - | correct |
| q38 | 2 | When did Gina go to a dance class with a group of friends? | 21 July 2023 | Last Friday | 0 | 0.00 | 0.00 | 1/1.00 | C | - | date evidence present but temporal reasoning failed |
| q39 | 4 | What is Gina's favorite style of dance? | Contemporary | contemporary dance | 0 | 0.67 | 0.67 | 1/1.00 | E | H | semantically close but too short / different wording |
| q40 | 4 | What is Jon's favorite style of dance? | Contemporary | contemporary | 1 | 1.00 | 1.00 | 1/1.00 | - | - | correct |
| q41 | 4 | What was Gina's favorite dancing memory? | Winning first place at a regionals dance competition | when my team won first place at a regionals at age fifteen | 0 | 0.44 | 0.44 | 1/1.00 | E | - | semantically close but too short / different wording |
| q42 | 4 | What kind of dance piece did Gina's team perform to win first place? | "Finding Freedom" | a contemporary piece called "Finding Freedom" | 0 | 0.57 | 0.57 | 1/1.00 | E | H | semantically close but too short / different wording |
| q43 | 4 | What do the dancers in the photo represent? | They are performing at the festival | the ones performing at the festival | 0 | 0.67 | 0.67 | 1/1.00 | E | H | semantically close but too short / different wording |
| q44 | 4 | What does Gina say about the dancers in the photo? | They look graceful | they're so graceful | 0 | 0.33 | 0.33 | 1/1.00 | E | - | semantically close but too short / different wording |
| q45 | 4 | What is Jon's attitude towards being part of the dance festival? | Glad | Glad to be part of it | 0 | 0.29 | 0.29 | 1/1.00 | E | H | semantically close but too short / different wording |
| q46 | 4 | What kind of flooring is Jon looking for in his dance studio? | Marley flooring | Marley flooring | 1 | 1.00 | 1.00 | 1/1.00 | - | - | correct |
| q47 | 4 | What did Gina find for her clothing store on 1 February, 2023? | The perfect spot for her store | a perfect spot for your store | 0 | 0.80 | 0.80 | 1/1.00 | E | H | semantically close but too short / different wording |
| q48 | 4 | What did Gina design for her store? | the space, furniture, and decor | a limited edition line | 0 | 0.00 | 0.00 | 1/1.00 | D | - | list answer incomplete |
| q49 | 4 | What did Gina want her customers to feel in her store? | cozy and comfortable | cozy and inviting | 0 | 0.67 | 0.50 | 1/1.00 | E | - | semantically close but too short / different wording |
| q50 | 4 | What did Jon say about Gina's progress with her store? | hard work's paying off | Looks like your store is growing | 0 | 0.00 | 0.00 | 1/1.00 | G | - | answer not supported by the provided context |
| q51 | 4 | What made Gina choose the furniture and decor for her store? | personal style and customer comfort | she wanted to make the place look like her own style and make her cus… | 0 | 0.19 | 0.21 | 1/1.00 | E | - | semantically close but too short / different wording |
| q52 | 4 | What did Jon say about creating a special experience for customers? | It's the key to making them feel welcome and coming back | Creating a special experience for customers is the key to making them… | 0 | 0.72 | 0.70 | 1/1.00 | E | - | semantically close but too short / different wording |
| q53 | 4 | What did Gina say about creating an experience for her customers? | making them want to come back | Just creating an experience that'll make 'em wanna come back | 0 | 0.27 | 0.40 | 1/1.00 | E | - | semantically close but too short / different wording |
| q54 | 4 | How is Gina's store doing? | The store is doing great. | doing great | 0 | 0.67 | 0.67 | 1/1.00 | E | H | semantically close but too short / different wording |
| q55 | 4 | What does Gina's tattoo symbolize? | Freedom and expressing herself through dance | freedom - dancing without worrying what people think | 0 | 0.15 | 0.33 | 1/1.00 | E | - | semantically close but too short / different wording |
| q56 | 4 | What did Jon and Gina compare their entrepreneurial journeys to? | dancing together and supporting each other | a roller coaster | 0 | 0.00 | 0.00 | 1/1.00 | G | - | answer not supported by the provided context |
| q57 | 4 | What advice does Gina give to Jon about running a successful business? | build relationships with customers, create a strong brand i… | Don't be scared to reach out to people in your field for help and con… | 0 | 0.00 | 0.00 | 0/0.00 | A | - | gold evidence missing from context |
| q58 | 4 | Why did Jon shut down his bank account? | for his business | for his biz | 0 | 0.67 | 0.67 | 1/1.00 | E | - | semantically close but too short / different wording |
| q59 | 4 | Why did Gina combine her clothing business with dance? | she is passionate about dance and fashion | she's passionate about dance and fashion so combining them lets her s… | 0 | 0.33 | 0.30 | 1/1.00 | E | - | semantically close but too short / different wording |
| q60 | 4 | What does Jon's dance make him? | happy | so happy | 0 | 0.67 | 0.67 | 1/1.00 | E | H | semantically close but too short / different wording |
| q61 | 4 | What did Gina receive from a dance contest? | a trophy | one of my trophies | 0 | 0.00 | 0.40 | 1/1.00 | E | H | semantically close but too short / different wording |
| q62 | 4 | How does Gina stay confident in her business? | By reminding herself of her successes and progress, having … | by reminding myself of my successes and progress, having a good suppo… | 0 | 0.68 | 0.65 | 1/1.00 | E | - | semantically close but too short / different wording |
| q63 | 4 | What kind of professional experience did Gina get accepted for on May 23, 2023? | fashion internship | No answer | 0 | 0.00 | 0.00 | 1/1.00 | F | - | gold evidence in context but model refused |
| q64 | 4 | Where is Gina's fashion internship? | fashion department of an international company | No answer | 0 | 0.00 | 0.00 | 1/1.00 | F | - | gold evidence in context but model refused |
| q65 | 4 | What book is Jon currently reading? | The Lean Startup | The Lean Startup | 1 | 1.00 | 1.00 | 1/1.00 | - | - | correct |
| q66 | 4 | What is Jon offering to the dancers at his dance studio? | One-on-one mentoring and training | dance classes and workshops, one-on-one mentoring and training | 0 | 0.67 | 0.67 | 1/1.00 | E | H | semantically close but too short / different wording |
| q67 | 4 | How does Jon use the clipboard with a notepad attached to it? | To set goals, track achievements, and find areas for improv… | No answer | 0 | 0.00 | 0.00 | 1/1.00 | F | - | gold evidence in context but model refused |
| q68 | 4 | What does Jon tell Gina he won't do? | quit | give up | 0 | 0.00 | 0.00 | 0/0.00 | A | B | gold session missing but answer content largely present |
| q69 | 4 | What did Jon take a trip to Rome for? | To clear his mind | to clear my mind a little | 0 | 0.67 | 0.67 | 1/1.00 | E | H | semantically close but too short / different wording |
| q70 | 4 | What is Jon working on opening? | a dance studio | a dance studio | 1 | 1.00 | 1.00 | 1/1.00 | - | - | correct |
| q71 | 4 | How does Gina describe the studio that Jon has opened? | amazing | amazing | 1 | 1.00 | 1.00 | 1/1.00 | - | - | correct |
| q72 | 4 | How does Jon feel about the opening night of his dance studio? | excited | excited | 1 | 1.00 | 1.00 | 1/1.00 | - | - | correct |
| q73 | 4 | How does Gina describe the feeling that dance brings? | magical | ain't nothing like the feeling it gives us | 0 | 0.00 | 0.00 | 0/0.00 | A | - | gold evidence missing from context |
| q74 | 4 | What does Jon plan to do at the grand opening of his dance studio? | savor all the good vibes | make some awesome memories | 0 | 0.00 | 0.00 | 1/1.00 | G | - | answer not supported by the provided context |
| q75 | 4 | What does Gina say to Jon about the grand opening? | Let's live it up and make some great memories | Can't wait for tomorrow's grand opening! | 0 | 0.00 | 0.00 | 1/1.00 | G | - | answer not supported by the provided context |
| q76 | 4 | What is the general sentiment about the upcoming grand opening? | excitement | excited and positive | 0 | 0.00 | 0.67 | 1/1.00 | E | H | semantically close but too short / different wording |
| q77 | 4 | What did Gina make a limited edition line of? | Hoodies | hoodie | 0 | 0.00 | 1.00 | 1/1.00 | E | H | semantically close but too short / different wording |
| q78 | 4 | According to Gina, what makes Jon a perfect mentor and guide? | His positivity and determination | His positivity and determination | 1 | 1.00 | 1.00 | 1/1.00 | - | - | correct |
| q79 | 5 | What temporary job did Jon take to cover expenses? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q80 | 4 | What plans does Jon have after receiving advice at the networking event? | Sprucing up his business plan, tweaking his pitch to invest… | sprucing up my biz plan and tweaking my pitch to investors, and worki… | 0 | 0.65 | 0.65 | 1/1.00 | D | - | list answer incomplete |
| q81 | 4 | What offer does Gina make to Jon regarding social media? | Helping with making content and managing his social media a… | help with making content or even managing your accounts | 0 | 0.53 | 0.67 | 1/1.00 | E | - | semantically close but too short / different wording |
| q82 | 5 | What is Jon's favorite style of painting? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q83 | 5 | What was Jon's favorite dancing memory? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q84 | 5 | What kind of dance piece did Jon's team perform to win first place? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q85 | 5 | What is Gina's attitude towards participating in the dance festival? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q86 | 5 | What kind of flooring is Gina looking for in her dance studio? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q87 | 5 | What did Jon find for his clothing store on 1 February, 2023? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q88 | 5 | What did Jon design for his store? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q89 | 5 | What did Jon want his customers to feel in her store? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q90 | 5 | What made Jon choose the furniture and decor for his store? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q91 | 5 | How is Jon's store doing? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q92 | 5 | What does Jon's tattoo symbolize? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q93 | 5 | Why did Gina shut down her bank account? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q94 | 5 | Why did Jon combine his clothing business with dance? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q95 | 5 | What did Gina receive from a dance contest? |  | one of my trophies | 0 | 0.00 | 0.00 | 1/1.00 | G | - | answered an adversarial question instead of refusing |
| q96 | 5 | What kind of professional experience did Jon get accepted for on May 23, 2023? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q97 | 5 | Where is Gina's HR internship? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q98 | 5 | Where is Jon's fashion internship? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q99 | 5 | What book is Gina currently reading? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q100 | 5 | How does Gina use the clipboard with a notepad attached to it? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q101 | 5 | What did Jon take a trip to Barcelona for? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q102 | 5 | What did Jon make a limited edition line of? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q103 | 5 | What temporary job did Gina take to cover expenses? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |
| q104 | 5 | What plans does Gina have after receiving advice at the networking event? |  | No answer | 1 | 1.00 | 0.00 | 1/1.00 | - | - | correct refusal on adversarial question |

标签：A 检索错误；B 证据不完整；C 时间推理；D 多事实整合；E 表达/部分正确；F 过度拒答；G 错误推断/幻觉；H 评测器/语义等价。


## 附录 H：诊断实验明细

### H.1 Oracle / Full-Context 汇总

| condition | n | EM | F1 | offF1 | refusal | ctx tokens |
|---|---|---|---|---|---|---|
| R2_P0 | 81 | 0.185 | 0.387 | 0.402 | 0.173 | 761 |
| R2_P7 | 81 | 0.272 | 0.603 | 0.627 | 0.062 | 761 |
| R3_P7 | 105 | 0.259 | 0.546 | 0.577 | 0.049 | 11796 |
| R3_P0 | 105 | 0.198 | 0.360 | 0.400 | 0.173 | 11435 |

### H.2 重复运行（temperature=0，禁用缓存）

**P0**：词面不稳定 14.3%，得分不稳定 14.3%，Jaccard 0.976，每次 EM [0.214, 0.286, 0.357]，多数票 0.286

| qid | cat | gold | 3 次答案 | 3 次 EM | Jaccard | 多数票 EM |
|---|---|---|---|---|---|---|
| q4 | 4 | He lost his job and decided to start his own busi… | Losing my job gave me the push to finally start my dream business: my… / Losing my job gave me the push to finally start my dream business: my… / Losing my job gave me the push to finally start my dream business: my… | [0, 0, 0] | 1.000 | 0 |
| q9 | 1 | Rome | No answer / No answer / No answer | [0, 0, 0] | 1.000 | 0 |
| q16 | 2 | 24 April, 2023 | No answer / No answer / No answer | [0, 0, 0] | 1.000 | 0 |
| q17 | 1 | She always loved fashion trends and finding uniqu… | After losing my job, I wanted to take control of my own destiny / After losing my job, I wanted to take control of my own destiny / After losing my job, I wanted to take control of my own destiny | [0, 0, 0] | 1.000 | 0 |
| q24 | 1 | fair, networking events, dance competition | a fair, networking events, a dance competition, a dance comp / a fair, networking events, a dance competition / a fair, networking events, a dance competition | [0, 1, 1] | 0.889 | 1 |
| q29 | 1 | Paris, Rome | Paris, Rome / Paris, Rome / Paris, Rome | [1, 1, 1] | 1.000 | 1 |
| q31 | 1 | six months | No answer / No answer / No answer | [0, 0, 0] | 1.000 | 0 |
| q33 | 2 | 20 June, 2023 | No answer / No answer / No answer | [0, 0, 0] | 1.000 | 0 |
| q39 | 4 | Contemporary | contemporary / contemporary / contemporary | [1, 1, 1] | 1.000 | 1 |
| q47 | 4 | The perfect spot for her store | a perfect spot for your store / a perfect spot for your store / a perfect spot for her store | [0, 0, 1] | 0.778 | 0 |
| q54 | 4 | The store is doing great. | doing great / doing great / doing great | [0, 0, 0] | 1.000 | 0 |
| q63 | 4 | fashion internship | No answer / No answer / No answer | [0, 0, 0] | 1.000 | 0 |
| q74 | 4 | savor all the good vibes | make some awesome memories / make some awesome memories / make some awesome memories | [0, 0, 0] | 1.000 | 0 |
| q85 | 5 |  | No answer / No answer / No answer | [1, 1, 1] | 1.000 | 1 |

**P7**：词面不稳定 28.6%，得分不稳定 0.0%，Jaccard 0.886，每次 EM [0.143, 0.143, 0.143]，多数票 0.143

| qid | cat | gold | 3 次答案 | 3 次 EM | Jaccard | 多数票 EM |
|---|---|---|---|---|---|---|
| q4 | 4 | He lost his job and decided to start his own busi… | Losing my job gave me the push to finally start my dream business: my… / Losing my job gave me the push to finally start my dream business: my… / Losing my job gave me the push to finally start my dream business: my… | [0, 0, 0] | 0.760 | 0 |
| q9 | 1 | Rome | No answer / No answer / No answer | [0, 0, 0] | 1.000 | 0 |
| q16 | 2 | 24 April, 2023 | No answer / No answer / No answer | [0, 0, 0] | 1.000 | 0 |
| q17 | 1 | She always loved fashion trends and finding uniqu… | After losing my job, I wanted to take control of my own destiny, I've… / After losing my job, I wanted to take control of my own destiny, I've… / After losing my job, I wanted to take control of my own destiny, I've… | [0, 0, 0] | 1.000 | 0 |
| q24 | 1 | fair, networking events, dance competition | networking events, a fair, the last networking event / networking events, a fair, the last networking event / networking events, a fair, a dance comp near me next month, a dance c… | [0, 0, 0] | 0.500 | 0 |
| q29 | 1 | Paris, Rome | Paris, Rome / Paris, Rome / Paris, Rome | [1, 1, 1] | 1.000 | 1 |
| q31 | 1 | six months | 4 months and 18 days / No answer / 4 months and 18 days | [0, 0, 0] | 0.333 | 0 |
| q33 | 2 | 20 June, 2023 | 21 June, 2023 / 21 June, 2023 / 21 June, 2023 | [0, 0, 0] | 1.000 | 0 |
| q39 | 4 | Contemporary | contemporary / contemporary / contemporary | [1, 1, 1] | 1.000 | 1 |
| q47 | 4 | The perfect spot for her store | the perfect spot for your store / the perfect spot for your store / the perfect spot for your store | [0, 0, 0] | 1.000 | 0 |
| q54 | 4 | The store is doing great. | The store's doing great! / The store's doing great! / The store's doing great! | [0, 0, 0] | 1.000 | 0 |
| q63 | 4 | fashion internship | No answer / No answer / No answer | [0, 0, 0] | 1.000 | 0 |
| q74 | 4 | savor all the good vibes | savor all the good vibes, make some awesome memories tomorrow at the … / savor all the good vibes, make some awesome memories tomorrow at the … / savor all the good vibes, make some awesome memories tomorrow at the … | [0, 0, 0] | 1.000 | 0 |
| q85 | 5 |  | Wow, they look great! Can't wait to see them rock the festival. Gonna… / Wow, they look great! Can't wait to see them rock the festival. Gonna… / Can't wait to see them rock the festival. Gonna be awesome! | [0, 0, 0] | 0.810 | 0 |

### H.3 官方 cat-5 多选协议（24 题）

| prompt | accuracy | n | parse |
|---|---|---|---|
| P0 | 0.833 | 24 | {'a': 9, 'b': 3, 'not_mentioned': 8, 'free': 4, 'error': 0} |
| P7 | 0.875 | 24 | {'a': 14, 'b': 5, 'not_mentioned': 3, 'free': 2, 'error': 0} |
| OFFICIAL | 0.875 | 24 | {'a': 14, 'b': 10, 'not_mentioned': 0, 'free': 0, 'error': 0} |

**OFFICIAL 逐题**

| qid | question | adversarial answer | (a) | (b) | raw output | correct |
|---|---|---|---|---|---|---|
| q79 | What temporary job did Jon take to cover expenses? | Not mentioned | Not mentioned | Not mentioned in the conversation | (a) Not mentioned | 0 |
| q82 | What is Jon's favorite style of painting? | Contemporary | Not mentioned in the conversation | Contemporary | (a) Not mentioned in the conversation | 1 |
| q83 | What was Jon's favorite dancing memory? | Winning first place at a regionals dance competit… | Not mentioned in the conversation | Winning first place at a regionals danc… | (a) Not mentioned in the conversation | 1 |
| q84 | What kind of dance piece did Jon's team perform to win first place? | "Finding Freedom" | "Finding Freedom" | Not mentioned in the conversation | (b) Not mentioned in the conversation. | 1 |
| q85 | What is Gina's attitude towards participating in the dance festival? | Glad | Glad | Not mentioned in the conversation | (a) Glad | 0 |
| q86 | What kind of flooring is Gina looking for in her dance studio? | Marley flooring | Marley flooring | Not mentioned in the conversation | (b) Not mentioned in the conversation. | 1 |
| q87 | What did Jon find for his clothing store on 1 February, 2023? | The perfect spot for her store | Not mentioned in the conversation | The perfect spot for her store | (a) Not mentioned in the conversation | 1 |
| q88 | What did Jon design for his store? | the space, furniture, and decor | the space, furniture, and decor | Not mentioned in the conversation | (b) Not mentioned in the conversation. | 1 |
| q89 | What did Jon want his customers to feel in her store? | cozy and comfortable | Not mentioned in the conversation | cozy and comfortable | (a) Not mentioned in the conversation | 1 |
| q90 | What made Jon choose the furniture and decor for his store? | personal style and customer comfort | Not mentioned in the conversation | personal style and customer comfort | (a) Not mentioned in the conversation | 1 |
| q91 | How is Jon's store doing? | The store is doing great. | Not mentioned in the conversation | The store is doing great. | (a) Not mentioned in the conversation | 1 |
| q92 | What does Jon's tattoo symbolize? | Freedom and expressing himself through dance | Freedom and expressing himself through … | Not mentioned in the conversation | (b) Not mentioned in the conversation. | 1 |
| q93 | Why did Gina shut down her bank account? | for her business | Not mentioned in the conversation | for her business | (a) Not mentioned in the conversation | 1 |
| q94 | Why did Jon combine his clothing business with dance? | he is passionate about dance and fashion | he is passionate about dance and fashion | Not mentioned in the conversation | (b) Not mentioned in the conversation. | 1 |
| q95 | What did Gina receive from a dance contest? | a trophy | Not mentioned in the conversation | a trophy | (b) a trophy. | 0 |
| q96 | What kind of professional experience did Jon get accepted for on May 23, 2023? | fashion internship | fashion internship | Not mentioned in the conversation | (b) Not mentioned in the conversation. | 1 |
| q97 | Where is Gina's HR internship? | fashion department of an international company | fashion department of an international … | Not mentioned in the conversation | (b) Not mentioned in the conversation. | 1 |
| q98 | Where is Jon's fashion internship? | fashion department of an international company | Not mentioned in the conversation | fashion department of an international … | (a) Not mentioned in the conversation | 1 |
| q99 | What book is Gina currently reading? | The Lean Startup | Not mentioned in the conversation | The Lean Startup | (a) Not mentioned in the conversation | 1 |
| q100 | How does Gina use the clipboard with a notepad attached to it? | To set goals, track achievements, and find areas … | To set goals, track achievements, and f… | Not mentioned in the conversation | (b) Not mentioned in the conversation. | 1 |
| q101 | What did Jon take a trip to Barcelona for? | To clear his mind | Not mentioned in the conversation | To clear his mind | (a) Not mentioned in the conversation | 1 |
| q102 | What did Jon make a limited edition line of? | Hoodies | Hoodies | Not mentioned in the conversation | (b) Not mentioned in the conversation. | 1 |
| q103 | What temporary job did Gina take to cover expenses? | Not mentioned | Not mentioned in the conversation | Not mentioned | (a) Not mentioned in the conversation | 1 |
| q104 | What plans does Gina have after receiving advice at the networking event? | Sprucing up her business plan, tweaking her pitch… | Not mentioned in the conversation | Sprucing up her business plan, tweaking… | (a) Not mentioned in the conversation | 1 |

**P0 逐题**

| qid | question | adversarial answer | (a) | (b) | raw output | correct |
|---|---|---|---|---|---|---|
| q79 | What temporary job did Jon take to cover expenses? | Not mentioned | Not mentioned | Not mentioned in the conversation | Answer: No answer | 0 |
| q82 | What is Jon's favorite style of painting? | Contemporary | Not mentioned in the conversation | Contemporary | Answer: (a) Not mentioned in the conversation | 1 |
| q83 | What was Jon's favorite dancing memory? | Winning first place at a regionals dance competit… | Not mentioned in the conversation | Winning first place at a regionals danc… | Answer: (a) Not mentioned in the conversation | 1 |
| q84 | What kind of dance piece did Jon's team perform to win first place? | "Finding Freedom" | "Finding Freedom" | Not mentioned in the conversation | Answer: Not mentioned in the conversation. | 1 |
| q85 | What is Gina's attitude towards participating in the dance festival? | Glad | Glad | Not mentioned in the conversation | Answer: Glad | 0 |
| q86 | What kind of flooring is Gina looking for in her dance studio? | Marley flooring | Marley flooring | Not mentioned in the conversation | Answer: Not mentioned in the conversation. | 1 |
| q87 | What did Jon find for his clothing store on 1 February, 2023? | The perfect spot for her store | Not mentioned in the conversation | The perfect spot for her store | Answer: Not mentioned in the conversation | 1 |
| q88 | What did Jon design for his store? | the space, furniture, and decor | the space, furniture, and decor | Not mentioned in the conversation | Answer: (b) Not mentioned in the conversation. | 1 |
| q89 | What did Jon want his customers to feel in her store? | cozy and comfortable | Not mentioned in the conversation | cozy and comfortable | Answer: (a) Not mentioned in the conversation | 1 |
| q90 | What made Jon choose the furniture and decor for his store? | personal style and customer comfort | Not mentioned in the conversation | personal style and customer comfort | Answer: (a) Not mentioned in the conversation | 1 |
| q91 | How is Jon's store doing? | The store is doing great. | Not mentioned in the conversation | The store is doing great. | Answer: (a) Not mentioned in the conversation | 1 |
| q92 | What does Jon's tattoo symbolize? | Freedom and expressing himself through dance | Freedom and expressing himself through … | Not mentioned in the conversation | Answer: (b) Not mentioned in the conversation. | 1 |
| q93 | Why did Gina shut down her bank account? | for her business | Not mentioned in the conversation | for her business | Answer: (a) Not mentioned in the conversation | 1 |
| q94 | Why did Jon combine his clothing business with dance? | he is passionate about dance and fashion | he is passionate about dance and fashion | Not mentioned in the conversation | Answer: No answer | 0 |
| q95 | What did Gina receive from a dance contest? | a trophy | Not mentioned in the conversation | a trophy | Answer: a trophy | 1 |
| q96 | What kind of professional experience did Jon get accepted for on May 23, 2023? | fashion internship | fashion internship | Not mentioned in the conversation | Answer: Not mentioned in the conversation. | 1 |
| q97 | Where is Gina's HR internship? | fashion department of an international company | fashion department of an international … | Not mentioned in the conversation | Answer: Not mentioned in the conversation | 1 |
| q98 | Where is Jon's fashion internship? | fashion department of an international company | Not mentioned in the conversation | fashion department of an international … | Answer: Not mentioned in the conversation | 1 |
| q99 | What book is Gina currently reading? | The Lean Startup | Not mentioned in the conversation | The Lean Startup | Answer: Not mentioned in the conversation | 1 |
| q100 | How does Gina use the clipboard with a notepad attached to it? | To set goals, track achievements, and find areas … | To set goals, track achievements, and f… | Not mentioned in the conversation | Answer: (b) Not mentioned in the conversation | 1 |
| q101 | What did Jon take a trip to Barcelona for? | To clear his mind | Not mentioned in the conversation | To clear his mind | Answer: (a) Not mentioned in the conversation | 1 |
| q102 | What did Jon make a limited edition line of? | Hoodies | Hoodies | Not mentioned in the conversation | Answer: Not mentioned in the conversation. | 1 |
| q103 | What temporary job did Gina take to cover expenses? | Not mentioned | Not mentioned in the conversation | Not mentioned | Answer: No answer | 0 |
| q104 | What plans does Gina have after receiving advice at the networking event? | Sprucing up her business plan, tweaking her pitch… | Not mentioned in the conversation | Sprucing up her business plan, tweaking… | Answer: (a) Not mentioned in the conversation | 1 |

**P7 逐题**

| qid | question | adversarial answer | (a) | (b) | raw output | correct |
|---|---|---|---|---|---|---|
| q79 | What temporary job did Jon take to cover expenses? | Not mentioned | Not mentioned | Not mentioned in the conversation | (a) Not mentioned | 0 |
| q82 | What is Jon's favorite style of painting? | Contemporary | Not mentioned in the conversation | Contemporary | (a) Not mentioned in the conversation | 1 |
| q83 | What was Jon's favorite dancing memory? | Winning first place at a regionals dance competit… | Not mentioned in the conversation | Winning first place at a regionals danc… | (a) Not mentioned in the conversation | 1 |
| q84 | What kind of dance piece did Jon's team perform to win first place? | "Finding Freedom" | "Finding Freedom" | Not mentioned in the conversation | (b) Not mentioned in the conversation. | 1 |
| q85 | What is Gina's attitude towards participating in the dance festival? | Glad | Glad | Not mentioned in the conversation | Glad | 0 |
| q86 | What kind of flooring is Gina looking for in her dance studio? | Marley flooring | Marley flooring | Not mentioned in the conversation | Not mentioned in the conversation. | 1 |
| q87 | What did Jon find for his clothing store on 1 February, 2023? | The perfect spot for her store | Not mentioned in the conversation | The perfect spot for her store | (a) Not mentioned in the conversation | 1 |
| q88 | What did Jon design for his store? | the space, furniture, and decor | the space, furniture, and decor | Not mentioned in the conversation | (b) Not mentioned in the conversation. | 1 |
| q89 | What did Jon want his customers to feel in her store? | cozy and comfortable | Not mentioned in the conversation | cozy and comfortable | (a) Not mentioned in the conversation | 1 |
| q90 | What made Jon choose the furniture and decor for his store? | personal style and customer comfort | Not mentioned in the conversation | personal style and customer comfort | (a) Not mentioned in the conversation | 1 |
| q91 | How is Jon's store doing? | The store is doing great. | Not mentioned in the conversation | The store is doing great. | (a) Not mentioned in the conversation | 1 |
| q92 | What does Jon's tattoo symbolize? | Freedom and expressing himself through dance | Freedom and expressing himself through … | Not mentioned in the conversation | (b) Not mentioned in the conversation. | 1 |
| q93 | Why did Gina shut down her bank account? | for her business | Not mentioned in the conversation | for her business | (a) Not mentioned in the conversation | 1 |
| q94 | Why did Jon combine his clothing business with dance? | he is passionate about dance and fashion | he is passionate about dance and fashion | Not mentioned in the conversation | No answer | 0 |
| q95 | What did Gina receive from a dance contest? | a trophy | Not mentioned in the conversation | a trophy | a trophy | 1 |
| q96 | What kind of professional experience did Jon get accepted for on May 23, 2023? | fashion internship | fashion internship | Not mentioned in the conversation | Not mentioned in the conversation. | 1 |
| q97 | Where is Gina's HR internship? | fashion department of an international company | fashion department of an international … | Not mentioned in the conversation | Not mentioned in the conversation. | 1 |
| q98 | Where is Jon's fashion internship? | fashion department of an international company | Not mentioned in the conversation | fashion department of an international … | (a) Not mentioned in the conversation | 1 |
| q99 | What book is Gina currently reading? | The Lean Startup | Not mentioned in the conversation | The Lean Startup | (a) Not mentioned in the conversation | 1 |
| q100 | How does Gina use the clipboard with a notepad attached to it? | To set goals, track achievements, and find areas … | To set goals, track achievements, and f… | Not mentioned in the conversation | (b) Not mentioned in the conversation. | 1 |
| q101 | What did Jon take a trip to Barcelona for? | To clear his mind | Not mentioned in the conversation | To clear his mind | (a) Not mentioned in the conversation | 1 |
| q102 | What did Jon make a limited edition line of? | Hoodies | Hoodies | Not mentioned in the conversation | (b) Not mentioned in the conversation. | 1 |
| q103 | What temporary job did Gina take to cover expenses? | Not mentioned | Not mentioned in the conversation | Not mentioned | (a) Not mentioned in the conversation | 1 |
| q104 | What plans does Gina have after receiving advice at the networking event? | Sprucing up her business plan, tweaking her pitch… | Not mentioned in the conversation | Sprucing up her business plan, tweaking… | (a) Not mentioned in the conversation | 1 |

### H.4 max_tokens=256 截断敏感性（512 重跑）

| file | qid | cat | gold | 256 EM/F1 | 512 EM/F1 | finish |
|---|---|---|---|---|---|---|
| R3_P7.jsonl | q50 | 4 | hard work's paying off | 0/0.04 | 0/0.05 | stop |
| R3_P7.jsonl | q68 | 4 | quit | 0/0.01 | 0/0.04 | stop |
| C2_P7.jsonl | q3 | 1 | They lost their jobs and decided to start th… | 0/0.03 | 0/0.17 | stop |
| cmpo_compp4_P7.jsonl | q3 | 1 | They lost their jobs and decided to start th… | 0/0.03 | 0/0.25 | stop |

### H.5 HY4 调用统计

| log | lines | ok | errors |
|---|---|---|---|
| hy4_callsco.jsonl | 11 | 11 | 0 |
| hy4_callsjo.jsonl | 237 | 237 | 0 |
| hy4_callsoa.jsonl | 96 | 96 | 0 |
| hy4_callsor.jsonl | 266 | 266 | 0 |
| hy4_callspo.jsonl | 450 | 450 | 0 |
| hy4_callsrp.jsonl | 98 | 84 | 14 |

注意：Comprehension 图的官方 LLM fact rerank 调用走 `outputs/raw_graphs/conv-30/llm_cache`，不在上表；无重排消融的检索调用为 0。


## 附录 I：完整实验日志

# conv-30 Prompt 优化实验执行日志

所有命令均在项目根目录 `/group/40150/licochen/Memory/RSI_GraphRAG` 下执行，
Python 使用 `.venv/bin/python`。每个阶段只写 `outputs/prompt_optimization_conv30/` 或
`outputs/raw_graphs/conv-30/variants/cmpo_*`，不修改历史产物。

## Phase 0 — 审计与口径（无 LLM）

```bash
# 0.2 官方 LoCoMo 评分器 vs 项目评分器（只用已有 pred_answer）
python scripts/prompt_opt/00_official_metrics.py

# 0.1 现有 Reader / Comprehension Prompt 的真实渲染与哈希
python scripts/prompt_opt/01_prompt_audit.py

# 0.3 105 题逐题错误归因（A–H），并落盘每题实际 Reader 上下文
python scripts/prompt_opt/02_error_analysis.py
```

产物：`audit/official_metric_comparison.{md,json,csv}`、`audit/prompt_audit.json`、
`audit/reader_prompt_q*.txt`、`audit/comp_prompt_group*.txt`、`audit/error_analysis_conv30.{jsonl,csv}`、
`audit/graw_contexts/qNNN.txt`。

## Phase 1 — Reader Prompt

```bash
# 1.0 固定 dev40 / val65 划分（seed 42）
python scripts/prompt_opt/03_make_splits.py

# 1.1 dev 上跑 P1–P6（P0 复用历史答案；40 题 × 6）
HY4_CACHE_SHARD=po python scripts/prompt_opt/04_reader_pilot.py \
    --split dev --pids P1,P2,P3,P4,P5,P6 --max-workers 6

# 1.1b dev-informed 综合 Prompt P7 与保守变体 P8
HY4_CACHE_SHARD=po python scripts/prompt_opt/04_reader_pilot.py --split dev --pids P7
HY4_CACHE_SHARD=po python scripts/prompt_opt/04_reader_pilot.py --split dev --pids P8

# 1.2 汇总（若脚本在 summary 阶段中断，用这个免 LLM 重算）
python scripts/prompt_opt/04b_summarize.py --subdir pilot --pids P1,P2,P3,P4,P5,P6,P7,P8

# 1.3 在看到 val 结果之前冻结选择（生成 selection.json + prompt 哈希）
#     （由 /tmp 内联脚本完成，内容见 reader/pilot/selection.json）

# 1.4 内部验证集：P7、P8（65 题 × 2）
HY4_CACHE_SHARD=po python scripts/prompt_opt/04_reader_pilot.py \
    --split val --pids P7,P8 --max-workers 6 --out-subdir final
```

## Phase 2 — Reader 诊断

```bash
# 2.2 Oracle Gold Evidence（R2）与 Full Conversation（R3）；R3/P0 复用历史 fullcontext
HY4_CACHE_SHARD=or python scripts/prompt_opt/08_oracle.py --conditions R2,R3 --pids P0,P7
python scripts/prompt_opt/08b_summarize_oracle.py

# 2.4 同输入重复 3 次（use_cache=False，84 次真实调用 + 14 次 429 重试）
HY4_CACHE_SHARD=rp python scripts/prompt_opt/09_repeats.py \
    --pids P0,P7 --qids 4,9,16,17,24,29,31,33,39,47,54,63,74,85 --repeats 3

# 2.5 官方 cat-5 二选一多选协议（评价协议对照，不是新方法）
HY4_CACHE_SHARD=oa python scripts/prompt_opt/13_official_adversarial.py --pids P0,P7
```

## Phase 3 — Comprehension Prompt

```bash
# 3.1 在冻结的 C2 分组与 C2 三元组上生成 COMP-P1..P5（每组一次调用；P5 额外一次 verify）
HY4_CACHE_SHARD=co python scripts/prompt_opt/05_comp_generate.py \
    --pids COMP-P1,COMP-P2,COMP-P3,COMP-P4,COMP-P5

# 3.3 关系质量审计（无 LLM）
python scripts/prompt_opt/12_comp_quality.py

# 3.2 每个 Prompt 建独立图（G_raw state + 固定 C2 三元组 + 该 Prompt 关系）
#     并跑 105 题检索（含官方 retrieve 的 fact rerank LLM 中间状态诊断）
python scripts/prompt_opt/06_comp_graphs_retrieval.py \
    --pids COMP-P1,COMP-P2,COMP-P3,COMP-P4,COMP-P5

# 3.2b 汇总（06 会覆盖 summary.json；这一步从 records 重建全部变体）
python scripts/prompt_opt/06c_resummarize.py

# 3.4 追踪新增关系是否进入最终 top-5 fact
python scripts/prompt_opt/06b_trace_facts.py

# 3.5 关系数量控制：P4 的 16 条按 confidence 保留前 10 条，重新建图 + 检索
python scripts/prompt_opt/16_make_count_control.py
python scripts/prompt_opt/06_comp_graphs_retrieval.py --pids COMP-P4-c10 --skip-baselines
python scripts/prompt_opt/06c_resummarize.py
```

## Phase 5 — 联合

```bash
# E2 = P4 + P0, E3 = P4 + P7, 以及 C2 + P7（同一 Reader，比较图）
HY4_CACHE_SHARD=jo python scripts/prompt_opt/10_joint.py \
    --variants cmpo_compp4 --pids P0,P7 --split all --max-workers 6
HY4_CACHE_SHARD=jo python scripts/prompt_opt/10_joint.py \
    --variants C2 --pids P7 --split all --max-workers 6
```

## 报告与汇总

```bash
python scripts/prompt_opt/15_cases.py --variant cmpo_compp4 --pid P7
python scripts/prompt_opt/14_build_reports.py     # reports/auto_tables.md, all_metrics.json
python scripts/prompt_opt/17_manifest.py          # reports/run_manifest.json
```

## HY4 调用量与配额

| shard | 用途 | 实际调用 | 错误 |
|---|---|---:|---:|
| `po` | Reader pilot + val | 450 | 0 |
| `co` | Comprehension 生成 + verify | 11 | 0 |
| `or` | Oracle R2/R3 | 266 | 0 |
| `oa` | 官方 cat-5 多选 | 24 | 0 |
| `rp` | 重复稳定性（禁用缓存） | 98 行（84 成功 + 14 次 429 重试） | 14 次重试后成功 |
| `jo` | 联合 E2/E3/C2+P7 | 237 | 0 |

另外，Comprehension 图的检索包含官方 HippoRAG 的逐题 LLM fact rerank；
这些调用走 `outputs/raw_graphs/conv-30/llm_cache`（变体间共享 symlink），
不在上表的 `HY4Client` 统计中。

## 已知不可复现/不稳定点

1. GPU embedding 存在批次级数值差异：105 题批量检索与单题检索的 ranked_docs 不完全一致；
   G_raw 的 diag 重跑与历史记录 93/105 题排名一致。
2. 官方 fact reranker 是 LLM 调用；同一图重跑时 top-5 fact 可能变化。
3. HY4 在 temperature=0 下仍非确定；重复实验见 `reader/repeats/summary.json`。

## 追加实验 — 关闭 LLM fact reranker（2026-11-09）

```bash
# 用 embedding top-k fact 直接作为 PPR 种子，不调用 LLM
#   top5  = 与官方 reranker 的最终 fact 数一致
#   top20 = 更宽的候选窗口（敏感性）
python scripts/prompt_opt/18_no_llm_rerank.py --graphs G_raw,C2,COMP-P4 --fact-ks 5,20

# 在无重排的检索上下文上跑 Reader P0/P7
HY4_CACHE_SHARD=jo python scripts/prompt_opt/10_joint.py \
    --records-root comprehension/retrieval_norerank \
    --variants G_raw_top5,COMP-P4_top5 --pids P0,P7 --split all --max-workers 6
```

产物：`comprehension/retrieval_norerank/{G_raw,C2,COMP-P4}_top{5,20}/`、
`joint/{G_raw_top5,COMP-P4_top5}_{P0,P7}.jsonl`。
无重排检索不产生 LLM 调用；QA 部分 420 次调用使用 shard `jo`，0 错误。
