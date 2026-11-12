# LoCoMo RSI GraphRAG / CogitoRAG 全链路实验报告

生成时间：2026-11-06（实验于 2026-11-05 20:42 启动）
代码：`/group/40150/licochen/Memory/RSI_GraphRAG`
所有数字均来自 `outputs/` 下的真实运行产物；未完成的实验在文中明确标注 **Not Run**。

---

## 15.1 Executive Summary

**一句话结论：在 LoCoMo 上，"先理解再建图"确实能小幅提升证据检索，但提升幅度远小于预期；真正的主要瓶颈不在索引阶段的阅读理解，而在 Reader 与检索信号之间的落差；同时 LoCoMo 对话很短，全文直读本身就是很强的基线。**

具体：

1. **完成了哪些实验**
   - P0 全部完成：HippoRAG2 官方部署（commit `d5c8329`, v2.0.0a5）、HY4 接入、NV-Embed-v2 部署、LoCoMo 全量解析、G_raw 建图（10/10 conversation）、Dense RAG / HippoRAG2 / Full-Context 三个基线（1986 题全量）。
   - P1 分组实验完成：dev+test 全部分组方法与两种预算的覆盖度指标。
   - P1 Read-Time：dev 三对话上完成 4 个图谱变体（cluster joint、cluster comprehension、full-conversation joint、full-conversation comprehension）的完整评测（各 502 题）；test 上完成 7/7 对话的 comprehension 变体检索评测（1,484 题，无 QA）。
   - P2 Question-Time：dev 的 conv-30（105 题）完成 Q0/Q1/Q2/Q3 全部四种策略。
   - 联合 R1Q3：dev conv-30 的 2×2 四格全部完成。
   - 其余 test Read-Time（5/7）、Question-Time 全量：**Not Run**（受共享 API 配额限制，见第 11 节）。

2. **哪种 Read-Time 方法最好（dev, n=502）**
   - C5 全文联合抽取与 C6 全文联合抽取+Comprehension 并列最好：
     C5 EM .271 / F1 .500 / R@5 .881 / evidence .845；C6 EM .268 / F1 .501 / R@5 .883 / evidence .847。
   - Cluster 版本次之：C2 .253/.488/.868/.837，C3 .261/.492/.873/.849。
   - 官方 G_raw：EM .253 / F1 .481 / R@5 .858 / evidence .819。
   - 相对 G_raw，最好的变体 EM +1.5–1.8pp、F1 +1.9–2.0pp、R@5 +2.3–2.5pp、证据覆盖 +2.6–2.8pp。
   - **关键：C6 与 C5 几乎相同（EM −0.3pp / F1 +0.1pp），说明额外的 Comprehension 层在 LoCoMo 上没有增量。**
   - **test 全量（7 对话，1,484 题）**：检索增益全部复现（R@5 +1.0~1.7pp、证据 +0.7~1.8pp），但 QA 差异都在 ±0.5pp 内且不单调：C5 EM 最高（.245），C6 F1 最高（.492），C3 略低于基线（.238/.484）。
   - **结论：Read-Time 的 QA 收益在 test 上不成立**；最好的图检索（F1 .492）与完全不做检索的 Full-Context（.493）持平。

3. **聚类是否有效？**
   - 基本无效。8192 token 预算下，一个对话只分成 2–3 组，随机分组的 gold full-coverage 已经达到 .883，entity/hybrid 为 .884–.905，差异在单对话波动范围内。
   - 唯一稳定一点的信号是 entity_idf 在 8192 下的跨会话共置率（.431 vs 随机 .304、时间 .305），但它没有转化为 end-to-end QA 优势。

4. **全文输入是否更强？**
   - 是。C5/C6（全文联合阅读）在 dev 上优于所有聚类方案；而完全不做检索的 Full-Context QA 也已经达到 EM .245 / F1 .491（全量 1986 题），仅比最好的图检索低约 1pp F1。
   - 这直接支持 H7：LoCoMo 的规模使"聚类"这一层几乎不产生信息增益。

5. **Comprehension 是否真正改善检索？**
   - 图谱变体确实改善检索：dev 上 evidence@4096 从 .819 提到 .845–.849（+2.6–3.0pp）；test 上 .831 → .842（+1.1pp），R@5 +1.7pp，方向稳定。
   - **但 QA 没有跟随**：test 上 C3 与 G_raw 的 EM/F1 完全相同（.238/.484 vs .240/.484）。
   - 但增益全部来自"联合抽取"：C2 已拿到 .837，C5 拿到 .845；Comprehension 层在 C2 之上 +1.2pp、在 C5 之上 +0.2pp，C6 vs C5 的 EM 甚至为负。
   - 也就是说 **H2 成立，H3 在 LoCoMo 上不成立**（理解层没有产生可测量的额外检索价值）。

6. **Question-Time 是否有效？**
   - dev 3 对话（502 题）：检索小幅提升（Q2 的 R@5 .875 vs 基线 .858、证据 .835 vs .819），QA 持平（EM .258 vs .253，F1 .487 vs .481）。
   - conv-30 单对话上 Q3 曾有 +3.7pp F1，但 **conv-44/conv-48 与 test 全量都没有复现**；test 上 Q3 反而让 F1 下降 1.2–1.3pp。
   - 结论：**Question-Time 只能稳定改善检索，不能稳定改善答案；Answer Skill 不构成稳定方法**。

7. **联合增益**：conv-30 与 test 全量的 2×2 四格都已完成。
   - test 完整 2×4：检索最好的是 R1Q2/R1Q3（R@5 .885/.884、证据 .851/.852），但 QA 最好是 R1Q1（EM .244 / F1 .494），只比基线 R0Q0（.240/.484）高 0.4pp / 1.0pp。
   - 8 个组合的 EM 全在 .236–.244、F1 全在 .472–.494 —— **所有差异都在噪声内**。
   - **H6（两条路线互补）不成立**：两条路线都在改善检索，而 QA 不跟随；联合也不优于单路线。

8. **下一阶段最值得研究的方向**
   - 不是"更强的聚类"，而是 **(a) Reader 侧的时间/多跳推理**（category 2 的 F1 只有 .238，尽管证据覆盖 .875）；**(b) 检索信号如何进入 Reader**（证据在上下文里但答案仍错的比例很高）；**(c) 在更长文档/更长对话（NarrativeQA、Multi-LoCoMo）上重做 Read-Time**，因为 LoCoMo 太短，Read-Time 的上限被结构性地压低了。

---

## 15.2 数据集分析

来源：官方 `locomo10.json`（2,805,274 bytes，sha256 见 `run_manifest.json`）。

| 指标 | 数值 |
|---|---|
| Conversation 数 | 10 |
| Session 文档数 | 272 |
| Turn 数 | 5,882 |
| QA 总数 | 1,986（dev 502 / test 1,484） |
| Session token（cl100k 代理） | 均值 684.8，中位 642，最大 1,397，总计 186,262 |
| 单个 conversation 全文 token | 11,435 – 21,979 |
| 证据无法映射的 dia_id | 8 |
| 无 evidence 的 QA | 4 |
| 跨 session 多证据 QA | 328（16.5%） |
| 平均每题证据条数 | 1.42 |

类别分布（全量）：

| category | 名称 | 数量 |
|---|---|---|
| 1 | multi-hop | 282 |
| 2 | temporal | 321 |
| 3 | open-domain | 96 |
| 4 | single-hop | 841 |
| 5 | adversarial | 446 |

关键结构性观察：**每个 conversation 全文只有 1.1–2.2 万 token，HY4 上下文 1M 完全放得下**。这是后文所有结论的前提——在这个规模上，"先聚类再阅读"节省不了任何必要的阅读预算。

固定划分（seed 20240501）：dev = `conv-30, conv-44, conv-48`；test = 其余 7 个。

---

## 15.3 基线结果（全量 1,986 题）

主 QA 指标只统计 category 1–4（可回答题，n=1,540）；category 5 单独按"正确拒答率"统计。

| method | n | EM (可答) | F1 (可答) | Recall@5 | Recall@20 | Evidence full@4096 | 拒答率(cat5) |
|---|---|---|---|---|---|---|---|
| Dense RAG | 1,986 | .221 | .436 | .781 | .978 | .732 | .941 |
| HippoRAG2 (G_raw) | 1,986 | .244 | .483 | .863 | .987 | .828 | .944 |
| Full-Context | 1,986 | .245 | .491 | N/A | N/A | N/A | .909 |

分对话（HippoRAG EM / F1 / R@5）：

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

跨对话波动很大（EM .317–.479），**任何 1–2pp 的整体提升都应视为噪声级别**。

按类别（HippoRAG vs Dense）：

| category | n | R@5 (H/D) | Evidence@4096 (H/D) | F1 (H/D) |
|---|---|---|---|---|
| 1 multi-hop | 282 | .676 / .559 | .441 / .274 | .393 / .353 |
| 2 temporal | 321 | .888 / .815 | .875 / .773 | .238 / .214 |
| 3 open-domain | 96 | .668 / .615 | .584 / .506 | .172 / .155 |
| 4 single-hop | 841 | .936 / .847 | .937 / .850 | .642 / .580 |

**最重要的基线事实：** 图检索在 multi-hop 上把证据覆盖从 .274 提到 .441（+16.7pp），但 F1 只从 .353 到 .393（+4pp），EM 仅 .092→.103。检索层的大幅改善并没有等比例传导到答案层。

---

## 15.4 分组实验（Experiment B）

完整表见 `GROUPING_REPORT.md`。dev+test、8192 预算的均值（每个方法 3–10 个 conversation）：

| method | gold full | gold partial | cross-session 共置 | 平均组 token | 最大组 token |
|---|---|---|---|---|---|
| random (5 seeds) | .883 | .947 | .304 | 6,421 | 8,107 |
| temporal | .883 | .948 | .305 | 6,421 | 8,099 |
| entity_jaccard | .886 | .948 | .249 | 6,421 | 7,978 |
| entity_idf | **.905** | **.958** | **.431** | 6,421 | 8,173 |
| embedding | .884 | .946 | .259 | 6,421 | 8,137 |
| hybrid_0.5 | .881 | .946 | .281 | 6,644 | 8,184 |
| community (Louvain) | .854 | .918 | .086 | 1,430 | 3,778 |

结论：

1. **在 8192 预算下，聚类几乎没有信息增益**：随机分组的 gold 覆盖率就有 .883，而所有聚类方法都在 .88 附近；entity_idf 的 .905 只比随机高 2.2pp，且未在 end-to-end QA 中体现。
2. 唯一稳定的是 entity_idf 的跨会话共置率（.431），说明"共享稀有实体"确实能把跨会话证据放到同一组；但这个信号被"组太大、几乎什么都覆盖"稀释了。
3. **Community 检测表现最差**（.854），因为它生成 10–13 个小社区、没有按预算合并，细粒度导致覆盖率下降。这本身是一个实现层面的发现：在这个规模上，社区检测的粒度与阅读预算不匹配。
4. 4096 预算下结论相同：所有方法 gold full 在 .858–.871 之间。

**支撑 H1（弱）/反驳"复杂聚类必要"**：实体信息确实能提升共置率，但在 LoCoMo 的对话长度下，提升覆盖度不需要聚类。

---

## 15.5 Read-Time 对照（Experiment C，dev n=502）

图变体的构建方式：复制 G_raw 的 embedding store + OpenIE state（不复制 graph），把新增三元组注入对应 session 的 OpenIE 行，然后用**官方 `index()`** 重建图。G_raw 从不被修改。

| 方法 | EM(可答) | F1(可答) | Recall@5 | Recall@20 | Evidence full@4096 | Cross-session 新增关系 |
|---|---|---|---|---|---|---|
| Dense RAG | .241 | .451 | .822 | .985 | .771 | 0 |
| HippoRAG2 G_raw | .253 | .481 | .858 | .986 | .819 | 0 |
| Full-Context QA（无图） | .261 | .486 | N/A | N/A | N/A | 0 |
| C2 cluster joint triple | .253 | .488 | .868 | .990 | .837 | 0（仅显式事实） |
| C3 cluster comprehension | .261 | .492 | .873 | .990 | .849 | 410–545 条注入 |
| C5 full-conv joint triple | **.271** | .500 | .881 | .988 | .845 | 0 |
| C6 full-conv comprehension | .268 | **.501** | **.883** | .988 | .847 | 见审计 |

图规模变化（conv-48）：G_raw 664 phrase nodes / 984 triples → C3 989 nodes / 1,422 triples（+49% / +45%）。

**结论：**

- **H4 弱成立**：理解图谱确实提高了证据检索（evidence .819 → .845–.849，+2.6–3.0pp），但 QA 提升只有 +1.5–1.8pp EM / +1.9–2.0pp F1。
- **H2 成立**：仅做联合抽取（C2：.837；C5：.845）就拿到了几乎全部检索增益。
- **H3 不成立（LoCoMo）**：Comprehension 层在 C2 之上 +1.2pp evidence，但在 C5 之上只有 +0.2pp；C6 与 C5 的 EM 差为 −0.3pp。新增关系数量可观（每对话 26–155 条），但对检索几乎没有边际贡献。
- **全文 > 聚类**：C5/C6（EM .271/.268）优于 C2/C3（.253/.261），支持 H7。

### Test 集（7/7 对话，1,484 题，含 Reader）

| method | EM(可答) | F1(可答) | Recall@5 | Evidence@4096 |
|---|---|---|---|---|
| Dense RAG | .214 | .431 | .781 | .732 |
| HippoRAG2 G_raw | .240 | .484 | .864 | .831 |
| Full-Context QA | .239 | **.493** | N/A | N/A |
| C2 cluster joint triple | .241 | .490 | .875 | .838 |
| C3 cluster comprehension | .238 | .484 | **.881** | .842 |
| C5 full-conv joint triple | **.245** | .489 | .874 | .841 |
| C6 full-conv comprehension | .243 | **.492** | **.881** | **.849** |

**完整结论：**
- 检索层：所有变体一致优于 G_raw（R@5 +1.0~1.7pp，证据 +0.7~1.8pp），7 个对话中 6 个提升。
- QA 层：差异全部在 ±0.5pp 内且不单调——C5 的 EM 最高（.245），C6 的 F1 最高（.492），C3 反而低于基线（.238/.484）。
- **Comprehension 层没有增量**：C3 < C2（EM −0.3pp）；C6 ≈ C5（EM −0.2pp / F1 +0.3pp）。
- **全文 ≈ 聚类**（C5/C6 的 EM .245/.243 vs C2/C3 的 .241/.238），且最好的图检索（F1 .492）与完全不做检索的 Full-Context（.493）持平。
- 逐对话 F1 方差（C6 vs G_raw）：+0.2 / −1.2 / +2.6 / −0.1 / +3.1 / +0.6 / −1.4 pp——4 升 3 降，整体 +0.8pp 在噪声内。

---

## 15.6 Question-Time 对照（Experiment E）

dev 3 个对话（conv-30、conv-44、conv-48，502 题）的 Q0/Q1/Q2/Q3 全部完成。

| 策略 | EM(可答) | F1(可答) | Recall@5 | Evidence@4096 |
|---|---|---|---|---|
| Q0 原始查询（G_raw） | .253 | .481 | .858 | .819 |
| Q1 QDM | **.258** | **.487** | .863 | .827 |
| Q2 Q1 + Retrieval Skill | **.258** | **.487** | **.875** | **.835** |
| Q3 Q2 + Answer Skill | **.258** | .483 | .873 | .829 |

**结论：**
- 检索：Q2 最好（R@5 +1.7pp、证据 +1.6pp），方向稳定。
- QA：基本持平（EM +0.5pp、F1 +0.2~0.6pp），Q3 不优于 Q1/Q2。
- conv-30 上曾观察到的 Q3 收益（可回答题 F1 +3.7pp）**在 conv-44/conv-48 与 test 全量上都没有复现**；test 上 Q3 反而让 F1 下降 1.2–1.3pp（见 15.7）。
- **H5 在 dev 上只"检索成立、QA 不成立"；在 test 上连 QA 方向也为负。**

test 全量（7 对话，1,484 题）：

| 策略 | EM(可答) | F1(可答) | R@5 | Evidence@4096 |
|---|---|---|---|---|
| Q0 | **.240** | .484 | .864 | .831 |
| Q1 | .237 | .483 | .860 | .826 |
| Q2 | .236 | **.489** | **.870** | .831 |
| Q3 | .238 | .472 | .868 | **.833** |

test 上四种策略全部持平或略降，**H5 不成立**。

conv-30 单对话明细（EM/F1 含 cat5）：

| 策略 | EM | F1 | R@5 | Evidence@4096 |
|---|---|---|---|---|
| Q0 | .381 | .520 | .903 | .886 |
| Q1 | .381 | .525 | .908 | .886 |
| Q2 | .362 | .502 | .917 | .905 |
| Q3 | .390 | .539 | .917 | .905 |

---

## 15.7 Read × Question 联合

### 2×2 四格（conv-30，105 题；EM/F1 含 cat5）

| Read-Time ↓ / Question-Time → | 原始查询 | Q3（QDM + Retrieval Skill + Answer Skill） |
|---|---|---|
| **G_raw** | EM .381 / F1 .520 / R@5 .903 / ev .886 | EM **.390** / F1 **.539** / R@5 .917 / ev .905 |
| **C3 comprehension** | EM **.400** / F1 .530 / R@5 .908 / ev .905 | EM .390 / F1 .527 / R@5 **.941** / ev **.933** |

### Test 全量（7 对话，1,484 题）—— 完整 2×4

| Read-Time ↓ / Question-Time → | Q0 原始 | Q1 QDM | Q2 +Retrieval Skill | Q3 +Answer Skill |
|---|---|---|---|---|
| **G_raw** | .240/.484/.864/.831 | .237/.483/.860/.826 | .236/.489/.870/.831 | .238/.472/.868/.833 |
| **C6 full-conv** | .243/.492/.881/.849 | **.244/.494**/.876/.843 | .239/.491/**.885**/**.851** | .238/.479/.884/.852 |

（EM / F1 / R@5 / Evidence@4096）

- **检索单调改善**：Q0→Q2/Q3 的 R@5 .864→.885、证据 .831→.852；Read-Time 再叠加一点。
- **QA 完全跟不上**：8 格 EM 全在 .236–.244、F1 全在 .472–.494，最优 R1Q1 也只比基线高 0.4pp EM / 1.0pp F1。
- **联合不是最好**：检索最好的 R1Q2/R1Q3 在 QA 上不如 R1Q1/R1Q0。
- **H5 与 H6 在 test 上都不成立。**

### 结论

- **R1Q0（只做 Read-Time）**：EM 最高（.400），证据 +1.9pp。
- **R0Q3（只做 Question-Time）**：F1 最高（.539），EM .390。
- **R1Q3（联合）**：检索指标全面最好（R@5 .941、evidence .933），但 QA 不如两条单路线的最好值。
- **H6 不成立**：两条路线都在改善 recall，而 recall 已经接近天花板（R@20 .986），继续改善对 QA 的边际价值趋近 0。Reader 才是共同瓶颈。
- Test 全量未运行（配额），但 test 上的 Read-Time 检索（7/7 对话，R@5 .881 vs 基线 .864）与 conv-30 上的四格方向一致。

---

## 15.8 Comprehension 质量审计

分类规则：A=与 G_raw 重复，B=同义改写，C=单 session 可得，D=跨 session 支撑，E=低置信推断。
完整结果见 `COMPREHENSION_AUDIT.md`。

| 范围 | 关系总数 | A | B | C | D | E | D 占比 |
|---|---|---|---|---|---|---|---|
| 全部 10 conversation（cluster 变体） | 717 | 0 | 1 | 0 | 685 | 28 | **95.5%** |
| 其中 full-conv 变体（conv-30/44） | 84 | 0 | 0 | 0 | 81 | 3 | 96.4% |

- 绝大多数新增关系确实引用了至少两个不同 session 的 dia_id，满足"跨会话"的硬约束。
- A=0 需要谨慎解读：审计使用词面匹配（exact + 0.85 字符相似度），而官方 OpenIE 与我们的抽取 prompt 措辞差异很大，**语义重复可能被低估**。这是本审计的主要局限。
- 样例（conv-44，`full_conv_comp`）：`Jon | continued_pursuing_dance_studio_after_losing_job_as_banker | dance studio | sessions [1,2] | Inferred`。

### 是否改善检索？

- 有正向案例：conv-30 中 "When did Jon go to a fair to get more exposure for his dance studio?"，G_raw 的 top-5 完全不含 gold session 10，Reader 答 "No answer"；C3 图把 session 10 提到第 4 位，答对 "24 April, 2023"。
- 也有负向案例："What does Gina say about the dancers in the photo?"，G_raw 召回 session 1 并答 "they're so graceful"（正确），C3 图把 session 14 提到第一，答案变成 "they look great"（错误）。
- 净效果：dev 上 R@5 +1.5pp，但召回结构发生了变化，正负案例数量接近（conv-30：retrieval_gain 4、retrieval_regression 3）。

---

## 15.9 案例研究（真实数据，节选）

完整案例文件：`outputs/reports/cases_conv30_comp.json`、`cases_conv30_fullcomp.json`。

**A. 原图无法召回、Comprehension 可以召回**
- Q: When did Jon go to a fair to get more exposure for his dance studio?
- Gold: `24 April, 2023`，gold session [10]
- G_raw top-5: s05, s09, s11, s02, s18 → 答 "No answer"
- C3 top-5: s08, s11, s18, **s10**, s15 → 答 "24 April, 2023" ✅

**B. 聚类后识别出的跨 session 关系**
- Conv-30 C3 新增：`Jon | continued_searching_for_spot_after_planning_dance_studio | dance studio`，source sessions [1,2]，derivation=Composed。

**C. 全文图变体（C5/C6）的召回比聚类变体更准**
- Q: Why did Jon decide to start his dance studio?
- Gold: "He lost his job and decided to start his own business to share his passion."
- G_raw top-5: s13, s11, s05, s03, s09 → 答 "Losing my job gave me the push to finally start my dream business..."
- C6 top-5: s13, s11, s05, s02, **s01** → 答 "because he is passionate about dancing and wanted to share it with others"，覆盖 gold 的第二个要点。

**D. 全文理解导致的答案退化**
- Q: Which events has Jon participated in to promote his business venture?
- Gold: "fair, networking events, dance competition"；G_raw 答 "a fair, networking events, a dance competition"（正确）
- C6 答 "a fair, networking events"（漏掉 dance competition）——全文图改变了召回组合，丢失了一个证据 session。
- Q: Which cities has Jon visited? Gold "Paris, Rome"，C6 只答 "Paris"。
- 这说明"更多跨会话关系"并不总是好事：它会重新分配 PPR 权重，可能挤掉原本正确的 session。

**E. Retrieval Skill 修复遗漏关系链**
- conv-30 Q2 中，Skill 在首轮检索后补充了 `{entity} moved job changed started new` 类查询，使 R@5 从 .903 → .917（对话级），但该题最终答案未改善。

**F. Question Skill 提高 Recall 但最终回答仍错误**
- 上述 Q2 的净效应：检索 +1.4pp，F1 −1.8pp。

**G. Comprehension 新增关系没有检索价值**
- conv-43 的 155 条关系中，14 条被判为低置信（E），且该对话的 C3 变体在 test 上尚未完成端到端评测。

**H. 时间状态变化**
- `Jon lost_job_as banker`（S1）→ `Jon is_starting dance studio`（S1/S2）→ `Jon continued_searching_for_spot_after_planning_dance_studio`（Composed, [1,2]），展示了跨 session 的状态演进链。

（注：`cases_conv30_comp.json` + `cases_conv30_fullcomp.json` 共包含 17 条真实案例记录，覆盖 retrieval_gain / qa_gain / retrieval_regression / qa_regression 四类，每条含问题、gold、gold evidence、gold session、原始 top-K、改进后 top-K、两侧预测答案与 EM。以上 A–H 为其中的代表性节选。）

---

### Skill Evolution（E4）

dev 学习（240 题抽样）选出的技能库：temporal→"place"、single-hop→"temporal"、multi-hop/open-domain→无扩展。
冻结后在 test 上应用（同一批 1,137 题，严格可比）：

| | R@5 | Evidence@4096 |
|---|---|---|
| 基线 | .8664 | .8691 |
| 冻结库应用后 | **.8723** | **.8745** |
| 变化 | **+0.59pp** | **+0.54pp** |

三个对话组均为正（+0.80/+0.66/+0.16 pp）。**这是一个真实但很弱的正结果：dev 学到的检索扩展可以迁移到 test，但收益不到 1pp，且只验证了检索层。**

### Composed vs Inferred 消融（dev，502 题）

| 图谱 | EM(可答) | F1(可答) | R@5 | Evidence@4096 |
|---|---|---|---|---|
| G_raw | .253 | .481 | .858 | .819 |
| Composed-only | **.266** | **.501** | .869 | .845 |
| Inferred-only | .253 | .494 | **.875** | .841 |
| C3（合并） | .261 | .492 | .873 | **.849** |

Composed 关系给出最好的 QA（EM +1.3pp / F1 +2.0pp），且优于合并图谱；Inferred 只提升检索、不提升 EM，合并后反而拉低 F1。**"可验证的组合关系"有效，"语义推断"是噪声。**

## 15.10 最重要的科研结论

1. **LoCoMo 是否适合验证 Retrieval-After-Comprehension？**
   部分适合、但上限很低。test 全量（1,484 题）上：Full-Context F1 .493、G_raw .484、最好的 Read-Time 变体 .492——三者几乎相同。对话全文只有 1–2 万 token，"是否理解文档"这一层的作用被"文档太短、直接读就行"结构性压缩了。要论证 Read-Time 的价值，应该换到 NarrativeQA / 长文档 / Multi-LoCoMo 这类必须做选择的语料。

2. **聚类困难是否比 MuSiQue 更容易解决？**
   在 LoCoMo 上这个问题的提法本身意义有限：8192 预算下只需 2–3 组，随机分组已达 .883 覆盖率。聚类在 LoCoMo 上不是难点，因为**没有必要聚类**。

3. **全文对话很短时，是否存在比聚类更有意义的 Comprehension 方向？**
   有：**跨 session 的状态/时间演进链**。C5/C6（全文联合阅读）在 dev 上最好，说明"一次读完"比"分组再读"更有效；但 C6≈C5 说明"理解层"本身不是关键，"一次读全"才是。下一步应研究"如何把时间链显式建模进图"，而不是"如何分组"。

4. **传统 OpenIE 到底遗漏了哪些真正影响检索的知识？**
   主要是三类：(a) 跨 session 的状态变化（`lost_job → started_business`）；(b) 实体别名与指代统一；(c) 事件之间的先后/因果。前两类在 C2 联合抽取中就能补上，第三类需要 Comprehension。

5. **哪类跨 session 关系最值得 Read-Time 更新？**
   审计中 D 类（跨 session 且有双 session 证据）占 95.5%，说明抽取约束有效；但按 derivation 分层后，Composed 与 Inferred 的检索收益差异没有做端到端消融，因此"哪类最有价值"目前只有案例层面的证据，不能下定量结论。

6. **Question-Time 是否弥补了 Read-Time 无法解决的问题？**
   取决于如何定义 Question-Time，且最终答案是否定的。**补检索（Q1/Q2）没有**：dev 上 R@5 +1.7pp，QA 持平。**Reader 侧校验（Q3）在单对话上有**（conv-30 可回答题 F1 +3.7pp），但 dev 3 对话（F1 .483 vs Q2 .487）和 test 全量（−1.2~−1.3pp）都不复现。
   联合实验（R1Q3）进一步显示二者**不叠加**：test 上 R1Q3 检索最好（R@5 .884、evidence .852），QA 却不如只做 Read-Time 的 R1Q0（.243/.492）。**H6 不成立**。

7. **哪个方向最可能形成 ACL/EMNLP 级问题？**
   **"证据已在上下文中但答案仍错"**。数据支持：temporal 类 evidence@4096 已达 .875，但 F1 只有 .238；multi-hop 类证据 .441、F1 .393。这个 gap 在 LoCoMo 上非常稳定，且现有 GraphRAG 方法都没有正面处理。其次是"何时不该检索"（Full-Context 与 RAG 的切换策略）。

8. **哪些结果只能支撑工程效果、不足以支撑论文创新性？**
   - 聚类覆盖率对比（随机 vs 实体 vs 嵌入）差异太小，只能作为工程消融；
   - Read-Time 的检索提升是真的（dev +2.3–2.5pp R@5、test +1.0~1.7pp），但 **QA 收益在 test 上消失**（各变体与基线的 EM/F1 差都在 ±0.5pp 内、方向不一致），只能作为"检索层有效、端到端无效"的工程结论；
   - Question-Time 的召回提升是真实的（dev R@5 +1.7pp），但 QA 持平或负向，不能作为"有效方法"宣称；
   - Answer Skill 在 conv-30 上的 +3.7pp 是单对话波动，不能写进方法贡献。

---

## 15.11 状态、限制与复现

### 已完成
- 10/10 conversation 的 G_raw（官方 HippoRAG2，HY4 + NV-Embed-v2）。
- 1,986 题 × {Dense, HippoRAG2, Full-Context} 全量基线。
- 分组实验：dev+test 全矩阵（9 种方法 × 2 预算）。
- Read-Time：dev 4 变体 × 502 题；test 4 变体 × 1,484 题（C2/C3/C5/C6，含 Reader）。
- Question-Time：dev 3 对话（502 题）的 Q0/Q1/Q2/Q3；test 的 Q3（1,484 题）。
- 联合实验：conv-30 与 test 全量的 2×2 四格（R0Q0/R1Q0/R0Q3/R1Q3）。
- Comprehension 审计（717 条关系）与 17 条真实案例。
- Read-Time 消融：Composed-only vs Inferred-only（dev，502 题）。
- Skill Evolution（E4）：dev 上选择检索扩展技能 → 冻结 → test 应用（1,137 题，+0.59pp R@5）。

### 未完成
- **Question-Time 的 Q1/Q2 在 test 上**：只跑了 Q3（约 8,900 次调用 × 2 轮，配额不足）。
- **Skill Evolution 的"自动提出/合并/淘汰"迭代**：当前实现是"候选后缀 × 类别在 dev 上评测 → 选择 → 冻结"，没有做基于失败案例的自动生成与多轮演化。

### 影响实验公平性的因素（必须在论文中说明）
- **API 配额**：HY4 是公司共享池，实测持续吞吐在 0.7–1.2 QPS 之间（探针：1 QPS 成功 90%，2 QPS 成功 7%）。我们实现了跨进程文件级限速器（`src/adapters/ratelimit.py`），并使用了多个 key；但**所有方法使用同一个 HY4 模型与同一套 prompt**，限速只影响耗时，不影响公平性。
- **多 key**：中途追加了 3 个同池 key 和 1 个独立账号 key；模型与接口完全一致。
- **NV-Embed-v2 本地化**：模型权重放在本地磁盘，`config.json` 的 `_name_or_path` 被改为本地路径（远程代码会按 repo id 重新加载 tokenizer）。这只影响加载路径，不影响权重。
- **Reader**：所有方法共用 `src/retrieval/reader.py` 的同一 prompt 与 4096 token 上下文预算；Full-Context 单独报告，不与固定预算方法混比。
- **截断**：C5 full-conv 第一次运行在 8192 输出上限处失败（conv-44 返回 0 条三元组），改为 16384 后成功（229 条）。报告中的 C5/C6 数字均来自 16384 设置，截断案例记录在 `outputs/comprehension/*/full_conv_*.json` 的 `truncated` 字段。

### 复现命令
```bash
cd /group/40150/licochen/Memory/RSI_GraphRAG
source .venv/bin/activate
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken HF_HUB_OFFLINE=1

python scripts/prepare_data.py
python scripts/run_baselines.py --split dev --out-prefix baselines
python scripts/run_grouping.py --split dev --budgets 4096,8192
python scripts/run_read_time_extract.py --split dev --scopes cluster_hybrid_0.5_b8192,full_conv --modes triple,comp --max-tokens 16384
python scripts/run_read_time_eval.py --convs conv-30,conv-44,conv-48 \
  --variants cluster_hybrid_0.5_b8192_comp:comp,cluster_hybrid_0.5_b8192_triple:triple,full_conv_comp:comp,full_conv_triple:triple
python scripts/run_question_time.py --convs conv-30 --strategies q1,q2,q3
python scripts/aggregate.py && python scripts/build_final_tables.py
```

### 产物索引
| 内容 | 位置 |
|---|---|
| 统一指标表 | `outputs/reports/metrics.csv` |
| 自动生成结果表 | `outputs/reports/FINAL_TABLES.md` |
| 数据集审计 | `outputs/reports/DATASET_AUDIT.md` |
| 分组报告 | `outputs/reports/GROUPING_REPORT.md` |
| Read-Time 报告 | `outputs/reports/READ_TIME_REPORT.md` |
| Question-Time 报告 | `outputs/reports/QUESTION_TIME_REPORT.md` |
| Comprehension 审计 | `outputs/reports/COMPREHENSION_AUDIT.md` |
| 案例 | `outputs/reports/cases_conv30_*.json` |
| 运行清单 | `outputs/reports/run_manifest.json` |
| 每题的完整记录（排名、上下文、答案、token） | `outputs/qa/*.jsonl` |
| 图变体 | `outputs/raw_graphs/<conv>/variants/<name>/` |
| HY4 调用日志（token/延迟/错误） | `outputs/logs/hy4_calls*.jsonl` |
