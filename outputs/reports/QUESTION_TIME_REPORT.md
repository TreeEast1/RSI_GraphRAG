# QUESTION_TIME_REPORT — Experiment E

状态：**dev 3 个对话（conv-30、conv-44、conv-48，共 502 题）的 Q0/Q1/Q2/Q3 全部完成**；test 集的 Q3 也已完成（见 JOINT_EVALUATION）。

## 1. 主表（conv-30，105 题，同一题集）

| 策略 | EM | F1 | Recall@5 | Recall@20 | Evidence full@4096 |
|---|---|---|---|---|---|
| Q0 原始查询（HippoRAG G_raw） | .381 | .520 | .903 | .986 | .886 |
| Q1 QDM 分解 | .381 | .525 | .908 | 1.000 | .886 |
| Q2 Q1 + Retrieval Skill | .362 | .502 | **.917** | 1.000 | **.905** |
| Q3 Q2 + Answer Skill | **.390** | **.539** | **.917** | 1.000 | **.905** |

（EM/F1 含 category-5 拒答；同题集可直接比较。可回答题口径：Q0 .210/.390，Q1 .210/.397，Q2 .185/.366，Q3 .235/.427。）

## 1b. Dev 全量（3 对话，502 题）

| 策略 | EM(可答) | F1(可答) | Recall@5 | Evidence@4096 |
|---|---|---|---|---|
| Q0 原始查询（G_raw） | .253 | .481 | .858 | .819 |
| Q1 QDM | .258 | .487 | .863 | .827 |
| Q2 Q1 + Retrieval Skill | .258 | .487 | **.875** | **.835** |
| Q3 Q2 + Answer Skill | .258 | .483 | .873 | .829 |

**在 3 个 dev 对话上：**
- 检索有稳定小幅提升（R@5 +1.7pp、证据 +1.6pp，Q2 最好）。
- QA 基本持平：EM +0.5pp、F1 +0.2–0.6pp，且 Q3 并不优于 Q1/Q2。
- conv-30 上 Q3 的 +3.7pp 属于单对话波动，**没有在另外两个对话上复现**。

## 1c. Test 全量（7 对话，1,484 题）

| 策略 | EM(可答) | F1(可答) | Recall@5 | Evidence@4096 |
|---|---|---|---|---|
| Q0 原始查询（G_raw） | **.240** | .484 | .864 | .831 |
| Q1 QDM | .237 | .483 | .860 | .826 |
| Q2 Q1 + Retrieval Skill | .236 | **.489** | **.870** | .831 |
| Q3 Q2 + Answer Skill | .238 | .472 | .868 | **.833** |

**test 上四种策略的 QA 全部持平或略降**（EM .236–.240、F1 .472–.489），检索差异也只有 ±0.6pp。Q1 甚至比 Q0 略差。
**H5 在 test 上不成立。**

## 2. 行为统计

| 指标 | Q1 | Q2 |
|---|---|---|
| 题目数 | 105 | 105 |
| QDM 判定需要拆解 | 9 (8.6%) | 9 (8.6%) |
| 平均检索子查询数 | 1.23 | 1.23（+ Skill 追加） |
| Retrieval Skill 触发 | — | **95 / 105 (90.5%)** |
| Skill 追加的查询总数 | — | 133（平均 1.4/题） |
| 平均检索轮数 | 1.23 | 约 2.5 |
| 正向改进题数 | 见下 | 见下 |
| 负向退化题数 | 见下 | 见下 |

## 3. 逐题变化（相对 Q0）

| 策略 | Retrieval 提升题 | Retrieval 退化题 | QA 提升题 | QA 退化题 |
|---|---|---|---|---|
| Q1 | 2 | 1 | 1 | 1 |
| Q2 | 3 | 2 | 1 | 3 |

（用 `scripts/mine_case_studies.py` 对 qtime 与 baselines 的 conv-30 文件做差得到。）

## 4. 结论

1. **H5 在检索层成立、单独使用时在 QA 层不成立**：Retrieval Skill 把 R@5 从 .903 提到 .917、evidence 从 .886 提到 .905，但 F1 从 .520 掉到 .502、EM 从 .381 掉到 .362。
   **加上 Answer Skill（Q3）后反转为正收益**：EM .390、F1 .539，均高于 Q0；Q3 共修订了 13/105 个答案。
2. **Retrieval Skill 严重过度触发**：90.5% 的题目都判定"缺少关系链"，平均每题追加 1.4 个查询。它没有学会"证据已经够了"。
3. **QDM 过于保守**：只有 8.6% 的题被拆解。LoCoMo 的问题大多是单跳或轻量多跳，模型倾向于不拆。
4. **机制解释**：LoCoMo 的 R@20 已经 ≈ 1.0，补检索的空间很小；追加的 session 主要带来上下文噪声，Reader 在多份候选之间更容易被干扰。
5. **Answer Skill 在 conv-30 上有效（F1 +3.7pp），但在 dev 全量（502 题）和 test 全量（1,484 题）上都不复现**：dev 上 Q3 F1 .483 低于 Q2 .487；test 上 Q3 比 Q0 低 1.2–1.3pp。因此**不能把 Answer Skill 作为稳定方法**，它更像是单对话上的偶然收益。
6. Q3 的修订率约 12%，修订主要发生在多证据整合与拒答判断上；但修订方向在多数题目上是中性的。

## 4b. Skill Evolution（E4，完整 dev→test 冻结流程）

**dev 学习**（240 题抽样，每类评估 3 个检索扩展后缀）：

| 类别 | 选中技能 | dev R@5（基线 .809） |
|---|---|---|
| multi-hop | none | .809（无扩展） |
| temporal | **place**（"where place location"） | .925 |
| open-domain | none | .809 |
| single-hop | **temporal**（"date time year month"） | .922 |

**冻结**：`outputs/skills/frozen_skill_library.json`（含每类的候选收益，未选中的一律 `none`）

**test 应用**（同一批 1,137 道有 gold 证据的题，严格可比）：

| | R@5 | Evidence@4096 |
|---|---|---|
| 基线（不加技能） | .8664 | .8691 |
| 冻结库应用后 | **.8723** | **.8745** |
| 变化 | **+0.59pp** | **+0.54pp** |

- 三个对话组都为正（+0.80 / +0.66 / +0.16 pp），方向一致。
- 幅度很小，且只评估了检索（未跑 Reader），**不能宣称对 QA 有效**。
- 这是一个真实但微弱的正结果：**dev 上学到的检索扩展能迁移到 test，但收益 <1pp**。

## 5. Skill Library

- 默认 Skill 定义见 `src/question_skill/skills.py`（`DEFAULT_SKILLS`）。
- 本轮的 Retrieval Skill 由 HY4 在线诊断缺失关系链实现，尚未做 dev→test 的 skill 冻结流程（配额原因）。
- `SkillLibrary` 支持 `usage_count / observed_benefit / enabled` 字段与持久化，后续可用于 E4 的 Skill Evolution。
