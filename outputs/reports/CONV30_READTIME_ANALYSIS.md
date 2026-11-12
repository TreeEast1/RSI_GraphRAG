# conv-30 Read-Time 进化效果专项分析（G_raw / C2 / C3 / C5 / C6）

> 分析对象：LoCoMo `conv-30`，19 个 session，105 道问题。
> 数据来源：`outputs/qa/` 下已完成的逐题记录 + `outputs/comprehension/conv-30/` 的知识抽取产物 + `outputs/logs/` 的建图日志。
> **本次没有重跑任何 LLM、检索或建图**；所有指标都是用项目自带的指标定义（`src/evaluation/pipeline.py::summarize`）对已落盘的逐题记录重新聚合得到的，并与仓库里已存的 dev 汇总互相校验（见 §1.3）。

---

## 0. 先给结论

1. **conv-30 上 Comprehension（C3/C6 比 C2/C5）确实存在"可定位"的独立增益，但只发生在检索层、且只有 1 道题真正把答案救回来。**
   - C2→C3：Evidence@4096 `89.5% → 90.5%`（+0.95pp = 净 +1 题，q16 进、无出），R@5 净 +1 题（q16/q17 进、q96 出），EM(1–4) `22.2% → 24.7%`（+2 题，其中 q33 的翻转经检验属 reader 抖动，见 §8）。
   - C5→C6：Evidence@4096 `88.6% → 89.5%`（+1 题，同样是 q16），但 EM(1–4) `21.0% → 19.8%`（−1 题，q29：全文变体把金标 s15 挤出 top-5）。
   - 唯一"机制可追溯"的正例是 **q16**：新增的 Composed 关系 `Jon --attended_fair_before--> worked_on_opening dance studio`（置信度 0.88，来源 session 10+15）让金标 session 10 从 rank 8/9 升到 rank 4（C6 为 rank 3），答案从 `No answer` 变成正确的 `24 April, 2023`。
2. **联合抽取（G_raw→C2 / G_raw→C5）的收益明显更稳，且主要体现在排序质量而不是"多捞回多少证据"**：R@3 `81.0% → 84.1%(C2) / 85.4%(C5)`，MRR `75.4% → 75.9% / 77.3%`，nDCG@10 `80.1% → 80.3% / 81.8%`。
3. **conv-30 的天花板本身很低，这是"增益看起来很小"的主因**：R@5 已经 90.3%、Evidence@4096 已经 88.6%、R@20 恒为 100%（只有 19 篇文档）、而且 conv-30 **一道 category-3 题都没有**，真正跨 session 的多跳题只有 11 道。
4. **答案层（EM）在 n=105 上不可用于判定方法优劣**：用"完全相同 prompt、不同次运行"的实测，EM 抖动幅度为 4–6 题/105（≈5pp），与所有方法之间的 EM 差距同量级。

---

## 1. 数据完整性核实（只汇总真正跑完的结果）

### 1.1 五种方法的 conv-30 产物

| 方法 | 图定义 | 结果文件 | 行数 | 问题集 | 有答案 | 完成 |
|---|---|---|---|---|---|---|
| **G_raw**（C0/C1） | 官方 per-session OpenIE 图 | `baselines_hipporag_conv-30.jsonl` | 105 | 105 唯一 qid | 105 | ✅ 2026-11-06 02:37 |
| **C2** | G_raw + 分组联合抽取三元组 | `readtime_cluster_hybrid_0.5_b8192_triple_conv-30.jsonl` | 105 | 与 G_raw 完全一致 | 105 | ✅ 11-06 03:22 |
| **C3** | C2 + 跨 session Comprehension 关系 | `readtime_cluster_hybrid_0.5_b8192_comp_conv-30.jsonl` | 105 | 完全一致 | 105 | ✅ 11-06 03:11 |
| **C5** | G_raw + 全文联合抽取三元组 | `readtime_full_conv_triple_conv-30.jsonl` | 105 | 完全一致 | 105 | ✅ 11-06 03:53 |
| **C6** | C5 + 全文 Comprehension 关系 | `readtime_full_conv_comp_conv-30.jsonl` | 105 | 完全一致 | 105 | ✅ 11-06 03:44 |

补充（不是主表要求，但同批跑完，用于消融）：

| 方法 | 结果文件 | 行数 | 完成 |
|---|---|---|---|
| C3-Composed-only | `readtime_ablation_..._comp_composed_conv-30.jsonl` | 105 | ✅ 11-08 00:13 |
| C3-Inferred-only | `readtime_ablation_..._comp_inferred_conv-30.jsonl` | 105 | ✅ 11-08 00:14 |

**核实要点**

- 五个文件 qid 集合 **逐题完全一致**（0 缺失 / 0 多出 / 0 重复），可以同题对比。
- 五个文件都有 `pred_answer`（无空回答、无 `finish_reason=length` 截断导致的空 answer）。
- G_raw 在 `outputs/raw_graphs/conv-30/` 根目录、C2/C3/C5/C6 在 `outputs/raw_graphs/conv-30/variants/` 下**各有独立目录**（各自带 `graph.pickle`、`openie_state.json`、`index_manifest.json`），说明检索确实在各目的图上跑过，不是复用同一份结果。
- **缺失/不可用项（明确说明，不补造）**：
  - `outputs/reports/readtime_summary.json` 已被后续 conv-44 的运行覆盖，**没有** conv-30 单独的 summary 文件；因此本文的 per-conv-30 数字是从逐题 jsonl 重新聚合的。
  - C3v（`comp_ver`，带 LLM 校验的变体）**没有** conv-30 的抽取产物，未参与比较。
  - conv-30 的 G_raw **图规模**在日志里是 `441 phrase nodes / 600 triples`；`READ_TIME_REPORT.md` §3 表里写的 `505 / 750` 实际是 **conv-26** 的数字（那份报告的行错位了）。本文一律采用日志 + `openie_state.json` 双向核对过的 441/600。

### 1.2 评测口径（全文统一，不混用）

- 检索：top-20 截断，document(=session) 级 Recall@1/3/5/10/20、MRR、nDCG@10。
- Evidence Coverage@4096 = `evidence_full@4096`：**所有**金标 turn 所属 session 是否都塞进 4096 token 的 context（贪心按 rank 装填，装不下就跳过该篇继续尝试后续）。
- QA：同一 reader、同一 prompt、`temperature=0`、`max_tokens=256`。
  - **EM/F1 只报 category 1–4（=81 题）**；
  - category 5（adversarial，24 题）单独报"拒答率"（回答命中 `no answer/cannot/...` 视为正确）。

### 1.3 交叉校验（证明我算的数与仓库既有一致）

拿本文重算的 `conv-30` 行，与 `outputs/reports/metrics_summary.json` 里已存的 `*|dev` 汇总的 `by_conversation["conv-30"]` 对比（该汇总本身覆盖全 dev 502 题）：

| 方法 | 已存 EM(all) / F1(all) / R@5 | 本文重算 | 一致 |
|---|---|---|---|
| G_raw | .3810 / .5199 / .9032 | .3810 / .5199 / .9032 | ✅ |
| C2 | .3810 / .5209 / .9032 | 同 | ✅ |
| C3 | .4000 / .5297 / .9079 | 同 | ✅ |
| C5 | .3810 / .5313 / .8984 | 同 | ✅ |
| C6 | .3714 / .5284 / .9056 | 同 | ✅ |
| C3-composed | .3810 / .5319 / .9079 | 同 | ✅ |
| C3-inferred | .3524 / .4936 / .9079 | 同 | ✅ |

---

## 2. conv-30 的结构性前提（读后面的表必须知道）

| 项 | 值 | 含义 |
|---|---|---|
| session 数（=候选文档数） | 19 | 检索池极小，top-20 = 全量 |
| 问题数 | 105 | cat1 multi-hop **11** / cat2 temporal **26** / cat3 open-domain **0** / cat4 single-hop **44** / cat5 adversarial **24** |
| 多 session 金标题 | 11（cat1 全部 + 2/3/4 session） | 只有这 11 题真正需要"跨 session 关联" |
| 4096 token 实际装填 | 平均 6.1 篇 / ≈3940 token（五方法一致） | 单篇 session ≈ 600 token，19 篇装不完 |
| R@20 | 五方法全为 **100%** | 该指标在 conv-30 上无区分度（文献里常见"R@20 已饱和"就是指这种情况） |

**重要推论**：Comprehension 的设计目标是"跨 session 的隐含关系"，而 conv-30 的跨 session 题只有 11 道、open-domain（最吃跨段推理的类别）为 0。**这个 graph 的可用上限在这里就已经被限死了**——下面所有"增益很小"的结论，都必须放在这个前提里读。

---

## 3. 主对照表（同 105 题）

### 3.1 主表

| 方法 | R@1 | R@3 | R@5 | R@10 | R@20 | Ev@4096 | MRR | nDCG@10 | **EM(1–4)** | **F1(1–4)** | 拒答率(cat5) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G_raw | .6103 | .8095 | **.9032** | .9738 | 1.000 | .8857 | .7542 | .8014 | .2099 | .3900 | .9583 (23/24) |
| C2 | .5944 | .8413 | .9032 | .9754 | 1.000 | .8952 | .7586 | .8026 | .2222 | .4036 | .9167 (22/24) |
| C3 | .5944 | .8317 | .9079 | .9802 | 1.000 | **.9048** | .7575 | .8041 | **.2469** | **.4151** | .9167 (22/24) |
| C5 | .6262 | .8540 | .8984 | .9786 | 1.000 | .8857 | .7730 | .8184 | .2099 | .4048 | .9583 (23/24) |
| C6 | **.6381** | **.8635** | .9056 | **.9833** | 1.000 | .8952 | **.7811** | **.8273** | .1975 | .4010 | .9583 (23/24) |

（区分度排序：MRR/nDCG/Evidence 更可信；R@20 无效；EM 噪声大，见 §8。）

### 3.2 分题型（EM / F1 / R@5 / Ev@4096）

**cat1 multi-hop（11 题，全部多 session）**

| 方法 | EM | F1 | R@5 | Ev@4096 |
|---|---|---|---|---|
| G_raw | .364 | .489 | .621 | .455 |
| C2 | .364 | .511 | .621 | .455 |
| C3 | .364 | .496 | **.667** | .455 |
| C5 | .182 | .458 | .576 | .364 |
| C6 | .182 | .459 | .553 | .364 |

**cat2 temporal（26 题）**

| 方法 | EM | F1 | R@5 | Ev@4096 |
|---|---|---|---|---|
| G_raw | .231 | .292 | .962 | .962 |
| C2 | .231 | .292 | .962 | .962 |
| C3 | **.308** | **.369** | **1.000** | **1.000** |
| C5 | .231 | .262 | .962 | .962 |
| C6 | .269 | .300 | **1.000** | **1.000** |

**cat4 single-hop（44 题）**

| 方法 | EM | F1 | R@5 | Ev@4096 |
|---|---|---|---|---|
| G_raw | .159 | .423 | .886 | .886 |
| C2 | .182 | .443 | .886 | .909 |
| C3 | .182 | .422 | .886 | .909 |
| C5 | **.205** | **.476** | .886 | .909 |
| C6 | .159 | .446 | .886 | .909 |

**cat5 adversarial（24 题，唯一指标=拒答率）**：G_raw .958 / C2 .917 / C3 .917 / C5 .958 / C6 .958。
C2、C3 各多错 1 题（q85，"definitely in for the show"）。但 **q85 在 C3 与 C3-composed 的 prompt 完全相同的情况下答案不同**，属 reader 抖动，不能归因于图谱（见 §8）。

---

## 4. 四组关键对比：联合抽取收益 vs 额外 Comprehension 收益

### 4.1 G_raw → C2（**联合抽取**，分组 hybrid_0.5 / 8192 预算）

| 指标 | G_raw | C2 | Δ |
|---|---|---|---|
| R@1 | .6103 | .5944 | −1.6pp |
| R@3 | .8095 | .8413 | **+3.2pp** |
| R@5 | .9032 | .9032 | 0.0 |
| Ev@4096 | .8857 | .8952 | **+0.95pp** |
| MRR | .7542 | .7586 | +0.44pp |
| nDCG@10 | .8014 | .8026 | +0.12pp |
| EM(1–4) | .2099 | .2222 | +1.2pp（1 题） |
| F1(1–4) | .3900 | .4036 | +1.4pp |

**解读**：R@1 掉、R@3 涨、R@5 不变 = 典型"中段排序被重排"的形态，说明联合抽取主要改善"第 2–3 名"的排序，而不是"多召回一篇"。

### 4.2 C2 → C3（**额外 Comprehension**，同三元组，只多 26 条跨 session 关系）

| 指标 | C2 | C3 | Δ |
|---|---|---|---|
| R@1 | .5944 | .5944 | 0.0 |
| R@3 | .8413 | .8317 | −1.0pp |
| R@5 | .9032 | .9079 | **+0.48pp（+0.5 题：q16 +1.0、q17 +0.5、q96 −1.0）** |
| Ev@4096 | .8952 | .9048 | **+0.95pp（+1 题：q16 进，无出）** |
| MRR | .7586 | .7575 | −0.11pp |
| nDCG@10 | .8026 | .8041 | +0.15pp |
| EM(1–4) | .2222 | .2469 | **+2.5pp（+2 题：q16、q33）** |
| F1(1–4) | .4036 | .4151 | **+1.2pp** |

逐题：金标 session 名次**改善 3 题**（q16 8→4、q17 6→4、q33 2→1），**退化 4 题**（q47 1→2、q80 2→3、q87 3→4、q96 4→6）；R@5 净翻转 = +1 题（q16、q17 进 / q96 出）；Ev@4096 只动了 1 题（q16 进）。

### 4.3 G_raw → C5（**全文联合抽取**）

| 指标 | G_raw | C5 | Δ |
|---|---|---|---|
| R@1 | .6103 | .6262 | **+1.6pp** |
| R@3 | .8095 | .8540 | **+4.4pp** |
| R@5 | .9032 | .8984 | −0.5pp |
| Ev@4096 | .8857 | .8857 | 0.0 |
| MRR | .7542 | .7730 | **+1.9pp** |
| nDCG@10 | .8014 | .8184 | **+1.7pp** |
| EM(1–4) | .2099 | .2099 | 0.0 |
| F1(1–4) | .3900 | .4048 | +1.5pp |

**解读**：全文联合抽取的排序收益是四组里最扎实的（MRR +1.9pp / nDCG +1.7pp / R@3 +4.4pp），但 Evidence@4096 一点没涨——就是"排序更好、覆盖面不变"。同时它把 cat1 多 session 题拖坏了（EM .364→.182、Ev .455→.364），因为全文抽取的三元组把多 session 题的证据挤出了 4096 预算。

### 4.4 C5 → C6（**全文 + Comprehension**）

| 指标 | C5 | C6 | Δ |
|---|---|---|---|
| R@1 | .6262 | .6381 | **+1.2pp** |
| R@3 | .8540 | .8635 | **+0.95pp** |
| R@5 | .8984 | .9056 | **+0.72pp（+0.75 题：q16 +1.0、q23 −0.25）** |
| Ev@4096 | .8857 | .8952 | **+0.95pp（+1 题：q16 进，无出）** |
| MRR | .7730 | .7811 | +0.81pp |
| nDCG@10 | .8184 | .8273 | +0.89pp |
| EM(1–4) | .2099 | .1975 | **−1.2pp（−1 题：q29）** |
| F1(1–4) | .4048 | .4010 | −0.4pp |

逐题：名次改善 3 题（q16 8→3、q31 4→1、q72 3→1），退化 2 题（q23 1→5、q17 8→7）。

### 4.5 一句话区分两种收益

| 收益来源 | 最稳的证据 | 量级 |
|---|---|---|
| **联合抽取本身**（多读几段一起抽） | R@3 +3.2~4.4pp、MRR +0.4~1.9pp、nDCG@10 +0.1~1.7pp | 排序层的系统性小幅改善，多个题一致受益（q4 四种变体都进 top-5） |
| **额外 Comprehension**（再找跨 session 隐含关系） | Evidence@4096 **各 +0.95pp（C2→C3、C5→C6 都恰好 +1 题，且都是 q16）**；R@1 不动、MRR 一升一降（−0.11 / +0.81pp） | 覆盖面各只多挤出 1 题；排序层无一致方向；EM 变化在噪声内 |

> 注意：C2→C3 与 C5→C6 的 +0.95pp 都等于 **1 题 / 105 题**（C3：q16 进；C6：q16 进）。在 105 题的样本上，"一题≈0.95pp"，所以这两个 +0.95pp 必须读成"1 题"，而不是"约 1 个百分点的系统性收益"。

---

## 5. 图谱规模与新增关系

### 5.1 节点 / 关系（全部来自 `get_graph_info()`，日志可查）

| 图 | phrase 节点 | passage 节点 | 总节点 | 抽取三元组 | 含 passage 的三元组 | 同义边 | 总边数 |
|---|---|---|---|---|---|---|---|
| G_raw | 441 | 19 | 460 | 600 | 545 | 741 | 1,733 |
| C2 | 552 | 19 | 571 | 787 | 688 | 1,090 | 2,343 |
| C3 | 570 | 19 | 589 | 813 | 742 | 1,108 | 2,434 |
| C5 | 532 | 19 | 551 | 735 | 657 | 901 | 2,095 |
| C6 | 559 | 19 | 578 | 753 | 723 | 938 | 2,213 |
| C3-Composed-only | 566 | 19 | 585 | 807 | 732 | 1,102 | 2,414 |
| C3-Inferred-only | 556 | 19 | 575 | 793 | 698 | 1,095 | 2,362 |

规模变化（相对 G_raw，按"抽取三元组"计）：C2 **+31%**（787 vs 600）、C3 **+36%**（813）、C5 **+23%**（735）、C6 **+26%**（753）。C3 比 C2 多 26 个 fact 节点，恰好等于 26 条关系（每条关系在图中就是 1 个 fact + 2 条连接边）；三组图的 passage 节点都是 19，说明变化全部来自新增知识。

### 5.2 注入记录与重复（严格区分三件不同的事）

| 项 | C2 | C3 | C5 | C6 |
|---|---|---|---|---|
| 注入三元组（去重后记录数） | 228 | 228 | 164 | 164 |
| 注入关系（Composed / Inferred） | 0 | **26（20 / 6）** | 0 | **18（9 / 9）** |
| 词面完全重复、被跳过写库（`skipped_duplicate`，按 chunk 计） | 24 | 24 | 19 | 19 |
| 实际写库插入次数（按 chunk 计） | 217 | 273 | 159 | 196 |
| 与 G_raw 语义重复（A 类完全重复） | — | **0** | — | **0** |
| 与 G_raw 语义重复（B 类改写重复） | — | **0** | — | **0** |
| 单 session 关系（不含跨 session，官方审计未计为 D 类） | — | 0 | — | 0 |
| 关系内部自身重复（(s,p,o) 相同或与三元组重复） | — | 0 | — | 0 |

（A/B 类来自 `outputs/reports/comprehension_audit.json`：conv-30 的 `cluster_hybrid_0.5_b8192_comp` 26 条关系全部判定为 **D 类"双/多 session 可支撑"**；`full_conv_comp` 18 条中 17 条 D 类、1 条 E 类"低置信推断"。）

**关系类型分布**

- C3（26 条）：`temporal_ordering` 17、`state_change` 5、`causal` 2、`identity_unification` 1、`enabling_link` 1；置信度 min/mean/max = 0.78 / 0.88 / 0.97。
- C6（18 条）：`temporal_ordering` 7、`causal_link` 5、`state_change` 4、`identity_unification` 2；置信度 0.55 / 0.84 / 0.96。

**结论**：conv-30 上的新增关系质量是"干净但冗余"——没有语义重复、全部跨 session，但 **65%（C3：17/26，C6 只有 39%：7/18）都是 temporal_ordering**，而 conv-30 的时间题本来 R@5 就有 96.2%，等于把新增预算砸在了已经饱和的题型上。

### 5.3 关系的"可用机会"（potential ≠ realized）

| 口径 | C3 | C6 |
|---|---|---|
| 金标 session 上至少挂着 1 条新关系的题 | **100 / 105** | 99 / 105 |
| 其中"关系文本与问题有词面重叠"的题（宽口径，单词命中即算） | **90 / 105** | 75 / 105 |
| 实际发生金标名次改善的题 | 3 | 3 |
| 实际把答案由错变对的题（可归因） | 1（q16） | 1（q16） |

**机会 90 题 → 兑现 1 题**，这是本次分析最重要的一个数字。

---

## 6. 逐题比较（相对 G_raw）

### 6.1 汇总

| 方法 | 检索改善(R@5↑) | 检索退化(R@5↓) | 持平 | Ev@4096↑ | Ev@4096↓ | **答案由错变对(cat1–4)** | **答案由对变错(cat1–4)** |
|---|---|---|---|---|---|---|---|
| C2 | 2 | 2 | 101 | 2 | 1 | 1（q74） | 0 |
| C3 | 4 | 3 | 98 | 3 | 1 | 3（q16, q33, q47） | 0 |
| C5 | 1 | 2 | 102 | 1 | 1 | 2（q39, q74） | 2（q24, q29） |
| C6 | 2 | 3 | 100 | 2 | 1 | 1（q16） | 2（q24, q29） |
| C3-composed | 4 | 3 | 98 | 3 | 1 | 2（q16, q74） | 2（q0, q24） |
| C3-inferred | 3 | 2 | 100 | 2 | 1 | 1（q47） | 3（q11, q24, q78） |

> **上表"答案翻转"一栏必须配合 §8 一起读**：conv-30 的 EM 翻转量级是 1~3 题，而 reader 在同一 prompt 下的抖动就有 4~6 题。逐条核查后，只有 **q16**（C3/C6）有检索层独立佐证；**q47 / q74 / q39 / q24 / q33 / q85 / q11 / q78 / q0** 的翻转都发生在 context 相同或几乎相同的情况下（详见案例 2/5/9/10 与 §8），不能记在图谱账上。**C5 唯一可归因的答案是 q29 的"由对变错"（金标 s15 被挤出 top-5）。**

金标 session 名次（最好的那个金标 session）移动统计：

| 方法 | 名次改善 | 名次退化 | 不变 |
|---|---|---|---|
| C2 | 18 | 13 | 74 |
| C3 | 17 | 12 | 76 |
| C5 | 13 | 8 | 84 |
| C6 | 16 | 9 | 80 |

> 注意两件事：(a) R@5 的翻转非常稀疏（每题 1 篇文档就是 0.5~1pp），(b) **名次移动 21~31 题，但真正跨过 R@5 边界的只有 3~5 题**——因为 conv-30 的金标 session 大多本来就在 top-5 内，名次从第 3 挪到第 2 对指标毫无影响。

### 6.2 检索层明细（R@5 或 Ev@4096 发生变化的全部题）

| 题 | 类 | 金标名次 G_raw→ | C2 | C3 | C5 | C6 | 备注 |
|---|---|---|---|---|---|---|---|
| q2 | 4 | 6 | **5 ✅R@5 0→1** | **5 ✅** | 6 | 6 | 仅"分组联合抽取"受益 |
| q4 | 4 | 7 | **4 ✅** | **4 ✅** | **5 ✅** | **5 ✅** | 四种变体都受益（联合抽取通用收益） |
| q16 | 2 | 9 | 8（dense 回退） | **4 ✅ EM 0→1** | 8（dense 回退） | **3 ✅ EM 0→1** | **Comprehension 唯一硬正例** |
| q17 | 1 | 8/12 | 6/11 | **4/11（R@5 0→.5）** | 8/11 | 7/11 | 多 session 题，只捞回 1/2 |
| q33 | 2 | 1 | 2 | **1 ✅ EM 0→1** | 1 | 1 | 名次 2→1；EM 翻转属抖动 |
| q44 | 4 | 5 | 6 ❌R@5 1→0 | 6 ❌ | 6 ❌ | 6 ❌ | **四种变体一致退化**（Ev 未变，仍 1.0） |
| q54 | 4 | 1 | **9 ❌ Ev 1→0** | **9 ❌** | 1 | 1 | 只有分组变体坏：触发 dense 回退 |
| q96 | 5 | 2 | 4 | **6 ❌R@5 1→0** | 3 | 3 | 对抗题，拒答仍正确 |
| q29 | 1 | s2 第1 / s15 第3 | 1/3 | 1/2 | **1/9 ❌Ev 1→0** | **1/9 ❌Ev 1→0** | 全文变体把 s15 挤出 top-5（两 session 题） |
| q23 | 1 | 1（dense 回退） | 1 | 1 | 1 | **5 ❌R@5 .5→.25** | C6 从 dense 回退切到 PPR 后反而变差 |
| q39 | 4 | 1 | 1 | 1 | 1 | 1 | 检索名次五法全为 1，但 C5 答 `contemporary`（EM 1）而其余答 `contemporary dance`（EM 0）→ **纯措辞/抖动，非图谱** |
| q74 | 4 | 2 | 3（EM 1） | 3（EM 0） | 2（EM 1） | 2（EM 0） | 检索名次反而变差却变对；C3 vs C3-composed 同 prompt 出现 1/0 分歧 → 抖动 |

---

## 7. 典型案例（含 Gold Evidence、各方法 Top-5、最终答案）

> 说明：`score` 在不同图之间**不可比**（PPR 分数未归一化，而 dense 回退路径会把 top-1 压到 1.0），所以只做名次比较。文档 id `sNN` = 第 NN 个 session。Triples 只进入图、**不进入 reader 的 prompt**（reader 只读 session 原文），因此下面所有"关系起作用"都只能通过检索排序生效，不存在文本泄漏。

### 案例 1（**正例，Comprehension 独立增益**）q16

- **问题**：`When did Jon go to a fair to get more exposure for his dance studio?`
- **Gold**：`24 April, 2023`；**Gold Evidence**：session 10，turn `D10:1`（单 session、单 turn，题目本身不难）

| 方法 | Top-5（名次） | 金标 s10 名次 | Ev@4096 | 答案 |
|---|---|---|---|---|
| G_raw | s05, s09, s11, s02, s18 | 9 | 0 | `No answer` ❌ |
| C2 | s18, s05, s01, s02, s17（**dense 回退**，top1=1.0） | 8 | 0 | `No answer` ❌ |
| **C3** | s08, s11, s18, **s10**, s15 | **4** | **1** | **`24 April, 2023`** ✅ |
| C5 | s18, s05, s01, s02, s17（dense 回退） | 8 | 0 | `No answer` ❌ |
| **C6** | s08, s11, **s10**, s18, s05 | **3** | **1** | **`24 April, 2023`** ✅ |

- **到底是哪条关系起的作用**：C3 关系 #15
  `Jon --attended_fair_before--> worked_on_opening dance studio`（Composed，conf 0.88，`source_sessions=[10,15]`，`temporal_ordering`）。
  该关系与问题的词面重叠 = `{fair, jon, dance, studio}`（4 个词）——在 s10 挂载的 3 条关系里是唯一同时覆盖 "fair" 和 "dance studio" 的一条，且它的 provenance 直接指向 session 10，因此它能把 PPR 质量引到 s10。
  C6 里起同样作用的是另一条：`Jon's dance studio --was_promoted_at--> fair`（Inferred，conf 0.55，`[10,12]`）。
- **消融验证**（这是"独立增益"最有力的证据）：只保留 Composed 关系的 C3-composed 复现了 C3 的名次（4）与正确答案；只保留 Inferred 关系的 C3-inferred **退回到 C2 的 dense 回退状态**（名次 8、`No answer`）。两个消融图的 triple 集合完全相同、只差关系子集，所以**这条 Composed 关系是唯一的作用源**。
- **反直觉细节**：C2 的失败不是"排错"，而是**"根本没进 PPR"**——它的 top-1 分数正好等于 1.0，这是 `dense_passage_retrieval` 的 min-max 归一化特征，意味着 recognition memory 之后没有任何 fact 存活，整题退化成纯向量检索。**推断**（产物里没有落盘 per-question 的 fact 候选列表，因此这是基于分数特征 + 消融的推断）：C3 新增的这条关系进入 fact 候选池后改变了 rerank 结果，使 PPR 重新被激活。

### 案例 2（**正例但答案翻转不可归因**）q33

- **问题**：`When did Jon visit networking events for his store?` / **Gold**：`20 June, 2023` / Evidence：session 16 `D16:6`
- 名次：G_raw 1 → C2 **2** → C3 **1** → C6 1。
- 起作用的关系：`Jon --attended_networking_events_before--> went_full-time_with biz idea`（Composed 0.82，`[16,17]`），与问题重叠 `{events, jon, networking}`。
- 但 **C3 与 C3-composed 的 Top-5 顺序与 token 数完全相同（3740），答案却是 `20 June, 2023` vs `No answer`** → 这次 EM 翻转是 reader 抖动，**不能算 Comprehension 的功劳**。检索名次 2→1 可以算。

### 案例 3（**正例，多 session 部分修复**）q17

- **问题**：`Why did Gina decide to start her own clothing store?` / **Gold**（长答案，需 s1+s6 两处证据）/ Ev 需要 s1 和 s6 同时在 4096 内
- 名次：s1/s6 = G_raw `12/8` → C2 `11/6` → **C3 `11/4`** → C6 `7/11`。
- 起作用的关系：`Gina --lost_job_before--> opened online clothing store`（Composed 0.93，`[6,14]`，与问题重叠 `{gina, clothing, store}`）把 s6 从 8 拉到 4。
- 结果：R@5 从 0 变成 0.5，**但 Ev@4096 仍为 0**（s1 在第 11 位，装不进预算）→ 答案依旧错。这是"关系只修好了一半"的典型：跨 session 题需要**两条**证据，救回一条不解决 EM。

### 案例 4（**联合抽取的通用正例，但 EM 没兑现**）q4

- **问题**：`Why did Jon decide to start his dance studio?` / **Gold**：`He lost his job and decided to start his own business to share his passion.` / Evidence：session 1
- 名次：G_raw 7 → C2 **4** → C3 **4** → C5 **5** → C6 **5**（Ev 0→1，四种变体全部受益）
- 但四种变体的答案都变成 `he's passionate about dancing...`（只回答了"share passion"半句，丢了"lost his job"），EM 仍 0、F1 0.33 左右。
- 说明：**联合抽取稳定提升"证据可达性"，但 reader 的综合/取舍能力才是 EM 的瓶颈**。

### 案例 5（**负例：分组变体触发 dense 回退，反而丢掉唯一金标**）q54

- **问题**：`How is Gina's store doing?` / **Gold**：`The store is doing great.` / Evidence：session 4 `D4:2`

| 方法 | Top-5 | 金标 s4 | Ev@4096 | 答案 |
|---|---|---|---|---|
| G_raw | **s04**, s08, s05, s03, s06（PPR） | 1 | 1 | `doing great`（F1 .667，EM 0：缺 "The store is"） |
| C2 | s08, s05, s06, s03, s02（**dense 回退，top1=1.0**） | **9** | **0** | 长句复述，F1 0 ❌ |
| C3 | 同 C2（dense 回退） | **9** | **0** | 同上 ❌ |
| C5 / C6 | s04 仍在第 1（PPR 正常） | 1 | 1 | `doing great` |

- **机理**：q54 是一个"短、泛、无实体"的问题（How is Gina's store doing?）。C2/C3 的金标 session 4 上**一条新关系都没有**（`[C3-rel] 0 records touch gold sessions`），但注入的新三元组改变了 fact 候选池，使 recognition memory 阶段没有任何 fact 存活 → 退化为纯向量检索 → 泛化问题下向量检索把 s8 顶到第一。
- **这是 C2/C3 相对 G_raw 唯一"真·检索退化"（Ev 1→0）的题**，而且它**不是** Comprehension 的错（C3 与 C2 完全相同），是"往图里塞知识"这一动作的副作用。

### 案例 6（**负例：四种变体一致退化，但证据没丢**）q44

- **问题**：`What does Gina say about the dancers in the photo?` / **Gold**：`They look graceful` / Evidence：session 1
- 名次：G_raw 5 → 全部变体 **6**（Ev@4096 全部仍为 1.0，R@5 从 1 掉到 0）
- 答案：G_raw `they're so graceful`（EM 0，因为 gold 是 "They look graceful"，**F1 = .667**；EM 判错是措辞差异）→ 变体 `they look great`（真的答错了）。
- 说明两件事：① R@5 的"退化"很多时候只是第 5/6 名互换、证据仍在预算内；② **EM 对 "they're/they look" 这类措辞非常脆**，用它评估方法会放大噪声。

### 案例 7（**负例：全文变体把多 session 证据挤出预算**）q29 / q24

- **q29** `Which cities has Jon visited?`（Gold `Paris, Rome`，Evidence s2 + s15）
  - G_raw：s2 第 1、s15 第 3 → Ev 1、EM 1（`Paris, Rome`）
  - **C5/C6：s15 掉到第 9 → Ev 0**，答案退化成 `Paris`（EM 0、F1 .667）
  - 原因：全文抽取新增的三元组把 s15 的排序挤下去了（s17/s04/s13/s18 顶上来）；这题**没有任何新关系**能 bridging s15（C6 关系全部与 "Jon visited Paris/Rome" 无关）。
- **q24** `Which events has Jon participated in...?`（3 个金标 session：s8/s10/s16）
  - G_raw/C2/C3 答案正确（`a fair, networking events, a dance competition`），**C5/C6 变成 `a fair, networking events`**（漏掉 dance competition）。
  - 但 C5/C6 的 Top-5 **集合与 G_raw 完全相同**（只是 s14 从第 5 顶到第 1，s16/s18/s08/s04 顺延），Ev@4096 也仍是 1.0 → 这个答案损失是 reader 抖动，不能记在全文抽取账上。

### 案例 8（**正例：检索大幅改善但 QA 无反应**）q31

- **问题**：`How long did it take for Jon to open his studio?` / **Gold**：`six months` / Evidence：s1 + s15（多 session）
- 名次：G_raw s15 第 4 / s1 第 11 → **C6 s15 第 1** / s1 第 9（Ev 仍为 0）
- 起作用的关系：`Jon's dance studio --opened_after--> Jon searched for ideal spot`（Composed 0.93，`[2,15]`）。
- 结果：名次从 4 直接到 1，**但答案仍然 `No answer`**——因为 "six months" 在 s15（`D15:13`），需要和 s1 的"开始找工作"对齐做时间差计算，reader 没有做。这是"检索已到位、reader/多跳推理不到位"的清晰样本。

### 案例 9（**负例：对抗题被"顶到第 1 的错误 session"带偏**）q85

- **问题**：`What is Gina's attitude towards participating in the dance festival?` / **Gold**：`None`（不可回答）
- G_raw：Top-1 = s11，s01 第 2 → `No answer` ✅
- C2/C3：**s01 被顶到第 1** → `definitely in for the show` ❌（幻觉）
- 但 q85 在 C3 与 C3-composed 之间 **Top-5 顺序、token 数（4094）完全一致**，答案却一个是幻觉一个是 `No answer` → 归因为抖动；只能说"Top-1 变成 s01 后，这个 prompt 更容易被诱导"，不能断言图谱导致。

### 案例 10（**消融：Composed vs Inferred 的分工**）q47 / q11 / q78

| 题 | 现象 | Composed-only | Inferred-only |
|---|---|---|---|
| q47 `What did Gina find for her clothing store on 1 Feb 2023?` | C3 EM 0→1（答案 `a perfect spot for **her** store` vs gold `The perfect spot for her store`；C2 答 `your store` → EM 0） | EM 0（`your store`） | EM 1（`her store`） |
| q11 `When did Gina get her tattoo?` | 名次完全相同（s05 都第 1） | EM 1 | **EM 0（`No answer`）** |
| q78 `According to Gina, what makes Jon a perfect mentor?` | 名次完全相同 | EM 1 | **EM 0（把 "His" 抄成 "Your"）** |

→ **q11/q78 的差异只能来自 reader 抖动**（检索完全一致），这正好给出本次实验的噪声标尺；**q47 的 EM 差异实际是代词措辞（your/her）**，也不该算 Comprehension 功劳。

---

## 8. 必须写进结论的测量噪声（否则会过度解读）

我在同一批产物里发现了一个可以**直接量化**的 reader 抖动：

**定义**：两次运行塞进 reader 的 session **集合完全相同**（同一批文档、同一 token 预算），但答案不同。下表再细分"顺序也完全相同"（= prompt 逐字节相同）的情形。

| 对比 | 同 context 集合的题数 | 其中 EM 不同 | 其中"连顺序都相同"仍 EM 不同 |
|---|---|---|---|
| C3 vs C3-composed | 98 / 105 | **6**（q0, q24, q33, q47, q74, q85） | **4**（q0, q33, q74, q85） |
| C3 vs C3-inferred | 85 / 105 | 3（q11, q24, q78） | 1（q78） |
| C3 vs G_raw | 53 / 105 | 2（q47, q85） | 0 |
| C5 vs C6 | 77 / 105 | 0 | 0 |

q0 / q33 / q74 / q85 这四组是**ranked_docs 全序列与 context_tokens 都完全一致**（prompt 逐字节相同）却得到不同答案——说明 HY4 在 `temperature=0` 下接口本身存在非确定性（不同批次走了不同 cache shard，都是真实调用，非缓存复用）。

**推论（对本文所有 EM 数字）**：EM 的噪声带宽 ≈ **4~6 题 / 105 ≈ 4~6pp**。因此：

- C3 相对 G_raw 的 EM +3 题 = 落在噪声带内，**不能宣称"答案层有显著提升"**；
- C2→C3 的 EM +2 题同样如此；
- 唯一超出噪声的 EM 证据是 **q16**（有检索名次 8/9→4/3 的独立佐证 + 消融复现）。

另一个"隐藏机制"也必须交代：**11~16 题/105 的检索会退化成纯 dense**（top-1 分数恰为 1.0 = min-max 归一化特征）：

| 方法 | dense 回退题数 | 与 G_raw 重叠 | 只在本文该方法回退的题 |
|---|---|---|---|
| G_raw | 13 | 13 | — |
| C2 | 16 | 12 | q9, q16, q54, q56 |
| C3 | 15 | 12 | q9, q54, q56 |
| C5 | 13 | 11 | q16, q56 |
| C6 | 11 | 10 | q56 |

- **9 道题在五种图上都恒定回退**（q67/q82/q83/q93/q97/q99/q100/q101/q102）——这些题的 query 与图的 fact 根本匹配不上，任何建图策略都无效。
- **并集 17 题**，也就是说有 8 道题会在"回退 / PPR"之间切换（例如 q16 在 C2/C5 回退、在 C3/C6 走 PPR；q54 在 C2/C3 回退、在 G_raw/C5/C6 走 PPR）。
- 这些 regime switch **恰好解释了本次分析里两个最大的单题波动**（q16 变好、q54 变坏）。读"Comprehension 增益"时，"有没有进 PPR"往往比"关系桥接得好不好"更决定结果。

---

## 9. 最终回答：Comprehension 在 conv-30 上是否有独立增益？

### 9.1 有，但只在检索层、且只有 1 题完成闭环

| 判据 | 结论 |
|---|---|
| C2→C3 检索层 | **有（弱）**：Ev@4096 +0.95pp（+1 题：q16 进、无出），R@5 +0.48pp（q16/q17 进、q96 出），名次改善 3 题 / 退化 4 题 |
| C5→C6 检索层 | **有（弱）**：Ev@4096 +0.95pp（+1 题：q16 进、无出），R@1 +1.2pp、MRR +0.81pp、nDCG +0.89pp（排序层这一组比 C2→C3 更像真改善） |
| 排序层一致性 | **无**：C2→C3 的 MRR/nDCG 基本不动（−0.11 / +0.15pp），说明不是普遍性的排序改善 |
| 答案层 | **不可证**：+1~+2 题，处在 4~6 题的 reader 噪声带内 |
| 机制可追溯 | **1 题**：q16（Composed 关系 `Jon --attended_fair_before--> worked_on_opening dance studio`，`[10,15]`，conf .88），且消融（Composed-only 复现、Inferred-only 失败）确认它是唯一作用源 |

### 9.2 具体体现在哪里（按可验证程度排序）

1. **把"本来进不了 PPR 的题"救回 PPR**：q16 在 C2/C5 里整题退化为 dense 检索（top-1=1.0），加了 relation 后 fact 候选池改变、PPR 重新激活（机理为基于分数特征 + 消融的**推断**，见案例 1），金标 session 从第 8/9 名直接进到第 3/4 名。**这是"关系带着 provenance 参与检索"而不是"关系变成新文本"的典型收益。**
2. **给多 session 题补一条桥**：q17（s6 8→4）、q33（2→1）、q31（4→1）都是"关系的 subject/object 正好覆盖问题关键词，且 `source_sessions` 含金标 session"的情形，名次改善幅度 1~3 名，方向正确。
3. **对已饱和的题型几乎没有作用**：C3 的 65% 新关系是 `temporal_ordering`，而 cat2 的 R@5 本来就 96.2%、Ev 96.2% → 边际收益为负（q96 就是这样被挤出去的）。

### 9.3 为什么增益这么小（三条硬约束）

1. **题面天花板**：R@5 90.3%、Ev@4096 88.6%、R@20 100%、cat3 = 0 题、跨 session 题仅 11 道。90/105 题的关系"有机会命中"，但其中大多数题**本来就已检索正确**，关系再好也只能重排前 5 名内部。
2. **4096 token 预算**：能装 6.1/19 篇 session；多条证据的题（多 session）几乎注定 Ev=0（cat1 Ev 只有 45.5%）。**关系能把某条证据拉进 top-5，但不能让两条证据同时进入 4096 预算**（q17/q31 就是这样卡住的）。
3. **reader 才是瓶颈**：q4 的 Ev@4096 明确 0→1、q31 的金标名次明确 4→1，答案仍然错；反过来 q24/q47/q74 这些答案翻转发生在 context 相同或几乎相同的情况下。**在 conv-30 上，EM 的上限由 reader 的措辞/多跳推理决定，不是由图谱决定。**

### 9.4 建议（如果要继续这条线，按性价比排序）

1. **换评测口径再判 Comprehension**：用 `Ev@4096 + R@3/MRR`（对图谱敏感、噪声小），EM 只作参考并报 bootstrap CI（n=105 的 ±1 题 ≈ ±0.95pp，现在所有 EM 结论都不显著）。
2. **把 Conv-30 的结论限制在"检索层"**，真正评 Comprehension 应该在 cat3/open-domain 或长对话（会话更长、需要多跳）的样本上做；conv-30 的有效题量太小。
3. **关系生成要"反饱和"**：现在 65%（C3）产出 temporal_ordering，而时间题已饱和。应显式要求产出 `causal/state_change/enabling` 且**必须跨越 2 个以上 session 且其中一个 session 是问题实体当前未命中的 session**，把预算投到 Ev@4096=0 的那些题（cat1 的 55% 缺口）。
4. **修掉 dense 回退的"暗礁"**：9 题恒定回退 + 8 题在不同图之间"回退↔PPR"来回切换，这类 regime switch 贡献了两个最大的单题波动（q16 变好、q54 变坏）。建议在 `rerank_facts` 后把 `facts_after_rerank` 数量落盘，作为"本次检索是否可信"的显式指标。
5. **消除 reader 抖动**：同一 prompt 多次运行取多数票（或缓存同一 shard），否则 n=105 上的 ±1~3 题差异永远无法归因。

---

## 附录 A：本次分析用到的文件清单

| 用途 | 路径 |
|---|---|
| 五方法逐题结果 | `outputs/qa/baselines_hipporag_conv-30.jsonl`、`readtime_cluster_hybrid_0.5_b8192_triple_conv-30.jsonl`、`readtime_cluster_hybrid_0.5_b8192_comp_conv-30.jsonl`、`readtime_full_conv_triple_conv-30.jsonl`、`readtime_full_conv_comp_conv-30.jsonl` |
| 消融结果 | `outputs/qa/readtime_ablation_cluster_hybrid_0.5_b8192_comp_{composed,inferred}_conv-30.jsonl` |
| 抽取产物（三元组/关系/derivation/置信度/source_sessions） | `outputs/comprehension/conv-30/{cluster_hybrid_0.5_b8192_triple,cluster_hybrid_0.5_b8192_comp,full_conv_triple,full_conv_comp}.json` |
| 图规模 | `outputs/logs/readtime_eval_conv-30.log`、`baselines_g2.log`（conv-30 G_raw）、`rt_smoke.log`（C3 建图）、`ablation_{composed,inferred}.log` |
| 注入/重复统计 | 同上日志中的 `injection: {added, chunks_touched, skipped_duplicate}` |
| 语义重复审计（A/B/C/D/E） | `outputs/reports/comprehension_audit.json`、`COMPREHENSION_AUDIT.md` |
| 交叉校验 | `outputs/reports/metrics_summary.json`（`*|dev` 的 `by_conversation.conv-30`） |

## 附录 B：口径与坑（复现时务必注意）

1. 所有指标都用已落盘的逐题 `retrieval.*` / `budget.4096.*` / `em` / `f1` 字段直接平均，与 `summarize()` 完全同定义；**未重新计算任何 embedding 或 PPR**。
2. `R@20` 在 conv-30 恒为 1.0（19 篇文档 < 20），**不要**把它写进结论表当有效指标。
3. `Ev@4096` 是"全部金标 turn 的 session 都在 context 内"，对多 session 题非常苛刻（cat1 只有 45.5%）。
4. 跨图比较**只看名次，不看 score**：dense 回退路径的 top-1 恒为 1.0，PPR 路径的分数是未归一化的 PPR 质量。
5. `injection.added` 是**按 chunk 计的插入次数**（一条跨 2 个 session 的关系会被插 2 次），不等于"新增记录数"；新增记录数请用 comprehension json 里的 228/26、164/18。
6. 图中 `num_extracted_triples`（HippoRAG 归一化后）与 `openie_state.json` 里 unique triple 数（600→804/830/745/763）不同，两个口径都存在，本文只用前者做图规模、后者做"注入增量"的旁证。
7. `READ_TIME_REPORT.md` §3 的 conv-30 行（505/750）是 conv-26 的数据，属于既有报告的行错位，本文已改用日志核对的 441/600。
