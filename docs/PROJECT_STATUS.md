# 项目进度台账

> 更新：2026-11-12。这里是唯一的进度摘要；详细数字以各实验 README 和 `outputs/` 原始报告为准。

## 当前主线

| 项 | 当前值 |
|---|---|
| 主线数据集 | DetectiveQA |
| 当前阶段 | Round 2 已完成；R2–R5 图注入和 Question-Time 闭环暂停 |
| 阻塞点 | 当前 split 主要在测 recency/reveal；gold 不能作为严格上限；关系 precision 只有 66.7% |
| 下一项实验 | anti-reveal 子集 + gold 重审，然后在该子集上重跑 Stage A/B |
| 继续图注入的条件 | 人工 kept precision ≥75%，且检索 R@10 或 All@20 的 CI 下界 > 0 |

## 实验状态总表

| 数据集 | 已完成 | 最重要结果 | 当前判断 | 主要未完成项 | 详情 |
|---|---|---|---|---|---|
| MuSiQue / Wiki | 二部图审计、零 API 分组、v4/v5 read-time、本地 Qwen pilot | 二部图最大连通分量占 97.99%；最好分组 313/1000，顺序切块 225 | 不适合验证阅读时跨文档理解；保留证据链和分层验证 | 没有和 HippoRAG2 R@5 74.7 可比的全库数字 | [01](../experiments/01_musique_wiki/README.md) |
| LoCoMo | 阶段 A 全量；阶段 B/C 只在 conv-30 | test R@5 `.864→.885`，F1 `.472–.494`；P7 F1 `+13.1pp` | Reader 和饱和基准掩盖了理解层信号 | B/C 没有推广到其余 9 段对话；没有 LLM-Judge | [02](../experiments/02_locomo/README.md) |
| QASPER | 100 篇 / 372 题冻结样本，M0–M6，QA，机制追踪 | Dense Any@5 `.740` > G_raw `.692`；M4 All@5 `−14.4pp` | unit 覆盖不足，等权融合会挤掉正确证据 | 281 篇全量、门控融合、section 级覆盖扩张 | [03](../experiments/03_qasper/README.md) |
| DetectiveQA | 6 部小说建图、prefix-safe、Round 1/2、read-time v3 | recent `.805/.805`；nonrecent `.439/.390`；kept precision 66.7% | 当前 split 无法测出图方法的长程收益 | anti-reveal、gold 重审、R2–R5、Question-Time、154 题扩展 | [04](../experiments/04_detectiveqa/README.md) |

## 假设台账

| ID | 假设 | 当前状态 | 证据 | 下一步是否继续 |
|---|---|---|---|---|
| H1 | 无监督分组能把相关文档放进同一组 | 弱成立 | LoCoMo entity_idf 共置 `.431` vs random `.304`；MuSiQue 最好 313/1000 | 不作为主线 |
| H2 | 联合抽取优于逐文档抽取 | 检索上成立 | LoCoMo C5 test R@5 `.864→.874` | 作为抽取对照 |
| H3 | 理解层能产生普通 triple 表达不了的新知识 | 未证实 | LoCoMo C6≈C5；QASPER M3 CI 跨 0；DetectiveQA precision 66.7% | 先提高 precision |
| H4 | 理解图能提升检索和 QA | 检索局部成立，QA 未成立 | LoCoMo 证据 +0.7~1.8pp，EM/F1 ≤±0.5pp | 只在 anti-reveal 子集重测 |
| H5 | Retrieval Skill 能补全关系链 | 检索弱成立，QA 未成立 | 触发率约 92%，过度触发 | 先做 sufficiency gate |
| H6 | Read-Time 与 Question-Time 互补 | 未成立 | LoCoMo R1Q2 检索最好但 QA 不如 R1Q1 | 暂停 |
| H7 | 全文直读或简单分组已经很强 | LoCoMo 成立 | Full-Context F1 `.493` ≈ 最好图 `.492` | 必须保留强基线 |
| H8 | Reader 是瓶颈 | LoCoMo/QASPER 成立；DetectiveQA 部分成立 | LoCoMo Oracle+P0 `.387`；QASPER Oracle F1 `.487` | 与检索问题分开报告 |
| H9 | 新关系必须进入 top-k 才有价值 | 机制成立 | LoCoMo 99 → 5 → 1 | 用于设计融合 gate |
| H10 | 反思技能能泛化 | 未成立 | 28 题命中：1 有益、1 有害、26 中性 | 等图方法过门槛后再做 |
| H11 | Unit 比 triple 更适合承载理解 | 只在覆盖子集成立 | covered `.567→.627`；uncovered `.514→.089` | 需要覆盖扩张 + 门控 |

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 11-03 | 停止 MuSiQue index-time 分组路线 | 无监督分组几乎无法让 gold 同组，实体图形成巨型分量 |
| 11-10 | LoCoMo 不作为主证据 | 候选池小、检索饱和，单题波动大；Reader 收益远大于理解层收益 |
| 11-10 | 转向更长、跨段依赖更强的数据 | QASPER 的段间关系弱，等权 unit 融合出现显著负迁移 |
| 11-12 | DetectiveQA 暂停 R2–R5 和 Question-Time | 关系 precision 没过门槛；当前 split 主要在测 recency/reveal |

## 结论强度

| 证据级别 | 标准 | 当前例子 |
|---|---|---|
| A | 全量或大样本，配对统计稳定，跨 seed / split 复现 | LoCoMo 阶段 A 的“检索提升但 QA 不动” |
| B | 小样本但效应大，有配对检验和 case 支撑 | DetectiveQA recent vs nonrecent |
| C | 单对话、小样本或只做 smoke | LoCoMo conv-30 RSI、DetectiveQA 关系类型细分 |

论文或周报中引用结论时，必须同时写出样本量、指标口径和证据级别。
