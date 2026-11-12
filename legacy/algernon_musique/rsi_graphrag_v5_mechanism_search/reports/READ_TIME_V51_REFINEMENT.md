# Read-Time V5.1

本轮没有扩大 A/B/C/D，也没有全量阅读或 Question-Time。G_raw 只读。Leiden 沿用上一轮冻住的单元，没有重划。新增 Luna 调用 3 次，输入 23,351 token，输出 5,979 token，其中推理 1,678 token。上限是 20 次、250,000 输入、50,000 输出。

## 1. 语义改写有没有被挡住？

有。开发集上的四类改写，旧的词面残留验证器会当成新知识，蕴含检查会拒收。

开发集是写规则时用的 10 条，包括 broadcast/air、first appearance/first appeared、appointed the new commander、同一句里的 from/until 端点，以及 Tristan 和 Kremer 两个正例。未见集是另外 14 条，规则写完后只评一次，没有为了它们再加同义词。

| 集合 | 条数 | 旧标签正确 | 新标签正确 |
| --- | --- | --- | --- |
| 开发 | 10 | 6 | 10 |
| 未见 | 14 | 12 | 13 |

具体从误接受到正确拒收的开发案例：

- Lipshitz。“not picked up for broadcast” 和三元组 “was not picked up for air” 是同一命题。旧验证器接受。新验证器标 `REDUNDANT`。
- Gemma。“first appearance ... broadcast on 20 June 1990” 被三元组 `first appeared on 20 June 1990` 蕴含。旧验证器当补漏接受。新验证器标 `REDUNDANT`。
- Law of the Plainsman。原句是 “aired ... from October 1, 1959, until May 5, 1960.” 候选只说前一个日期更早。新标签是 `TRIVIAL_DERIVATION`，事实成立，但不入库。
- Golovko。三元组已经写了 Ostapenko 首先指挥、后来升任，以及 Golovko 被任命为司令。候选里的 “new” 和 “after” 没有多出一条三元组之外的命题。新标签是 `REDUNDANT`。

未见集上，Cristine Phoebe 那句原文是 “the new trainer, Cristine Phoebe for replacing Juliana Mikael”。旧验证器说证据里没有接任动词，判成事实不成立。新验证器认为替换成立，但 OpenIE 已经覆盖，标 `REDUNDANT`。这是误拒绝被改成“支持但不新增”。

未见集仍有一条类型不一致，不是事实误判。Glockner 的候选写 “two-year contract to play Mason Turner”。引文句是两年合同，以及 “filming his first scenes as Mason”，没有 “play”。新标签是 `SEMANTIC_ABSTRACTION`，旧标签是 `EXTRACTION_REPAIR`。两年期限仍然会被保留，只是价值类型不同。

事实是否成立和有没有信息增量是分开的。区间端点比较是真的，标签是 `TRIVIAL_DERIVATION`，不是 `UNSUPPORTED`。

## 2. 证据链生成能不能在更低开销下写完？

能。变的是输出结构，不是把输出上限加大。

对照单元是已经冻住的中簇 `hist_medium`，以及上次正文为空的 overlap 单元 `new_overlap`。每次只改一个因素。

| 设置 | 单元 | 推理强度 | 完成 | 提出/旧验证器接受 | 推理 token | 输出 token | 实际留下的内容 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 原版 C | 中簇 | medium | 完成 | 3/3 | 1,536 | 2,966 | 十周休假等补漏。Gemma 的首播日期在新检查下是重复 |
| 原版 C | overlap | medium | 空正文 | 0/0 | 3,500 | 3,500 | `finish_reason=length`，推理占满上限 |
| 只改强度 | 中簇 | low | 完成 | 3/3 | 408 | 2,340 | 三条日期先后，十周休假消失 |
| 只改强度 | overlap | low | 完成 | 3/1 | 246 | 1,408 | 接受的是 “Great Quittacas Pond is a lake, reservoir, or pond.” |
| 只改短 schema | overlap | medium | 完成 | 3/1 | 1,024 | 2,231 | 接受的是 Lee Kreindler “founder of air disaster law” |

低推理强度能写完，但中簇上的质量下降：有用的补漏被换成日期比较。短 schema 在原来的 medium 强度下也能写完，并且留下 Kreindler 这条补漏，另外两条被正确判成重复。它没有恢复 Juventas 那条，所以不是无损压缩，但比低推理强度更接近原版 C 的补漏行为。

overlap 上，短 schema 这一次调用的输入是 9,950 token，输出 2,231。按新检查，有效新增是 1 条补漏。

## 3. 哪些理解真有跨事实增量？

在新检查下会入库的跨事实关系很少。

- Tristan。23 October 2009 与 22 December 2009 来自两篇传记，先后关系不在任何一条三元组里。标签 `COMPOSITIONAL_UNDERSTANDING`。
- Aidan 的选角公布日 19 June 2011 早于首播 19 August 2011。两件事在同一篇，不是一个 from/until 区间的两端。标签也是组合，检索价值低。
- 低推理强度在中簇上写的 “Mason Turner began filming earlier than his first screen appearance” 属于同一类。

Kremer 的 “first all-women booth” 和 Kreindler 的 “founder of air disaster law” 是原句补漏，不是跨文档组合。Golovko 的接任在新检查下不再算新增。

## 4. 理解节点有没有改变原文召回？

没有提高。在 85 道能做实体链接的开发题上，R0、R1、R3 和等规模乱接边的对照组数字相同。

这是全库 Luna 关系图上的 Personalized PageRank，不是上次那 93 篇的局部探针。开发题 100 道，gold 都在 11,622 篇语料里。15 道题链不上长度足够的多词实体，单独标成不可评估，没有算进 Recall。

| 组 | Recall@1 | Recall@3 | Recall@5 | 五篇内召回全部 gold |
| --- | --- | --- | --- | --- |
| R0 原图 | 0.192 | 0.413 | 0.472 | 0.271 |
| R1 只加补漏节点 | 0.192 | 0.413 | 0.472 | 0.271 |
| R2 把理解文本的词面加分送回源文 | 0.153 | 0.395 | 0.472 | 0.271 |
| R3 理解节点和图边 | 0.192 | 0.413 | 0.472 | 0.271 |
| R3 对照：同样数量的边接到随机原文 | 0.192 | 0.413 | 0.472 | 0.271 |

85 道题里，R0 和 R3 的逐题指标没有一道不同。R2 让 7 道题的 Recall@1 变差。10 道题在词面上碰到了理解节点，但那些节点的源文不是该题的 gold。

## 5. 为什么没有提升？

主因是覆盖，不是接口没把节点接进图。

入库的理解有 20 条：14 条补漏、3 条语义抽象、3 条组合。它们的源文和 100 道开发题的 gold 原文交集是 0。R3 的边已经连到这些源文和实体上；因为 gold 不在这些源文里，边没有把别的 gold 原文顶上来。等规模乱接边的结果一样，说明也不是“多了几条边就会涨分”。

R2 的下降说明，把理解文本的词面分数加回源文，会把非 gold 的源文排得更靠前。

索引状态需要分开看。MuSiQue 的 11,656 条 passage NV-Embed 向量在 `case_study_musique/traditional/graph/chunk_embeddings`，维度 4096。同一目录里的实体和事实向量属于 gpt-4o-mini 的 OpenIE，不是 Luna 的 G_raw，所以没有拿来当这张图的检索索引。Luna 图没有 synonymy。四台 GPU 的空闲显存都低于 4 GB，这次没有加载 NV-Embed，也没有结束别人的进程。因此上表不是 HippoRAG2 的向量检索分。就“这些理解节点的源文是否覆盖开发题 gold”这一点，向量检索也不会把 Recall 算高：gold 和源文没有交集，把源文排到前面不会增加 gold 命中。

## 6. 下一步

不要再搜索理解 Prompt。C 的短输出就够用：保留算子、逐字引文和 passage id，去掉长说明。推理强度保持 medium。低推理强度会把补漏换成低价值日期比较。

Read-Time 的当前假设需要改。按图结构抽到的阅读簇，补上的是这些簇自己的原句和很少的日期比较。这些源文不出现在开发题的 gold 里，所以不会提高这套开发题的召回。要看到检索收益，阅读单元必须覆盖问题真正需要的原文；用 gold 去挑选阅读簇会变成 Question-Time，这一轮没有那样做。

在没有这种覆盖之前，继续改增强图的打分方式没有对象。下一阶段如果还做检索，应该先回答阅读单元如何在不看 gold 标签的前提下覆盖会被问到的原文，而不是再加一种理解提示。
