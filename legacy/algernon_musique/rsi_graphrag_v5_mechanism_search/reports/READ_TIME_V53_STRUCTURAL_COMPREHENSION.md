# Read-Time V5.3 结构理解

G_raw 没有改。没有重跑 OpenIE，没有 Question-Time，也没有全量阅读。6 个新单元按冻结的 Leiden passage 集合和稳定连通分量选取，没有读题或 gold。诊断查询在生成之前写好。

新增 Luna 调用 12 次，输入 32,409 token，输出 22,991 token，其中推理 10,749 token。上限是 15 次、180,000 输入、40,000 输出。

## 检索基线：PPR-only 和 HippoRAG2

官方 `retrieve()` 的顺序是：查询到事实的向量、Recognition Memory 重排事实、稠密原文向量、把事实里的实体和原文向量一起作为 PPR 种子、再按原文节点排序。没有事实留下时，退回稠密原文检索。

| 组件 | 官方 HippoRAG2 | 当前 PPR-only |
| --- | --- | --- |
| Query-to-Fact | 查询向量点乘事实向量 | 没有。种子是题面里的多词实体字符串 |
| Fact Embedding | 需要与当前三元组对齐的向量 | Luna 图没有。`case_study_musique` 里有 128,834 条事实向量，内容是 gpt-4o-mini 三元组，不能套到 Luna 的三元组 id 上 |
| Recognition Memory | `rerank_facts` 再调用 `rerank_filter` | 没有跑 |
| Passage Embedding | 查询向量点乘原文向量，并作为 PPR 的原文种子 | 向量文件能复用，查询向量没有算 |
| Dense Passage Seeds | 有，权重是 `passage_node_weight` | 没有。链不上实体的题没有种子 |
| Synonymy | `add_synonymy_edges` | Luna 图没有 |
| PPR 与排序 | 实体种子加原文种子，再读原文节点 | 只有实体字符串种子，再读原文节点 |
| 链不上实体 | 退回稠密原文检索 | 标成不可做图检索。100 道开发题里有 15 道 |

Passage 向量可以复用。11,656 条缓存里包含全部 11,622 个 Luna passage id，多出的 34 条不在 Luna OpenIE 里。抽查 `chunk-387019fe59c39942feead0deff701010`，缓存正文和 Luna 原文一致。要真正用上它，还得给查询算 NV-Embed。四台 GPU 空闲显存都低于 4 GB，这次没有加载模型，也没有占用那些进程。

所以完整 HippoRAG2 现在跑不起来的部分是：Luna 事实向量、Luna 实体向量、synonymy、查询向量和 Recognition Memory。能跑的是字符串种子上的 PPR。

## 结构引导有没有更容易找到跨事实关系？

没有。对照是同一批单元上的短 schema 证据链。结构引导先用程序找“两实体没有直接边、但有共同邻居”的缺口，再只把缺口、路径和相关原文交给模型，并允许 NO_OP。

| | 证据链 | 结构引导 |
| --- | --- | --- |
| 提出的 ADD | 16 | 4 |
| 旧验证器 ACCEPT | 11 | 3 |
| 蕴含检查后保留 | 10 | 3 |
| 其中跨 Passage | 0 | 0 |
| 补漏 / 语义抽象 | 5 / 5 | 0 / 3 |

结构引导在 6 个单元里有 4 个主要返回 NO_OP。它明确拒绝了原文不支持的桥，例如 USB 的速率和供电、Holman 与 Halfway Tree 只是同一天、Mount Ginini 和各个公园之间没有新的直接关系。证据链则在单篇里继续补句子，例如 Siim Kallas 先前不从政、Alex Vreeke 被困 20 年。两边都没有留下跨 Passage 的新边。

## 这些关系是否被原文支持？

留下的 3 条结构引导结果都只引用一篇原文，而且是语义抽象，不是新的跨文档边。

- 支持，但不是缺口所要的跨篇关系。2011 年拉脱维亚选举原文写着不要把候选人 Andris Bērziņš 和曾经属于 Latvian Way 的前总理弄混。模型写成 “a candidate distinct from the former prime minister associated with Latvian Way”。这句被原文支持。它没有使用爱沙尼亚改革党那一篇，所以没有建立两篇之间的新路径。
- 支持，单篇。Jumanji 原文写 Nick Jonas 演 Jefferson McDonough，而这个角色是 Alex Vreeke 的 avatar，Colin Hanks 演成年 Alex。两条都在 `chunk-9635e26db187b5fc88ce4cb87d79a8cd`。
- 正确的拒绝。证据链曾把 Halfway Tree 的发行日和 Holman Bible Outreach 的成立日收成同一天的跨篇关系。验证器因引用不完整拒绝。结构引导对这个缺口直接 NO_OP，理由是两篇没有跨文档关系。这是结构引导比证据链更稳的地方，但它拒绝的是假关系，没有换成一条真关系。

## 可达性和 Recall

诊断集 10 条，生成前冻结。1 条 USB 问句链不上实体。其余 9 条上，R0、证据链图、结构引导图、以及结构引导边的随机改接，Recall@1/3/5/20 完全一样：Recall@1 为 0.556，Recall@5 和 Recall@20 都是 0.889。

已经排在前面的题没有被新边改动。例如拉脱维亚选举日期、Holman 的成立日、Jumanji 的 avatar、ACT 移交年份和 gang-gang cockatoo，支持原文在原图上就是第 1 或第 2。

失败的是改写问句 “Which peak inside the Australian Capital Territory ranks third in height”。Mount Ginini 那篇在原图排第 113，结构引导和图边都没有把它提前。程序列出的 Ginini 缺口被模型判为原文没有新关系。

开发集不能按 gold 选簇。85 道可链接题的 Recall@1/5/20 在四组里都是 0.192 / 0.472 / 0.569。100 道里另外 15 道没有图种子，没有从主表删掉。结构引导源文和这 100 道题的 gold 交集是 0。R0 和结构引导图没有逐题 Recall 变化。随机改接也一样。

词面检索这次没有单独加分。诊断集和开发集上的数字变化都是 0，所以没有“词面造成的提升”可以报告。上一轮里，词面加分曾经把 Lill-Babs 的一篇源文从约第 11000 名拉到第 1，同时拉低了开发题的 Recall@1。那是文本命中，不是新图关系。本轮的图关系没有复现那种拉动。

## 结论

1. 结构引导没有比证据链发现更多真正的跨事实关系。它更常正确地说没有新关系。
2. 留下的 3 条受单篇原文支持，但都不是跨 Passage 新边。
3. 它们没有改善证据可达性。需要两篇的诊断题，原图已经能排到前 2 的仍然前 2；原图排在 113 的仍然是 113。
4. 因此没有 Recall 增益。
5. 本轮没有来自新图关系的提升，也没有来自词面检索的提升。
6. 最值得继续看的不是再写一种理解提示。程序找到的“缺边”大多不是原文里的缺事实。检索上，字符串 PPR 也还缺少事实向量、原文种子和 synonymy；在那些组件补上之前，无法判断完整 HippoRAG2 会不会给同一条边更高的权重。

这是负结果。没有加调用，也没有再换提示。配置和 trace 在 `artifacts/v53/`。
