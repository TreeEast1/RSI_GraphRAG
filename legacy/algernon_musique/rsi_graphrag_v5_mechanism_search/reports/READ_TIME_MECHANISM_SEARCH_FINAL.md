# Read-Time 机制搜索

## 1. 总结

目前最值得留下的机制是证据链生成，加上写库前的分层程序验证。它相对保守生成和推理生成的改进，是把“原句被 OpenIE 漏掉”和“两篇里的日期可以比较”分开，并且能挡住把 “and” 说成 “before”、把已有三元组再收一遍、以及引用了不存在的 passage id。它没有证明模型会稳定产生多跳新语义。把跨事实推理硬排到显式补漏前面之后，有用的原句补漏消失了，换来的大多是同一句话里两个日期谁更早。

## 2. 方法比较

所有方法读同一份冻住的单元，温度沿用部署默认值，推理强度 medium，输出上限 3,500 token。判定用同一套分层验证器。G_raw 的 OpenIE 文件在实验开始后没有被改写。

| 方法 | 做了什么 | 看到的结果 | 代价和限制 |
| --- | --- | --- | --- |
| A 保守 | 只收原文直接支持、三元组没写的句子 | 条数少。小簇历史结果是 NO_OP。新的大切片上它收下了 Kremer 的全女性解说组，也收下了 Morley 的十周休假。新单元上仍是补漏。带 overlap 的单元能在 token 上限内写完。 | 几乎不产生跨段落关系。 |
| B 推理 | 允许把多条证据合成一条 | 条数最多，里面既有补漏，也有把第二季续订和首播日期重新说一遍的重复，还有一条引文对不上原文。 | 同样的输出上限下，推理 token 更多。不比 C 更可靠。 |
| C 证据链 | 每条候选要写算子、引文和 passage id | 大、中、小三个对照单元上 8 条接受，其中 7 条是补漏，1 条是 Tristan 早于 Ludwig。新单元上接受的 9 条全是补漏。引文和 id 比 D 准。 | 在 49 个核心实体加 19 个 overlap 的单元上，3,500 个输出 token 全部耗在推理上，正文为空。 |
| D 图缺口 | 先列缺口，再只填缺口 | 小簇直接 NO_OP，漏掉了唯一一条跨段落比较。大簇把 Kremer 的 id 写错，另外两条填的是已有三元组。中簇的三条补漏是对的。 | 多出来的结构没有减少幻觉 id，也没有增加组合关系。 |
| C2 先跨事实 | 只改 C 的顺序：先日期和接任，至多一条补漏 | 保留了 Tristan。丢掉了 Kremer 和十周休假。新增的接受是一部剧的开播日早于停播日，以及选角公布早于首播。 | 这是负结果。不要用这个顺序替换 C。 |

没有跑可选的检索导向方法 E。A–D 的结果表明，模型在这些单元上缺的不是“再想一个方便检索的捷径”，而是把原句补全和把两个日期比较这两件事分开。E 会再引入一种目标，和已经看到的失败模式叠在一起。

## 3. 验证器

旧的 Luna 验证器把单元截到 12,000 个字符。大单元大约 8,000 多 token，原文在实体和三元组之后，所以 Kremer、Sajak、Lipshitz 的句子在生成器里看得到，在验证器里看不到。它因此拒绝了原文里明明有的句子，又把 KPBD 开播、Shades of Blue 第三季、Morley 和 Chris 的两跳口头化接受成新事实。Azawad 那条把 “took control of by April and declared” 接受成 “before”。

分层验证器分五步，不看生成器的自我分类。

- V1：引文必须是指定原文里的连续片段。对不上就标 `EVIDENCE_MISMATCH`，不把事实判死。D 写错的 Kremer id 停在这里。
- V2：候选里的专名必须出现在引文所在句，或由篇名和句首代词接上。Gemma 那句只有 “her aunt”，没有 Madge Bishop，所以拒绝。Wang Shen'ai 同样。
- V3：两个完整日期的先后用代码比较。“before” 不能从 “and” 推出来。接任必须在引文里有 succeeded、replaced、replacing、appointed the new、debuted as host 这类说法。
- V4：用该原文的全部 OpenIE 三元组做新颖性，不是只用单元切片里的三元组。KPBD、第三季、修会驻地、Lipshitz 未播出、Aidan 与 Chris 的恋爱关系因此被标成重复。
- V5：只有句子支持并且还有三元组没覆盖的内容词，才 ACCEPT。重复是 REJECT，不是另一种 unsupported。

14 条历史候选里，这套规则和人工“该不该入库”对齐 13 条。三条人造对照也按预期拒绝。词表最初漏了 replacing。补上之后，全部已保存候选里只有 C2 的 Cristine Phoebe 那条改变了失败环节：句子 “the new trainer, Cristine Phoebe for replacing Juliana Mikael” 确实在说替换，但这条已经被 OpenIE 覆盖，所以仍然不入库。

还没挡住的误接受是改写。Lipshitz 的 “broadcast” 对 “air”，Gemma 的 “first appearance ... broadcast on” 对 “first appeared on”，验证器会把多出来的词当成新信息。接受率不能当成事实正确率。

## 4. 典型案例

### 值得留下的理解

1. 跨段落，小簇 C 和 C2 都得到。Tristan 的原文是 “He first appeared on 23 October 2009.” Ludwig 的原文是 “He appeared from 22 December 2009 to the character's death on 23 April 2013.” 两边的日期三元组都在，先后关系不在。这是这一轮唯一干净的跨段落新关系。历史协议 B 也得到过它，协议 A 在同一单元上是 NO_OP。

2. 显式补漏，大簇 A 和 C。Andrea Kremer 的原文把 Thursday Night Football、Amazon Prime Video 和 “the first all-women booth to call any major men's team sport” 写在同一句。OpenIE 有共同解说和平台，没有 “first all-women booth”。旧验证器因为截断说没有原文。

3. 显式补漏，大簇 C。“Next Caller was an unaired television comedy series.” 缺口列表里的原句是 “Next Caller is an unaired American television comedy series”。这是三元组没写的地位，不是推断。

4. 显式补漏，中簇 A、B、C、D 都独立写出。Aidan Foster 那篇写着 “Morley took a ten-week break from Neighbours in April 2012 to appear in the feature film, Blinder.” 已有三元组写了休假和月份，没有 “ten-week” 和 “to appear in”。

5. 显式补漏，新单元 cc-00172，A 和 C。原文写乳牙在产前第六到第八周开始形成，恒牙在第二十周开始形成。这是冻结单元之外的复现，说明补漏不依赖冒烟时看过的那几篇。

Golovko 仍算一条被原文支持的接任，但那是历史协议 B 的结果，这一轮的新单元没有再产生同类句子。原文是 “The ASDFB was first commanded by ... Oleg Ostapenko, who was promoted to Deputy Minister of Defence in November 2012. On 24 December 2012, Aleksandr Golovko was appointed the new commander.” “the new commander” 加上 “first commanded by” 比两个日期强。

### 失败、拒绝和重复

1. Azawad。原句是 “northern Mali, which Tuareg rebels took control of by April and declared the secession of a new state, Azawad.” 历史候选的引文删掉了 “northern Mali, which”，又把 “and” 说成 “before”。占领和独立已经各有一条三元组。旧验证器接受。分层验证器拒绝。

2. D 的 Kremer。缺口内容是对的，passage id 写成了不存在的 `chunk-c30c6747497487f36f06ffa6bed8a8`。这是证据错位，不是这条地位判断为假。C 对同一句给了正确 id。

3. C2 的日期比较。Law of the Plainsman 的原句已经是 “aired ... from October 1, 1959, until May 5, 1960.” 候选只说 1959-10-01 早于 1960-05-05。验证器因为 “earlier” 不在三元组里接受了。这是低价值的区间端点比较。

4. Gemma 的 Madge。引文句是 “She moves to Erinsborough to live with her aunt, following her mother's death.” “following” 支持 “after”，不支持关系名里的 motivated。Madge Bishop 在上一句。旧验证器接受了完整的动机句。分层验证器拒绝。

5. 重复三元组。KPBD 的 OpenIE 已有 “began broadcasting in June 2005”。Shades of Blue 已有第三季和 2017-03-17。Sisters of St Joseph of Nazareth 已有 “has main centre in Whanganui”。Lipshitz 已有 “was not picked up for air”。旧验证器把前两句当新事实，又说 Lipshitz 的原句不在文中。

Sajak 是旧验证器的误拒绝。原文写 Chuck Woolery 主持到 1981-12-25 且合同未续，Pat Sajak 于 1981-12-28 “made his debut as the host”。这是同一职位的下一任。分层验证器接受。它不是跨文档推断。

## 5. 检索

这是 93 篇试点原文上的结构探针，不是全库 HippoRAG2，也没有加载 NV-Embed-v2。四台 GPU 的空闲显存都低于 4 GB，现有加载代码要求 12 GB 以上才用 GPU。

R0 是原三元组和原文。R1 只加补漏节点。R2 把理解文本当额外文档，再把分数送回源原文。R3 把理解节点连到实体和源原文。

诊断问句是人工写的，单独标记，不能当作 MuSiQue 成绩。

- “first all-women booth ... major men's team sport”：Kremer 原文在 R0 就是第 1。R1–R3 不变。
- “Which character appeared earlier, Tristan von Lahnstein or Ludwig von Lahnstein?”：两篇原文在 R0 是第 1 和第 2。理解节点没有改变名次。两个人名已经能把两篇原文召回来。
- “When do permanent teeth begin to form before birth?”：目标原文在 R0 就是第 1。

开发集按题面是否包含试点里的实体来选，没有看 gold 文档。两道题只匹配到泛指短语 “body of water”，它们的 gold 原文都不在这 93 篇里。没有可以计算的试点 Recall。

因此这一轮不能说理解节点改善了原文召回。能说的是：这些节点的文字和源句很接近时，词面检索已经能命中源句，多出来的图边没有新的召回。

## 6. 成本

没有读到这个部署的单价，下面只有 token，没有美元。

| 方法 | 新调用 | 本轮缓存 | 输入 token | 输出 token |
| --- | --- | --- | --- | --- |
| A | 7 | 1 | 33,945 | 7,345 |
| B | 2 | 1 | 12,598 | 4,525 |
| C | 8 | 1 | 37,106 | 17,087 |
| C2 | 3 | 0 | 14,407 | 3,384 |
| D | 3 | 0 | 14,173 | 6,024 |
| 合计 | 23 | 3 | 112,229 | 38,365 |

输出 token 里有 18,296 是推理 token。小簇历史 A/B 没有调用。23 次都返回了 `gpt-5.6-luna-2026-07-09`。

C 的接受大多是补漏。按“接受条数 / 调用”比较会把补漏算成收益，这个数不能当单位成本。更有用的观察是：C 在一个 overlap 单元上花了 10,182 输入 token 和 3,500 输出 token，正文为空。A 在同一单元、同一上限下写完了，2 条接受、3 条被正确判成重复。

全量阅读没有启动。量级上，这次用 23 次调用覆盖了 8 个单元和五种提示。若以后每个阅读单元只跑一次 C，调用数会比 23 大一个数量级以上；切分后的单元总数这次没有逐个点清，所以不写精确的全库调用数。

## 7. 下一版建议

已经有实验支持的做法：

- 阅读单元继续用核心节点加 1-hop overlap，并且把单元文本冻住。Leiden 不设种子时社区会变，这一轮两次运行分别得到 106 和 115 个社区。小簇 `cc-00029` 可以按连通分量复现，中簇和大簇不行。
- 生成用证据链。每条候选带算子、逐字引文和 passage id。算子里保留 `EXPLICIT_GAP` 和 `TEMPORAL_COMPARE`。显式补漏单独入库，不要和跨段落比较混成一种“理解节点”。
- 写进 G_comp 之前用分层验证器。日期比较用代码。新颖性对照该原文的全部 OpenIE，而不是单元切片。验证器不要再截断到 12,000 字符。
- 丢掉 D 这种先写缺口再填的单次格式，至少在它会改写 passage id 的时候丢掉。丢掉 C2 那种“先跨事实、少补漏”的顺序。
- G_raw 保持不可写。每种提示从空的 G_comp 开始，不要让后一种提示读到前一种提示新写的节点。

还需要验证的假设：

- 补漏节点只在问句用词和源句不同时才会改变召回。这次的诊断问句用词太接近源句，看不出差别。
- 把 C 的输出上限提高到足以在 overlap 单元上写完 JSON，会不会只是多收补漏，而不是多收跨段落关系。这次为了和 A 公平，没有单独加长 C 的上限。
- replacing 这类词形已经补进接任词表，但是改写残留造成的误接受还没有修。

工程上合理、但这次没有效果证据的设计：

- 把理解节点放进 Personalized PageRank。探针里 R3 没有改变目标原文的名次。
- 用 NV-Embed-v2 做全库向量。权重在本地，这次没有加载。
- 检索导向的方法 E。没有跑。

不要做的事：

- 不要把 ACCEPT 条数当成知识质量。
- 不要用测试题或 gold 文档挑选阅读簇，也不要按 gold 答案补图。
- 不要在验证器、提示、单元和评价指标同时改动之后宣布某一种提示胜出。
- 不要启动全量 Read-Time 或 Question-Time。下一步应该先修改写残留，并在问句用词不同于源句的诊断集上重测召回。
