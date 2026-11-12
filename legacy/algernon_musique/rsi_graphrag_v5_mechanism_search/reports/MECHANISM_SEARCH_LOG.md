# Read-Time Mechanism Search Log

模型是接口返回的 `gpt-5.6-luna-2026-07-09`。G_raw 只读。没有全量阅读，没有 Question-Time。理解只写在本目录的运行结果里。

预算：新增调用 23 / 60，输入 112,229 / 750,000 token，输出 38,365 / 150,000 token。另外有 3 次命中本轮自己的缓存，不计入新增调用。小簇的历史 A/B 没有重调，因为格式化后的阅读单元和 V4 缓存键一致。

## Round 0 — 历史 14 条

假设：旧验证器的误判主要来自两件事。一是阅读单元被截到 12,000 个字符，原文排在实体和三元组后面。二是新颖性只对照单元里看得见的三元组，不对照该原文在 G_raw 里的全部 OpenIE。

没有新的 Luna 调用。人工标签在 `scripts/round0_audit.py`。

| ID | 候选 | 人工类别 | 旧验证器 | 分层验证器 |
| --- | --- | --- | --- | --- |
| small B | Tristan 2009-10-23 早于 Ludwig 2009-12-22 | Correct & Novel，跨段落 | ACCEPT，对 | ACCEPT |
| med A | Bobby 签六个月演 Aidan | 时长是补漏，角色不在引文句里 | ACCEPT | REJECT，引用不完整 |
| med A | Glockner 两年合同，并拍 Mason | Extraction Repair | REJECT，引文太窄 | ACCEPT |
| med A | Sisters 的中心在 Whanganui | 正确但重复 | REJECT，主语认错 | REJECT，重复 |
| med A | KPBD 2005 年 6 月开播 | 正确但重复 | ACCEPT，误当新事实 | REJECT，重复 |
| med B | Gemma 母亲去世后搬家，和 Madge 同住 | 句子成立，Madge 不在引文句 | ACCEPT | REJECT，引用不完整 |
| med B | Morley 扮演的 Aidan 是 Chris 的恋爱对象 | 两条三元组的口头化 | ACCEPT | REJECT，重复 |
| med B | Golovko 任新司令 | Correct & Novel | ACCEPT | ACCEPT |
| med B | 占领之后宣布 Azawad 独立 | “and” 推不出 “before” | ACCEPT，误接受 | REJECT |
| large A | Shades of Blue 第三季 | 正确但重复 | ACCEPT，误当新事实 | REJECT，重复 |
| large A | Lipshitz 未播出 | 正确但重复，原文里有这句 | REJECT，说原文没有 | REJECT，重复 |
| large B | Kremer 与 Storm 的全女性解说组 | Correct & Novel | REJECT，说没有原文 | ACCEPT |
| large B | Sajak 在 Woolery 之后主持 | Correct & Novel | REJECT，说没有引文 | ACCEPT |
| large B | Lipshitz 获准试播但未播出 | 两句都已是三元组 | REJECT，说没有原文 | ACCEPT，改写残留 |

分层验证器和“该不该入库”的人工标签对齐 13/14。剩下的一条是 Lipshitz：候选把 “air” 写成 “broadcast”，验证器把这个改写当成了新词。三条合成对照按预期处理：重复日期拒绝，从日期推出死因拒绝，不存在的引文标成 `EVIDENCE_MISMATCH`。

## Round 1 — 同一批冻结单元上的 A/B/C/D

假设：要求先写证据链（C），或先列图缺口再填（D），会比直接生成更少无依据推断，并留下跨段落比较。

小簇 `cc-00029` 与历史缓存键完全一致。Leiden 没有固定种子，中簇和大簇重跑后社区变了，缓存键对不上。对照用的是同一次 Leiden 抽签冻住的单元：中簇切片含 Gemma、Glockner、Morley、Sisters 四篇锚点原文；大簇切片含 Kremer、Lipshitz、Shades of Blue、Wheel of Fortune。A/B 在这两个新单元上重跑，保证和 C/D 读到同一段输入。

| 单元 | A 提出/接受 | B | C | D |
| --- | --- | --- | --- | --- |
| 小 | 历史 NO_OP，未重调 | 历史 1/1 | 1/1 | 0/0 |
| 中 | 1/1 | 3/2 | 3/3 | 3/3 |
| 大 | 1/1 | 5/3 | 4/4 | 3/0 |

接受下来的几乎都是 `EXTRACTION_REPAIR`。唯一的跨段落接受是小簇 C：Tristan 早于 Ludwig。D 在小簇返回 NO_OP，因为它只找“某一句里有、三元组里没有”的命题，不会比较两篇里各自已经抽出的日期。D 在大簇把 Kremer 的 passage id 写成了不存在的 `chunk-c30c6747497487f36f06ffa6bed8a8`，验证器标 `EVIDENCE_MISMATCH`。同一缺口在 C 里 id 和引文都是对的。

## Round 2 — 验证器

验证器在生成之前就固定，后面没有为了抬高接受率再改规则。唯一一次事后修补是词形：C2 引用的句子写了 “replacing”，词表里只有 “replaced”，先被判成无证据。补上 `replac(e|ed|es|ing)` 之后，对全部已保存候选重算，只有这一条的失败环节变了。它变成“句子支持替换，但 OpenIE 已经有这条”，决策仍是不入库。没有新的 Luna 调用。

仍会漏的一类是改写残留：Gemma 的 “first appearance ... broadcast on 20 June 1990” 对应已有三元组 `first appeared on 20 June 1990`，验证器因为多出来的 appearance、episode、broadcast 接受了它。

## Round 3 — 只改生成顺序，然后换单元

假设：如果要求先做日期比较和显式接任，再允许至多一条显式补漏，跨事实关系会增加，重复补漏会下降。

这是 C2。单元、验证器、token 上限都没变。三个单元共 3 次新调用。

- 小簇仍得到 23 October 2009 早于 22 December 2009。
- 中簇只留下 19 June 2011 早于 19 August 2011，也就是 Aidan 的选角公布早于首播。十周休假那条显式补漏消失了。
- 大簇留下 “1959-10-01 早于 1960-05-05”，这是 Law of the Plainsman 同一句里的开播日和停播日。Kremer 的全女性解说组没有了。Cristine Phoebe “replacing” Juliana Mikael 被验证器认出后，判定为已有三元组。

C2 确实压掉了补漏，但补掉的包括有信息量的原句，换来的多是同一区间两端的日期比较。因此不采用 C2。

然后只换阅读单元，在 5 个未参与历史冒烟的单元上复跑 A 和 C。种子 20261030，按结构抽样，没有读题。

| 新单元 | A 提出/接受 | C 提出/接受 | 接受类型 |
| --- | --- | --- | --- |
| cc-00172，牙齿，8 实体 | 3/3 | 4/4 | 全是补漏 |
| cc-00083，9 实体 | 0/0 | 0/0 | 两边都认为没有缺口 |
| cc-00646，三条河，15 实体 | 3/3 | 4/4 | 全是补漏 |
| stable-0026，74 实体 | 1/0 | 2/1 | C 接受 “Gao Wei became emperor at age eight”；Wang Shen'ai 因引文句没有人名被拒 |
| overlap u003，49 核心 + 19 overlap | 5/2 | 空输出 | C 的 3,500 输出 token 全部用在推理上，`finish_reason=length`，正文为空 |

新单元上的接受结果仍然是单篇补漏，没有新的跨段落关系。

## Round 4 — 局部检索

93 篇试点原文，17 个方法 C 接受的节点，其中 16 个是补漏，1 个是 Tristan 的跨段落比较。GPU 空闲内存都低于 4 GB，没有加载 NV-Embed-v2。排序是词面重叠加实体命中，不是 HippoRAG2。

诊断问句里，Kremer、Tristan、恒牙形成这三问的目标原文在原始图上已经是前两名。加上理解节点后名次不变。开发集里只有两道题碰到试点实体，匹配到的是泛指的 “body of water”，它们的 gold 原文都不在这 93 篇里。不能把这个结果说成 MuSiQue 检索提升。

## 停止

C2 是一次明确的负结果，新单元复现了“主要产物是补漏”。再改 prompt 抬接受率没有新的机制信息。按预算和停止条件结束。
