# Read-Time 冒烟 Case

这是第二次冒烟的完整记录，对应 `artifacts/smoke/smoke_summary.json`。模型是 `gpt-5.6-luna-2026-07-09`。三个簇按图结构抽取，没有读题目。每个簇只把一个阅读单元送给模型。

协议 A 偏保守。协议 B 允许把多条证据合成一条关系。两边用同一个独立验证器。只有验证器给出 ACCEPT，并且证据规则通过，才写入隔离的理解层。`uncertain_hypothesis` 和验证器的 REJECT 都不入库。

这一轮没有跑检索，也没有跑 Question-Time。三个簇的原始 OpenIE 文件和簇上的 G_raw 哈希都没有变。

中簇协议 B 接受了 4 条。摘要文件当时只留了前 2 条的正文。后 2 条和验证理由从同一次调用缓存里补回，和摘要里的接受计数一致。

## 总表

| 簇 | 单元 | 协议 | 模型输出 | 接受 | 不接受 |
| --- | --- | --- | --- | --- | --- |
| 小，cc-00029，10 实体，2 篇 | 整簇，1,165 token | A | 1 条 NO_OP | 0 | 0 |
| 同上 | 同上 | B | 1 条 ADD | 1 | 0 |
| 中，leiden-0099，99 实体，24 篇 | 整簇，9,584 token | A | 4 条 ADD | 2 | 2 |
| 同上 | 同上 | B | 4 条 ADD | 4 | 0 |
| 大，leiden-0088，670 实体，196 篇，切成 26 个单元 | u010，50 个核心实体加 13 个 overlap，8,460 token | A | 2 条 ADD | 1 | 1 |
| 同上 | 同上 | B | 3 条 ADD | 0 | 3 |

大簇其余 25 个单元只做了切分检查，没有调用模型。1-hop overlap 保住了 43 条跨单元边。

## 小簇 cc-00029

两篇原文，10 个实体，10 条三元组，不用切分。

### 协议 A：不新增

模型返回 `NO_OP`，没有提出关系。

理由：原文只是在重复图里已经有的事实，包括人物是虚构的、属于哪部肥皂剧、由谁出演、以及出场日期。相对冻结图谱，没有直接支持的新事实。

### 协议 B：接受 1 条

**Tristan von Lahnstein 的首次出场早于 Ludwig von Lahnstein。** 日期分别是 2009-10-23 和 2009-12-22。

- 关系：`earlier first appearance than`
- 类型：derived
- 证据一：`chunk-5531a128df4a1d94732e682316b73430`，`He first appeared on 23 October 2009.`
- 证据二：`chunk-429ba54878382d30bf5cb2e822a01def`，`He appeared from 22 December 2009 to the character's death`
- 模型说明：两篇各给一个日期，比较之后得到先后。原图没有这条先后关系。
- 验证器：ACCEPT。faithfulness = supported，novelty = new，consistency = consistent。原文分别给出两个日期，没有任何一条三元组陈述这个顺序。

这是这一轮最干净的跨段落推导。

## 中簇 cc-00000-leiden-0099

99 个实体，24 篇，110 条三元组。整簇 9,584 token，仍放得进一个单元，没有 overlap。

### 协议 A：接受 2 条，拒绝 2 条

两条接受都是单句补漏，不是多跳。

**接受。Bobby Morley 起初签了六个月来演 Aidan Foster。**

- 关系：合同时长，类型 explicit_restatement
- 证据：`chunk-56664711c1c4de235b820a051107fb2f`，`Morley was initially contracted for six months`
- 验证器：ACCEPT。原句直接说了六个月，三元组 `triple-2b8bd66f576fee8bca73822ecf494fbf` 只说明 Morley 演 Aidan Foster，没有合同时长。

**接受。KPBD 于 2005 年 6 月开始播出。**

- 关系：开播时间，类型 explicit_restatement
- 证据：`chunk-ed22e931f32904f9af5775fa847dedd6`，`The station began broadcasting in June 2005`
- 验证器：ACCEPT。原句写了开播月份。三元组 `triple-7006d2ea2c59b028d7c6cb96b97fa51a` 把 KPBD 认成这家 station。已有三元组写了频率、执照和后来停播，没有写开播时间。

**拒绝。Taylor Glockner 签了两年合同来演 Mason Turner。**

- 类型：explicit_restatement
- 模型引用的证据只说 Glockner 和 Neighbours 签了两年，没有说他演 Mason Turner。
- 验证器：REJECT。faithfulness = unsupported。合同对象和角色被模型接在了一起，原文没有这个连接。

**拒绝。The Sisters of St Joseph of Nazareth 的中心在新西兰 Whanganui。**

- 类型：explicit_restatement
- 模型引用的原文是 `have their main centre in Whanganui, New Zealand`。
- 验证器：REJECT。这句话的主语没有被证明是 Sisters of St Joseph of Nazareth。图里对应的实体是 Sisters of St Joseph Whanganui，名字对不上。

### 协议 B：接受 4 条

**接受。Gemma Ramsay 在母亲去世后搬到 Erinsborough，和姑妈 Madge Bishop 同住。**

- 关系：`move motivated by family circumstances`
- 类型：derived
- 证据：`chunk-1a2f9e9e4d0a9f1c7cdb872540133c08`，`She moves to Erinsborough to live with her aunt, following her mother's death.`
- 验证器：ACCEPT。原句把搬家、原因和姑妈放在一起。已有三元组只是分开写了搬家和 Madge 的选角。

**接受。Bobby Morley 扮演的 Aidan Foster，是作为 Chris Pappas 的恋爱对象被引进 Neighbours 的。**

- 关系：`actor-character plot association`
- 类型：derived
- 证据：`chunk-56664711c1c4de235b820a051107fb2f`，`Aidan was introduced to "Neighbours" as a love interest for established character Chris Pappas (James Mason).`
- 来源三元组包括 `triple-2b8bd66f576fee8bca73822ecf494fbf`
- 验证器：ACCEPT。原句写了 Aidan 和 Chris 的关系，三元组写了 Morley 演 Aidan。没有任何一条三元组同时包含演员和这条剧情关系。

**接受。Oleg Ostapenko 升任国防部副部长之后，Aleksandr Golovko 成为空天防御部队新司令。**

- 关系：`commander succession`
- 类型：derived
- 证据：`chunk-bdff4e9a69ee9bf8c09cf8574391ba71`，`Oleg Ostapenko, who was promoted to Deputy Minister of Defence in November 2012. On 24 December 2012, Aleksandr Golovko was appointed the new commander.`
- 验证器：ACCEPT。原文把晋升写在任命之前。分开的晋升三元组和任命三元组本身没有写出 Golovko 接替 Ostapenko。

**接受。图阿雷格叛军控制马里北部之后，于 2012 年 4 月宣布 Azawad 独立。**

- 关系：`territorial takeover preceding secession`
- 类型：derived
- 证据：`chunk-9498c40a3a45abafaadf924367b5e688`，`Tuareg rebels took control of by April and declared the secession of a new state, Azawad.`
- 验证器：ACCEPT。原句把占领和随后宣布独立连在一起。已有三元组分开写了占领和独立，没有写先后。

注意第三条和第四条的证据句子本身不完整：Azawad 那句是 `took control of by April`，中间缺了宾语。验证器仍然接受了。这是验证偏松的一个例子。

## 大簇 cc-00000-leiden-0088

670 个实体，196 篇，825 条三元组，估计 26,899 token。切成 26 个阅读单元。送进模型的是 `cc-00000-leiden-0088-u010`：50 个核心实体，13 个 overlap 实体，输入 8,460 token。

### 协议 A：接受 1 条，拒绝 1 条

**接受。Shades of Blue 在 2017-03-17 被续订第三季。**

- 关系：`renewed for third season`
- 类型：explicit_restatement
- 证据：`chunk-11b375ea69e06d1107ef5ec409d4cb32`，`On March 17, 2017, the series was renewed for a third season.`
- 验证器：ACCEPT。原句写了第三季和日期。已有三元组只写了续订 13 集的第二季。

**拒绝。Lipshitz Saves the World 没有被播出。**

- 关系：`not picked up for air`
- 类型：explicit_restatement
- 模型给的证据是 `The show was not picked up for air.`，passage 是 `chunk-c48d38af96b2654659443d02f43dc84f`
- 验证器：REJECT。faithfulness = unsupported。它认为这句不在提供的原文里。已有三元组只写了拍摄和批准，没有写未播出。验证器同时说这条不和已有三元组冲突。

### 协议 B：3 条全部拒绝

**拒绝。Andrea Kremer 和 Hannah Storm 在为 Amazon Prime Video 解说 Thursday Night Football 时，成为第一组解说主要男子团队运动的全女性解说组合。**

- 关系：`part of first all-women major-sport broadcast booth`
- 类型：derived
- 验证器：REJECT。faithfulness = partial。三元组支持她们解说 Thursday Night Football，也支持 Kremer 为 Amazon Prime Video 工作。提供的原文没有支持「第一组全女性解说」这个说法。
- 生成器其实引用了 `making sports history, along with Hannah Storm, by becoming the first all-women booth...`。验证器没有把这句当成足够的原文支持。这是生成器和验证器对同一段证据判断不一致的例子。

**拒绝。Chuck Woolery 在 1981-12-25 离开后，Pat Sajak 接任 Wheel of Fortune 主持人。**

- 类型：derived
- 验证器：REJECT。faithfulness = partial。三元组有 Woolery 在 1981-12-25 结束主持，以及 Sajak 在 1981-12-28 首次主持。没有一句被引用的原文把「接任」说完整。

**拒绝。NBC 在 2006 年 7 月批准了 Lipshitz Saves the World 的试播，但这档节目没有被播出。**

- 类型：derived
- 验证器：REJECT。faithfulness = partial。批准试播有三元组支持。没有原文支持「没有被播出」这半句。这和协议 A 被拒绝的那条是同一件事，只是 B 又把批准日期拼了进去。

## 从这些 case 能看到的问题

保守协议会收下 OpenIE 漏掉的原句。Bobby Morley 的六个月合同、KPBD 的开播月、Shades of Blue 的第三季续订，都是这种补漏。它们有原文，但不是多跳理解。

组合协议会把分开的三元组接成一条。小簇的出场先后、Golovko 接任、Azawad 独立，是这种。大簇上它也会把证据不够的半句接进去，然后被验证器拒绝。

验证器会抓住两类错误：角色或机构名对不上，以及引用的句子不在原文里。它也会放过不完整的句子，比如 Azawad 那句缺了宾语。

同一事实会在两个协议里重复出现。Lipshitz 未播出在 A 和 B 各被拒绝一次。
