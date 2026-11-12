# V5.3 新增知识有效性审计

只读冻结的 Luna OpenIE 和对应原文。没有重跑 OpenIE，没有读评测题或 Gold，没有改 G_raw，没有新的模型调用。

分类：

- A：原图已有，或只是换了一种说法。
- B：没有这条直接边，但同篇里的现有边已经把同一知识走通；新句子是捷径或压缩。
- C：同篇原文写了，OpenIE 没有等价三元组，也没有能推出它的路径。
- D：必须两篇以上才能得到。
- E：证据不够。

## Jumanji：`(Jefferson McDonough, avatar_of, Alex Vreeke)`

这条不是新知识。

原文 `chunk-9635e26db187b5fc88ce4cb87d79a8cd`：

> Nick Jonas as Jefferson “Seaplane” McDonough, a young aircraft pilot in Jumanji who is Alex Vreeke's avatar.

G_raw 已有直接边：

- `(Jefferson "Seaplane" McDonough, is the avatar of, Alex Vreeke)`

名称差在引号和中间名 Seaplane，关系差在 “is the avatar of” 和 “avatar_of”。这是同一条边。

同篇还有：

- `(Nick Jonas, portrays, Jefferson "Seaplane" McDonough)`
- `(Mason Guccione, portrays, Alex Vreeke)`
- `(Colin Hanks, plays the adult version of, Alex Vreeke)`

另一篇 `chunk-df987a543dd7e90ab1e535a77f8965fb` 没有 avatar 这句，也没有这条三元组。avatar 知识只在第一篇，而且已经被抽出。

因此这条三元组是 A，不是单篇补漏。

## 结构引导保留的 3 条

### 1. 拉脱维亚选举里的另一位 Andris Bērziņš

新增句子：The election included a candidate distinct from the former prime minister associated with Latvian Way.

来源：`chunk-b4d60fe460691ba03ad83c18b248456e`。

原文：not to be confused with Andris Bērziņš, former PM from Latvia's First Party/Latvian Way.

已有三元组：

- `(Andris Bērziņš, was not the same person as the former prime minister from, Latvian Way)`
- `(Andris Bērziņš, was not the same person as the former prime minister from, Latvia's First Party)`

分类：A。候选只是把 “was not the same person” 说成 “distinct”。

### 2. Colin Hanks 演成年 Alex，其 avatar 是 Jefferson McDonough

来源：同上 Jumanji 第一篇。

已有：

- `(Colin Hanks, plays the adult version of, Alex Vreeke)`
- `(Jefferson "Seaplane" McDonough, is the avatar of, Alex Vreeke)`

两跳是 Jefferson → Alex → Colin，每一步原文都在。候选没有增加第三种事实，只是把这两条边写进一句。

分类：B。其中 avatar 那一半本身是 A。

### 3. Jefferson 通过 Nick Jonas 的饰演出现在电影里

来源：同一篇。

已有：

- `(Nick Jonas, portrays, Jefferson "Seaplane" McDonough)`
- `(Nick Jonas, appears in, Jumanji: Welcome to the Jungle)`
- `(Jefferson "Seaplane" McDonough, is a pilot in, Jumanji)`

没有 `(Jefferson, appears in, Jumanji: Welcome to the Jungle)` 这一条直接边。“pilot in Jumanji” 指的是游戏里的身份，不是电影片名。电影出演是由“Nick 饰演 Jefferson”和“Nick 出现在这部电影”推出来的，两步都有原文。

分类：B。

## 证据链保留的 10 条

### 4. 中学之后进入有高中部的学校或初级学院

来源：`chunk-71cd4c22aa403bdbbfe9652bebaa8a32`。

原文：After completing secondary education, students enroll in schools or junior colleges with a higher secondary facility.

这篇 OpenIE 有学校隶属、授课语言、证书和 EAM-CET，没有这句入学安排，也没有能走到它的边。

分类：C。

### 5. 希腊本土姓氏是固定的，中间名仍用动态父名

来源：`chunk-3d55bd5399bacbeebbfb1ec4fbe25c05`。

已有：

- `(Greek surnames in Greece, are generally, static)`
- `(Greece, has, dynamic patronymic usage in middle names)`

分类：A。两半都已是三元组，候选只是用 while 把它们并到一句。

### 6. 希腊姓氏取代了用父亲的名字称呼的古代习惯

来源：同一篇。

原文：Greek surnames were widely in use by the 9th century supplanting the ancient tradition of using the father’s name.

已有的是 `(Greeks, used Greek surnames widely by, 9th century)`，以及塞浦路斯仍然按父亲的名字给姓。后者说的是相反的现行习惯，推不出“希腊姓氏取代了这一传统”。

分类：C。

### 7. 首都领地移交早于堪培拉被命名为首都

来源：`chunk-9d7d1d0ba6e55fc44634c8b002385912`。

原文：transferred … in 1911, two years prior to the naming of Canberra as the national capital in 1913.

已有两个日期：

- `(Australian Capital Territory, was transferred in, 1911)`
- `(Canberra, was named the national capital in, 1913)`

图上没有 “prior to / preceded” 边。两个年份本身也不构成一条关系路径；比较年份需要额外的数值规则。原文则直接写了 two years prior。

分类：C。增量只是这个先后，两个年份已经在图里。

### 8. Siim Kallas 此前不从政

来源：`chunk-c42d87d5f4a98f330d69805bf509c890`。

原文：who had been President of the Bank of Estonia and previously uninvolved in politics.

已有他领导该党、曾任银行行长、监督克朗发行、与 Mart Laar 的政府无牵连。没有“不从政”。和 Laar 无关推不出他此前不从政。

分类：C。

### 9. Andris Bērziņš 在提名截止前两天被提名

来源：`chunk-b4d60fe460691ba03ad83c18b248456e`。

原文：nominated … just two days before the nomination deadline.

已有他被五名议会议员和绿党与农民联盟提名。没有“截止前两天”。

分类：C。

### 10. USB 3.0 高功率设备最多取六个单位负载

来源：`chunk-e46a1203dbcbc2aa7d6fb7bb0abe3dea`。

已有 `(USB 3.0, permits a device to draw up to, six unit loads)`。

高功率设备“最多取标准允许的最大单位负载”是下一句。把“标准允许的最大值”换成“六个”，是把这两句接上，不是图里没有的数量。

分类：B。

### 11. USB 2.0 高功率设备最多取五个单位负载

同一篇。已有 `(USB 2.0, permits a device to draw up to, five unit loads)`。结构和上一条相同。

分类：B。

### 12. Alex Vreeke 被困在 Jumanji 里 20 年

来源：`chunk-df987a543dd7e90ab1e535a77f8965fb`。

原文：was trapped inside the Jumanji video game for 20 years.

已有 `(Alex Vreeke, was trapped inside, Jumanji)`，没有时长。另一篇写的是 two decades，而且那篇的三元组同样只有 trapped inside，没有时长。没有别的边写出 20 年。

分类：C。

### 13. 微利售卖和捐赠使 Holman 能把圣经送给买不起的人

来源：`chunk-22cdfe585380b9bb5eb5bbaba4adadf6`。

已有三条：

- `(Holman Bible Outreach International, sells Bibles for, minimal profit)`
- `(Holman Bible Outreach International, receives donations from, individuals and ministries)`
- `(Holman Bible Outreach International, gives Bibles to, people who may not be able to afford them)`

原文用 With that profit and donations … they are able to give 把这三句连成因果。图里有三件事实，没有 “enable” 这条边。三件事并列并不自动推出因果，但候选的实体和结果都已经在图里，新的只是把同句拆开的三元组再压回去。

分类：B。

## 汇总

| 类 | 条数 | 是哪些 |
| --- | --- | --- |
| A 重复 | 2 | 拉脱维亚“不是同一人”；希腊姓氏固定且中间名仍用父名 |
| B 捷径或压缩 | 5 | Jumanji 的两条；两条 USB 单位负载；Holman 的因果压缩 |
| C 单篇补漏 | 6 | 海德拉巴升学；希腊姓氏取代父名传统；堪培拉先后；Kallas 不从政；提名提前两天；被困 20 年 |
| D 跨篇新知识 | 0 | 无 |
| E 证据不足 | 0 | 保留集里没有 |

13 条里，真正新增的是 6 条，全部是单篇补漏。跨篇新知识是 0。其余 7 条是重复或把已有边压成一句。

Read-Time 目前留下的主要不是跨篇新关系。保留集里大约一半是单篇补漏，另一半是重复或捷径。验证器偏松：它把 “was not the same person as” 收成新的语义抽象，也把 “is the avatar of” 和已经抽出的出演关系再收一次。它只看还有没有未覆盖的词，没有先问这条边是否已经以别的谓词存在，也没有先走同篇的两跳。

后续不要把这几类都当成新关系。入库前先做三步：谓词和实体归一后查直接边；再查同篇两跳，并且每一步都能在原文对上，才标成捷径而不是新知识；只有直接边和这种路径都不蕴含时，才标成单篇补漏。跨篇新知识必须至少两篇，而且任何一篇单独都不蕴含结论。捷径可以单独存放，但不能和补漏混在同一个“新关系”计数里。
