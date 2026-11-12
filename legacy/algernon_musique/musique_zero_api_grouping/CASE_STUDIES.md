# 案例

案例从三个冻结方法里抽：无监督选中的向量上限 16，事后 gold 最高的 1.0× 融合上限 16，以及 1.5× 融合复制。抽取规则是组大小 3–8、至少两个 title、组内余弦较高，外加一个中位组。没有按 gold 挑组。完整 JSON 在 `artifacts/cases.json`。

这些组能看出主题，不能看成多跳证据已经对齐。

## 同一人物，跨条目

向量上限 16 里有一个 3 chunk 组，平均余弦 0.54。共享实体是 Antonio Stradivari 和 Cremona，DF 都是 3。

- `Hellier Stradivarius`：约 1679 年 Antonio Stradivari 在 Cremona 做的琴。
- `Hammer Stradivarius`：同一制琴师的另一把琴。
- `Molitor Stradivarius`：1697 年、Golden period 的琴。

这是真正的跨 title 连接，关系是同一制作者的不同作品。这种组对单篇抽取是重复的传记事实，不是必须把三篇合在一次抽取里才能得到的桥。

## 同一地理系统

1.5× 融合组里，Sullivan County 的几条溪流被放在一起：`Ore Run`、`Hunts Run`、`Lead Run` 都是 Davidson Township 的支流，共享 `Sullivan Branch`、`Huntley Mountain Formation`。文本结构几乎一样。这是地名词典式重复，不是跨文档推理。

同一方法里的三艘扫雷舰 `USS Dextrous`、`USS Prevail`、`USS Acme` 共享 `United States Navy`，句子模板相同。向量把套话聚在一起。

## 同一条目的重复 chunk，再混进邻近主题

`(Ghost) Riders in the Sky` 的两个 chunk 写 Stan Jones 1948 年的录音和 Eden Ahbez、Burl Ives。组里还有 `Unchained Melody`。歌名和年代接近，后一篇是另一首歌。

`Molotov–Ribbentrop Pact` 的三个 chunk 与 `Malta Summit` 分在一组。共享词包括 pact、Finland、Britain、Soviet Union、Germany。马尔他峰会和 1939 年条约不是同一事件。DF 99 的 Soviet Union 和 DF 265 的 Germany 足够把它们串上。

## 容量把弱边也合并进来

融合上限 16 的一个组包含两篇 `Alex O'Loughlin`，共享 Hawaii Five-0 和 Steve McGarrett。同组还有 `Alex MacFarlane` 和 `Yao Defen`。后两篇与 O'Loughlin 没有那些 DF 2 的实体。贪心合并是传递的：A 连 B、B 连 C 之后，C 会留在同一个容量还没用完的组里。上限 16 把这种链拉得很长。

向量组里六个“美国广播电视人物”只稳定共享 `American`（DF 887）和有时出现的 `CBS`。`The Russian Room` 是一部 1997 年短片，被余弦邻居带进来。

`Nolensville`、`Coalmont`、`Middle Valley` 三座田纳西城镇共享 `Tennessee` 和 2010 census 的句式。它们没有跨城镇的事件关系。

## 和上一轮的关系

上一轮 Jaccard 小组能留下 Nature Publishing Group、Schneersohn 这种稀有字符串。本轮容量组更常留下向量邻域：同一工匠、同一县的河流、同一句式的军舰，以及被传递合并进来的无关页。两种现象都有。它们都不是 MuSiQue 那种要等题目出现才确定的支持文档组合。
