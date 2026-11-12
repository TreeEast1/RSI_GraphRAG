# 案例

案例从统计结果里按规则抽出，不按题目挑选。小图在 `plots/*_case_*.png`。原始 NER 分量里不存在 6–20 个 chunk 的分量；中等大小的例子来自 Jaccard ≥ 0.20 的硬聚类，并标明它不是原始连通分量。

这些例子说明共享字符串为什么会连边。它们不是分组准确率。

## 1. 最大连通分量

gpt-4o-mini B1 的最大分量有 11,422 个 chunk、64,871 个实体、9,620 个 title。Luna B1 是 11,365 个 chunk、9,564 个 title。

分量内部频率最高的实体是 `United States`（1,173）、`American`（887）、`Canada`（360）、`France`（314）、`Australia`（272）、`Germany`（265）、`2010`（250）。把前 15 名列全会看到年份和城市，见 `tables/hipporag2_gpt4o_mini_top_entities.csv`。

一个只共享 `United States` 的对：

- `Lionel Messi`，`chunk-151654358c8c4b3c7582d9182aaeb0bf`。传记页，NER 里同时有大量年份。
- `Member states of NATO`，`chunk-41e0cf7a0711db3927947a68b172c46d`。原文写 “two are located in North America (Canada and the United States)”。

两个 title 不同。共享实体只有 `United States`，DF 1,173。gpt-4o-mini 上这对没有共同的三元组实体。把这种边保留下来，分量就会吞进足球传记和北约成员表。

## 2. 原始图里没有 5–20 chunk 的分量

gpt-4o-mini B1 在巨型分量之外，最大的是 3 个 chunk，然后是若干 2 chunk。6–20、21–50、51–500 的分量个数都是 0。Luna 同样是 0。

因此不能从原始二部图里抽一个“典型中等证据组”。中等大小只出现在把弱边砍掉之后，见下一节。

## 3. 两个 chunk、一个稀有桥

Luna B1 里，`Rabbi Menachem Mendel Schneersohn` 的 DF 是 2，两个 title 不同。

- `Chaya Mushka Schneersohn`：她是 Rabbi Dovber Schneuri 的女儿，也是第三代 Rebbe Rabbi Menachem Mendel Schneersohn 的妻子。
- `Derech Mitzvosecha`：这本书由 Rabbi Menachem Mendel Schneersohn（1789–1866）写成。

这是同一人物把两篇不同条目连起来的例子。它落在巨型分量外面，是因为这两篇的其他实体没有再碰到高频网。这样的分量在 B1 里只有十几篇 chunk 的量级，覆盖不了语料。

另一个 DF 2 的桥是 `Nature Publishing Group`，gpt-4o-mini 和 Luna 都有：

- `European Journal of Human Genetics`：由 Nature Publishing Group 替 European Society of Human Genetics 出版。
- `The Journal of Antibiotics`：由 Nature Publishing Group 替 Japan Antibiotics Research Association 出版。

出版社相同，研究主题没有交集。共享实体可以是真的组织关系，同时对跨文档事实抽取价值很低。

## 4. 泛化实体和同形异义

`Princess of Orange` 在 gpt-4o-mini B1 的 DF 是 2，连的是：

- `Monarchy of the Netherlands`：现任继承人 Catharina-Amalia, Princess of Orange。
- `Anna of Lorraine`：她因与 René of Châlon 的婚姻成为 Princess of Orange。

这是同一个头衔的两个不同人。字符串把 16 世纪和当代王室连进一个分量。

`Macmillan` 经过大小写折叠后 DF 为 2：

- `A Prisoner of Birth`：Jeffrey Archer 的小说，2008 年由 Macmillan 出版。
- `McMillan (surname)`：原文起句就是 “MacMillan, Macmillan, McMillen or McMillan are variants of a Scottish surname.”

出版社和姓氏被收成同一个 key。这正是未做消歧时应该保留的歧义，不能把它当成别名合并的成功。

`Las Palmas` 的 DF 是 2，而且只在 NER 上把两篇连起来，两篇没有共同的三元组实体：

- `Lionel Messi`，`chunk-7b2db166f07ccb8e39f5d677bbfef87a`：原文是 “Barcelona's match against Las Palmas”，指的是比赛对手。
- `Yaiza (municipality)`：Lanzarote 的市镇，“forms part of the province of Las Palmas”。

两处原文都真有这个字符串，指的不是同一个对象。

## 5. NER 能连、三元组实体不能连

上一节的 Las Palmas 就是 gpt-4o-mini 上的这种对。`Las Palmas` 出现在 Yaiza 的三元组端点里，没有出现在 Messi 那篇的三元组端点里，两篇的三元组实体集合也不相交。

Luna 上的对应例子是 `Europa League`，DF 2：

- `Iker Muniain`：职业生涯在 Athletic Bilbao，原文后部写到 Europa League。
- `Liverpool F.C.–Manchester United F.C. rivalry`：一张胜负表，其中一行是 “Europa League 0”。

比赛名称把球员传记和另一场德比的统计表连起来。NER 边存在，共同的三元组实体不存在。这种差集有 1,287 个 gpt-4o-mini 实体，但它们改变不了巨型分量。

## 6. 两跳稀有实体

gpt-4o-mini 上，两个端点没有共同的 NER 实体，也没有共同的三元组实体：

1. `El Clásico` 与一篇 `Lionel Messi` 共享 `Supercopa de España`（DF 2）。El Clásico 页写的是 2017 年 8 月 16 日的 Supercopa。
2. 同一篇 Messi 与 `2002 FIFA World Cup Final` 共享 `Yokohama`（DF 2）。Messi 原文是 2015 年 FIFA Club World Cup 决赛 “in Yokohama”；世界杯决赛页是 2002 年 6 月 30 日在横滨的德国对巴西。

路径是真实的字符串重叠。端点是 2017 年的国家德比和 2002 年的世界杯决赛，中间经过一场 2015 年的俱乐部决赛举办地。把它当成证据组，会把三场不同比赛送进同一次抽取。

Luna 上的两跳更明显地不构成关系：`FC Barcelona` 与 `My Fair Lady (film)` 共享 `£5 million`，后者再与 `Sony Music` 共享 `William S. Paley`。一笔转会费和一笔电影版权费被收成同一个金额字符串。

## 7. Jaccard ≥ 0.20 才出现的中等组

这不是原始连通分量。阈值和分组在 gold 之前就定了。gpt-4o-mini 上，这个阈值有 666 个跨 chunk 组，最大 20 chunk，全部低于 16,000 token，只覆盖 15.5% 的 chunk。

抽到的一个 6 chunk 组有 3 个 title、989 个估计 token。组内较稀有的共享名包括 `Damion Poitier`（DF 2）、`Natasha Romanova`、`Black Widow`、`Thanos`、`Avengers: Infinity War`。成员是：

- `Black Widow (Natasha Romanova)` 的三个 chunk
- `Thanos` 的两个 chunk
- `Samuel L. Jackson` 一个 chunk，写他出演 Marvel Cinematic Universe 里的 Nick Fury

组的主体是同一 title 的重复 chunk，再加上相邻的漫威页面。它像系列页聚类，不像一道多跳题所需的跨条目证据包。

Luna 上 Jaccard ≥ 0.20 的一个 6 chunk、6 title、1,142 token 的组以 Shrek 电影页为主。`Kung Fu Panda` 也被连进来，因为它自己的原文写首映周末票房 “behind Shrek 2, Shrek the Third, and Shrek Forever After”。比较句里的片名足够让加权重叠跨过 0.20。票房比较已经写在 Kung Fu Panda 这一篇里面；把其他 Shrek 页一起送去抽三元组，读到的是系列介绍，不是这句比较所缺的事实。
