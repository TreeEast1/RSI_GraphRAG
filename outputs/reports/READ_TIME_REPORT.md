# READ_TIME_REPORT — Experiment C

所有方法使用同一 Reader、同一 top-20 截断、同一 4096 token 阅读预算。
主指标为 category 1–4（可回答题，n=1,540）；dev 上 n=502。

## 1. Dev 主表（3 conversation，502 题，全部完成）

| 方法 | EM (可答) | F1 (可答) | Recall@5 | Recall@20 | Evidence full@4096 |
|---|---|---|---|---|---|
| Dense RAG | .241 | .451 | .822 | .985 | .771 |
| HippoRAG2 G_raw（C0/C1） | .253 | .481 | .858 | .986 | .819 |
| Full-Context QA（无图） | .261 | .486 | N/A | N/A | N/A |
| C2 cluster joint triple | .253 | .488 | .868 | .990 | .837 |
| C3 cluster comprehension | .261 | .492 | .873 | .990 | .849 |
| C5 full-conv joint triple | **.271** | .500 | .881 | .988 | .845 |
| C6 full-conv comprehension | .268 | **.501** | **.883** | .988 | .847 |

（同口径全行 EM：G_raw .392 / C2 .400 / C3 .410 / C5 .418 / C6 .416 / Full-Context .395。）

## 2. 结论

1. **H2 成立**：只把多个 session 放在一起做联合抽取（C2/C5），就能拿到几乎全部检索增益。
   - G_raw → C5：R@5 .858 → .881（+2.3pp）、evidence .819 → .845（+2.6pp）、EM .253 → .271。
2. **H3 不成立（在 LoCoMo 上）**：额外的 Comprehension 层几乎没有增量。
   - C3 vs C2：EM +0.8pp、F1 +0.4pp、evidence +1.2pp。
   - C6 vs C5：EM −0.3pp、F1 +0.1pp、evidence +0.2pp。
   - 也就是说，"发现跨 session 隐含关系"这一步没有产生可测量的检索/QA 收益。
3. **H7 成立**：全文联合阅读（C5/C6）优于聚类阅读（C2/C3），EM 高约 1pp、F1 高约 0.8pp。
   原因是 LoCoMo 对话全文仅 1.1–2.2 万 token，一次读完不需要任何选择。
4. **H4 弱成立**：相对 G_raw，最好的 Read-Time 变体提升 evidence +2.6–2.8pp、EM +1.5–1.8pp。方向一致，但幅度接近跨对话方差。

## 3. 图规模

| conv | G_raw phrase nodes / triples | C3 phrase nodes / triples | 注入新增记录 |
|---|---|---|---|
| conv-30 | 505 / 750 | 570 / 813 | 228 triples + 26 relations |
| conv-44 | 548 / 885 | 767 / 1,224 | 385 + 27 |
| conv-48 | 664 / 984 | 989 / 1,422 | 453 + 67 |

C3 平均增加约 45% 的 phrase node 与 45% 的 fact，但只换来 ~1pp 的 evidence recall。这本身说明"新增知识"的边际检索价值很低。

## 4. Test 集（7/7 对话，1,484 题，含 Reader）

| method | EM(可答) | F1(可答) | Recall@5 | Evidence@4096 |
|---|---|---|---|---|
| Dense RAG | .214 | .431 | .781 | .732 |
| HippoRAG2 G_raw（C0/C1） | .240 | .484 | .864 | .831 |
| Full-Context QA | .239 | **.493** | N/A | N/A |
| C2 cluster joint triple | .241 | .490 | .875 | .838 |
| C3 cluster comprehension | .238 | .484 | **.881** | .842 |
| C5 full-conv joint triple | **.245** | .489 | .874 | .841 |
| C6 full-conv comprehension | .243 | **.492** | **.881** | **.849** |

**结论：**
- **检索层一致改善**：所有变体的 R@5 比 G_raw 高 1.0–1.7pp，证据覆盖高 0.7–1.8pp；dev 与 test 方向一致。
- **QA 层没有一致改善**：差异都在 ±0.5pp 内，且不单调——C5 的 EM 最高（.245），C6 的 F1 最高（.492），C3 反而略低于基线（.238/.484）。
- **理解层（Comprehension）没有增量**：C3 < C2（EM −0.3pp）、C6 ≈ C5（EM −0.2pp / F1 +0.3pp）。新增的跨会话关系在 test 上没有转化为检索或答案收益。
- **全文 vs 聚类**：C5/C6 略好于 C2/C3（EM +0.4pp / F1 +0.1pp），但差距在噪声内；配合 Full-Context（F1 .493）几乎等于最好的图检索，再次说明 LoCoMo 的规模使"聚类"没有意义。

## 4b. Composed vs Inferred 消融（dev，502 题）

把 C3 注入的跨会话关系按 `derivation` 拆成两组，分别建图（都保留原始三元组）：

| 图谱 | EM(可答) | F1(可答) | R@5 | Evidence@4096 |
|---|---|---|---|---|
| G_raw | .253 | .481 | .858 | .819 |
| Composed-only | **.266** | **.501** | .869 | .845 |
| Inferred-only | .253 | .494 | **.875** | .841 |
| C3（两者合并） | .261 | .492 | .873 | **.849** |

**结论：**
- **Composed 关系带来最好的 QA**（EM +1.3pp、F1 +2.0pp vs G_raw），且优于两者合并的 C3。
- **Inferred 关系主要提升检索**（R@5 .875 最高），但对 EM 没有贡献；合并后 F1 反而比 Composed-only 低 0.9pp。
- 这说明「可验证的组合关系」是有效的，"语义推断"类关系更像检索噪声——与案例研究中的观察一致。

## 5. 已知问题

- **C5 首次运行在 8192 输出 token 处截断**（conv-44 返回 0 条三元组）。改为 16384 后成功（229 条）。所有 C5/C6 数字均来自 16384 设置，截断标志记录在各 `full_conv_*.json` 的 `truncated` 字段。
- **A/B 类重复检测是词面的**，Comprehension 的"新知识"可能被高估。
- 图谱变体通过复制 G_raw 的 embedding store + OpenIE state、再调用官方 `index()` 构建，**G_raw 永不被修改**。
