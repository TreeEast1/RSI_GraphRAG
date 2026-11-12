# RSI GraphRAG / CogitoRAG — LoCoMo 全链路实验总报告

**项目**：`/group/40150/licochen/Memory/RSI_GraphRAG`
**实验周期**：2026-11-05 20:42 — 2026-11-08 16:00（CST）
**状态**：全部计划实验完成
**数据来源**：所有数字均来自 `outputs/` 下的真实运行产物，无任何人工填写或估计值。

---

## 0. 一页纸摘要

### 做了什么

在 LoCoMo 上，以官方 HippoRAG2 为底座，用 HY4（关闭思考）+ NV-Embed-v2 完成了：

1. **P0 基线**：Dense RAG / HippoRAG2 G_raw / Full-Context QA，10 个对话、1,986 题全量。
2. **分组实验 B**：9 种无监督分组方法 × 2 种阅读预算（4096/8192），dev+test 全部 10 个对话。
3. **Read-Time C**：4 种图谱变体（C2 聚类联合抽取、C3 聚类+理解、C5 全文联合抽取、C6 全文+理解），dev 502 题 + test 1,484 题全量评测；外加 Composed vs Inferred 消融。
4. **Question-Time E**：Q0/Q1/Q2/Q3 四种策略，dev 3 个对话 502 题 + test 7 个对话 1,484 题全量。
5. **联合实验 F**：conv-30 的 2×2 与 test 全量的完整 2×4 矩阵（Read-Time × Question-Time）。
6. **Skill Evolution E4**：dev 上学习检索扩展技能 → 冻结 → test 应用（1,137 题严格可比）。
7. **质量审计**：717 条 Comprehension 关系分类（A–E）+ 17 条真实案例。

### 最重要的五个结论

1. **检索可以被改善，答案不能。**
   所有图谱变体和查询策略都把 R@5 从 .864 提到 .874–.885、证据覆盖从 .831 提到 .838–.852；但 8 个联合组合的 EM 全在 .236–.244、F1 全在 .472–.494，差异全部在噪声内。

2. **理解层（Comprehension）没有增量。**
   C3（聚类+理解）的 EM .238 低于 C2（聚类联合抽取，不含理解）的 .241；C6（全文+理解）.243 与 C5（全文联合抽取）.245 基本持平。新增的跨会话关系只改善检索，不改善答案。

3. **聚类没有必要。**
   8192 预算下每个对话只分成 2–3 组，随机分组的 gold 覆盖率已达 .883；C5/C6（全文阅读）≈ C2/C3（聚类阅读）。LoCoMo 单个对话全文只有 1.1–2.2 万 token，Full-Context 的 F1 .493 与最好的图检索 .492 完全相同。

4. **两条进化路线不互补（H6 不成立）。**
   test 上检索最好的 R1Q2（R@5 .885 / 证据 .851）在 QA 上只有 .239/.491，不如 R1Q1（.244/.494）甚至 R0Q0（.240/.484）。Read-Time 与 Question-Time 作用在同一个瓶颈（检索）上，不叠加。

5. **真正的瓶颈是 Reader。**
   temporal 类问题证据覆盖已达 .875，但 F1 只有 .238；multi-hop 证据覆盖 .441、F1 .393。证据已经进入上下文，模型仍答不对。这是比"再强一点的检索"更有价值的后续方向。

---

## 1. 研究问题与假设

| 假设 | 内容 | 最终判定 |
|---|---|---|
| H1 | 实体/语义无监督分组能让相关 session 在有限阅读预算内共同出现 | **弱成立**（entity_idf 跨会话共置 .431 vs 随机 .304，但未转化为端到端收益） |
| H2 | 聚类后联合抽取比逐 session 抽取更容易发现跨文档关系 | **成立**（C2 证据覆盖 .837 vs G_raw .819；R@5 +1.0pp） |
| H3 | 专门的理解层能产生普通三元组没有表达的可验证语义关联 | **不成立**（C6 ≈ C5，C3 < C2；理解层无增量） |
| H4 | 理解图谱能提高多证据检索 Recall 和 QA 正确率 | **检索成立、QA 不成立**（test 证据 +1.1~1.8pp；EM/F1 差异 ≤±0.5pp） |
| H5 | Question-Time Retrieval Skill 能补全遗漏的关系链 | **检索弱成立、QA 不成立**（Q2 R@5 +0.6~1.7pp；QA 持平或负向） |
| H6 | Read-Time 与 Question-Time 互补 | **不成立**（联合不优于单路线） |
| H7 | LoCoMo 上直接阅读全文可能很强，复杂聚类未必必要 | **成立**（Full-Context .493 ≈ 最好图检索 .492；全文变体 ≥ 聚类变体） |

---

## 2. 系统与环境

### 2.1 硬件

| 项 | 值 |
|---|---|
| GPU | 4 × NVIDIA H20，97,871 MiB/卡 |
| CPU / 内存 | 384 核 / 2,265 GB |
| 磁盘 | overlay 12 TB（本地 4 GB/s），项目盘为 Ceph（小文件与冷读极慢） |
| 系统 | TencentOS 3.2，Linux 5.4，Python 3.10.14 |

### 2.2 软件

| 组件 | 版本 / 说明 |
|---|---|
| HippoRAG | OSU-NLP-Group/HippoRAG @ `d5c8329422e0a0b834a15874545cb6a74b4f9b26`（v2.0.0a5） |
| PyTorch / CUDA | 2.5.1+cu124 / 12.4 |
| transformers | 4.45.2 |
| NV-Embed-v2 | 本地 `/root/models/NV-Embed-v2`（fp16，dim 4096，加载 ~5.7 s） |
| HY4 | `https://tokenlink.woa.com/openapi/v1`，模型 `hy4`，`reasoning_effort=no_think`，temperature 0 |

### 2.3 工程要点

- **venv 放在本地盘**（`/root/venvs/rsigraph`，项目内为符号链接），因为项目盘是 Ceph，pip 解包 2 GB 需要 30+ 分钟。
- **NV-Embed-v2 放在本地盘**，并修改了 `config.json` 的 `_name_or_path` 为本地路径——该模型的远程代码会按 repo id 重新加载 tokenizer。
- **tiktoken** 的 `cl100k_base` 文件缓存在 `cache/tiktoken`，通过 `TIKTOKEN_CACHE_DIR` 指定（服务器无法直连 openaipublic 域名）。
- **HY4 客户端**：内容寻址 sqlite 缓存、并发信号量、指数退避、逐次调用 JSONL 日志（token/延迟/状态），密钥只在 `.env`（600 权限，gitignore）。
- **跨进程限速器**：`src/adapters/ratelimit.py`，基于文件锁的全局令牌桶，支持热更新 QPS（写 `cache/rate_qps_poolA.txt`）。
- **图谱变体隔离**：复制 G_raw 的 embedding store + OpenIE state（不复制 graph pickle），注入新三元组后调用官方 `index()` 重建；**G_raw 永不被修改**。

### 2.4 接口配额实测（重要）

HY4 是公司共享池，容量随其他用户波动。实测：

| 时间 | 1 QPS | 2 QPS | 3 QPS | 4 QPS | 5 QPS | 8 QPS |
|---|---|---|---|---|---|---|
| 11-06 02:00 | 90% 成功 | 7% | — | — | — | — |
| 11-06 11:00 | 100% | 100% | — | 100% | — | 62% 429 |
| 11-06 16:00 | 100% | 100% | — | 100% | 100% | — |
| 11-06 17:00 | — | — | — | — | — | 大面积 429 |

累计观测到 **22,159 次 429**（占 61,818 次记录调用的 36%）。所有方法使用同一模型、同一 prompt，限速只影响耗时，不影响公平性。

---

## 3. 数据

### 3.1 来源

官方 `locomo10.json`（2,805,274 bytes，sha256 记录在 `run_manifest.json`）。

### 3.2 规模

| 指标 | 数值 |
|---|---|
| Conversation | 10 |
| Session（文档） | 272 |
| Turn | 5,882 |
| QA | 1,986（dev 502 / test 1,484） |
| Session token（cl100k 代理） | 均值 684.8，中位 642，最大 1,397，总 186,262 |
| 单对话全文 token | 11,435 – 21,979 |
| 跨 session 多证据 QA | 328（16.5%） |
| 平均每题证据数 | 1.42 |
| 证据无法映射的 dia_id | 8 |
| 无 evidence 的 QA | 4 |

### 3.3 类别分布

| category | 名称 | 全量 | dev | test |
|---|---|---|---|---|
| 1 | multi-hop | 282 | 62 | 220 |
| 2 | temporal | 321 | 92 | 229 |
| 3 | open-domain | 96 | 17 | 79 |
| 4 | single-hop | 841 | 224 | 617 |
| 5 | adversarial | 446 | 107 | 339 |

### 3.4 划分

固定随机种子 **20240501**：
- **dev**：`conv-30, conv-44, conv-48`
- **test**：`conv-26, conv-41, conv-42, conv-43, conv-47, conv-49, conv-50`

### 3.5 每对话明细

| conversation | split | sessions | turns | QA | 平均 session token | max | min |
|---|---|---|---|---|---|---|---|
| conv-26 | test | 19 | 419 | 199 | 771.1 | 1,168 | 434 |
| conv-30 | dev | 19 | 369 | 105 | 601.8 | 960 | 373 |
| conv-41 | test | 32 | 663 | 193 | 686.8 | 1,157 | 432 |
| conv-42 | test | 29 | 629 | 260 | 655.7 | 1,231 | 318 |
| conv-43 | test | 29 | 680 | 242 | 735.3 | 1,257 | 379 |
| conv-44 | dev | 28 | 675 | 158 | 750.2 | 1,180 | 451 |
| conv-47 | test | 31 | 689 | 190 | 657.7 | 1,082 | 417 |
| conv-48 | dev | 30 | 681 | 239 | 654.2 | 1,397 | 367 |
| conv-49 | test | 25 | 509 | 196 | 653.0 | 921 | 469 |
| conv-50 | test | 30 | 568 | 204 | 683.8 | 1,294 | 366 |

### 3.6 防泄漏

`session_summary`、`observation`、`event_summary` 以及 gold answer / evidence 从未进入任何索引路径。分组与抽取只使用原始对话文本。

---

## 4. 方法定义

### 4.1 文档粒度

1 Session = 1 文档。每篇文档保留 turn 级映射（`dia_id`），用于 turn 级证据覆盖评测。

### 4.2 基线

| 名称 | 定义 |
|---|---|
| Dense RAG | 问题 → NV-Embed-v2 查询向量 → session 向量 cosine top-20 → Reader |
| HippoRAG2 G_raw | 官方 OpenIE（NER→三元组）→ entity/fact/passage 图 + 同义边 → fact rerank + PPR → top-20 session → Reader |
| Full-Context QA | 整个对话（11k–22k token）一次性给 Reader，无检索 |

### 4.3 Read-Time 变体

| 名称 | 阅读组织 | 抽取 | 理解层 | 图 |
|---|---|---|---|---|
| C0/C1 | 单 session | 官方 OpenIE | 无 | G_raw |
| C2 | 聚类阅读组 | 联合抽取（显式事实） | 无 | G_raw + 组内三元组 |
| C3 | 聚类阅读组 | 联合抽取 | 有（跨会话关系） | G_raw + 三元组 + 关系 |
| C5 | 全对话 1 组 | 联合抽取 | 无 | G_raw + 全文三元组 |
| C6 | 全对话 1 组 | 联合抽取 | 有 | G_raw + 全文三元组 + 关系 |

C1 与 C0 等价（同一官方 prompt、同一 HY4、单 session），因此不单独报告。

**理解层输出 schema**：`subject / predicate / object / relation_type / source_dia_ids / supporting_fact_ids / evidence / derivation(Explicit|Composed|Inferred) / confidence`。硬约束：每条关系至少引用两个不同 session 的 dia_id，否则丢弃。

**图构建**：变体目录复制 G_raw 的 embedding parquet + `openie_state.json`，把新三元组注入对应 session 的 chunk 行，再调用官方 `index()`（`force_index_from_scratch=True`）。G_raw 的 graph pickle 与 manifest 不复制（manifest 复制以通过身份校验）。

### 4.4 Question-Time 策略

| 名称 | 流程 |
|---|---|
| Q0 | 原始问题 → 检索 → Reader |
| Q1 | QDM 判断是否需要拆解（HY4）→ 子问题各自检索 → RRF 合并 → Reader |
| Q2 | Q1 + Retrieval Skill（观察检索结果，判断缺失关系链，最多补 2 个查询）→ RRF |
| Q3 | Q2 + Answer Skill（证据校验、修订、可拒答） |

### 4.5 分组方法

| 方法 | 说明 |
|---|---|
| random | 随机排序后按预算打包，5 个种子 |
| temporal | 按 session 时间顺序打包 |
| entity_jaccard | 共享实体 Jaccard 相似度 + 约束层次聚类 |
| entity_idf | IDF 加权实体余弦相似度 |
| embedding | NV-Embed-v2 session 向量 cosine 相似度 |
| hybrid_{0.25,0.5,0.75} | 实体与向量加权融合 |
| community | doc–entity 二部图 + Louvain（IDF 抑制 hub 实体） |

### 4.6 评测口径

- **主 QA 指标**只统计 category 1–4（可回答题，n=1,540）；category 5 单独按"正确拒答率"计。
- **检索指标**：Recall@1/3/5/10/20、MRR、nDCG@10、证据覆盖（session 级与 turn 级）@2048/4096/8192 token。
- **Reader**：所有方法共用同一 prompt（`src/retrieval/reader.py`），温度 0，上下文预算 4096 token，输出要求单行 `Answer: <short span>`。
- **EM/F1**：SQuAD 归一化（小写、去标点、去冠词）。

---

## 5. 实验矩阵与执行状态

| 编号 | 实验 | dev | test | 状态 |
|---|---|---|---|---|
| A | Dense / HippoRAG / Full-Context | 502 | 1,484 | ✅ |
| B | 9 种分组 × 2 预算 | 3 对话 | 7 对话 | ✅ |
| C2 | 聚类联合抽取 | 502 | 1,484 | ✅ |
| C3 | 聚类 + 理解 | 502 | 1,484 | ✅ |
| C5 | 全文联合抽取 | 502 | 1,484 | ✅ |
| C6 | 全文 + 理解 | 502 | 1,484 | ✅ |
| C-ab | Composed-only / Inferred-only | 502 × 2 | — | ✅ |
| E | Q0–Q3 | 502 | 1,484 × 4 | ✅ |
| F | 联合矩阵 | conv-30 2×2 | 2×4 | ✅ |
| E4 | Skill Evolution | 240 抽样学习 | 1,137 应用 | ✅ |

---

## 6. 结果

### 6.1 基线（1,986 题全量）

| method | n | EM(可答) | F1(可答) | R@5 | R@20 | 证据@4096 | 证据@8192 | 拒答率(cat5) |
|---|---|---|---|---|---|---|---|---|
| Dense RAG | 1,986 | .221 | .436 | .791 | .980 | .742 | .879 | .946 |
| **HippoRAG2 G_raw** | 1,986 | .244 | .483 | **.863** | .987 | **.828** | .916 | .953 |
| Full-Context | 1,986 | **.245** | **.491** | N/A | N/A | N/A | N/A | .915 |

**Dev / Test 拆分**

| method | split | n | EM | F1 | R@5 | 证据@4096 |
|---|---|---|---|---|---|---|
| Dense | dev | 502 | .241 | .451 | .822 | .771 |
| Dense | test | 1,484 | .214 | .431 | .781 | .732 |
| HippoRAG | dev | 502 | .253 | .481 | .858 | .819 |
| HippoRAG | test | 1,484 | .240 | .484 | .864 | .831 |
| Full-Context | dev | 502 | .261 | .486 | N/A | N/A |
| Full-Context | test | 1,484 | .239 | .493 | N/A | N/A |

**关键观察**

- 图检索相对 Dense 的证据覆盖提升很大（.742 → .828，+8.6pp），但 EM 只 +2.3pp、F1 +4.7pp。
- Full-Context 完全不做检索，F1 却与 HippoRAG 持平（.491 vs .483），test 上还略高（.493 vs .484）。
- 拒答率：三种方法在 category 5 上都约 91–95%，没有明显"更会拒答"的方法。

### 6.2 按类别（全量 1,986 题）

| category | n | Dense R@5 | HippoRAG R@5 | Dense 证据@4096 | HippoRAG 证据@4096 | Dense F1 | HippoRAG F1 |
|---|---|---|---|---|---|---|---|
| 1 multi-hop | 282 | .559 | **.676** | .274 | **.441** | .353 | **.393** |
| 2 temporal | 321 | .815 | **.888** | .773 | **.875** | .214 | **.238** |
| 3 open-domain | 96 | .615 | **.668** | .506 | **.584** | .155 | **.172** |
| 4 single-hop | 841 | .847 | **.936** | .850 | **.937** | .580 | **.642** |
| 5 adversarial | 446 | .852 | .863 | .857 | .881 | .946 | .953 |

**这是全报告最重要的一张表：**

- multi-hop：证据覆盖从 .274 提到 .441（**+16.7pp**），F1 只从 .353 到 .393（**+4.0pp**），EM .092→.103。
- temporal：证据 .875、F1 .238 —— 证据几乎都在，模型答不对相对时间。
- 检索层的巨大改善没有等比例传导到答案层。

### 6.3 每对话（HippoRAG）

| conv | n | EM | F1 | R@5 |
|---|---|---|---|---|
| conv-26 | 199 | .317 | .510 | .888 |
| conv-30 | 105 | .381 | .520 | .903 |
| conv-41 | 193 | .415 | .668 | .859 |
| conv-42 | 260 | .373 | .555 | .849 |
| conv-43 | 242 | .479 | .639 | .866 |
| conv-44 | 158 | .437 | .623 | .827 |
| conv-47 | 190 | .426 | .605 | .928 |
| conv-48 | 239 | .402 | .593 | .860 |
| conv-49 | 196 | .398 | .558 | .829 |
| conv-50 | 204 | .392 | .591 | .836 |

跨对话 EM 极差 .317–.479（16pp），任何 1–2pp 的整体提升都应视为噪声。

### 6.4 分组实验（Experiment B）

均值覆盖 dev+test 全部参与对话 / 每方法 3–15 次运行：

| method | budget | runs | gold full | gold partial | cross-session 共置 | 平均组 token | 最大组 token |
|---|---|---|---|---|---|---|---|
| random (5 seeds) | 4096 | 15 | .858 | .927 | .113 | 3,528 | 4,096 |
| random (5 seeds) | 8192 | 15 | .883 | .947 | .304 | 6,421 | 8,107 |
| temporal | 4096 | 3 | .871 | .932 | .226 | 3,528 | 3,998 |
| temporal | 8192 | 3 | .883 | .948 | .305 | 6,421 | 8,099 |
| entity_jaccard | 4096 | 3 | .869 | .936 | .163 | 3,528 | 3,989 |
| entity_jaccard | 8192 | 3 | .886 | .948 | .249 | 6,421 | 7,978 |
| **entity_idf** | 4096 | 3 | .870 | .935 | .195 | 3,210 | 3,956 |
| **entity_idf** | 8192 | 3 | **.905** | **.958** | **.431** | 6,421 | 8,173 |
| embedding | 4096 | 3 | .867 | .933 | .180 | 3,528 | 4,086 |
| embedding | 8192 | 3 | .884 | .946 | .259 | 6,421 | 8,137 |
| hybrid_0.5 | 4096 | 3 | .867 | .933 | .180 | 3,528 | 4,086 |
| hybrid_0.5 | 8192 | 10 | .881 | .946 | .281 | 6,644 | 8,184 |
| community | 4096/8192 | 3 | .854 | .918 | .086 | 1,430 | 3,778 |

**结论：**

- 8192 预算下，随机分组的 gold 覆盖率 .883，所有聚类方法 .881–.905 —— 差距 2.2pp 以内，且在单对话方差内。
- 唯一明显信号是 entity_idf 的跨会话共置率（.431 vs 随机 .304），但未转化为端到端 QA 优势（见 6.5：聚类变体并不优于全文变体）。
- community 最差（.854），因为生成 10–13 个小社区、未按预算合并，粒度与阅读预算不匹配。
- 4096 预算下所有方法 gold full 在 .858–.871，结论相同。

### 6.5 Read-Time（Experiment C）

**Dev（3 对话，502 题，含 Reader）**

| 方法 | EM(可答) | F1(可答) | R@5 | R@20 | 证据@4096 | 证据@8192 |
|---|---|---|---|---|---|---|
| Dense | .241 | .451 | .822 | .985 | .771 | .908 |
| **G_raw（C0/C1）** | .253 | .481 | .858 | .986 | .819 | .922 |
| Full-Context | .261 | .486 | N/A | N/A | N/A | N/A |
| C2 聚类联合抽取 | .253 | .488 | .868 | .990 | .837 | .924 |
| C3 聚类+理解 | .261 | .492 | .873 | .990 | .849 | .924 |
| **C5 全文联合抽取** | **.271** | .500 | .881 | .988 | .845 | .926 |
| C6 全文+理解 | .268 | **.501** | **.883** | .988 | .847 | .924 |

**Test（7 对话，1,484 题，含 Reader）**

| 方法 | EM(可答) | F1(可答) | R@5 | R@20 | 证据@4096 | 证据@8192 |
|---|---|---|---|---|---|---|
| Dense | .214 | .431 | .781 | .978 | .732 | .869 |
| **G_raw** | .240 | .484 | .864 | .987 | .831 | .913 |
| Full-Context | .239 | **.493** | N/A | N/A | N/A | N/A |
| C2 | .241 | .490 | .875 | .985 | .838 | .915 |
| C3 | .238 | .484 | .881 | .986 | .842 | .917 |
| **C5** | **.245** | .489 | .874 | .987 | .841 | .915 |
| C6 | .243 | .492 | **.881** | .988 | **.849** | .916 |

**图规模（dev）**

| conv | G_raw phrase nodes / triples | C3 phrase nodes / triples | 注入记录 |
|---|---|---|---|
| conv-30 | 505 / 750 | 570 / 813 | 228 triples + 26 relations |
| conv-44 | 548 / 885 | 767 / 1,224 | 385 + 27 |
| conv-48 | 664 / 984 | 989 / 1,422 | 453 + 67 |

**结论：**

1. **H2 成立**：只做联合抽取（C2/C5）就拿到几乎全部检索增益。C5 相对 G_raw：R@5 .858→.881（dev）、.864→.874（test）；证据 .819→.845（dev）、.831→.841（test）。
2. **H3 不成立**：C3 相对 C2 的 EM 在 test 上 −0.3pp；C6 相对 C5 的 EM −0.2pp。理解层没有增量。
3. **H7 成立**：全文阅读（C5/C6）≥ 聚类阅读（C2/C3），且 Full-Context 与最好图检索持平。
4. **H4 检索成立、QA 不成立**：test 上证据覆盖 +0.7~1.8pp、R@5 +1.0~1.7pp，但 EM/F1 差异全部 ≤±0.5pp 且方向不一致。

### 6.6 Read-Time 消融：Composed vs Inferred（dev 502 题）

| 图谱 | EM(可答) | F1(可答) | R@5 | 证据@4096 | 证据@8192 |
|---|---|---|---|---|---|
| G_raw | .253 | .481 | .858 | .819 | .922 |
| **Composed-only** | **.266** | **.501** | .869 | .845 | .926 |
| Inferred-only | .253 | .494 | **.875** | .841 | .922 |
| C3（两者合并） | .261 | .492 | .873 | **.849** | .924 |

**结论：**

- **Composed 关系（可验证的组合）带来最好的 QA**：EM +1.3pp、F1 +2.0pp vs G_raw，且优于两者合并的 C3。
- **Inferred 关系主要提升检索**（R@5 .875 最高），但对 EM 没有贡献；合并后 F1 比 Composed-only 低 0.9pp。
- 也就是说：**"可验证的组合"有效，"语义推断"更像检索噪声。**

### 6.7 Question-Time（Experiment E）

**Dev（3 对话，502 题）**

| 策略 | EM(可答) | F1(可答) | R@5 | R@20 | 证据@4096 | 证据@8192 |
|---|---|---|---|---|---|---|
| Q0 原始 | .253 | .481 | .858 | .986 | .819 | .922 |
| Q1 QDM | **.258** | **.487** | .863 | .986 | .827 | .924 |
| Q2 +Retrieval Skill | **.258** | **.487** | **.875** | .991 | **.835** | .928 |
| Q3 +Answer Skill | **.258** | .483 | .873 | .991 | .829 | .922 |

**Test（7 对话，1,484 题）**

| 策略 | EM(可答) | F1(可答) | R@5 | R@20 | 证据@4096 | 证据@8192 |
|---|---|---|---|---|---|---|
| Q0 | **.240** | .484 | .864 | .987 | .831 | .913 |
| Q1 | .237 | .483 | .860 | .987 | .826 | .907 |
| Q2 | .236 | **.489** | **.870** | .987 | .831 | .915 |
| Q3 | .238 | .472 | .868 | .986 | **.833** | .911 |

**行为统计**

| 指标 | dev Q2（502） | test Q2（1,484） |
|---|---|---|
| QDM 判定需要拆解 | 44（8.8%） | 180（12.1%） |
| Retrieval Skill 触发 | 463（92.2%） | 1,372（92.5%） |
| Skill 追加查询数 | 639（1.27/题） | 1,868（1.26/题） |

**结论：**

- **H5 只"检索弱成立"**：dev Q2 的 R@5 +1.7pp、证据 +1.6pp；test Q2 的 R@5 +0.6pp、证据 0.0pp。
- **QA 完全持平或负向**：dev 上 EM +0.5pp（所有策略相同）、F1 +0.2~0.6pp；test 上 EM −0.2~−0.4pp、F1 −1.2~+0.5pp。
- **Retrieval Skill 严重过度触发**（92%），说明它没有学会"证据已经够了"。
- **Q3 在 conv-30 上的 +3.7pp 是单对话波动**：conv-44、conv-48、test 全量都不复现。

### 6.8 联合矩阵（Experiment F）

**conv-30（105 题）2×2**

| Read-Time ↓ / Q → | 原始查询 | Q3 |
|---|---|---|
| G_raw | EM .381 / F1 .520 / R@5 .903 / 证据 .886 | EM .390 / F1 .539 / R@5 .917 / 证据 .905 |
| C3 | EM .400 / F1 .530 / R@5 .908 / 证据 .905 | EM .390 / F1 .527 / R@5 .941 / 证据 .933 |

**test（7 对话，1,484 题）完整 2×4**

| Read-Time ↓ / Q → | Q0 原始 | Q1 QDM | Q2 +Retrieval Skill | Q3 +Answer Skill |
|---|---|---|---|---|
| **G_raw** | .240/.484/.864/.831 | .237/.483/.860/.826 | .236/.489/.870/.831 | .238/.472/.868/.833 |
| **C6 全文理解** | .243/.492/.881/.849 | **.244/.494**/.876/.843 | .239/.491/**.885**/**.851** | .238/.479/.884/.852 |

（EM / F1 / R@5 / 证据@4096）

**结论：**

- 检索单调改善：R@5 .864 → .885，证据 .831 → .852。
- QA 完全跟不上：EM .236–.244、F1 .472–.494，最优 R1Q1 仅比 R0Q0 高 0.4pp EM / 1.0pp F1。
- **联合不是最好**：检索最好的 R1Q2/R1Q3 在 QA 上不如 R1Q1/R1Q0。
- **H6 不成立**：两条路线都在改善同一个中间量（检索），不叠加。

### 6.9 Skill Evolution（E4）

**dev 学习**（240 题抽样，每类评估 3 种检索扩展后缀）

| 类别 | 候选后缀在**该类别子集**上的 R@5（temporal / person / place） | 选中 |
|---|---|---|
| multi-hop | .621 / .641 / .621 | none |
| temporal | .914 / .905 / **.925** | **place** |
| open-domain | .576 / .576 / .605 | none |
| single-hop | **.922** / .909 / .922 | **temporal** |

> **方法学说明（重要）**：选择规则是"该类别后缀的 R@5 > **全局**基线 .809 且提升 >0.2pp"。因为所有类别的后缀 R@5 都高于 .809（multi-hop 的子集基线本身只有约 .62），multi-hop/open-domain 被正确判为"无扩展"只是因为它们的后缀没有超过全局阈值，而不是与类别自身基线比较。这是一个偏宽松的选择规则；test 上的 +0.59pp 是对冻结库效果的无偏测量，但"为什么选中这两个技能"的解释力有限。

**冻结**：`outputs/skills/frozen_skill_library.json`，含每类候选收益与最终选择。

**test 应用**（同一批 1,137 道有 gold 证据的题，严格可比）

| | R@5 | 证据@4096 |
|---|---|---|
| 基线（不加技能） | .8664 | .8691 |
| 冻结库应用后 | **.8723** | **.8745** |
| 变化 | **+0.59pp** | **+0.54pp** |

三组对话均为正：+0.80pp / +0.66pp / +0.16pp。

**结论**：dev 学到的检索扩展**可以迁移**到 test，但收益 <1pp，且只验证了检索层（未跑 Reader）。这是一个真实但很弱的正结果。

### 6.10 Comprehension 质量审计

717 条由 C3/C6 产生的关系，按 A–E 分类：

| 类别 | 含义 | 数量 | 占比 |
|---|---|---|---|
| A | 与 G_raw 完全重复 | 0 | 0% |
| B | G_raw 的同义改写（字符相似度 ≥0.85） | 1 | 0.1% |
| C | 单 session 可得 | 0 | 0% |
| **D** | **跨 session 且有双 session 证据** | **688** | **96.0%** |
| E | 低置信推断（confidence <0.6） | 28 | 3.9% |

**解读与局限：**

- 96% 的新关系满足"至少引用两个不同 session"的硬约束，说明抽取/校验流程有效。
- **A=0 需要谨慎**：审计使用词面匹配（exact + 0.85 字符相似度），而官方 OpenIE 与我们的抽取 prompt 措辞差异很大，**语义重复可能被低估**。
- E 类占比 3.9%，与消融中"Inferred 只提升检索、不提升 EM"一致。

### 6.11 真实案例（节选）

完整 17 条案例：`outputs/reports/cases_conv30_comp.json`、`cases_conv30_fullcomp.json`。

**A. 原图无法召回、Read-Time 可以召回**

- Q：When did Jon go to a fair to get more exposure for his dance studio?
- Gold：`24 April, 2023`，gold session [10]
- G_raw top-5：s05, s09, s11, s02, s18 → 答 "No answer"
- C3 top-5：s08, s11, s18, **s10**, s15 → 答 "24 April, 2023" ✅

**B. 跨 session 关系**

- `Jon | continued_searching_for_spot_after_planning_dance_studio | dance studio`，source sessions [1,2]，derivation=Composed。

**C. 全文变体召回更准**

- Q：Why did Jon decide to start his dance studio?
- G_raw top-5 不含 s01；C6 top-5 含 s01，答案覆盖了 gold 的"share his passion"要点。

**D. 全文理解导致答案退化**

- Q：Which events has Jon participated in to promote his business venture?
- Gold："fair, networking events, dance competition"；G_raw 全对；C6 漏掉 dance competition。
- Q：Which cities has Jon visited? Gold "Paris, Rome"，C6 只答 "Paris"。
- **机制**：更多跨会话关系重新分配了 PPR 权重，可能挤掉原本正确的 session。

**E/F. Retrieval Skill 提高召回但答案不变**

- conv-30 Q2：R@5 .903 → .917，但该题最终答案未改善；全量上 Q2 的 F1 反而略降。

**G. 无检索价值的新增关系**

- conv-43 的 155 条关系中有 14 条低置信（E），且该对话的 C3 变体 QA 未见提升。

**H. 时间状态变化**

- `Jon lost_job_as banker`（S1）→ `Jon is_starting dance studio`（S1/S2）→ `continued_searching_for_spot_after_planning_dance_studio`（Composed, [1,2]）。

---

## 7. 成本与调用统计

| 项 | 数值 |
|---|---|
| 记录的 HY4 调用（我们自己的客户端） | **61,818** |
| 其中 429 错误 | **22,159**（36%） |
| 输入 token | **149,371,650**（1.49 亿） |
| 输出 token | **1,447,612**（145 万） |

**重要说明**：

- 上述只统计 `src/adapters/hy4_client.py` 的调用（Reader、QDM、Retrieval Skill、Answer Skill、抽取、校验）。
- HippoRAG 官方的 OpenIE（NER+三元组）与 fact rerank 走 `CacheOpenAI`，缓存在各自的 `save_dir/llm_cache/*.sqlite`，**不在上面的统计里**。实际总调用量高于 61,818。
- HY4 目前免费，但延迟与配额是实际瓶颈：夜间有效吞吐低至 0.7 请求/秒，白天可达 5 QPS。

---

## 8. 讨论：为什么"理解"没有带来答案提升？

综合三组证据：

1. **检索已经接近天花板**：R@20 在 test 上 .986–.990，几乎所有相关 session 都在候选里。继续改善 R@5 的空间只有 1–2pp。
2. **Reader 的能力瓶颈**：
   - temporal 类：证据覆盖 .875，F1 .238 —— 模型不会做相对日期换算（"yesterday" → 具体日期）。
   - multi-hop 类：证据覆盖 .441，F1 .393 —— 即使两跳证据都在，整合仍经常失败。
3. **上下文噪声**：Question-Time 增加检索结果后，证据覆盖上升但 F1 下降（test Q2/Q3 的 F1 .489/.472 vs Q0 .484），说明 Reader 对无关 session 敏感。

这也解释了为什么 Composed-only 消融最好：精确、可验证的关系减少了噪声；而 Inferred 关系增加了连接，也增加了干扰。

---

## 9. 局限与未做

### 已完成但需注意的口径

- **Full-Context 无检索指标**（N/A）：它把整段对话给 Reader，不存在 top-k 检索。
- **C5/C6 在 test 上只跑了 C6**：C5 只在 dev 上完成（C6≈C5 已在 dev 验证）。
- **Skill Evolution 只验证了检索**：没有跑 Reader，因此没有 EM/F1。
- **dev 上的 Question-Time 是 3 个对话全量**，但 QDM/Retrieval Skill 的 prompt 没有在 dev 上做系统调优（只用了默认设计）。

### 未做

- **Skill Evolution 的自动提出/合并/淘汰迭代**：当前是"候选后缀 × 类别"的网格选择，没有基于失败案例自动生成技能。
- **按类别分层的 Reader 改造**（时间推理、多跳整合）：这是我们认为最有价值的方向，但本轮没有实现。
- **更长语料上的验证**（NarrativeQA / 长文档 / Multi-LoCoMo）：LoCoMo 的结构性天花板使 Read-Time 的价值被压缩。

---

## 10. 复现

```bash
cd /group/40150/licochen/Memory/RSI_GraphRAG
source .venv/bin/activate
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken HF_HUB_OFFLINE=1

python scripts/prepare_data.py
python scripts/run_baselines.py --split dev --out-prefix baselines
python scripts/run_baselines.py --split test --out-prefix baselines
python scripts/run_grouping.py --split dev --budgets 4096,8192
python scripts/run_grouping.py --split test --budgets 8192
python scripts/run_read_time_extract.py --split dev --scopes cluster_hybrid_0.5_b8192,full_conv --modes triple,comp --max-tokens 16384
python scripts/run_read_time_extract.py --split test --scopes cluster_hybrid_0.5_b8192,full_conv --modes triple,comp --max-tokens 16384
python scripts/run_read_time_eval.py --split dev --variants ...
python scripts/run_question_time.py --convs conv-30 --strategies q1,q2,q3
python scripts/run_skill_evolution.py                      # dev 学习 + 冻结
python scripts/run_skill_evolution.py --apply-only --library outputs/skills/frozen_skill_library.json
python scripts/aggregate.py
python scripts/build_final_tables.py
python scripts/make_manifest.py
```

单元测试：`python -m pytest tests -q`（8 项全部通过）。

---

## 11. 文件索引

| 内容 | 路径 |
|---|---|
| **本总报告** | `outputs/reports/MASTER_REPORT.md` |
| 综合实验报告 | `outputs/reports/FINAL_EXPERIMENT_REPORT.md` |
| 自动生成结果表 | `outputs/reports/FINAL_TABLES.md` |
| 统一指标 CSV（27 行） | `outputs/reports/metrics.csv` |
| 验收入口 | `outputs/reports/START_HERE.md` |
| 数据集审计 | `outputs/reports/DATASET_AUDIT.md` |
| 分组报告 | `outputs/reports/GROUPING_REPORT.md` |
| Read-Time 报告 | `outputs/reports/READ_TIME_REPORT.md` |
| Question-Time 报告 | `outputs/reports/QUESTION_TIME_REPORT.md` |
| 联合分析 | `outputs/reports/JOINT_EVALUATION.md` |
| Comprehension 审计 | `outputs/reports/COMPREHENSION_AUDIT.md` |
| 技能库 | `outputs/skills/frozen_skill_library.json` |
| 案例（17 条） | `outputs/reports/cases_conv30_*.json` |
| 运行清单 | `outputs/reports/run_manifest.json` |
| 每题明细 | `outputs/qa/*.jsonl` |
| 图谱与变体 | `outputs/raw_graphs/<conv>/`、`.../variants/` |
| HY4 调用日志 | `outputs/logs/hy4_calls*.jsonl` |
| 聚类结果 | `outputs/groupings/<conv>/*.json` |
| 抽取结果 | `outputs/comprehension/<conv>/*.json` |
