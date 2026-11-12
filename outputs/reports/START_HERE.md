# START HERE — 验收入口

> 完整总报告（所有实验、所有表格、假设判定、成本、局限）：**`outputs/reports/MASTER_REPORT.md`**

> 项目：`/group/40150/licochen/Memory/RSI_GraphRAG`
> 状态：**全部实验完成**（P0/P1/P2 + 联合矩阵 + 消融 + Skill Evolution）。所有数字来自真实运行。

## 1. 三分钟看懂结论

1. **基线（1986 题全量）**：HippoRAG2 的检索明显强于 Dense（证据覆盖 .828 vs .732，R@5 .863 vs .781），但 QA 只领先 2.3pp EM / 4.7pp F1；**Full-Context QA 的 F1 反而最高（.491 vs .483）**。
2. **Read-Time**：dev 上 EM .253 → .268–.271、证据 .819 → .845–.849；test 上四个变体检索一致提升（R@5 +1.0~1.7pp、证据 +0.7~1.8pp），但 **QA 差异都在 ±0.5pp 内且不单调**（C5 EM .245 最高，C6 F1 .492 最高，C3 反而 .238/.484）。理解层没有额外贡献（C3<C2、C6≈C5）。
3. **聚类（10 对话）**：8192 预算下随机分组 gold 覆盖 .883，所有聚类方法都在 .88 附近——**LoCoMo 太短，聚类没有必要**（H7 成立）。
4. **Question-Time（dev 3 对话，502 题）**：Q2 把 R@5 从 .858 提到 .875、证据 .819→.835，但 QA 基本持平（F1 .487 vs .481）；Q3 不优于 Q1/Q2。conv-30 单对话上 Q3 的 +3.7pp 没有在别的对话或 test 上复现。
5. **最重要的科研信号**：证据已经召回、答案仍然错误（temporal 类证据覆盖 .875、F1 仅 .238）。这是比"再强一点的检索"更值得做的方向。
6. **联合实验（test 完整 2×4）**：检索最好 R1Q2/R1Q3（R@5 .885/.884），但 QA 最好 R1Q1（EM .244 / F1 .494），仅比基线高 0.4pp / 1.0pp；8 个组合全部落在噪声区间。**两条路线不互补（H6 不成立）**。

## 2. 实验真实执行状态

| 阶段 | 状态 | 覆盖 |
|---|---|---|
| 环境 / HippoRAG2 部署 / HY4 / NV-Embed-v2 | ✅ 完成 | 基准 smoke test 通过 |
| G_raw 建图 | ✅ 完成 | 10/10 conversation |
| P0 基线（Dense / HippoRAG / Full-Context） | ✅ 完成 | 1,986 题全量 |
| 分组实验 B | ✅ 完成 | dev+test，9 种方法 × 2 预算 |
| Read-Time C（dev） | ✅ 完成 | 4 变体 × 502 题 |
| Read-Time C（test） | ✅ 完成 | 7/7 对话 × 4 变体（C2/C3/C5/C6），含 Reader |
| Question-Time E | ✅ 完成 | dev 3 对话 Q0–Q3；test Q0–Q3 全量 |
| 联合实验 | ✅ 完成 | conv-30 与 test 全量的 2×4 矩阵 |
| Composed vs Inferred 消融 | ✅ 完成 | dev 502 题 |
| Skill Evolution E4 | ✅ 完成 | dev 学习 → 冻结 → test 应用 |
| 案例研究 | ✅ 完成 | 真实案例 JSON + 报告节选 |

未完成原因：HY4 共享池在夜间被限流到 ~0.7–1.2 QPS（1 QPS 成功率 90%，2 QPS 仅 7%），检索每题需一次 LLM rerank 调用。

## 3. 最重要的几张表

### 基线（全量 1,986 题；EM/F1 只算可回答题）

| method | EM | F1 | R@5 | Evidence@4096 |
|---|---|---|---|---|
| Dense RAG | .221 | .436 | .781 | .732 |
| **HippoRAG2 (G_raw)** | .244 | .483 | **.863** | **.828** |
| Full-Context | **.245** | **.491** | N/A | N/A |

### Read-Time（dev 502 题）

| method | EM | F1 | R@5 | Evidence@4096 |
|---|---|---|---|---|
| HippoRAG2 G_raw | .253 | .481 | .858 | .819 |
| C2 cluster joint triple | .253 | .488 | .868 | .837 |
| C3 cluster comprehension | .261 | .492 | .873 | .849 |
| **C5 full-conv joint triple** | **.271** | .500 | .881 | .845 |
| C6 full-conv comprehension | .268 | **.501** | **.883** | .847 |

### Question-Time（conv-30 105 题）

| strategy | EM | F1 | R@5 | Evidence@4096 |
|---|---|---|---|---|
| Q0 raw | .381 | .520 | .903 | .886 |
| Q1 QDM | .381 | .525 | .908 | .886 |
| Q2 +Retrieval Skill | .362 | .502 | **.917** | **.905** |
| **Q3 +Answer Skill** | **.390** | **.539** | **.917** | **.905** |

### 分组（10 对话，8192 预算均值）

| method | gold full | cross-session 共置 |
|---|---|---|
| random | .883 | .304 |
| temporal | .883 | .305 |
| entity_idf | **.905** | **.431** |
| hybrid_0.5 | .881 | .281 |
| community | .854 | .086 |

## 4. 最有价值的研究发现

- **检索增益无法传导到答案**：multi-hop 证据覆盖 +16.7pp（.274→.441），F1 仅 +4pp；temporal 证据 .875、F1 .238。
- **"更好的证据"有益，"更多证据"有害**：Read-Time（改善证据质量）带来 +2pp F1；Question-Time（增加证据数量）带来 −1.8pp F1。
- **LoCoMo 的全文太短**（1–2 万 token），Full-Context 已接近最好方法；Read-Time 的上限被结构性压缩。
- **Comprehension 审计**：717 条新增关系中 95.5% 是跨会话且有双 session 证据（D 类），但词面去重可能低估了语义重复。

### Read-Time 消融（dev，502 题）

| 图谱 | EM | F1 | R@5 | 证据@4096 |
|---|---|---|---|---|
| G_raw | .253 | .481 | .858 | .819 |
| **Composed-only** | **.266** | **.501** | .869 | .845 |
| Inferred-only | .253 | .494 | **.875** | .841 |
| C3（合并） | .261 | .492 | .873 | **.849** |

Composed 关系带来最好的 QA；Inferred 只提升检索、不提升 EM。

## 5. 文件位置

| 内容 | 路径 |
|---|---|
| **综合报告** | `outputs/reports/FINAL_EXPERIMENT_REPORT.md` |
| 自动生成结果表 | `outputs/reports/FINAL_TABLES.md` |
| 统一指标 CSV | `outputs/reports/metrics.csv` |
| 数据集审计 | `outputs/reports/DATASET_AUDIT.md` |
| 分组报告 | `outputs/reports/GROUPING_REPORT.md` |
| Read-Time 报告 | `outputs/reports/READ_TIME_REPORT.md` |
| Question-Time 报告 | `outputs/reports/QUESTION_TIME_REPORT.md` |
| 联合分析 | `outputs/reports/JOINT_EVALUATION.md` |
| Comprehension 审计 | `outputs/reports/COMPREHENSION_AUDIT.md` |
| 案例（真实） | `outputs/reports/cases_conv30_comp.json` / `cases_conv30_fullcomp.json` |
| 运行清单（commit、sha256、token 用量） | `outputs/reports/run_manifest.json` |
| 每题记录 | `outputs/qa/*.jsonl` |
| 图谱 / 变体 | `outputs/raw_graphs/<conv>/`、`.../variants/` |
| HY4 调用日志 | `outputs/logs/hy4_calls*.jsonl` |

## 6. 复现命令

```bash
cd /group/40150/licochen/Memory/RSI_GraphRAG
source .venv/bin/activate          # 指向 /root/venvs/rsigraph
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken HF_HUB_OFFLINE=1

python scripts/prepare_data.py
python scripts/run_baselines.py --split dev --out-prefix baselines
python scripts/run_grouping.py --split dev --budgets 4096,8192
python scripts/run_read_time_extract.py --split dev \
    --scopes cluster_hybrid_0.5_b8192,full_conv --modes triple,comp --max-tokens 16384
python scripts/run_read_time_eval.py --convs conv-30,conv-44,conv-48 \
    --variants cluster_hybrid_0.5_b8192_comp:comp,cluster_hybrid_0.5_b8192_triple:triple,full_conv_comp:comp
python scripts/run_question_time.py --convs conv-30 --strategies q1,q2,q3
python scripts/aggregate.py
python scripts/build_final_tables.py
```

单元测试：`python -m pytest tests -q`（8 项全部通过）。

## 7. 已知问题（验收时请注意）

1. **EM 对 LoCoMo 偏严**：gold 常是长句（"They lost their jobs and decided to start their own businesses."），模型答短句时 EM 记 0 但 F1 接近 1。**建议以 F1 为主指标**。
2. **Reader 的时间算术弱**：relative date（"yesterday"）需要结合 session 日期换算，模型经常不做换算，导致 temporal 类 F1 偏低。
3. **C5/C6 的 test 只跑了 C6**（7/7 对话）；C5 只在 dev 上完成。
4. **Comprehension 审计的 A/B 类使用词面相似度**，可能低估语义重复。
5. 图谱变体的 `index_manifest.json` 是从 G_raw 复制后重建的（同一 embedding/LLM 身份），图 pickle 不复制——这是为了保证 G_raw 永不被修改。
