# DetectiveQA Pilot Report

日期：2026-11-11  
状态：Pilot（6 部人工标注小说：27, 79, 117, 118, 121, 124）。所有数字来自 `outputs/detectiveqa/` 下的真实输出文件；未运行的实验明确标注“未运行”。

## 0. 摘要

1. **Pilot 全链路跑通，防泄漏门通过。** 6 部小说建图 2,925,138 token；41 道人工题在 chunk/fact/entity/边权重/重复检索五个维度上 0 失败。
2. **Graph 没有稳定优于 Dense。** qo 条件下 graph MRR 比 dense 低 11.5pp（CI 排除 0），R@10 低 2.9pp；只有 All@5 +7.3pp（边缘）和 q-only R@20 +2.3pp（CI 排除 0）。没有达到“扩大”的门槛。
3. **主要瓶颈在 Reader，不在检索。** dense 的 12 个错误中 9 个是 gold 已到上下文却答错；oracle 仍有 8 题答错。
4. **最近前缀是最强基线。** long_prefix 80.5% = oracle 80.5% > dense 70.7% ≈ graph 68.3%；15 题在没有任何 gold chunk 时答对。
5. **记忆基线很高。** 严格 options-only 29.3%，宽松允许先验知识的 options-only-memory 46.3%（+17.1pp，CI 排除 0）。后续收益必须相对 46.3% 报告。
6. **Comprehension 方向可用但还不稳。** 30 条人工抽查中 19 条是正确的跨块关系（63%），kept 集 14/18（78%）正确、无 claim-as-fact；但抽样中 verifier 误杀 5/19 条好关系，全部 30 个包只产出 39 条 kept。
7. **结论：暂不扩大到 23 部。** 先修 verifier、提高关系产量、把 long-prefix/记忆基线写进主表，再按预注册门槛决定是否扩大。R2–R5 图变体与 Question-Time 闭环未运行。

## 1. 环境与版本

| 项 | 值 |
|---|---|
| 项目路径 | `/group/40150/licochen/Memory/RSI_GraphRAG` |
| Python | 3.10.14 (`/root/venvs/rsigraph`) |
| PyTorch / CUDA | 2.5.1+cu124 / CUDA 12.4 |
| GPU | 4 × NVIDIA H20 102 GB |
| igraph | 0.11.8 |
| transformers / sentence-transformers | 4.45.2 / 3.4.1 |
| Embedding | NV-Embed-v2, `/root/models/NV-Embed-v2`, fp16 |
| LLM | HY4 `no_think`, temperature 0 |
| HippoRAG | `third_party/HippoRAG` 官方实现，OpenIE prompt 与图构建未改 |
| 限流 | `HY4_RATE_FILE=detectiveqa`, `HY4_GLOBAL_QPS=1`（跨进程共享） |

说明：GPU 机器离线，tiktoken 只有 `cl100k_base` 缓存；Reader 的 4096-token 上下文预算用 `cl100k_base` 估算，HY4 实际分词可能略有差异。

## 2. 数据审计

| 项目 | 本次运行 | 预期 |
|---|---:|---:|
| 小说数 | 62 ✅ | 62 |
| 题目数 | 600 ✅ | 600 |
| human_anno | 154 ✅ | 154 |
| AIsup_anno | 446 ✅ | 446 |
| chunk 数 | 14166 ✅ | 14166 |
| eval_human | 154 ✅ | 154 |
| within_practice | 161 ✅ | 161 |
| cross_practice | 208 ✅ | 208 |
| cross_dev | 77 ✅ | 77 |
| unique qids | 600 ✅ | 600 |

- 每题线索中位数 5，线索跨度中位数 645 段，P90 2220 段。
- 切块：目标 200 词、最多 320 词，每个问题 cutoff 前强制断开。每题可见块中位数 170，gold 块中位数 4。
- 有更早同小说练习题的 eval_human 题：112/154。

**发现并修复的数据问题：** `human_anno/149.json` 是小说 117 的第二份英文标注（其 `novel_id` 仍为 117），与 `117.json` 的 5 道题产生 qid 冲突。原 loader 用 `source:novel:index` 生成 qid，`load_prepared` 的 qid-dict 会静默丢掉其中 5 道，导致 pilot 只剩 36 道人工题。现已对后续文件加文件名判别符（如 `human_anno:117:149:00`），保留全部 154 道；600 个 qid 全部唯一，pilot 恢复为 41 道人工题 / 37 道同小说练习题。两份 117 标注是近重复题（题干措辞不同、答案与线索位置相同），报告同时在敏感性分析中给出剔除 5 道重复题的结果。

其他数据检查：0 道题无 gold chunk；0 道题无可见 chunk；原始线索 3,803 条，其中 21 条 ≥ cutoff 被丢弃（与计划一致）；6 个选项带 `200:` 前缀，已按规则清理。

## 3. 实际运行命令

```bash
cd /group/40150/licochen/Memory/RSI_GraphRAG
source .venv/bin/activate
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken HF_HUB_OFFLINE=1
export HY4_RATE_FILE=detectiveqa HY4_GLOBAL_QPS=1
python -m unittest discover -s tests -p test_detectiveqa.py -v
python scripts/detectiveqa/prepare_data.py
# novel 27 先串行跑完；其余 5 部用同一 rate file 并行（--out 避免输出文件竞争）
python scripts/detectiveqa/build_raw_graphs.py --split pilot --novels 27
for n in 79 117 118 121 124; do
  CUDA_VISIBLE_DEVICES=<gpu> python scripts/detectiveqa/build_raw_graphs.py \
    --split pilot --novels $n --out raw_graph_builds_$n.json &
done
python scripts/detectiveqa/graph_build_stats.py --novels 27,79,117,118,121,124
python scripts/detectiveqa/check_prefix_safety.py --split eval_human \
  --novels 27,79,117,118,121,124 --k 20 --out prefix_safety
python scripts/detectiveqa/run_retrieval.py --split eval_human \
  --novels 27,79,117,118,121,124 --query-mode both --top-k 20 \
  --out retrieval_eval_human_raw_both
python scripts/detectiveqa/run_retrieval.py --split eval_human \
  --novels 27,79,117,118,121,124 --query-mode qo --top-k 20 --llm-rerank \
  --out retrieval_eval_human_raw_qo_rerank
python scripts/detectiveqa/run_reader.py --split eval_human \
  --novels 27,79,117,118,121,124 \
  --conditions options_only,options_only_memory,dense,graph,oracle,long_prefix \
  --out reader_pilot
python scripts/detectiveqa/run_readtime_smoke.py --novels 27,79,117,118,121,124
# 分析/报告（全部从上面的输出文件读取）
python scripts/detectiveqa/summarize_retrieval.py --input retrieval_eval_human_raw_both.jsonl
python scripts/detectiveqa/analyze_alias_failures.py --retrieval retrieval_eval_human_raw_both.jsonl
python scripts/detectiveqa/analyze_reader.py --input reader_pilot.jsonl
python scripts/detectiveqa/dump_audit_candidates.py --n 30 --kept-max 18
python scripts/detectiveqa/make_pilot_report.py
```

最小适配改动（不改变官方方法）：
1. `data.py`：修复上述 117 qid 冲突；新增 `unique_qids/duplicate_qids` 审计字段。
2. `build_raw_graphs.py`：新增 `--out`，使 5 部小说可并行建图而不争用同一输出文件。
3. `run_retrieval.py`：新增 `--query-mode {q,qo,both}`、gold chunk 排名、耗时与 fallback 统计。
4. 新增 `check_prefix_safety.py`、`run_reader.py`、`run_readtime_smoke.py` 等实验脚本。

**方法合规**：Dense/HippoRAG 检索、Reader、Read-Time 抽取都只使用问题、选项和 cutoff 前的原文；不读取 `annotator_reasoning`。`clue_paragraphs` 只用于计算检索指标和 `oracle` 诊断条件，不作为任何可部署方法的输入。

## 4. HippoRAG 建图统计与成本

| novel | chunks | passage | phrase | total nodes | extracted triples | total triples | patched chunks | build s | prompt tok | completion tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 27 | 316 | 316 | 5259 | 5575 | 9035 | 26347 | 1 | 1321.1 | 480644 | 207654 |
| 79 | 262 | 262 | 4436 | 4698 | 6206 | 28632 | 0 | 4207.6 | 399841 | 150031 |
| 117 | 187 | 187 | 2946 | 3133 | 4951 | 11977 | 1 | 3730.3 | 282222 | 119172 |
| 118 | 204 | 204 | 2996 | 3200 | 5167 | 14570 | 0 | 3960.4 | 305102 | 117107 |
| 121 | 182 | 182 | 3078 | 3260 | 4918 | 13647 | 0 | 3653.2 | 276290 | 113894 |
| 124 | 230 | 230 | 3433 | 3663 | 5349 | 15313 | 1 | 4139.6 | 346501 | 126680 |
| **合计** | **1381** | | | **23529** | **35626** | **110486** | **3** | **21012.2** | **2090600** | **834538** |

- 总 token：2,925,138 （prompt 2,090,600 + completion 834,538）。`cache_rows` 合计 2762，等于每 chunk 一次 NER + 一次 triple。
- OpenIE 失败：`patched_chunks` 合计 3 个 chunk；失败调用被容错包装为空实体/空三元组，不修改官方 prompt，也不把错误当成功。
- 建图时间包含 NV-Embed 编码；5 部小说并行共享 1 QPS 配额，单部耗时因此包含等待时间。

## 5. Prefix-safe 防泄漏检查

对 41 道 pilot eval_human 题逐题重放，独立重算可见 passage/entity、fact provenance 与边权重：

| 检查 | 结果 |
|---|---|
| 检索 chunk 全部 `end_pid < cutoff` | 通过 |
| top facts 至少有一个可见来源 | 通过 |
| PPR 子图无未来 passage node | 通过 |
| 未来-only entity 不进入子图 | 通过 |
| 边权重只由可见来源重算 | 通过 |
| 同 cutoff 重复检索顺序稳定 | 通过 |

- 失败题数：0 / 41；失败类型：无。
- 检索模式：{'prefix_ppr': 41}；空检索：0。
- 1 道题存在“同一 fact 在 cutoff 后再次出现”的情况。这不是泄漏：该 fact 的文本已由 cutoff 前的 chunk 完整支持，且边权重只累加可见来源；检查器逐边验证了这一点。
- 重复检索的 PPR 分数存在 ~1e-15 量级浮点差（igraph prpack 求和顺序），但 chunk 顺序与 fact 集完全一致，因此判定为数值稳定、非信息泄漏。

## 6. Dense vs HippoRAG 检索结果

四种条件：`q_dense`（仅问题）、`q_graph`（HippoRAG2 prefix-safe，仅问题）、`qo_dense`（问题+选项）、`qo_graph`（HippoRAG2 prefix-safe，问题+选项）。默认关闭 LLM fact reranker。

| condition | Any@5 | Any@10 | Any@20 | R@5 | R@10 | R@20 | All@5 | All@10 | All@20 | MRR | Hit@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q_dense | 0.610 | 0.756 | 0.854 | 0.307 | 0.418 | 0.515 | 0.098 | 0.146 | 0.220 | 0.543 | 0.854 |
| q_graph | 0.658 | 0.780 | 0.878 | 0.262 | 0.425 | 0.538 | 0.024 | 0.146 | 0.244 | 0.496 | 0.878 |
| qo_dense | 0.707 | 0.732 | 0.829 | 0.292 | 0.436 | 0.527 | 0.024 | 0.220 | 0.244 | 0.599 | 0.829 |
| qo_graph | 0.658 | 0.780 | 0.878 | 0.311 | 0.407 | 0.537 | 0.098 | 0.195 | 0.244 | 0.484 | 0.878 |

MRR 按全部题目计算，未召回的题记 0。

检索耗时（每题，CPU 端 PPR + 已缓存 embedding）：qo dense 0.096s，qo graph 0.441s；graph 无 LLM 调用，全部 `prefix_ppr`，dense fallback = 0.0%。官方 LLM reranker 每题约 4.4s（见下表）。

完整表（含去重敏感性、逐小说与排名案例）见 `outputs/detectiveqa/retrieval_summary.md`：

# DetectiveQA retrieval summary

Input: `retrieval_eval_human_raw_both.jsonl`

All metrics are over all questions; MRR counts a missed gold chunk as 0.

### All pilot human questions (n=41)

| method | Any@5 | Any@10 | Any@20 | R@5 | R@10 | R@20 | All@5 | All@10 | All@20 | MRR | Hit@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q_dense | 0.610 | 0.756 | 0.854 | 0.307 | 0.418 | 0.515 | 0.098 | 0.146 | 0.220 | 0.543 | 0.854 |
| q_graph | 0.658 | 0.780 | 0.878 | 0.262 | 0.425 | 0.538 | 0.024 | 0.146 | 0.244 | 0.496 | 0.878 |
| qo_dense | 0.707 | 0.732 | 0.829 | 0.292 | 0.436 | 0.527 | 0.024 | 0.220 | 0.244 | 0.599 | 0.829 |
| qo_graph | 0.658 | 0.780 | 0.878 | 0.311 | 0.407 | 0.537 | 0.098 | 0.195 | 0.244 | 0.484 | 0.878 |

### Sensitivity: drop the five duplicate novel-117 annotations (n=36)

| method | Any@5 | Any@10 | Any@20 | R@5 | R@10 | R@20 | All@5 | All@10 | All@20 | MRR | Hit@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q_dense | 0.611 | 0.750 | 0.861 | 0.325 | 0.433 | 0.534 | 0.111 | 0.167 | 0.250 | 0.531 | 0.861 |
| q_graph | 0.639 | 0.750 | 0.861 | 0.273 | 0.440 | 0.554 | 0.028 | 0.167 | 0.278 | 0.486 | 0.861 |
| qo_dense | 0.722 | 0.750 | 0.833 | 0.314 | 0.467 | 0.554 | 0.028 | 0.250 | 0.278 | 0.597 | 0.833 |
| qo_graph | 0.639 | 0.778 | 0.861 | 0.337 | 0.439 | 0.559 | 0.111 | 0.222 | 0.278 | 0.479 | 0.861 |

### Dense fallback (graph mode)

| condition | fallback ratio |
|---|---:|
| q_graph | 0.000 |
| qo_graph | 0.000 |

### Per-novel Recall@10 (qo)

| novel | n | dense R@10 | graph R@10 | dense MRR | graph MRR |
|---|---:|---:|---:|---:|---:|
| 27 | 7 | 0.399 | 0.363 | 0.429 | 0.437 |
| 79 | 5 | 0.230 | 0.247 | 0.443 | 0.385 |
| 117 | 10 | 0.219 | 0.173 | 0.633 | 0.509 |
| 118 | 7 | 0.524 | 0.524 | 0.576 | 0.517 |
| 121 | 6 | 0.486 | 0.479 | 0.498 | 0.361 |
| 124 | 6 | 0.861 | 0.778 | 1.000 | 0.667 |

### Rank cases (qo): graph better 6, dense better 17

**graph_better**
- human_anno:27:00: dense rank 2, graph rank 1, gold 4
- human_anno:27:06: dense rank 0, graph rank 17, gold 15
- human_anno:79:01: dense rank 11, graph rank 1, gold 3
- human_anno:117:149:01: dense rank 0, graph rank 5, gold 4
- human_anno:118:03: dense rank 2, graph rank 1, gold 6
- human_anno:121:02: dense rank 11, graph rank 9, gold 6

**dense_better**
- human_anno:27:05: dense rank 1, graph rank 2, gold 8
- human_anno:79:02: dense rank 1, graph rank 3, gold 4
- human_anno:79:03: dense rank 8, graph rank 11, gold 4
- human_anno:79:04: dense rank 1, graph rank 2, gold 5
- human_anno:117:02: dense rank 4, graph rank 7, gold 6
- human_anno:117:04: dense rank 1, graph rank 3, gold 12
- human_anno:117:149:04: dense rank 1, graph rank 3, gold 12
- human_anno:118:00: dense rank 3, graph rank 7, gold 4


### 官方 LLM fact reranker 对照（qo，41 题）

| 指标 | graph, no rerank | graph, official rerank | Δ |
|---|---:|---:|---:|
| any@5 | 0.658 | 0.658 | +0.000 |
| any@10 | 0.780 | 0.780 | +0.000 |
| any@20 | 0.878 | 0.878 | +0.000 |
| recall@5 | 0.311 | 0.305 | -0.006 |
| recall@10 | 0.407 | 0.415 | +0.007 |
| recall@20 | 0.537 | 0.531 | -0.006 |
| all@5 | 0.098 | 0.098 | +0.000 |
| all@10 | 0.195 | 0.195 | +0.000 |
| all@20 | 0.244 | 0.244 | +0.000 |
| gold_rank_mrr | 0.484 | 0.526 | +0.042 |
| gold_rank_at_20 | 0.878 | 0.878 | +0.000 |
| fallback_ratio | 0.000 | 0.049 | +0.049 |
| seconds_mean | 0.441 | 4.427 | +3.986 |

官方 reranker 提高 MRR，但略降 Recall@5/20，并在 4.9% 的题上把 fact 过滤为空触发 dense fallback；每题多一次 LLM 调用。它没有把 graph 的 top-1 拉回到 dense 的水平。


## 7. question-only vs question+options 对照

| 指标 | dense q | dense q+o | Δ | graph q | graph q+o | Δ |
|---|---:|---:|---:|---:|---:|---:|
| recall@5 | 0.307 | 0.292 | -0.015 | 0.262 | 0.311 | +0.050 |
| recall@10 | 0.418 | 0.436 | +0.018 | 0.425 | 0.407 | -0.018 |
| recall@20 | 0.515 | 0.527 | +0.012 | 0.538 | 0.537 | -0.002 |
| all@10 | 0.146 | 0.220 | +0.073 | 0.146 | 0.195 | +0.049 |
| mrr | 0.543 | 0.599 | +0.056 | 0.496 | 0.484 | -0.011 |

结论按上表读数给出；任何差异都要结合 n=41 的噪声带（1 题 ≈ 2.4pp）解读。

## 8. Reader QA 对照

统一 Reader：同一 HY4 模型、temperature 0、同一 4096-token 上下文预算、同一按故事顺序渲染方式、同一答案解析。

| 条件 | n | Accuracy | 无效答案率 | insufficient | Evidence Any | Evidence Recall | 引用有效率 | 平均 prompt tok | 总 tok | 平均耗时 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| options_only | 41 | 0.293 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 198.5 | 10647 | 5.09 |
| options_only_memory | 41 | 0.463 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 205.5 | 11155 | 5.34 |
| dense | 41 | 0.707 | 0.000 | 0.195 | 0.634 | 0.261 | 0.976 | 4248.5 | 178387 | 6.26 |
| graph | 41 | 0.683 | 0.000 | 0.195 | 0.683 | 0.273 | 1.000 | 4227.0 | 177487 | 6.24 |
| oracle | 41 | 0.805 | 0.000 | 0.512 | 0.976 | 0.509 | 0.976 | 2300.3 | 98478 | 6.62 |
| long_prefix | 41 | 0.805 | 0.000 | 0.317 | 0.415 | 0.193 | 0.976 | 4136.9 | 173716 | 5.93 |

**失败分解（gold 是否进入 Reader 上下文）**

| 条件 | n | 准确率 | gold 到达 | 到达但仍错 | 未到达 | 未到达且错 | insufficient |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 41 | 0.707 | 30 | 9 | 11 | 3 | 8 |
| graph | 41 | 0.683 | 32 | 10 | 9 | 3 | 8 |
| oracle | 41 | 0.805 | 41 | 8 | 0 | 0 | 21 |
| long_prefix | 41 | 0.805 | 21 | 3 | 20 | 5 | 13 |

- dense 的 12 个错误里 9 个是“gold 已到达但答错”，只有 3 个是未到达；graph 的 13 个错误里 10 个是到达但答错。因此本 pilot 的主要损失在 Reader，不在检索。
- oracle 的 41 题全部有 gold，但仍错 8 题；这 8 题里包含 27:00（gold 集漏掉决定性句）、118:06（角色归属）、124:05（Mr./Mrs. Brand 指代）——oracle 不是严格上界。
- long_prefix 只有 21/41 题 gold 到达，却答对 33/41；其中 15 题是在没有任何 gold chunk 的情况下答对，说明答案证据经常在最近前缀里，而 gold clue 没标到它。

**配对检验（41 题，bootstrap 5000）**

| 对比 | Δ accuracy | 95% CI | 胜/负题数 |
|---|---:|---|---:|
| options_only → options_only_memory | +17.1pp | [+2.4, +31.7] | 9 / 2 |
| dense → graph | −2.4pp | [−12.2, +7.3] | 2 / 3 |
| dense → oracle | +9.8pp | [−7.3, +26.8] | 8 / 4 |
| dense → long_prefix | +9.8pp | [−4.9, +24.4] | 7 / 3 |
| graph → long_prefix | +12.2pp | [−2.4, +26.8] | 8 / 3 |

n=41（去重 36）时 1 题 = 2.4pp；除记忆基线外，所有 reader 差异的 CI 都跨 0，只能作为方向性证据，不能宣称某个上下文条件显著更好。

**117 近重复题一致性**：dense/graph 在 5 对题上给出 5/5 相同答案、4/5 同时正确；oracle 4/5 相同；long_prefix 5/5 相同且 5/5 同时正确；宽松记忆基线只有 1/5 相同。说明检索/Reader 对题干译名变体是稳定的，记忆基线不稳定。

- `options_only` 是严格的参数记忆/选项偏差基线：沿用同一 Reader system prompt（禁止使用书中记忆）。
- `options_only_memory` 是诊断上界：无原文，但 system prompt 明确允许使用对小说的先验知识；它只用于估计“模型是否记住了这本书”，不作为方法。
- `oracle` 只给标注线索所在的可见 chunk，是诊断上限，不是可部署方法。
- `long_prefix` 保留 cutoff 前最近的 4096 token（先丢更早文本），并明确记录截断方式。
- Evidence Any/Recall 按 Reader 自报 `evidence_ids` 与 gold chunk 的交集计算；引用有效率是自报 id 中确实出现在上下文里的比例。

## 9. Read-Time prompt smoke test 与人工抽查

- Prompt 版本：`detectiveqa-v2`。
- Prompt 迭代：v1 的 Joint OpenIE 在 6-chunk 包上会生成 150+ facts，4k 输出被截断、8k 仍可截断；v2 把 Joint 控制臂限制为最多 40 facts（Comprehension prompt 不变），之后所有 30 个包都正常完成（无 joint/comprehension/verifier 错误）。这是控制臂的可用性修复，不是对官方 HippoRAG OpenIE 的修改。
- 阅读包数：30（每部小说最多 5 个，每包最多 6 个 chunk）。
- Joint OpenIE facts：1345；Comprehension 原始候选：280；规则校验通过：251；LLM verifier 保留：39。
- 错误：joint 0，comprehension 0，verifier 0。
- 漏斗：原始候选 280 → 规则校验通过 251 → verifier 保留 39。规则拒绝 29 条（`needs_two_chunks` 22、`quote_not_found` 7）；verifier 拒绝 212 条，理由集中在“关系不被引用支持/谓词或宾语太泛/引用与事实不匹配/status 标错”。完整逐条理由见 `outputs/detectiveqa/readtime_smoke_summary.json`。
- 重复/事实改写检查：规则层用 normalized `(subject, predicate, object)` 与同 packet 的 Joint facts 精确匹配，匹配则拒绝（`duplicate_fact`）；“普通事实改写”由 verifier 的规则和人工抽查判断，没有做 embedding 相似度去重。
- 保留关系统计：`{"n": 39, "types": {"method_or_capability": 8, "alibi_or_opportunity": 3, "timeline_or_causal": 5, "alias_or_identity": 8, "state_change": 6, "contradiction_or_deception": 4, "motive_or_interest": 5}, "status": {"supported": 17, "claimed": 10, "suspected": 12}, "median_chunk_gap": 68}`。

**人工抽查：30 条候选**（抽样框：18 条 verifier-kept + 12 条 verifier-dropped；覆盖 6 部小说；不是只挑成功案例）。分类定义：

- `correct_cross_chunk`：关系为真，且确实需要组合 ≥2 个 chunk；
- `correct_but_fact_rewrite`：关系为真，但只是单 chunk 原子事实的改写/拼接；
- `insufficient_evidence`：引用证据不足以支持结论；
- `entity_alignment_error`：实体/别名/说话人错配；
- `claim_as_fact`：把证词/怀疑当事实；本样本 0 条；
- `too_generic`：关系太泛，无法检索；本样本 0 条。

| 标签 | 数量 | 占比 |
|---|---:|---:|
| correct_cross_chunk | 19 | 63% |
| correct_but_fact_rewrite | 5 | 17% |
| insufficient_evidence | 5 | 17% |
| entity_alignment_error | 1 | 3% |

- verifier 保留集的正确跨块率：14/18（78%）；4 条是普通事实改写，没有错误关系。
- verifier 拒绝集里仍有 5/12 条正确跨块关系被误杀（a21、a22、a25、a26、a28）；verifier 偏保守，且没有利用它自己给出的 `fixed_status`。

**代表案例（真实输出）**

- ✅ a14（kept，117:004）：尸体戴绿色纸帽，而衣柜里只有猩红/淡黄两顶纸帽 → 真实物证矛盾，必须跨 chunk。
- ✅ a06（kept，79:004）：Hayama 手里的原钥匙 + 从外面用 Ito 的钥匙锁门 → 密室手法链，retrieval 价值高。
- ✅ a07（kept，117:000）：Linda 自己提供时间证据 + 首次审讯态度可疑 → alibi/opportunity 关系。
- ⚠️ a21/a22（verifier 拒绝，27:000）：Van 声称 Clinton 已死 + 后文证明 Van 是骗子/最后一具尸体是 Clinton → 内容正确，只是 object 措辞模糊或 status 应改为 supported；verifier 应保留并修 status。
- ⚠️ a25（verifier 拒绝，27:001）：Aroo/Arroyo 是同一村庄的译名变体 → 正确的别名关系，但 status 应为 suspected。
- ❌ a18（拒绝）：Clinton 力气大 ≠ 能把尸体摆成特定姿势 → 证据不足，verifier 正确。
- ❌ a29（拒绝）：原文是 Ellery 说“第一个男人是凶手”，不是绷带男自称凶手 → 实体对齐错误，verifier 正确。

**判定**：Comprehension 方向可用——kept 集 78% 是正确跨块关系，且没有 claim-as-fact；但在 30 条抽查样本内，verifier 误杀了 5/19 条好关系，39 条 kept / 30 packets 的数量也太稀疏。因此本报告**不进入第五阶段图变体对照**；先把 verifier 校准（修 status、放宽别名/说话人证据）和阅读包覆盖做上去，再测 R2–R5。

## 10. 已完成与未完成实验

| 实验 | 状态 |
|---|---|
| 6 部小说 raw HippoRAG2 图 | 已完成（见第 4 节） |
| Prefix-safe 泄漏检查（6 部） | 已完成（见第 5 节） |
| Dense vs Graph × question/question+options 检索 | 已完成（见第 6–7 节） |
| 官方 LLM fact reranker 对照 | 已完成（qo，41 题，见第 6 节） |
| options-only（严格 + 宽松记忆）/ dense / graph / oracle / long-prefix Reader | 已完成（见第 8 节） |
| Read-Time smoke test（30 packets） | 已完成（见第 9 节） |
| 人工抽查 ≥30 条 comprehension 候选 | 已完成（30 条，见第 9 节） |
| R2–R5 图变体与检索/QA 对照 | 未运行：关系质量可用但 verifier 误杀、kept 数量仅 39/30 packets，未达进入门槛 |
| Question-Time 自进化闭环 | 未运行（按计划在 pilot 稳定前不启动） |
| 扩大到 23 部 human 小说 | 未运行（本报告第 12 节给建议） |

R0（Dense）与 R1（官方 HippoRAG2）已由第 6–8 节的 dense/graph 条件覆盖；R2–R5 需要构建图变体，按第 9 节的质量判定未运行。

### 10.1 Token 与调用成本（来自实际输出/log）

| 阶段 | 调用数 | prompt tok | completion tok | total tok |
|---|---:|---:|---:|---:|
| HippoRAG 建图（6 部） | 2762 | 2,090,600 | 834,538 | 2,925,138 |
| Reader（6 条件 × 41 题） | 245 | 623,637 | 21,779 | 645,416 |
| Read-Time smoke（最终 30 packets） | 88 | 369,819 | 122,098 | 491,917 |

注：Read-Time 的 HY4 日志合计 100 次调用 / 565,717 token，包含 v1 截断和 v2 调参的开发开销；上表只列最终成功运行的 30 packets（88 次调用 / 491,917 token）。官方 reranker 的 41 次调用写入各图自己的 `llm_cache`，未单独拆账。

## 11. 关键失败案例

### 11.1 检索层：gold clue 未召回，但答案证据已经被召回

- **27:06 "Who is the real killer?"**（15 个 gold chunk）：qo dense 完全未召回，qo graph 只把 1 个 gold 排到 17。但 graph top-1 `dq-27-00248` 直接写 "The killer is Vilga Crocet..."，对应选项 A "Wileg Crocetta"；dense/graph/oracle/long_prefix 四种条件都答对 A。gold clue 是标注者的推理链（15 段），不是直接给出答案的句子，因此 gold rank 0 没有阻止答题。
- **117:01 / 117:149:01**（同一题的两种英文写法）：dense 两版都未召回 gold；graph 在 "Christen" 版排到 5，"Kristen" 版未召回。graph top-1 `dq-117-00172` 是 "Kristen ... tamper with Linda's watch"，对应选项 B；但该题 gold 选 A（恐高/木板），四种 Reader 条件都答对 A。这说明题目可能有多条“真”选项（watch 也是另一条谎言），属于标注歧义/多真选项，不是检索失败。
- 其余 dense miss：27:02、27:04、79:00、118:06，均为多证据/跨段落题。逐题证据见 `outputs/detectiveqa/retrieval_failure_cases.json`。

**含义**：DetectiveQA 的 gold clue 更接近“标注者认为相关的段落”，不是“最小充分证据”。只用 gold-chunk Recall 会低估已经拿到答案证据的方法；需要同时报告 answer-evidence hit 或人工复核。

### 11.2 Reader：gold 已到上下文，仍然答错

- qo dense：41 题中 30 题 gold 进入上下文，其中 **9 题仍答错**；只有 3 题是“gold 未到且答错”。qo graph：32 题 gold 到达，**10 题仍答错**。错误主要不在检索。
- **27:00 oracle**：gold 4 个 chunk，graph/oracle 选 D；dense 和 long_prefix 上下文额外包含 `dq-27-00015`（"ordering caviar!"，说明 Andrew 曾有更富裕的生活），因此选对 B。gold 集漏掉了决定性的“鱼子酱”句 → oracle 不是严格上界。
- **118:06 oracle**：oracle 选 C（Francis），dense/graph/long_prefix 都选 A（Rolly）并答对。**124:05**：四种条件都选 B（Mrs. Brand），gold 是 A（Mr. Brand）——这是系统性的角色归属/标注歧义，不是检索缺失。这两题说明：gold clue 不足时 oracle 会错；而角色指代歧义会让所有 reader 一起错。

### 11.3 记忆与选项偏差（本 pilot 最大的混淆）

- 严格 `options_only`（同一 system prompt，禁止用书中记忆）：29.3%；宽松 `options_only_memory`（无原文，但允许先验知识）：**46.3%**（19/41），配对差 +17.1pp，CI [+2.4, +31.7]。
- 宽松记忆基线答对了 27:06（真凶）、117:04（Alyona 的真凶）、118:00、118:02 等题。这些小说（Agatha Christie / Ellery Queen 等）明显被模型记住了一部分。后续任何方法的收益必须相对 46.3% 报告，而不是相对 25% 随机。

### 11.4 译名与实体对齐

- 41 题中 17 题至少有一个题干/选项名字在可见原文中完全找不到，25 题有近似变体；例如 117 的 Kristen / Christen / Christine / Kristine，以及 Aroo / Arroyo。
- 但 7 个 dense miss 中 **0 个**、5 个 graph miss 中 **0 个** 含 absent name；译名不一致很普遍，却不是本 pilot 检索失败的主因。它更多影响关系抽取（a25 是正确别名但被 verifier 拒绝；a29 是说话人错配）。

### 11.5 工程/方法事故（已修）

- Joint OpenIE 控制臂在 6-chunk 包上生成 150+ facts，4k 输出被截断，8k 仍可能截断；第一轮 smoke test 的 joint facts 为 0。修复：v2 prompt 限制最多 40 facts。如果只看“comprehension 有输出”就会漏掉控制臂为空，这是必须保留的对照检查。
- Verifier 误杀：30 条抽查中 5/19 条正确跨块关系被拒绝（a21/a22/a25/a26/a28），且没有使用自己输出的 `fixed_status`。这说明 verifier 的“拒绝”不等于“关系错误”。

### 11.6 关键正向信号：最近前缀基线

- `long_prefix`（cutoff 前最近 4096 token）与 oracle 同为 80.5%，高于 dense 70.7% / graph 68.3%；其中 15 题在没有任何 gold chunk 的情况下答对。这说明 DetectiveQA 的答案证据经常就在问题位置附近，而按检索分数选 chunk 反而会把近处证据挤掉。任何 graph/dense 方法都应把“最近前缀”作为必报基线，而不是只和随机/无上下文比。

## 12. 是否值得扩大到 23 部 human 小说

**结论：现在不建议直接扩大。** 不是 pilot 跑不通，而是 pilot 还没有给出“扩大后能回答新问题”的证据。

### 12.1 不支持立即扩大的证据

1. **Graph 没有稳定优于 Dense。** qo 条件下 graph 的标准 MRR（未召回记 0）比 dense 低 11.5pp（0.484 vs 0.599；配对 bootstrap 95% CI [−21.7, −1.3]），R@10 低 2.9pp（CI 跨 0），All@20 完全相同（0.244）；只有 All@5 +7.3pp（CI [0.0, +17.1]，边缘）和 Any@20 +4.9pp（边缘）。q-only 条件下 graph 的 R@20 +2.3pp（CI [+0.6, +4.9]）是唯一稳定的正向，但 R@5 −4.5pp（CI 跨 0）。按 pilot 计划写的门槛（Recall@10 或 All@20 有明确差异），**没有达到**。
2. **Reader 是主要瓶颈。** dense 的 12 个错误中 9 个是“gold 已到上下文但答错”，只有 3 个是未到；oracle（只给 gold clue）仍有 8 题答错。扩大检索不太可能修复这部分。
3. **最近前缀基线更强。** long_prefix 80.5% = oracle 80.5% > dense 70.7% ≈ graph 68.3%。如果扩大后 graph 仍低于“直接读最近 4096 token”，扩大只会更贵，不会更有解释力。
4. **记忆基线太高。** 宽松 options-only 46.3%、严格 29.3%。在 41 题上，dense−memory = 24.4pp、graph−memory = 22.0pp；扩大前必须先决定如何控制“模型记得这本书”，否则 23 部小说上的 QA 差异无法归因。
5. **Comprehension 数量太稀疏、verifier 偏保守。** kept 39 条 / 30 packets；抽查 19/30 是正确跨块（63%），kept 集 14/18（78%）正确，但抽查样本内 verifier 误杀 5/19 条好关系。这个规模不足以让 R3 在 41 题上产生可检测的检索差异。
6. **Gold clue 不是最小充分证据。** 27:00、27:06、117:01 都出现“gold 未召回但答案证据已召回”。扩大到 23 部会把这种度量偏差一起放大；需要先加 answer-evidence hit 或人工复核口径。

### 12.2 扩大前应做的四件事（按优先级）

1. **修 verifier 并复测 30 条。** 使用 `fixed_status`；对“说话人明确声称的别名/身份”允许 claimed；对译名变体（Aroo/Arroyo、Kristen/Christen）允许 suspected。目标是 kept 集 ≥80% 正确、且不再误杀 ≥5/19 的好关系。
2. **提高阅读包覆盖与关系产量。** 当前每部 5 个包、每包 5-6 chunk，30 包只产出 39 条 kept。先在同一 6 部小说上把包数/实体覆盖提高，直到 kept ≥100 条，再跑 R2 vs R3 vs R4。
3. **把最近前缀和记忆基线写进主表。** 每个 reader 实验同时报 options_only、options_only_memory、long_prefix；否则无法区分“方法有效”和“模型记得/近处就有答案”。
4. **固定扩大门槛（预注册）。** 建议：graph 相对 dense 在 R@10 和 All@20 上的配对 95% CI 均排除 0；oracle 至少高于 dense 5pp；comprehension kept 精度 ≥80% 且 ≥100 条；reader 错误中“reached-but-wrong”仍占多数。四项同时满足再扩到 23 部。

### 12.3 成本估计

Pilot 建图 2.93M token（6 部 / 1,381 chunk），Reader 0.65M token，Read-Time 0.57M token。按 chunk 数线性外推，23 部 human 小说约 4-5 倍，即建图 ~12-15M token；加上阅读包、verifier、reader 对照，完整一轮约 18-22M token。在 1 QPS 共享配额下，这不是一次可以盲目启动的开销；应先完成 12.2 的 1-3 项。

## 附录 A：逐题 gold chunk 排名与 top-5 PPR facts

### A.1 每题 gold chunk 排名（qo；0 = 未进 top-20）

| qid | n_gold | dense gold ranks | graph gold ranks |
|---|---:|---|---|
| human_anno:27:02 | 3 | 0, 0, 0 (hits 0/3) | 0, 0, 0 (hits 0/3) |
| human_anno:27:00 | 4 | 3, 7, 2, 5 (hits 4/4) | 1, 9, 5, 3 (hits 4/4) |
| human_anno:27:01 | 4 | 9, 8, 0, 2 (hits 3/4) | 9, 12, 0, 2 (hits 3/4) |
| human_anno:27:05 | 8 | 0, 2, 0, 0, 3, 1, 0, 0 (hits 3/8) | 0, 4, 0, 0, 2, 3, 19, 0 (hits 4/8) |
| human_anno:27:04 | 7 | 0, 0, 0, 0, 0, 0, 0 (hits 0/7) | 0, 0, 0, 0, 0, 0, 0 (hits 0/7) |
| human_anno:27:03 | 6 | 11, 1, 9, 0, 2, 3 (hits 5/6) | 1, 3, 12, 0, 4, 2 (hits 5/6) |
| human_anno:27:06 | 15 | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... (hits 0/15) | 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, ... (hits 1/15) |
| human_anno:79:01 | 3 | 11, 0, 13 (hits 2/3) | 1, 0, 15 (hits 2/3) |
| human_anno:79:00 | 3 | 0, 0, 0 (hits 0/3) | 0, 0, 0 (hits 0/3) |
| human_anno:79:02 | 4 | 4, 0, 0, 1 (hits 2/4) | 6, 16, 0, 3 (hits 3/4) |
| human_anno:79:03 | 4 | 8, 0, 0, 0 (hits 1/4) | 11, 0, 0, 0 (hits 1/4) |
| human_anno:79:04 | 5 | 0, 10, 0, 1, 0 (hits 2/5) | 0, 2, 0, 3, 0 (hits 2/5) |
| human_anno:117:00 | 7 | 18, 6, 3, 11, 0, 0, 1 (hits 5/7) | 0, 15, 11, 10, 0, 0, 1 (hits 4/7) |
| human_anno:117:149:00 | 7 | 0, 8, 4, 7, 0, 0, 1 (hits 4/7) | 0, 15, 13, 9, 0, 0, 1 (hits 4/7) |
| human_anno:117:02 | 6 | 4, 19, 0, 0, 0, 0 (hits 2/6) | 7, 17, 0, 0, 0, 0 (hits 2/6) |
| human_anno:117:149:02 | 6 | 12, 15, 0, 0, 0, 0 (hits 2/6) | 13, 12, 0, 0, 0, 0 (hits 2/6) |
| human_anno:117:03 | 7 | 20, 11, 9, 4, 0, 0, 1 (hits 5/7) | 0, 14, 11, 9, 0, 0, 1 (hits 4/7) |
| human_anno:117:149:03 | 7 | 0, 13, 10, 4, 0, 0, 1 (hits 4/7) | 0, 15, 12, 9, 0, 0, 1 (hits 4/7) |
| human_anno:117:04 | 12 | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... (hits 2/12) | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... (hits 2/12) |
| human_anno:117:149:04 | 12 | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... (hits 2/12) | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... (hits 2/12) |
| human_anno:117:01 | 4 | 0, 0, 0, 0 (hits 0/4) | 0, 0, 0, 0 (hits 0/4) |
| human_anno:117:149:01 | 4 | 0, 0, 0, 0 (hits 0/4) | 5, 0, 0, 0 (hits 1/4) |
| human_anno:118:01 | 2 | 8, 1 (hits 2/2) | 1, 2 (hits 2/2) |
| human_anno:118:02 | 3 | 6, 7, 1 (hits 3/3) | 1, 4, 2 (hits 3/3) |
| human_anno:118:00 | 4 | 7, 8, 13, 3 (hits 4/4) | 9, 10, 11, 7 (hits 4/4) |
| human_anno:118:03 | 6 | 0, 0, 0, 0, 0, 2 (hits 1/6) | 0, 12, 14, 0, 0, 1 (hits 3/6) |
| human_anno:118:06 | 9 | 0, 0, 0, 0, 0, 0, 0, 0, 0 (hits 0/9) | 0, 0, 0, 0, 0, 0, 0, 0, 0 (hits 0/9) |
| human_anno:118:04 | 4 | 0, 0, 5, 0 (hits 1/4) | 0, 0, 7, 0 (hits 1/4) |
| human_anno:118:05 | 2 | 0, 1 (hits 1/2) | 0, 3 (hits 1/2) |
| human_anno:121:00 | 6 | 3, 5, 0, 7, 8, 0 (hits 4/6) | 6, 10, 0, 13, 15, 0 (hits 4/6) |
| human_anno:121:01 | 4 | 6, 9, 5, 2 (hits 4/4) | 7, 4, 3, 2 (hits 4/4) |
| human_anno:121:04 | 4 | 6, 2, 1, 5 (hits 4/4) | 1, 4, 2, 8 (hits 4/4) |
| human_anno:121:03 | 8 | 19, 0, 0, 0, 0, 1, 4, 20 (hits 4/8) | 3, 0, 0, 0, 0, 4, 8, 17 (hits 4/8) |
| human_anno:121:02 | 6 | 0, 11, 0, 0, 0, 0 (hits 1/6) | 0, 9, 0, 0, 0, 0 (hits 1/6) |
| human_anno:121:05 | 3 | 15, 0, 19 (hits 2/3) | 0, 0, 19 (hits 1/3) |
| human_anno:124:00 | 3 | 5, 6, 1 (hits 3/3) | 3, 2, 4 (hits 3/3) |
| human_anno:124:03 | 3 | 1, 3, 7 (hits 3/3) | 2, 4, 8 (hits 3/3) |
| human_anno:124:02 | 2 | 3, 1 (hits 2/2) | 1, 4 (hits 2/2) |
| human_anno:124:04 | 4 | 1, 2, 6, 4 (hits 4/4) | 2, 4, 11, 13 (hits 4/4) |
| human_anno:124:01 | 3 | 2, 1, 0 (hits 2/3) | 3, 1, 0 (hits 2/3) |
| human_anno:124:05 | 6 | 13, 8, 1, 0, 0, 3 (hits 4/6) | 13, 4, 2, 0, 0, 5 (hits 4/6) |

完整逐 chunk 排名见 `retrieval_eval_human_raw_both.jsonl` 的 `qo_dense_gold_chunk_ranks` / `qo_graph_gold_chunk_ranks`。

### A.2 top-5 PPR fact seeds（选中案例）

**human_anno:27:06** — qo dense gold rank 0, qo graph gold rank 17

- `['ellery', 'identified killer as', 'vilga crocetta']`
- `['crosetti', 'is likely killer of', 'vane']`
- `['crosett', 'thought he killed', 'andrew van']`
- `['ellery', 'confirmed that', 'croscos killed clinton']`
- `['crosett', 'had motive and desire to kill', 'andrew van']`

**human_anno:117:149:01** — qo dense gold rank 0, qo graph gold rank 5

- `['kristen', 'tampered with', 'linda s watch']`
- `['christin', 'had opportunity to tamper with', 'linda marshall s watch']`
- `['christine', 'has', 'alibi']`
- `['poirot', 'said', 'christine redfern s testimony gives linda marshall an alibi']`
- `['christine', 'lied about', 'acrophobia']`

**human_anno:117:01** — qo dense gold rank 0, qo graph gold rank 0

- `['kristen', 'tampered with', 'linda s watch']`
- `['kristen', 'had alibi that', 'kristen s hands were too small to have the strength to commit the crime']`
- `['kristen redfern', 'pretended to be', 'elena']`
- `['kristen', 'planned with crime partners to frame', 'kenneth marshall']`
- `['christine', 'has', 'alibi']`

**human_anno:27:00** — qo dense gold rank 2, qo graph gold rank 1

- `['andrew van', 'lived on', 'arroyo street']`
- `['andrew van', 'was considered model teacher by villagers of', 'arroyo']`
- `['van', 'had rural life in', 'arroyo']`
- `['andrew van', 'had contributions deemed satisfactory by', 'arroyo village board']`
- `['andrew', 'stayed in', 'arroyo']`

**human_anno:124:05** — qo dense gold rank 1, qo graph gold rank 2

- `['mrs brand s sister', 'is mastermind behind', 'murder plot']`
- `['mr corry', 'was', 'murdered man']`
- `['miss pemerton', 'may have killed', 'mr corry']`
- `['harry', 'told name of murdered man', 'mr corry']`
- `['mr corry', 'was', 'murdered']`

**human_anno:118:06** — qo dense gold rank 0, qo graph gold rank 0

- `['rolly', 'is', 'the murderer']`
- `['lynn', 'is', 'rolly s victim']`
- `['lynn', 'murmured', 'why did david kill andrae']`
- `['francis', 'felt something was not quite right between', 'rolly and lynn']`
- `['rolly', 'asks who killed', 'andré']`


