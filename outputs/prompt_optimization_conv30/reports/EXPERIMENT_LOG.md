# conv-30 Prompt 优化实验执行日志

所有命令均在项目根目录 `/group/40150/licochen/Memory/RSI_GraphRAG` 下执行，
Python 使用 `.venv/bin/python`。每个阶段只写 `outputs/prompt_optimization_conv30/` 或
`outputs/raw_graphs/conv-30/variants/cmpo_*`，不修改历史产物。

## Phase 0 — 审计与口径（无 LLM）

```bash
# 0.2 官方 LoCoMo 评分器 vs 项目评分器（只用已有 pred_answer）
python scripts/prompt_opt/00_official_metrics.py

# 0.1 现有 Reader / Comprehension Prompt 的真实渲染与哈希
python scripts/prompt_opt/01_prompt_audit.py

# 0.3 105 题逐题错误归因（A–H），并落盘每题实际 Reader 上下文
python scripts/prompt_opt/02_error_analysis.py
```

产物：`audit/official_metric_comparison.{md,json,csv}`、`audit/prompt_audit.json`、
`audit/reader_prompt_q*.txt`、`audit/comp_prompt_group*.txt`、`audit/error_analysis_conv30.{jsonl,csv}`、
`audit/graw_contexts/qNNN.txt`。

## Phase 1 — Reader Prompt

```bash
# 1.0 固定 dev40 / val65 划分（seed 42）
python scripts/prompt_opt/03_make_splits.py

# 1.1 dev 上跑 P1–P6（P0 复用历史答案；40 题 × 6）
HY4_CACHE_SHARD=po python scripts/prompt_opt/04_reader_pilot.py \
    --split dev --pids P1,P2,P3,P4,P5,P6 --max-workers 6

# 1.1b dev-informed 综合 Prompt P7 与保守变体 P8
HY4_CACHE_SHARD=po python scripts/prompt_opt/04_reader_pilot.py --split dev --pids P7
HY4_CACHE_SHARD=po python scripts/prompt_opt/04_reader_pilot.py --split dev --pids P8

# 1.2 汇总（若脚本在 summary 阶段中断，用这个免 LLM 重算）
python scripts/prompt_opt/04b_summarize.py --subdir pilot --pids P1,P2,P3,P4,P5,P6,P7,P8

# 1.3 在看到 val 结果之前冻结选择（生成 selection.json + prompt 哈希）
#     （由 /tmp 内联脚本完成，内容见 reader/pilot/selection.json）

# 1.4 内部验证集：P7、P8（65 题 × 2）
HY4_CACHE_SHARD=po python scripts/prompt_opt/04_reader_pilot.py \
    --split val --pids P7,P8 --max-workers 6 --out-subdir final
```

## Phase 2 — Reader 诊断

```bash
# 2.2 Oracle Gold Evidence（R2）与 Full Conversation（R3）；R3/P0 复用历史 fullcontext
HY4_CACHE_SHARD=or python scripts/prompt_opt/08_oracle.py --conditions R2,R3 --pids P0,P7
python scripts/prompt_opt/08b_summarize_oracle.py

# 2.4 同输入重复 3 次（use_cache=False，84 次真实调用 + 14 次 429 重试）
HY4_CACHE_SHARD=rp python scripts/prompt_opt/09_repeats.py \
    --pids P0,P7 --qids 4,9,16,17,24,29,31,33,39,47,54,63,74,85 --repeats 3

# 2.5 官方 cat-5 二选一多选协议（评价协议对照，不是新方法）
HY4_CACHE_SHARD=oa python scripts/prompt_opt/13_official_adversarial.py --pids P0,P7
```

## Phase 3 — Comprehension Prompt

```bash
# 3.1 在冻结的 C2 分组与 C2 三元组上生成 COMP-P1..P5（每组一次调用；P5 额外一次 verify）
HY4_CACHE_SHARD=co python scripts/prompt_opt/05_comp_generate.py \
    --pids COMP-P1,COMP-P2,COMP-P3,COMP-P4,COMP-P5

# 3.3 关系质量审计（无 LLM）
python scripts/prompt_opt/12_comp_quality.py

# 3.2 每个 Prompt 建独立图（G_raw state + 固定 C2 三元组 + 该 Prompt 关系）
#     并跑 105 题检索（含官方 retrieve 的 fact rerank LLM 中间状态诊断）
python scripts/prompt_opt/06_comp_graphs_retrieval.py \
    --pids COMP-P1,COMP-P2,COMP-P3,COMP-P4,COMP-P5

# 3.2b 汇总（06 会覆盖 summary.json；这一步从 records 重建全部变体）
python scripts/prompt_opt/06c_resummarize.py

# 3.4 追踪新增关系是否进入最终 top-5 fact
python scripts/prompt_opt/06b_trace_facts.py

# 3.5 关系数量控制：P4 的 16 条按 confidence 保留前 10 条，重新建图 + 检索
python scripts/prompt_opt/16_make_count_control.py
python scripts/prompt_opt/06_comp_graphs_retrieval.py --pids COMP-P4-c10 --skip-baselines
python scripts/prompt_opt/06c_resummarize.py
```

## Phase 5 — 联合

```bash
# E2 = P4 + P0, E3 = P4 + P7, 以及 C2 + P7（同一 Reader，比较图）
HY4_CACHE_SHARD=jo python scripts/prompt_opt/10_joint.py \
    --variants cmpo_compp4 --pids P0,P7 --split all --max-workers 6
HY4_CACHE_SHARD=jo python scripts/prompt_opt/10_joint.py \
    --variants C2 --pids P7 --split all --max-workers 6
```

## 报告与汇总

```bash
python scripts/prompt_opt/15_cases.py --variant cmpo_compp4 --pid P7
python scripts/prompt_opt/14_build_reports.py     # reports/auto_tables.md, all_metrics.json
python scripts/prompt_opt/17_manifest.py          # reports/run_manifest.json
```

## HY4 调用量与配额

| shard | 用途 | 实际调用 | 错误 |
|---|---|---:|---:|
| `po` | Reader pilot + val | 450 | 0 |
| `co` | Comprehension 生成 + verify | 11 | 0 |
| `or` | Oracle R2/R3 | 266 | 0 |
| `oa` | 官方 cat-5 多选 | 24 | 0 |
| `rp` | 重复稳定性（禁用缓存） | 98 行（84 成功 + 14 次 429 重试） | 14 次重试后成功 |
| `jo` | 联合 E2/E3/C2+P7 | 237 | 0 |

另外，Comprehension 图的检索包含官方 HippoRAG 的逐题 LLM fact rerank；
这些调用走 `outputs/raw_graphs/conv-30/llm_cache`（变体间共享 symlink），
不在上表的 `HY4Client` 统计中。

## 已知不可复现/不稳定点

1. GPU embedding 存在批次级数值差异：105 题批量检索与单题检索的 ranked_docs 不完全一致；
   G_raw 的 diag 重跑与历史记录 93/105 题排名一致。
2. 官方 fact reranker 是 LLM 调用；同一图重跑时 top-5 fact 可能变化。
3. HY4 在 temperature=0 下仍非确定；重复实验见 `reader/repeats/summary.json`。

## 追加实验 — 关闭 LLM fact reranker（2026-11-09）

```bash
# 用 embedding top-k fact 直接作为 PPR 种子，不调用 LLM
#   top5  = 与官方 reranker 的最终 fact 数一致
#   top20 = 更宽的候选窗口（敏感性）
python scripts/prompt_opt/18_no_llm_rerank.py --graphs G_raw,C2,COMP-P4 --fact-ks 5,20

# 在无重排的检索上下文上跑 Reader P0/P7
HY4_CACHE_SHARD=jo python scripts/prompt_opt/10_joint.py \
    --records-root comprehension/retrieval_norerank \
    --variants G_raw_top5,COMP-P4_top5 --pids P0,P7 --split all --max-workers 6
```

产物：`comprehension/retrieval_norerank/{G_raw,C2,COMP-P4}_top{5,20}/`、
`joint/{G_raw_top5,COMP-P4_top5}_{P0,P7}.jsonl`。
无重排检索不产生 LLM 调用；QA 部分 420 次调用使用 shard `jo`，0 错误。
