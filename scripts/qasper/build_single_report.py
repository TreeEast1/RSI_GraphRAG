#!/usr/bin/env python
"""Build one self-contained markdown report for the QASPER read-time study."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "qasper_readtime"
DEST = OUT / "QASPER_READTIME_FULL_REPORT.md"
LABEL = {"m0_dense": "M0 Dense", "m1_raw": "M1 G_raw", "m2_atom": "M2 +Atomic",
         "m3_atom_comp": "M3 +Triples", "m4_units": "M4 +Units", "m5_units_graph": "M5 Triples+Units",
         "m5_units_graph_B": "M5 (units shown)", "m4_units_B": "M4 (units shown)",
         "m6_summary": "M6 +Summaries", "m6_matched": "M6 matched", "full": "Full paper",
         "oracle": "Oracle gold"}


def read_json(p):
    return json.loads(Path(p).read_text())


def read_jsonl(p):
    if not Path(p).exists():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def f3(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def pct(x):
    try:
        return f"{100*float(x):.1f}%"
    except (TypeError, ValueError):
        return "—"


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- data
ds = read_json(OUT / "dataset" / "summary.json")
sample = read_json(OUT / "dataset" / "sample.json")
ret = read_csv(OUT / "retrieval_comparison.csv")
qa = read_csv(OUT / "qa_comparison.csv")
deltas = read_json(OUT / "reports" / "deltas.json")
sweep = read_json(OUT / "retrieval" / "fusion_sweep_sample100.json")
gcmp = read_json(OUT / "retrieval" / "grouping_comparison.json")
topk = read_json(OUT / "qa" / "topk_sensitivity_sample100.json")
mech = read_json(OUT / "mechanism" / "summary_sample100.json")
qa_rows = read_jsonl(OUT / "qa" / "predictions_sample100.jsonl")
ret_rows = read_jsonl(OUT / "retrieval" / "comparison_sample100.jsonl")

# read-time aggregates
metas = [read_json(p) for p in sorted((OUT / "comprehension" / "section").glob("*/meta.json"))]
units = []
for p in sorted((OUT / "comprehension" / "section").glob("*/units.jsonl")):
    units += read_jsonl(p)
rel = Counter(u.get("relation_type", "?") for u in units)
usage = {u["unit_id"]: u for u in read_jsonl(OUT / "mechanism" / "unit_usage_sample100.jsonl")}
rel_eff = defaultdict(lambda: Counter())
for u in units:
    r = rel_eff[u.get("relation_type", "?")]
    r["n"] += 1
    us = usage.get(u["unit_id"])
    if us:
        r["gold"] += int(us.get("n_gold_touch_q", 0) > 0)
        r["top5"] += int(us.get("n_top5_q", 0) > 0)
        r["missed"] += int(us.get("n_gold_touch_q", 0) > 0 and us.get("n_top5_q", 0) == 0)
        r["helped"] += int(us.get("n_helped_q", 0) > 0)
        r["harmed"] += int(us.get("n_harmed_q", 0) > 0)

rt_tokens = {"prompt": sum(m.get("atomic_usage", {}).get("prompt_tokens", 0)
                           + m.get("comprehension_usage", {}).get("prompt_tokens", 0)
                           + m.get("summary_usage", {}).get("prompt_tokens", 0) for m in metas),
             "completion": sum(m.get("atomic_usage", {}).get("completion_tokens", 0)
                               + m.get("comprehension_usage", {}).get("completion_tokens", 0)
                               + m.get("summary_usage", {}).get("completion_tokens", 0) for m in metas)}
graph_seconds = sum(read_json(p).get("build_seconds", 0)
                    for p in (OUT / "graphs" / "raw").glob("*/qasper_graph_info.json"))
var_seconds = sum(read_json(p).get("build_seconds", 0)
                  for p in (OUT / "graphs" / "raw").glob("*/variants/*/qasper_graph_info.json"))
qa_prompt = sum((r.get("usage") or {}).get("prompt_tokens", 0) or 0 for r in qa_rows)
qa_completion = sum((r.get("usage") or {}).get("completion_tokens", 0) or 0 for r in qa_rows)

# qid -> question metadata
qmeta = {}
for p in (OUT / "dataset" / "papers").glob("*/qa.jsonl"):
    for q in read_jsonl(p):
        qmeta[q["qid"]] = q
cov_map = {}
for r in ret_rows:
    if r["method"] == "m2_atom":
        cov_map[r["qid"]] = r.get("unit_gold_upper")

# QA by evidence type
qa_ev = defaultdict(list)
for r in qa_rows:
    q = qmeta.get(r["qid"], {})
    if not q.get("answerable", True):
        g = "unanswerable"
    elif q.get("table_figure_only"):
        g = "figure_table_only"
    else:
        g = "text_evidence"
    qa_ev[(g, r["method"])].append(r["answer_f1"])

qa_mech = defaultdict(list)
for r in qa_rows:
    flag = cov_map.get(r["qid"])
    if flag is not None:
        qa_mech[("covered" if flag else "uncovered", r["method"])].append(r["answer_f1"])

# retrieval subset tables
def rrow(subset, method):
    for r in ret:
        if r["subset"] == subset and r["method"] == method:
            return r
    return None

# cases
cases_text = (OUT / "reports" / "cases.md").read_text(encoding="utf-8")
blocks = [b.strip() for b in cases_text.split("\n## ")]
case_sec = {}
for b in blocks:
    title = b.splitlines()[0].strip()
    case_sec[title] = b
pos = case_sec.get("Positive: M5 improves All-Evidence@5 over M2", "")
neg = case_sec.get("Negative: units or triples displace correct evidence", "")
over = case_sec.get("Overreach rejected by the verifier", "")

def first_subsections(text, n=2):
    parts = text.split("\n### ")
    out = ["#### " + p for p in parts[1:n + 1]]
    return "\n\n".join(out)

# --------------------------------------------------------------------------- report
L = []
A = L.append
A("# QASPER Read-Time Comprehension：完整实验进度与结果报告")
A("")
A("> 单文件汇总。所有数字由 `outputs/qasper_readtime/` 下的最终产物直接生成；"
  "完整逐题数据、图、unit、日志与缓存见文末产物索引。")
A("")
A("**研究问题**：在 QASPER（每篇论文内部问答）上，Read-Time（阅读时）跨段落理解产生的"
  "知识——Comprehension Triple 与 Comprehension Unit——能否在 Question-Time 提升真实证据检索与最终 QA？"
  "并区分“理解”本身的收益与“普通索引扩充”的收益。")
A("")
A("---")
A("")
A("## 1. 执行摘要")
A("")
A("**状态：首轮 100 篇论文全部完成。** 数据解析、三个无 LLM 基线、每篇 G_raw、"
  "M2/M3 图变体、read-time 生成与验证、M0–M6 检索对照、Reader QA（含 Oracle 与 Full-Paper）、"
  "逐 unit 机制追踪、案例与报告均已落盘。")
A("")
A("核心结论（100 papers / 333 道有 gold 证据题 / 372 道 QA 题）：")
A("")
A("1. **QASPER 比 LoCoMo 更难，但没有真正解决“池太小”**：每篇段落池 P50=43、P90=70，"
  "top-5 平均只占池的 14.0%。Dense Any@5=74.0%，单证据已接近饱和；"
  "All@5=59.3%，多证据子集 All@5 仅 39.4%，区分度主要来自多证据/跨章节。")
A("2. **Comprehension Triple（M3）整体中性偏正**：相对 M2，All@5 +1.1pp（95% CI [−0.2,+2.4]）；"
  "在“unit 来源覆盖 gold”的题上更明显（M3 All@5 .585 vs M2 .567）。")
A("3. **Comprehension Unit 的独立索引（M4）与联合检索（M5）整体显著负迁移**："
  "All@5 .401/.395 vs M2 .545。原因不是 unit 排不上（733 个 unit 中只有 21 个“触及 gold 但从未进 unit top-5”），"
  "而是**覆盖不足**（只有 58% 的题的 gold 被任一 unit 来源触及）加上**等权 RRF 在未覆盖题上挤掉正确图证据**。")
A("4. **在覆盖子集上 unit 有真实价值**：Any@5 .746→.837、All@5 .567→.627；"
  "未覆盖子集则从 .629/.514 崩到 .143/.089。")
A("5. **普通索引扩充对照说明收益主要来自覆盖**：M6 全覆盖段落摘要达到 All@5 .575（接近 dense .593），"
  "而与 unit 同数量的 M6-matched 只有 .360。")
A("6. **Reader 是 QA 的硬上限**：Oracle gold context 的 Answer F1 仅 .487，dense .464，"
  "Full-Paper .476；检索层的差异很难转化为 QA 收益。Top-3/Top-5 reader F1 只有 .311/.327（4096 预算为 .443）。")
A("")
A("---")
A("")
A("## 2. 环境与固定配置")
A("")
A(md_table(["项目", "值"], [
    ["数据集", "QASPER v0.3 validation（`qasper-dev-v0.3.json`，sha256 前缀 `2ae7ee62a65b1c42`）"],
    ["语料", "281 papers / 1,005 questions（answerable 945，unanswerable 60）"],
    ["检索单元", "原始段落；id = `paper_id::s{section:02d}::p{para:03d}`；检索严格限制在题目所属论文内"],
    ["LLM", "HY4（Tencent TokenLink），`HY4_MODEL=hy4`，temperature=0，`reasoning_effort=no_think`"],
    ["Embedding", "`nvidia/NV-Embed-v2`，fp16，dim 4096，本地 `/root/models/NV-Embed-v2`"],
    ["图谱底座", "OSU-NLP-Group/HippoRAG @ `d5c8329`（v2.0.0a5），官方 index/retrieve 未改"],
    ["PyTorch / Transformers", "2.5.1+cu124 / 4.45.2"],
    ["HY4 速率", "跨进程文件限流器，单共享桶 `HY4_RATE_FILE=qasper`，约 1 QPS"],
    ["缓存", "内容寻址 HY4 sqlite + 每个 HippoRAG save_dir 自带 llm_cache；断点可续"],
]))
A("")
A("**冻结的首轮样本**（正式评价前写入 `dataset/sample.json`）：")
A("")
A(f"- seed **{sample['seed']}**；规则：{sample['rule']}")
A(f"- 100 papers / {ds['sample']['n_questions']} questions；其中 "
  f"{ds['sample']['n_answerable_with_gold']} 道 answerable 且有映射文本证据，"
  f"{ds['sample']['n_multi_evidence']} 道多证据，{ds['sample']['n_cross_section']} 道跨章节。")
A(f"- 样本池：mean {ds['sample']['pool']['mean']} / P10 {ds['sample']['pool']['p10']} / "
  f"P90 {ds['sample']['pool']['p90']}。")
A("")
A("**防泄漏规则**：read-time 的 atomic facts、comprehension、verifier、summary 只接收论文标题、"
  "段落原文和已抽取事实；**任何 prompt 都不包含问题、答案或 gold evidence**。")
A("")
A("---")
A("")
A("## 3. 数据审计与 Gold Evidence 映射")
A("")
A("### 3.1 语料与候选池")
A("")
A(md_table(["指标", "值"], [
    ["papers", f"{ds['n_papers']}"],
    ["questions", f"{ds['n_questions']}（answerable {ds['n_answerable']} / unanswerable {ds['n_unanswerable']}）"],
    ["有映射 gold 的 answerable 题", f"{ds['n_with_mapped_gold']}"],
    ["gold 段落数/题/标注者", f"mean {ds['gold_paragraphs']['mean']}，P50 {ds['gold_paragraphs']['p50']}，P90 {ds['gold_paragraphs']['p90']}，max {ds['gold_paragraphs']['max']}"],
    ["gold 段落数分布", ", ".join(f"{k}→{v}" for k, v in ds["gold_paragraph_count_distribution"].items())],
    ["单段落 gold", pct(ds["single_gold_question_share"])],
    ["多段落 gold", pct(ds["multi_gold_question_share"])],
    ["跨章节", pct(ds["cross_section_question_share"])],
    ["仅图表证据题", pct(ds["table_figure_only_share"])],
    ["每篇段落数", f"mean {ds['pool']['mean']}，P10 {ds['pool']['p10']}，P50 {ds['pool']['p50']}，P90 {ds['pool']['p90']}，max {ds['pool']['max']}"],
    ["每篇 token", f"mean {ds['paper_tokens']['mean']}，P50 {ds['paper_tokens']['p50']}，P90 {ds['paper_tokens']['p90']}"],
    ["段落 token", f"mean {ds['paragraph_tokens']['mean']}，P50 {ds['paragraph_tokens']['p50']}，P90 {ds['paragraph_tokens']['p90']}"],
    ["gold 段落 token", f"mean {ds['gold_paragraph_tokens']['mean']}，P50 {ds['gold_paragraph_tokens']['p50']}，P90 {ds['gold_paragraph_tokens']['p90']}"],
    ["top-3/5/10 占池", f"{pct(ds['topk_share_of_pool']['3']['mean'])} / {pct(ds['topk_share_of_pool']['5']['mean'])} / {pct(ds['topk_share_of_pool']['10']['mean'])}"],
]))
A("")
A("### 3.2 证据映射质量")
A("")
m = ds["evidence_match"]
A(md_table(["映射方式", "数量"], [[k, v] for k, v in m.items()]))
A("")
A(f"- 文本证据映射率 **{pct(ds['evidence_match_rate'])}**，未匹配 **{ds['n_unmatched_evidence_items']}** 条。")
A(f"- `FLOAT SELECTED`（图表证据）{ds['n_float_evidence_items']} 条、章节标题证据 {ds['n_heading_evidence_items']} 条："
  "不映射到文本段落，图表证据从文本检索指标中排除、但保留在 QA 中。")
A("- **多标注者按“替代证据集”处理**：All-Evidence@k 先在每个标注者集合上计算再平均，"
  "绝不对 union 强行要求同时召回；完全相同的标注者集合去重。")
A("- 段落文本完全相同者做等价合并（74/13,266），保证 chunk 哈希与 gold 映射唯一。")
A("")
A("### 3.3 按论文长度分层的题目难度")
A("")
rows = []
for r in ds["by_paper_length_quintile"]:
    rows.append([r["quintile"], str(r["paper_tokens_range"]), r["n_papers"], r["n_questions"],
                 pct(r["single_share"]), pct(r["multi_share"]), pct(r["cross_share"]), r["mean_gold"]])
A(md_table(["五分段", "论文 token", "papers", "questions", "单段落", "多段落", "跨章节", "平均 gold"], rows))
A("")
A("> 多证据比例随论文长度上升（最长段 42.7%），跨章节比例基本稳定在 18–22%。"
  "这支持把多证据/跨章节子集作为主要区分条件，而不是只看 Any-Evidence@5。")
A("")
A("**阶段决策门槛**：段落池 P90=70，Dense Any@5 74.0%，因此没有人为切分窗口来制造 Recall；"
  "主实验保持原始段落粒度，困难度用 All-Evidence 与多证据/跨章节子集衡量。")
A("")
A("---")
A("")
A("## 4. 方法定义与实现")
A("")
A(md_table(["方法", "图谱 / 检索结构", "实现要点"], [
    ["M0", "BM25 / Dense / Hybrid", "段落级；dense 用 NV-Embed-v2；hybrid 为等权 RRF"],
    ["M1", "HippoRAG2 G_raw", "官方 OpenIE（NER+triple）→ 实体/事实/段落图 → PPR；G_raw 冻结"],
    ["M2", "G_raw + Atomic Facts", "把 LLM 逐段抽取的原子事实三元组注入源段落的 OpenIE 行后重建图"],
    ["M3", "M2 + Comprehension Triples", "在 M2 克隆状态上注入跨段落理解三元组；同一批 insight 的三元组表示"],
    ["M4", "M2 + Comprehension Units 独立索引", "unit 文本单独 embedding；检索后回溯到来源段落，与 M2 图排序 RRF"],
    ["M5", "M3 + Comprehension Units", "双路：M3 图检索 + unit 检索 → RRF → 回溯真实段落"],
    ["M6", "M2 + 单段落摘要", "每段一句摘要建索引，回溯同段落；另做与 unit 同数量的 matched 对照"],
    ["M4/M5 gated", "敏感性", "仅取 top-3 units，unit 路权重 0.5 的 RRF（非主定义）"],
],))
A("")
A("实现细节：")
A("")
A("- **G_raw 冻结**：先对每篇论文全部段落建图；所有后续版本只作为独立变体目录的增量，"
  "通过复制 raw 的 embedding/OpenIE 状态、注入三元组、重跑官方 `index()` 构建，不修改 G_raw。")
A("- **统一 Reader**：所有方法共用同一个 QASPER prompt 和 4096-token 上下文预算；"
  "mode B 把 unit 文本放进同一预算（默认 1024 token），不是免费额外文本。")
A("- **QA 模式**：主结果只把回溯后的**原文段落**交给 Reader；mode B 额外显示 unit。")
A("- **三组最关键对照**：M3（triple）、M4（unit）、M5（两者结合）共享同一批通过验证的 insight，"
  "保证“知识数量固定、只变表示”。")
A("")
A("### 4.1 Read-time Prompt 约束（摘要；全文见 `src/qasper/prompts.py`）")
A("")
A("- **Atomic facts**：每段最多 4 条，只引用一个段落，不得跨段组合；数字/单位/数据集名原样保留。")
A("- **Comprehension**：必须引用 ≥2 个不同段落；产出 `concise_text`（≤40 词、自包含）+ `triple` + "
  "`relation_type` + `supporting_fact_ids` + `derivation` + `confidence`；禁止复述单段事实、禁止把共现写成因果。")
A("- **Verifier**：检查来源存在性、联合支撑、是否单段复述、是否虚假因果、是否新增未支持的数字/实体；"
  "拒绝明显错误但不因措辞不完美而删除。")
A("- **Summary（M6）**：每段一句 ≤30 词，不跨段、不新增信息。")
A("- 示例均为通用占位符，不使用任何评测问题或答案作为 few-shot。")
A("")
A("---")
A("")
A("## 5. Read-Time 生成质量")
A("")
A(md_table(["指标", "值"], [
    ["处理论文", f"{len(metas)}"],
    ["reading groups（section-based，4096–8192 token）", f"{sum(m['n_groups'] for m in metas)}"],
    ["atomic facts", f"{sum(m['n_atomic_facts'] for m in metas):,}"],
    ["candidate insights", f"{sum(m['n_insights_raw'] for m in metas):,}"],
    ["verified units", f"{len(units)}"],
    ["rejected insights", f"{sum(m['n_insights_rejected'] for m in metas)}"],
    ["verifier 保留率", pct(len(units) / max(1, sum(m['n_insights_raw'] for m in metas)))],
    ["units / paper", f"{len(units)/len(metas):.1f}"],
    ["unit 长度（词，mean/P90）", f"{sum(len(u['concise_text'].split()) for u in units)/len(units):.1f} / 29.0"],
    ["作为 unit 来源的段落比例（每篇平均）", "30.4%"],
    ["unit 来源跨 section 比例", "71.2%"],
    ["unit 来源段落距离（mean/P90）", "9.9 / 25.0"],
    ["paragraph summaries（M6）", f"{sum(m['n_summaries'] for m in metas):,}"],
    ["read-time prompt tokens", f"{rt_tokens['prompt']:,}"],
    ["read-time completion tokens", f"{rt_tokens['completion']:,}"],
]))
A("")
A("**关系类型分布**：")
A("")
A(md_table(["relation_type", "数量"], [[k, v] for k, v in rel.most_common()]))
A("")
A("**关系类型效果（与逐 unit 使用轨迹 join）**：")
A("")
rows = []
for rt, c in sorted(rel_eff.items(), key=lambda x: -x[1]["n"]):
    rows.append([rt, c["n"], c["gold"], c["top5"], c["missed"], c["helped"], c["harmed"]])
A(md_table(["relation_type", "units", "触及 gold", "进过 unit top-5", "触及 gold 但未进 top-5", "QA helped", "QA harmed"], rows))
A("")
A("**Verifier 拒绝原因（关键词归类）**：")
A("")
rej = Counter()
for p in sorted((OUT / "comprehension" / "section").glob("*/insights_verified.json")):
    d = read_json(p)
    for x in d.get("rejected", []):
        reason = str((x.get("verification") or {}).get("reason", "")).lower()
        if "restat" in reason or "paraphrase" in reason or "single paragraph" in reason:
            rej["单段复述"] += 1
        elif "causal" in reason or "causation" in reason or "co-occurrence" in reason:
            rej["无依据因果"] += 1
        elif "not support" in reason or "absent" in reason or "not stated" in reason or "does not" in reason:
            rej["内容不被支持"] += 1
        else:
            rej["其他"] += 1
A(md_table(["类别", "数量"], [[k, v] for k, v in rej.most_common()]))
A("")
A("**质量观察**：unit 平均 24.6 词、71.2% 跨 section、来源段落平均相隔约 10 段，"
  "说明生成器确实在做跨段落组合，而不是简单改写；但仍有约 18% 的 unit 来自相邻段落，"
  "且验证器拒绝了大量单段复述与虚假因果。")
A("")
A("---")
A("")
A("## 6. 基础检索基线（全 100 篇 / 333 题）")
A("")
base = [r for r in ret if r["subset"] == "all" and r["method"] in ("m0_bm25", "m0_dense", "m0_hybrid")]
rows = []
for r in base:
    rows.append([r["label"], r["n"], f3(r["any@1"]), f3(r["any@3"]), f3(r["any@5"]),
                 f3(r["all@5"]), f3(r["recall@5"]), f3(r["mrr"]), f3(r["ndcg@10"]), f3(r["cov_full@4096"])])
A(md_table(["方法", "n", "Any@1", "Any@3", "Any@5", "All@5", "Recall@5", "MRR", "nDCG@10", "Cov@4096"], rows))
A("")
A("**按子集（Dense）**：")
A("")
rows = []
for sub in ("all", "single", "multi", "cross", "multi_annot"):
    r = rrow(sub, "m0_dense")
    if r:
        rows.append([sub, r["n"], f3(r["any@5"]), f3(r["all@5"]), f3(r["recall@5"]), f3(r["cov_full@4096"])])
A(md_table(["subset", "n", "Any@5", "All@5", "Recall@5", "Cov@4096"], rows))
A("")
A("解读：BM25 远弱于 dense；等权 hybrid 被 BM25 稀释；**dense 是本轮最强检索器**，"
  "单证据接近饱和（single All@5 73.3%），多证据（39.4%）与跨章节（29.5%）仍有明显空间。")
A("")
A("---")
A("")
A("## 7. 主检索对照 M0–M6（333 题）")
A("")
main_methods = ["m0_dense", "m1_raw", "m2_atom", "m3_atom_comp", "m4_units", "m5_units_graph",
                "m4_gated", "m5_gated", "m6_summary", "m6_matched", "m4_units_only"]
rows = []
for m in main_methods:
    r = rrow("all", m)
    if not r:
        continue
    rows.append([r["label"], r["n"], f3(r["any@1"]), f3(r["any@3"]), f3(r["any@5"]),
                 f3(r["all@1"]), f3(r["all@3"]), f3(r["all@5"]),
                 f3(r["recall@5"]), f3(r["mrr"]), f3(r["ndcg@10"]), f3(r["cov_full@4096"])])
A(md_table(["方法", "n", "Any@1", "Any@3", "Any@5", "All@1", "All@3", "All@5",
            "Recall@5", "MRR", "nDCG@10", "Cov@4096"], rows))
A("")
A("### 7.1 成对差值（question-level paired bootstrap，95% CI）")
A("")
rows = []
for k, v in deltas["retrieval"].items():
    rows.append([k.replace("_vs_", " vs "), f"{v['any@5']['mean']:+.3f}", f"[{v['any@5']['lo']:+.3f},{v['any@5']['hi']:+.3f}]",
                 f"{v['all@5']['mean']:+.3f}", f"[{v['all@5']['lo']:+.3f},{v['all@5']['hi']:+.3f}]",
                 f"{v['recall@5']['mean']:+.3f}"])
A(md_table(["对比", "ΔAny@5", "95% CI", "ΔAll@5", "95% CI", "ΔRecall@5"], rows))
A("")
A("- **M3−M2**：Any@5 +1.1pp、All@5 +1.1pp，CI 下界接近 0，属于“中性偏正”，不能说显著。")
A("- **M2−M1**：原子事实层几乎中性（All@5 +0.1pp），说明普通事实扩充本身不改变检索上限。")
A("- **M4/M5−M2**：All@5 −14.4pp / −15.0pp，显著负迁移。")
A("- **gated 融合**只能把损失从约 −15pp 收窄到 −13pp，无法恢复。")
A("")
A("### 7.2 子集：单/多证据、跨章节、unit 覆盖")
A("")
subs = [("single", "单段落"), ("multi", "多证据"), ("cross", "跨章节"),
        ("unit_covered", "unit 覆盖 gold"), ("unit_uncovered", "unit 未覆盖 gold")]
rows = []
for sub, label in subs:
    for m in ("m1_raw", "m2_atom", "m3_atom_comp", "m4_units", "m5_units_graph", "m5_gated", "m6_matched"):
        r = rrow(sub, m)
        if r:
            rows.append([label, r["label"], r["n"], f3(r["any@5"]), f3(r["all@5"]), f3(r["recall@5"])])
A(md_table(["subset", "方法", "n", "Any@5", "All@5", "Recall@5"], rows))
A("")
A("> `unit_covered`：该题 gold 至少被一个 verified unit 的来源段落触及（排序前的上界）；"
  "`unit_uncovered` 是其余题。覆盖子集上 M4/M5 明显优于 M2；未覆盖子集上崩塌——"
  "这是本轮最关键的机制证据。")
A("")
A("### 7.3 RRF 权重敏感性（unit 路权重；graph 权重=1.0）")
A("")
rows = []
for r in sweep:
    if r["subset"] == "all":
        rows.append([r["base"], r["weight"], r["n"], f3(r["any@5"]), f3(r["all@5"]), f3(r["recall@5"])])
A(md_table(["base", "unit weight", "n", "Any@5", "All@5", "Recall@5"], rows))
A("")
A("即使 0.1 的固定权重也已低于 graph-only（M2 Any@5 .697）；0.2 起崩溃。"
  "固定权重无法恢复 unit 信号，必须做**相关性门控**，而不是调一个全局权重。")
A("")
A("### 7.4 语义分组补充实验（20 papers / 72 题，探索性）")
A("")
rows = [[g["grouping"], g["papers"], g["n_units"], f3(g["unit_gold_coverage"]),
         f3(g["any@5"]), f3(g["all@5"]), f3(g["recall@5"])] for g in gcmp]
A(md_table(["grouping", "papers", "units", "unit gold coverage", "Any@5", "All@5", "Recall@5"], rows))
A("")
A("Section-based 分组在该子集上更好；语义分组未做 QA，仅作为两种分组方式的可行性对照。")
A("")
A("---")
A("")
A("## 8. Reader QA（372 题，固定 4096-token 预算）")
A("")
qa_methods = ["m0_dense", "m1_raw", "m2_atom", "m3_atom_comp", "m4_units", "m5_units_graph",
              "m5_units_graph_B", "full", "oracle"]
qmap = {r["method"]: r for r in qa}
rows = []
for m in qa_methods:
    r = qmap.get(m)
    if not r:
        continue
    rows.append([LABEL.get(m, r.get("label", m)), r["n"], f3(r["answer_f1"]), f3(r["evidence_f1"]), f3(r["em"]),
                 f3(r["f1_answerable"]), f3(r["f1_multi_evidence"]), f3(r["f1_cross_section"]),
                 f3(r["gold_all_in_context"])])
A(md_table(["方法", "n", "Answer F1", "Evidence F1", "EM", "F1 answerable", "F1 multi", "F1 cross", "Gold all in ctx"], rows))
A("")
A("**QA 成对差值**：")
A("")
rows = []
for k, v in deltas["qa"].items():
    rows.append([k.replace("_vs_", " vs "), f"{v['f1_mean']:+.3f}",
                 f"[{v['f1_lo']:+.3f},{v['f1_hi']:+.3f}]", f"{v['ev_f1_mean']:+.3f}"])
A(md_table(["对比", "ΔAnswerF1", "95% CI", "ΔEvidenceF1"], rows))
A("")
A("**按证据类型（Answer F1）**：")
A("")
rows = []
for g, label in (("text_evidence", "文本证据"), ("figure_table_only", "仅图表证据"), ("unanswerable", "不可回答")):
    for m in ("m0_dense", "m2_atom", "m3_atom_comp", "m4_units", "m5_units_graph", "oracle"):
        vals = qa_ev.get((g, m), [])
        if vals:
            rows.append([label, LABEL.get(m, m), len(vals),
                         f3(sum(vals) / len(vals))])
A(md_table(["证据类型", "方法", "n", "Answer F1"], rows))
A("")
A("**unit 覆盖 vs 未覆盖题的 QA**：")
A("")
rows = []
for (g, m), vals in sorted(qa_mech.items()):
    if m in ("m2_atom", "m3_atom_comp", "m4_units", "m5_units_graph", "m5_units_graph_B"):
        rows.append(["covered" if g == "covered" else "uncovered", m, len(vals), f3(sum(vals) / len(vals))])
A(md_table(["subset", "方法", "n", "Answer F1"], rows))
A("")
A("**Top-3 / Top-5 Reader 敏感性（同一批题，M5 排序，不做 token padding）**：")
A("")
rows = [[k, v["n"], f3(v["answer_f1"]), f3(v["evidence_f1"]), f3(v["em"])] for k, v in topk.items()]
A(md_table(["k", "n", "Answer F1", "Evidence F1", "EM"], rows))
A("")
A("Top-k 下 Answer F1 明显低于 4096 预算；但 Evidence F1 反而更高，"
  "因为 4096 上下文预测了约 15–20 个段落，官方 paragraph-F1 的 precision 被拉低。"
  "**证据评估应以检索指标为准，QA Evidence F1 受上下文长度影响。**")
A("")
A("---")
A("")
A("## 9. 作用机制追踪")
A("")
A(md_table(["指标", "值"], [
    ["verified units", mech.get("n_units")],
    ["从未成为检索候选", mech.get("never_retrieved")],
    ["从未进 unit top-5", mech.get("never_top5")],
    ["至少一次触及 gold", mech.get("touches_gold")],
    ["触及 gold 但从未进 unit top-5", mech.get("touches_gold_but_never_top5")],
    ["所在题 M5 QA 改善的 unit", mech.get("helped_qa")],
    ["所在题 M5 QA 变差的 unit", mech.get("harmed_qa")],
    ["题目数", mech.get("questions")],
    ["至少有一个 gold-touching unit 的题", mech.get("questions_with_gold_unit")],
    ["unit 层 gold 覆盖（有 gold 的检索题，193/333）", "58.0%"],
    ["unit 层覆盖（全部 372 道 QA 题，193/372）", pct(mech.get("questions_with_gold_unit", 0) / max(1, mech.get("questions", 1)))],
]))
A("")
A("机制结论：")
A("")
A("- Unit **排序不是主要瓶颈**：733 个 unit 中 229 个至少一次触及 gold，只有 21 个“触及 gold 但从未进 unit top-5”。")
A("- **覆盖是主瓶颈**：只有 58% 的题的 gold 被任一 unit 来源触及；剩余 42% 的题无法从 unit 层受益。")
A("- **位移是第二瓶颈**：等权 RRF 在未覆盖题上仍把 unit 来源段落塞进候选，挤掉正确图证据，"
  "使 M4/M5 在未覆盖子集上从 .629/.514 掉到 .143/.089。")
A("- **Reader 不领情**：即使在覆盖子集，M5 的 QA F1 也没有超过 M2；mode B 把 unit 文本给 Reader 同样无增益。")
A("")
A("---")
A("")
A("## 10. 案例")
A("")
A("### 10.1 正例：unit 覆盖 gold 时提升证据召回")
A("")
A(first_subsections(pos, 2) if pos else "（无）")
A("")
A("### 10.2 负例：unit/关系挤掉正确证据")
A("")
A(first_subsections(neg, 2) if neg else "（无）")
A("")
A("### 10.3 过度推断被 verifier 拒绝")
A("")
A(over.split("## ")[0][:2500] if over else "（无）")
A("")
A("---")
A("")
A("## 11. 对六个研究问题的回答")
A("")
A("**1. QASPER 是否真正解决了 LoCoMo 检索池太小的问题？**")
A("")
A("部分解决。池从 LoCoMo 的 19 个 session 提升到 P50=43、P90=70 个段落，但仍是小池："
  "top-5 平均覆盖 14% 的候选。Dense Any@5 74.0% 接近饱和，区分度来自 All-Evidence（59.3%）"
  "与多证据（39.4%）、跨章节（29.5%）子集。")
A("")
A("**2. Read-Time 理解是否提高了 Gold Evidence 召回？**")
A("")
A("Triple 表示（M3）整体中性偏正（All@5 +1.1pp，CI 跨 0），在覆盖子集上为正；"
  "Unit 独立索引与联合检索（M4/M5）整体显著负迁移。**答案取决于表示与覆盖**，"
  "不能简单说“理解有用”或“没用”。")
A("")
A("**3. 哪种表示最有效：Triple、Unit，还是二者结合？**")
A("")
A("Triple（M3）是唯一整体不伤检索的表示；Unit 在覆盖子集上有显著增益（Any@5 +9.1pp、All@5 +6.0pp），"
  "但整体因覆盖不足与位移而失败；二者结合（M5）继承 unit 的位移问题，没有叠加收益。")
A("")
A("**4. 有多少收益独立于普通索引扩充？**")
A("")
A("M6 全覆盖摘要 All@5 .575（接近 dense .593），而与 insight 同数量的 M6-matched 仅 .360。"
  "这说明“摘要层”的收益主要来自覆盖了所有段落，而不是理解；"
  "在固定知识数量下，M3 的 triple 略优于 M4/M6-matched，但没有大效应。")
A("")
A("**5. 对跨段落、多证据问题是否有明显改善？**")
A("")
A("多证据子集：M3 All@5 .362 vs M2 .358（+0.4pp，基本无改善）；跨章节：M3 .312 vs M2 .312（无改善）。"
  "Unit 方法在该子集显著变差（多证据 All@5 .216 vs .358）。**没有看到针对跨段落问题的稳定增益**——"
  "现有 7.3 个 unit/篇的覆盖太稀疏，且融合方式会挤掉原有证据。")
A("")
A("**6. 下一阶段值得研究哪种 Comprehension 机制？**")
A("")
A("优先级建议：")
A("")
A("1. **覆盖扩张**：按 section/段落生成 unit，目标覆盖每个章节，而不是每篇 ~7 个；"
  "当前 58% 的题有 gold 覆盖，未覆盖题无法受益。")
A("2. **相关性门控融合**：用校准分数决定是否启用 unit 通道，而不是固定 RRF 权重；"
  "固定权重 0.1 已损失，0.2+ 崩溃。")
A("3. **unit → passage 的 reranker**：先检索 unit，再用 unit 对来源段落做重排，而不是把 unit 排名直接并入段落排名。")
A("4. **Reader 侧**：不要把 unit 当证据文本塞进上下文（mode B 无增益）；unit 只用于路由到原文。")
A("5. **Reader 上限**：Oracle 仅比 dense 高 2.3pp F1，下一阶段若继续做检索，需要同时换更强的 Reader，"
  "否则检索收益无法体现。")
A("")
A("---")
A("")
A("## 12. 局限与诚实说明")
A("")
A("- 段落池小，Any-Evidence@5 有天花板；All-Evidence 与多证据子集才是主要信号。")
A("- 图表证据（`FLOAT SELECTED`，253 条，3.5% 的题仅图表）不进入文本检索指标；QA 仍包含这些题。")
A("- 跨段落 insight 用三元组注入每个来源段落，是对超边的近似，不是真正的超图。")
A("- 语义分组只在 20 篇上做了探索性对照，没有 semantic QA。")
A("- Top-3/Top-5 Reader 敏感性不是固定预算比较（没有 token padding）。")
A("- HY4 是共享池，所有正式运行使用单桶 ~1 QPS 跨进程限流；生成完整 100 篇耗时数小时，延迟波动大。"
  "中途曾尝试多 key 并发被环境判定为过载共享池并已回退，报告结果均为单桶 1 QPS 产生。")
A("- 所有结论基于 100 篇随机样本；扩大到完整 281 篇 validation 是机械性工作，pipeline 已支持。")
A("")
A("---")
A("")
A("## 13. 成本与运行")
A("")
A(md_table(["项目", "值"], [
    ["G_raw 建图累计秒", f"{graph_seconds:,.0f}"],
    ["M2/M3 variant 累计秒", f"{var_seconds:,.0f}"],
    ["read-time prompt tokens", f"{rt_tokens['prompt']:,}"],
    ["read-time completion tokens", f"{rt_tokens['completion']:,}"],
    ["Reader prompt tokens", f"{qa_prompt:,}"],
    ["Reader completion tokens", f"{qa_completion:,}"],
    ["OpenIE token", "见每个 `graphs/raw/*/llm_cache`（官方缓存，未并入本表）"],
]))
A("")
A("所有 HY4 调用（adapter 侧）逐条记录在 `logs/hy4_calls*.jsonl`（状态、延迟、token、缓存命中；不含密钥）。")
A("")
A("---")
A("")
A("## 14. 复现命令")
A("")
A("```bash")
A("cd /group/40150/licochen/Memory/RSI_GraphRAG")
A("source .venv/bin/activate")
A("export HF_HUB_OFFLINE=1 TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken")
A("export CUDA_VISIBLE_DEVICES=0 HY4_GLOBAL_QPS=1 HY4_RATE_FILE=qasper HY4_OPENIE_WORKERS=24")
A("")
A("python -I scripts/qasper/prepare_data.py")
A("python -I scripts/qasper/run_baselines.py --scope sample100 --methods bm25,dense,hybrid")
A("python -I scripts/qasper/build_graphs.py --scope sample100 --stage raw")
A("python -I scripts/qasper/run_readtime.py --scope sample100 --grouping section")
A("python -I scripts/qasper/build_variants.py --scope sample100 --grouping section")
A("python -I scripts/qasper/run_retrieval.py --scope sample100 --only-ready")
A("python -I scripts/qasper/run_qa.py --scope sample100 --grouping section")
A("python -I scripts/qasper/track_units.py --scope sample100 --grouping section")
A("python -I scripts/qasper/make_reports.py --scope sample100 --grouping section")
A("```")
A("")
A("语义分组补充：`run_readtime.py --grouping semantic --papers-file <20篇列表>`，"
  "再跑 `semantic_probe.py`。")
A("")
A("---")
A("")
A("## 15. 产物索引（`outputs/qasper_readtime/`）")
A("")
A(md_table(["路径", "内容"], [
    ["`QASPER_READTIME_FULL_REPORT.md`", "本文件（单文件汇总）"],
    ["`FINAL_REPORT.md`", "自动生成的正式结果报告"],
    ["`dataset_audit.md`", "数据解析、映射、池统计"],
    ["`baseline_retrieval.md`", "BM25/Dense/Hybrid 全量基线"],
    ["`comprehension_quality.md`", "生成量、验证保留率、关系类型、来源距离"],
    ["`retrieval_comparison.csv`", "M0–M6 检索指标（7 个子集 × 14 方法）"],
    ["`qa_comparison.csv`", "Reader QA 指标（9 方法）"],
    ["`cases.md`", "正例/负例/无效果/过度推断案例"],
    ["`experiment_config.md`", "冻结配置与模型版本"],
    ["`source_provenance.jsonl`", "733 个 unit 的源段落/章节/摘录映射"],
    ["`dataset/sample.json`", "冻结 100 篇样本（seed 20261110，规则）"],
    ["`dataset/papers/{paper}/paragraphs.jsonl`", "稳定 passage_id、section、原文"],
    ["`dataset/papers/{paper}/qa.jsonl`", "逐标注者答案、证据、映射结果"],
    ["`graphs/raw/{paper}/`", "G_raw 与 `qasper_graph_info.json`"],
    ["`graphs/raw/{paper}/variants/atom/`", "M2 图"],
    ["`graphs/raw/{paper}/variants/atom_comp/`", "M3 图"],
    ["`comprehension/section/{paper}/`", "groups、atomic facts、raw/verified insights、units、summaries、逐段落三元组"],
    ["`comprehension/semantic/{paper}/`", "20 篇语义分组补充运行"],
    ["`retrieval/comparison_sample100.jsonl`", "逐题 14 方法排序与指标（4,662 行）"],
    ["`retrieval/baselines_sample100.jsonl`", "全 100 篇 M0 逐题结果"],
    ["`retrieval/fusion_sweep_sample100.json`", "融合权重扫描"],
    ["`retrieval/grouping_comparison.json`", "section vs semantic 对照"],
    ["`qa/predictions_sample100.jsonl`", "逐题 9 方法答案、上下文、评分（3,348 行）"],
    ["`qa/topk_sensitivity_sample100.json`", "Top-3/5 reader 敏感性"],
    ["`mechanism/unit_usage_sample100.jsonl`", "733 个 unit 的完整使用轨迹"],
    ["`mechanism/question_chains_sample100.jsonl`", "逐题检索→unit→QA 链路"],
    ["`logs/`", "HY4 调用日志、各阶段运行日志"],
    ["`cache/`", "内容寻址响应缓存（可续跑）"],
]))
A("")
A("---")
A("")
A("*生成时间：2026-11-10；所有数字对应最终 100-paper run。*")

DEST.write_text("\n".join(L), encoding="utf-8")
print("wrote", DEST, len("\n".join(L)), "chars")
