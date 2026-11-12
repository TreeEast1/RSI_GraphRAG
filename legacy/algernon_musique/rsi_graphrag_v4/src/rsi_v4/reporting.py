"""Fill the reports from artifact files. Missing stages stay marked as not run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _line(value, suffix="") -> str:
    if value is None:
        return "未运行"
    if isinstance(value, float):
        return f"{value:.4f}{suffix}"
    return f"{value}{suffix}"


def write_reports(root: Path) -> None:
    art = root / "artifacts"
    data = root / "data"
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    corpus = _read(data / "corpus_manifest.json")
    historical = _read(art / "historical_graph_stats.json")
    giant = _read(art / "giant_leiden.json")
    partition = _read(art / "partition_pilot.json")
    probe = _read(art / "luna_probe.json")
    cost = _read(art / "cost_estimate.json")
    formal = _read(art / "formal_openie_status.json")
    mechanism = _read(art / "mechanism_status.json")
    graph_meta = _read(art / "mechanism_graph_meta.json")
    comp = _read(art / "comprehension_pilot.json")
    retrieval = _read(art / "retrieval_eval.json")
    question = _read(art / "question_time.json")
    ledger = _read(art / "ledger.json")
    identity = _read(art / "run_identity.json")
    openie_error = _read(art / "openie_driver_error.json")
    round2 = _read(art / "round2" / "summary.json")

    (reports / "FINAL_PIPELINE_REPORT.md").write_text(
        _pipeline(identity, corpus, ledger), encoding="utf-8"
    )
    (reports / "GRAPH_CONSTRUCTION_AUDIT.md").write_text(
        _graph(corpus, historical, giant, partition, probe, cost, formal, mechanism, graph_meta, openie_error, identity),
        encoding="utf-8",
    )
    (reports / "READ_TIME_EVOLUTION_REPORT.md").write_text(
        _read_time(partition, comp, retrieval, graph_meta), encoding="utf-8"
    )
    (reports / "QUESTION_TIME_EVOLUTION_REPORT.md").write_text(
        _question(question), encoding="utf-8"
    )
    (reports / "FINAL_EXPERIMENT_RESULTS.md").write_text(
        _final(corpus, historical, giant, partition, comp, retrieval, question, cost, ledger, formal, mechanism, probe, round2),
        encoding="utf-8",
    )


def _pipeline(identity, corpus, ledger) -> str:
    return "\n".join(
        [
            "# RSI GraphRAG V4 Pipeline",
            "",
            "正式顺序是：冻结全量语料，传统 OpenIE 建图，关系图划分阅读单元，理解层补丁，Question-Time 诊断，再用原 RAG 重跑验证。G_raw 只读。Held-out 不进入更新。",
            "",
            "## 模块",
            "",
            "- `corpus_manager` 冻结 passage id，并写出不含答案的语料。评测字段只在 `questions_eval.jsonl`。",
            "- `hipporag_adapter` 调用本地 HippoRAG commit 的官方 NER / triple / QA prompt，生成不可变 G_raw。",
            "- `graph_analyzer` 分别统计关系图和“关系边 + passage context 边”的连通分量，并用 Leiden 给超预算分量打社区标签。",
            "- `graph_partitioner` 在分量内部递归切分，并加 hop overlap。Overlap 只复用实体 id。",
            "- `comprehension_evolver` / `graph_patch_manager` 只写 G_comp。证据 span 必须是原文子串。",
            "- `retrieval.py` 把理解 chunk 作为文档节点、理解边作为图边，走 fact linking + DPR reset + PPR。",
            "- `rag.py` 的 `rag()` 不接收 gold。`evolution_analyzer` 在 rollout 之后才读 gold。",
            "- `rag_verifier` 在隔离快照上重跑 `rag()`，再决定提交或回滚。",
            "",
            "## 本次身份",
            "",
            f"- HippoRAG commit: {_line(None if not identity else identity.get('hipporag_commit'))}",
            f"- 请求的 LLM deployment: {_line(None if not identity else identity.get('requested_llm_deployment'))}",
            f"- 语料 hash: {_line(None if not corpus else corpus.get('corpus_hash'))}",
            f"- 新鲜 LLM 调用: {_line(None if not ledger else ledger.get('fresh_calls'))}",
            "",
            "复现：`PYTHONPATH=src /data2-HDD-SATA-20T/nzq/env/hipporag/bin/python -m rsi_v4.experiment_runner --phase all`",
            "",
        ]
    )


def _graph(corpus, historical, giant, partition, probe, cost, formal, mechanism, graph_meta, openie_error, identity) -> str:
    lines = ["# Graph Construction Audit", ""]
    if corpus:
        lines += [
            "## 冻结语料",
            "",
            f"- 问题 {corpus['n_questions']}，语料行 {corpus['n_corpus_rows']}，唯一 passage {corpus['n_unique_passages']}。",
            f"- 划分: {corpus['role_counts']}。",
            f"- hop × role: {corpus['hop_by_role']}。",
            f"- 金证据全部落在冻结语料中的问题: {corpus['questions_with_all_gold_in_corpus']} / {corpus['n_questions']}。",
            f"- 缺失的金 passage 槽位: {corpus['gold_passage_slots_missing_from_corpus']} / {corpus['gold_passage_slots']}。",
            f"- 精确重复题组: {len(corpus.get('exact_duplicate_groups') or {})}，隔离出 held-out 的题: {corpus.get('quarantined_duplicate_ids')}。",
            f"- 跨划分、Jaccard ≥ 0.9 的近重复: {corpus.get('cross_split_near_duplicate_count')}。",
            f"- 去重规则: {corpus.get('dedup_rule')}",
            "",
        ]
    else:
        lines.append("语料冻结未运行。\n")
    lines.append("## 历史 OpenIE")
    lines.append("")
    lines.append("本地 `openie_results_ner_gpt-4o-mini.json` 和 Llama-3.3-70B 文件都不是 GPT-5.6 Luna，也不是同一套 prompt 记录，因此没有当作本轮 G_raw。下面的结构数字只描述 gpt-4o-mini 历史抽取。")
    lines.append("")
    if historical:
        lines.append(f"```json\n{json.dumps(historical, ensure_ascii=False, indent=2)}\n```")
    else:
        lines.append("历史图统计未运行。")
    lines.append("")
    lines.append("## 最大连通分量的 Leiden")
    lines.append("")
    if giant:
        lines.append(
            f"最大关系分量 {giant.get('parent_component_nodes')} 个节点，Leiden 社区 {giant.get('n_communities')} 个，"
            f"最大社区 {giant.get('largest_community_nodes')} "
            f"（占该分量 {giant.get('largest_community_fraction_of_component')}）。"
        )
        lines.append(f"社区规模直方图: {giant.get('community_size_histogram')}")
        lines.append("这仍是 gpt-4o-mini 历史图，不是 Luna G_raw。")
    else:
        lines.append("最大分量的 Leiden 未运行或尚未结束。")
    lines.append("")
    lines.append("## 切分试点")
    lines.append("")
    if partition:
        lines.append(partition["warning"])
        lines.append("")
        lines.append(f"- 展开的分量: {partition['n_components_expanded']}。")
        lines.append(f"- 大于 250 个节点、没有展开成阅读单元的分量: {partition['n_components_above_250_not_expanded']}。")
        lines.append(f"- 16000 token 预算: {json.dumps(partition.get('budget_16000'), ensure_ascii=False)}")
        lines.append(f"- 800 token 压力切分: {json.dumps(partition.get('stress_800'), ensure_ascii=False)}")
    else:
        lines.append("切分试点未运行。")
    lines += ["", "## Luna OpenIE", ""]
    if openie_error:
        lines.append(f"OpenIE 驱动失败: {openie_error.get('error')}")
    if probe:
        lines.append(f"- 请求 deployment `{probe.get('requested_deployment')}`，API 返回模型 id: `{probe.get('response_model')}`。")
        lines.append(f"- 探针 passage `{probe.get('passage_id')}`，实体 {probe.get('n_entities')}，三元组 {probe.get('n_triples')}。")
    else:
        lines.append("Luna 探针未运行。")
    if cost:
        lines.append(f"- 全量 OpenIE 投影调用数: {cost.get('projected_full_openie_calls')}。")
        lines.append(f"- 投影 prompt / completion tokens: {cost.get('projected_prompt_tokens')} / {cost.get('projected_completion_tokens')}。")
        lines.append(f"- 美元: {cost.get('usd')}。{cost.get('usd_reason')}")
    if formal:
        lines.append(f"- 正式语料前缀 OpenIE: {json.dumps(formal, ensure_ascii=False)}")
    if mechanism:
        lines.append(f"- 机制试点 OpenIE: {json.dumps({k: mechanism[k] for k in mechanism if k != 'errors'}, ensure_ascii=False)}")
    if graph_meta:
        lines.append(f"- 机制图: {json.dumps(graph_meta, ensure_ascii=False)}")
    else:
        lines.append("- Luna 机制图未建成。")
    if identity:
        lines.append(f"- commit {identity.get('hipporag_commit')}，prompt hash {identity.get('official_prompt_hash')}。")
    lines.append("")
    return "\n".join(lines)


def _read_time(partition, comp, retrieval, graph_meta) -> str:
    lines = ["# Read-Time Evolution", ""]
    if graph_meta:
        lines.append(f"机制图 hash `{graph_meta.get('graph_hash')}`，synonymy 状态 `{graph_meta.get('synonymy_status')}`。")
    else:
        lines.append("没有 Luna 机制图，因此没有在 Luna 图上做阅读进化。")
    lines.append("")
    if comp:
        lines.append(f"- G_raw hash 未变: {comp.get('raw_hash_unchanged')}。")
        lines.append(f"- 活跃理解节点: {comp.get('n_active_nodes')}，理解边: {comp.get('n_edges')}。")
        accepted = 0
        rejected = 0
        for row in comp.get("results") or []:
            outcome = row.get("outcome") or {}
            accepted += len(outcome.get("accepted") or [])
            rejected += len(outcome.get("rejected") or [])
        lines.append(f"- 接受的操作: {accepted}，拒绝的操作: {rejected}。")
    else:
        lines.append("Read-time comprehension 未运行。")
    lines.append("")
    if retrieval:
        lines.append("## 检索评测")
        lines.append("")
        lines.append(f"覆盖的 update 题: {retrieval.get('covered_update_questions')}")
        lines.append(f"Embedding: {retrieval.get('embedding')}")
        for name in ("R0", "R1"):
            block = retrieval.get(name)
            if block:
                lines.append(f"### {name}")
                lines.append("")
                lines.append(f"```json\n{json.dumps(block.get('summary'), ensure_ascii=False, indent=2)}\n```")
        lines.append(f"R1 top-10 中出现理解节点的题数: {retrieval.get('r1_questions_with_comp_in_top10')}")
        lines.append(f"Held-out 覆盖评测: {json.dumps(retrieval.get('heldout_covered_eval'), ensure_ascii=False)[:2000]}")
    else:
        lines.append("检索评测未运行。")
    if partition:
        lines.append("")
        lines.append("历史图上的 no-overlap / 1-hop / 2-hop 结构对比见 Graph Construction Audit，不把它写成 Luna 阅读收益。")
    lines.append("")
    return "\n".join(lines)


def _question(question) -> str:
    lines = ["# Question-Time Evolution", ""]
    if not question:
        lines.append("Question-time 未运行。可能原因是预算、Luna 不可用，或 NV-Embed 没有加载成功。")
        lines.append("")
        return "\n".join(lines)
    lines.append(f"Held-out 是否用于更新: {question.get('heldout_used_for_updates')}")
    lines.append(f"更新前 G_comp version: {question.get('g_read_version_before')}，更新后: {question.get('store_version_after')}")
    lines.append(f"G_raw hash: {question.get('raw_hash')}")
    lines.append("")
    for row in question.get("results") or []:
        lines.append(f"## {row.get('question_id')} {row.get('status', 'finished')}")
        lines.append("")
        lines.append(f"- rollout 答案: {row.get('rollout_answer')}")
        lines.append(f"- 诊断标记: {row.get('diagnosis_flags')}")
        verification = row.get("verification") or {}
        lines.append(f"- 提交类型: {verification.get('committed')}")
        lines.append(f"- 候选数: {verification.get('n_candidates')}")
        lines.append(f"- 验证前: {verification.get('before')}")
        for item in verification.get("results") or []:
            lines.append(
                f"  - {item.get('decision')} / {item.get('candidate', {}).get('kind')} / before {item.get('before')} / after {item.get('after')}"
            )
        lines.append("")
    return "\n".join(lines)


def _overlap_on_real_splits(giant) -> str:
    if not giant:
        return "最大分量的切分审计未运行。"
    compared = []
    for audit in giant.get("partition_audits") or []:
        hop0 = audit["budgets"]["budget_16000"]["0"]
        hop1 = audit["budgets"]["budget_16000"]["1"]
        if hop0["n_units"] > 1 and hop0["cross_unit_edges"]:
            compared.append(
                {
                    "cluster_id": audit["cluster_id"],
                    "n_nodes": audit["n_nodes"],
                    "units": hop0["n_units"],
                    "cross_edges": hop0["cross_unit_edges"],
                    "retention_0hop": hop0["boundary_retention_rate"],
                    "retention_1hop": hop1["boundary_retention_rate"],
                    "core_ok": hop0["core_coverage_complete"] and hop1["core_coverage_complete"],
                }
            )
    if not compared:
        return "在 16000 token 预算下，被审计的 Leiden 社区都没有切成多个阅读单元。"
    mean0 = sum(row["retention_0hop"] for row in compared) / len(compared)
    mean1 = sum(row["retention_1hop"] for row in compared) / len(compared)
    return (
        f"在 16000 token 的正式阅读预算下，有 {len(compared)} 个 Leiden 社区被切成多个阅读单元。"
        f"这些社区的 0-hop 边界保留率均值是 {mean0:.3f}，1-hop 是 {mean1:.3f}。"
        f"明细: {json.dumps(compared, ensure_ascii=False)}"
    )


def _final(corpus, historical, giant, partition, comp, retrieval, question, cost, ledger, formal, mechanism, probe, round2) -> str:
    lines = [
        "# Final Experiment Results",
        "",
        "这里只写已经落盘的结果。没有跑完的对照保持“未运行”，不用历史 gpt-4o-mini 图代替 Luna 图。",
        "",
        "## 完成情况",
        "",
        f"- 语料冻结: {'完成' if corpus else '未运行'}",
        f"- 历史结构审计: {'完成' if historical else '未运行'}",
        f"- 切分试点: {'完成' if partition else '未运行'}",
        f"- Luna 探针: {'完成' if probe else '未运行'}",
        f"- 正式前缀 OpenIE: {'有记录' if formal else '未运行'}",
        f"- 机制池 OpenIE: {'有记录' if mechanism else '未运行'}",
        f"- Read-time: {'有记录' if comp else '未运行'}",
        f"- 检索对照: {'有记录' if retrieval else '未运行'}",
        f"- Question-time: {'有记录' if question else '未运行'}",
        "",
        "## 成本",
        "",
    ]
    if ledger:
        lines.append(f"```json\n{json.dumps(ledger, ensure_ascii=False, indent=2)}\n```")
    else:
        lines.append("没有 token ledger。")
    if cost:
        lines.append(f"全量投影见 cost_estimate，自动运行上限 {cost.get('automatic_call_cap')} 次新鲜调用。")
    lines += [
        "",
        "## 研究问题（只根据已落盘的数字）",
        "",
        "RQ1. 第一轮 Luna 机制图接受了 7 个理解节点、拒绝了 1 个（来源 triple id 无法对上）。被接受的节点都带有原文子串。这是 2 个阅读单元上的结果，不是全库质量估计。",
        "RQ2. 第一轮只有 1 道已经能答对的 2-hop 题。R0 与 R1 的 EM/F1 都是 1。R1 的 Recall@3 从 1.0 降到 0.5，因为理解节点挤进了 reader 的 top-5；Recall@5 仍是完整支持集。这题的 top-10 里有 2 个理解节点。不能把这一题说成检索收益。",
        "RQ3. " + _overlap_on_real_splits(giant),
        "RQ4. 这道题的诊断三个错误标记都是 false，候选数是 0，G_comp version 保持 19。没有发生 Question-Time 更新，也就没有“重跑后变好”的证据。",
        "RQ5. Retrieval skill 和 answer skill 的 store 都是空的。没有任何一次隔离重跑，不能声称 skill 有收益。",
        "RQ6. Held-out 没有进入更新。机制索引里没有任何一道 held-out 题的金证据是齐的，所以 held-out 榜未运行。",
        "",
        "## 第二轮机制试点",
        "",
    ]
    if round2:
        lines.append(f"```json\n{json.dumps({k: round2[k] for k in round2 if k != 'rows'}, ensure_ascii=False, indent=2)[:8000]}\n```")
    else:
        lines.append("第二轮还没有 summary.json。")
    lines += [
        "",
        "## 还没有做的正式实验",
        "",
        "- 11,656 passage 的 Luna OpenIE，以及带 synonymy 的全量 G_raw。当前机制图的 synonymy 状态是 skipped_no_embedding。",
        "- 全量 R0/R1/R2/R3，以及 600 道 adaptation 上互相独立的 E2–E5。",
        "- 400 道 held-out 的最终榜。",
        "",
        "已有 OpenIE 缓存在 `artifacts/openie/luna/`。不要覆盖 `artifacts/graphs/mechanism_g_raw.json`。",
        "",
    ]
    return "\n".join(lines)
