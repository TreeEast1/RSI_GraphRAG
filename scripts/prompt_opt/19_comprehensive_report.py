#!/usr/bin/env python
"""Assemble one comprehensive document with all results and details.

Part I  = the full narrative report (FINAL_REPORT.md, headings demoted)
Part II = generated appendices: prompt texts, 105-question master tables,
          every flip, every relation, all 20 cases, full error attribution,
          oracle/repeat/adversarial/truncation details, experiment log.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import PATHS  # noqa: E402
from src.prompt_optimization import comprehension_prompts as CP  # noqa: E402
from src.prompt_optimization import reader_prompts as RP  # noqa: E402
from src.prompt_optimization.runner import evaluate  # noqa: E402

OUT = ROOT / "outputs" / "prompt_optimization_conv30"
BASE = [json.loads(l) for l in open(PATHS["qa"] / "baselines_hipporag_conv-30.jsonl", encoding="utf-8")]
SPLIT = json.loads((OUT / "splits" / "split_dev40_val65.json").read_text())
DEV, VAL = set(SPLIT["dev_qids"]), set(SPLIT["val_qids"])


def esc(x, limit=None):
    s = str(x if x is not None else "").replace("|", "\\|").replace("\n", " ").strip()
    if limit and len(s) > limit:
        s = s[: limit - 1] + "…"
    return s


def load_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def read_answers(pid):
    out = {}
    for sub in ("pilot", "final"):
        p = OUT / "reader" / sub / f"results_{pid}.jsonl"
        for r in load_jsonl(p):
            out[r["qidx"]] = r.get("pred_answer")
    return out


def joint_answers(variant):
    p = OUT / "joint" / f"{variant}.jsonl"
    return {r["qidx"]: r.get("pred_answer") for r in load_jsonl(p)}


def row_metrics(ans, q):
    b = BASE[q]
    return evaluate(ans, b["gold_answer"], int(b["category"]))


def demote(md: str) -> str:
    out, in_fence = [], False
    for line in md.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence and line.startswith("#"):
            out.append("#" + line)
        else:
            out.append(line)
    return "\n".join(out)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main():
    # ---------- load answers ----------
    p0 = {i: r.get("pred_answer") for i, r in enumerate(BASE)}
    p7, p8 = read_answers("P7"), read_answers("P8")
    e2 = joint_answers("cmpo_compp4_P0")
    e3 = joint_answers("cmpo_compp4_P7")
    c2p7 = joint_answers("C2_P7")
    nr_g0 = joint_answers("G_raw_top5_P0")
    nr_g7 = joint_answers("G_raw_top5_P7")
    nr_p0 = joint_answers("COMP-P4_top5_P0")
    nr_p7 = joint_answers("COMP-P4_top5_P7")

    # ---------- retrieval records ----------
    def recs(root, name):
        p = OUT / root / name / "records.jsonl"
        return {r["qid"]: r for r in load_jsonl(p)} if p.exists() else {}

    R = {
        "G_raw": recs("comprehension/retrieval", "G_raw"),
        "C2": recs("comprehension/retrieval", "C2"),
        "P4": recs("comprehension/retrieval", "cmpo_compp4"),
        "P4c10": recs("comprehension/retrieval", "cmpo_compp4c10"),
        "P4nr": recs("comprehension/retrieval_norerank", "COMP-P4_top5"),
        "C2nr": recs("comprehension/retrieval_norerank", "C2_top5"),
        "Gnr": recs("comprehension/retrieval_norerank", "G_raw_top5"),
    }

    def ev(d, q):
        r = d.get(q)
        return r["budget"]["4096"]["evidence_full"] if r else None

    def ranks(d, q):
        r = d.get(q)
        if not r:
            return []
        return [r["ranked_docs"].index(g) + 1 for g in r["gold_doc_ids"] if g in r["ranked_docs"]]

    def fallback(d, q):
        r = d.get(q)
        return r["diag"]["fallback_dense"] if r else None

    def r5(d, q):
        r = d.get(q)
        return round(r["retrieval"]["recall@5"], 3) if r else None

    def delta_table(title, a_ans, b_ans, a_name, b_name, only_em=True):
        rows = []
        for q in range(105):
            ca, cb = int(BASE[q]["category"]), int(BASE[q]["category"])
            ea, eb = row_metrics(a_ans.get(q), q), row_metrics(b_ans.get(q), q)
            if only_em and ea["em"] == eb["em"]:
                continue
            rows.append([f"q{q}", str(ca), esc(BASE[q]["gold_answer"], 55),
                         esc(a_ans.get(q), 60), f"{ea['em']:.0f} / {ea['f1']:.2f}",
                         esc(b_ans.get(q), 60), f"{eb['em']:.0f} / {eb['f1']:.2f}"])
        md = [f"### {title}", "",
              table(["qid", "cat", "gold", a_name, f"{a_name} EM/F1", b_name, f"{b_name} EM/F1"], rows),
              ""]
        md.append(f"共 {len(rows)} 题 EM 发生变化。")
        return "\n".join(md)

    L = []
    L.append("# conv-30 Prompt 优化实验：完整结果与全部细节\n")
    L.append("> 本文档把本轮所有实验（Reader Prompt P0–P8、Comprehension Prompt COMP-P0–P5、"
             "关系数量控制、Oracle/Full-Context 诊断、重复运行、官方评分口径、官方 cat-5 协议、"
             "以及关闭 LLM fact reranker 的追加消融）整合到一个文件里。\n"
             "> 范围：LoCoMo conv-30，19 sessions / 105 questions（81 可答、24 对抗）。"
             "基座：官方 HippoRAG2 + HY4（no_think, temperature=0）+ NV-Embed-v2。\n"
             "> 所有历史产物只读；新产物在 `outputs/prompt_optimization_conv30/`。\n")
    L.append("## 目录\n")
    L.append("""
**Part I — 叙述报告与结论**（原 `FINAL_REPORT.md` 全文）

- §0 执行摘要
- §1 原始 Prompt 审计（Reader / Comprehension / 检索路径）
- §2 QA 低分原因分析（原始指标、官方口径、错误归因、Oracle、波动、截断）
- §3 Reader Prompt 实验结果（dev / val / 全量 / 最佳方案）
- §4 Comprehension Prompt 实验结果（关系质量、检索、分层、机制、数量控制、**关闭 LLM reranker 消融**）
- §5 Graph × Reader 联合实验（E0–E3、C2+P7）
- §6 正负案例（20 题）
- §7 Q1–Q10
- §8 复现信息与限制

**Part II — 数据附录**

- 附录 0：全部指标总览（Reader / Comprehension / 无重排 / 联合 / Oracle / 重复 / 官方 cat-5）
- 附录 A：所有 Prompt 原文（Reader P0–P8、Comprehension COMP-P0–P5）
- 附录 B：105 题总表（问题、金标、各系统答案与 EM/F1）
- 附录 C：105 题检索状态总表（C2 / P4 / P4-c10 / P4-no-rerank 的 Evidence、gold rank、fallback、R@5）
- 附录 D：所有 EM 翻转逐题清单（P7 vs P0、E2 vs E0、E3 vs E1、no-rerank vs 正常）
- 附录 E：所有 Comprehension 关系清单（C3 / C6 / COMP-P1–P5 / P4-c10）
- 附录 F：20 题完整案例（含每图 top-5、gold rank、Ev、fallback、top facts、四种 Reader 答案）
- 附录 G：105 题完整错误归因（A–H 标签与理由）
- 附录 H：Oracle、重复运行、官方 cat-5、截断敏感性、HY4 调用统计
- 附录 I：完整实验日志（命令、shard、产物）

---
""")

    # ---------------- Part I ----------------
    L.append("# Part I — 叙述报告与结论\n")
    part1 = demote((OUT / "reports" / "FINAL_REPORT.md").read_text(encoding="utf-8"))
    part1 = part1.replace("### 附录 A：本案的核心数字速查", "### 原报告附录 A：核心数字速查")
    part1 = part1.replace("### 附录 B：关键文件索引", "### 原报告附录 B：关键文件索引")
    L.append(part1)
    L.append("\n\n---\n\n# Part II — 数据附录\n")

    # ---------------- Appendix 0: all metric overview ----------------
    L.append("## 附录 0：全部指标总览\n")
    L.append("### 0.1 Reader Prompt（P0–P8）\n")
    L.append("dev=40 题（含全部 11 道 cat1），val=65 题，all=105 题。P1–P6 只在 dev 上跑；P7/P8 在 dev+val 上跑。\n")
    answer_sets = {"P0": p0, "P7": p7, "P8": p8}
    for pid in ("P1", "P2", "P3", "P4", "P5", "P6"):
        answer_sets[pid] = read_answers(pid)

    def rmetrics(ans, qids):
        ans_rows = [q for q in qids if int(BASE[q]["category"]) != 5]
        adv_rows = [q for q in qids if int(BASE[q]["category"]) == 5]
        ms = [row_metrics(ans.get(q), q) for q in ans_rows]
        ma = [row_metrics(ans.get(q), q) for q in adv_rows]
        by_c = {}
        for c in (1, 2, 3, 4):
            sub = [q for q in ans_rows if int(BASE[q]["category"]) == c]
            if sub:
                mm = [row_metrics(ans.get(q), q) for q in sub]
                by_c[c] = (float(np.mean([m["f1"] for m in mm])),
                           float(np.mean([m["em"] for m in mm])))
        return {
            "n": len(qids), "em": float(np.mean([m["em"] for m in ms])),
            "f1": float(np.mean([m["f1"] for m in ms])),
            "off": float(np.mean([m["official_f1"] for m in ms])),
            "ref": float(np.mean([m["refused"] for m in ms])),
            "adv": float(np.mean([m["em"] for m in ma])) if ma else float("nan"),
            "words": float(np.mean([len(str(ans.get(q) or "").split()) for q in qids])),
            "by_cat": by_c,
        }

    rows = []
    for pid in ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
        split_map = {"dev": DEV}
        if pid in ("P0", "P7", "P8"):
            split_map["val"] = VAL
            split_map["all"] = set(range(105))
        for split, qs in split_map.items():
            m = rmetrics(answer_sets[pid], qs)
            rows.append([pid, split, str(m["n"]), f"{m['em']:.3f}", f"{m['f1']:.3f}",
                         f"{m['off']:.3f}", f"{m['ref']:.3f}",
                         f"{m['adv']:.3f}" if not np.isnan(m["adv"]) else "—",
                         f"{m['words']:.1f}",
                         f"{m['by_cat'].get(1,(float('nan'),)*2)[0]:.3f}" if 1 in m["by_cat"] else "—",
                         f"{m['by_cat'].get(2,(float('nan'),)*2)[0]:.3f}" if 2 in m["by_cat"] else "—",
                         f"{m['by_cat'].get(4,(float('nan'),)*2)[0]:.3f}" if 4 in m["by_cat"] else "—"])
    L.append(table(["pid", "split", "n", "EM", "F1", "offF1", "refusal", "adv refusal", "words",
                    "cat1 F1", "cat2 F1", "cat4 F1"], rows) + "\n")

    L.append("### 0.2 Comprehension 检索（105 题）\n")
    cs = json.loads((OUT / "comprehension" / "retrieval" / "summary.json").read_text())["summaries"]
    rel_counts = {}
    for pid in ("COMP-P1", "COMP-P2", "COMP-P3", "COMP-P4", "COMP-P5"):
        p = OUT / "comprehension" / "extractions" / f"{pid}.json"
        if p.exists():
            rel_counts[f"cmpo_{pid.replace('-', '').lower()}"] = json.loads(p.read_text())["n_relations"]
    p = OUT / "comprehension" / "extractions" / "COMP-P4-c10.json"
    if p.exists():
        rel_counts["cmpo_compp4c10"] = json.loads(p.read_text())["n_relations"]
    rows = []
    for name, s in cs.items():
        rows.append([name, str(rel_counts.get(name, 0)),
                     f"{s['recall@1']:.4f}", f"{s['recall@3']:.4f}", f"{s['recall@5']:.4f}",
                     f"{s['mrr']:.4f}", f"{s['ndcg@10']:.4f}", f"{s['evidence_full@4096']:.4f}",
                     str(s.get("fallback_dense_count", ""))])
    L.append(table(["graph", "relations", "R@1", "R@3", "R@5", "MRR", "nDCG@10", "Ev@4096", "fallback"], rows) + "\n")

    L.append("### 0.3 无 LLM rerank 消融（105 题）\n")
    nr = json.loads((OUT / "comprehension" / "retrieval_norerank" / "summary.json").read_text())["summaries"]
    rows = []
    for name, s in nr.items():
        rows.append([name, f"{s['recall@3']:.4f}", f"{s['recall@5']:.4f}", f"{s['mrr']:.4f}",
                     f"{s['ndcg@10']:.4f}", f"{s['evidence_full@4096']:.4f}",
                     str(s.get("fallback_dense_count", ""))])
    L.append(table(["graph", "R@3", "R@5", "MRR", "nDCG@10", "Ev@4096", "fallback"], rows) + "\n")

    L.append("### 0.4 联合 QA（105 题）\n")
    js = json.loads((OUT / "joint" / "summary.json").read_text())["summaries"]
    rows = []
    for k, s in js.items():
        rows.append([esc(k.replace("|", " | ")), str(s["n"]), f"{s['project_em']:.3f}",
                     f"{s['project_f1']:.3f}", f"{s['official_f1']:.3f}",
                     f"{s['refusal_rate_answerable']:.3f}",
                     f"{s['adversarial_refusal_project']:.3f}"])
    L.append(table(["graph\\|reader", "n", "EM", "F1", "offF1", "refusal", "adv"], rows) + "\n")

    L.append("### 0.5 Oracle / 重复 / 官方 cat-5（汇总数字）\n")
    oracle_s = json.loads((OUT / "reader" / "oracle" / "summary.json").read_text())["summaries"]
    rep_s = json.loads((OUT / "reader" / "repeats" / "summary.json").read_text())["summaries"]
    adv_s = json.loads((OUT / "reader" / "official_adversarial" / "summary.json").read_text())["summaries"]
    rows = [["Oracle " + k, f"{v['project_em']:.3f}", f"{v['project_f1']:.3f}",
             f"{v['official_f1']:.3f}", f"{v['refusal_rate_answerable']:.3f}",
             f"{v.get('mean_context_tokens',0):.0f}"] for k, v in oracle_s.items()]
    L.append(table(["condition", "EM", "F1", "offF1", "refusal", "ctx tokens"], rows) + "\n")
    rows = [["repeat " + k, f"{v['wording_unstable_rate']:.3f}", f"{v['scored_unstable_rate']:.3f}",
             f"{v['mean_pair_jaccard']:.3f}", str([round(x, 3) for x in v["em_per_run"]]),
             f"{v['majority_vote_em']:.3f}"] for k, v in rep_s.items()]
    L.append(table(["run", "wording unstable", "scored unstable", "Jaccard", "EM/run", "majority EM"], rows) + "\n")
    rows = [["official cat5 " + k, f"{v['accuracy']:.3f}", str(v["n"])] for k, v in adv_s.items()]
    L.append(table(["condition", "accuracy", "n"], rows) + "\n")

    # ---------------- Appendix A: prompts ----------------
    L.append("## 附录 A：所有 Prompt 原文\n")
    L.append("### A.1 Reader Prompt P0–P8\n")
    for p in RP.ALL_PROMPTS:
        L.append(f"#### {p.pid} — {p.name}\n")
        L.append(f"*notes*: {p.notes}\n")
        L.append("**System**\n\n```text\n" + p.system + "\n```\n")
        L.append("**User template**\n\n```text\n" + p.template + "\n```\n")
    L.append("### A.2 Comprehension Prompt COMP-P0–P5\n")
    for p in CP.ALL:
        L.append(f"#### {p.pid} — {p.name}\n")
        L.append("**System**\n\n```text\n" + p.system + "\n```\n")
        L.append("**Instructions**\n\n```text\n" + p.instructions + "\n```\n")
    L.append("### A.3 官方 LoCoMo QA Prompt（原文对照）\n")
    L.append("""```text
# 普通问题（cat 1/2/3/4）
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:

# cat-5 对抗题
Based on the above context, answer the following question.

Question: {} Short answer:
```

官方在 cat-2 的 question 后追加：`Use DATE of CONVERSATION to answer with an approximate date.`；
cat-5 则改写成 `(a) Not mentioned in the conversation (b) <诱导答案>` 的二选一。
""")

    # ---------------- Appendix B: master QA table ----------------
    L.append("## 附录 B：105 题总表（问题、金标、答案与 EM/F1）\n")
    L.append("列说明：E0=G_raw+P0，E1=G_raw+P7，E2=P4+P0，E3=P4+P7；"
             "no-rerank 两列是 COMP-P4+no-rerank top5 配 P0/P7。答案过长时截断到 90 字符。\n")
    rows = []
    for q in range(105):
        b = BASE[q]
        m0, m1 = row_metrics(p0.get(q), q), row_metrics(p7.get(q), q)
        m2, m3 = row_metrics(e2.get(q), q), row_metrics(e3.get(q), q)
        mn0, mn3 = row_metrics(nr_p0.get(q), q), row_metrics(nr_p7.get(q), q)
        rows.append([f"q{q}", str(b["category"]), esc(b["question"], 90), esc(b["gold_answer"], 70),
                     esc(p0.get(q), 90), f"{m0['em']:.0f}/{m0['f1']:.2f}",
                     esc(p7.get(q), 90), f"{m1['em']:.0f}/{m1['f1']:.2f}",
                     esc(e2.get(q), 90), f"{m2['em']:.0f}/{m2['f1']:.2f}",
                     esc(e3.get(q), 90), f"{m3['em']:.0f}/{m3['f1']:.2f}",
                     esc(nr_p0.get(q), 80), f"{mn0['em']:.0f}/{mn0['f1']:.2f}",
                     esc(nr_p7.get(q), 80), f"{mn3['em']:.0f}/{mn3['f1']:.2f}"])
    L.append(table(["qid", "cat", "question", "gold", "E0 ans", "E0 EM/F1", "E1 ans", "E1 EM/F1",
                    "E2 ans", "E2 EM/F1", "E3 ans", "E3 EM/F1", "P4nr+P0 ans", "EM/F1",
                    "P4nr+P7 ans", "EM/F1"], rows))
    L.append("\n")

    # ---------------- Appendix C: retrieval master ----------------
    L.append("## 附录 C：105 题检索状态总表\n")
    L.append("Ev=Evidence@4096（1 表示全部金标 turn 所在 session 都进了 4096 上下文）；"
             "rank 是每个金标 session 在 ranked_docs 中的名次；fb=dense fallback；R@5 是 session 级 recall。\n")
    rows = []
    for q in range(105):
        qid = BASE[q]["qid"]
        rows.append([f"q{q}", str(BASE[q]["category"]),
                     f"{ev(R['C2'], qid)}", f"{ev(R['P4'], qid)}", f"{ev(R['P4c10'], qid)}",
                     f"{ev(R['P4nr'], qid)}", f"{ev(R['C2nr'], qid)}",
                     str(ranks(R['C2'], qid)), str(ranks(R['P4'], qid)), str(ranks(R['P4nr'], qid)),
                     f"{r5(R['C2'], qid)}", f"{r5(R['P4'], qid)}", f"{r5(R['P4nr'], qid)}",
                     str(fallback(R['C2'], qid)), str(fallback(R['P4'], qid)), str(fallback(R['P4nr'], qid))])
    L.append(table(["qid", "cat", "C2 Ev", "P4 Ev", "P4c10 Ev", "P4noRR Ev", "C2noRR Ev",
                    "C2 gold rank", "P4 gold rank", "P4noRR gold rank",
                    "C2 R@5", "P4 R@5", "P4noRR R@5", "C2 fb", "P4 fb", "P4noRR fb"], rows))
    L.append("\n")

    # ---------------- Appendix D: flips ----------------
    L.append("## 附录 D：所有 EM 翻转逐题清单\n")
    L.append(delta_table("D.1 P7 vs P0（G_raw，同 context，全 105 题）", p0, p7, "P0", "P7"))
    L.append("\n" + delta_table("D.2 E2 vs E0（P4 图 + P0 vs G_raw + P0）", p0, e2, "E0", "E2"))
    L.append("\n" + delta_table("D.3 E3 vs E1（P4 图 + P7 vs G_raw + P7）", p7, e3, "E1", "E3"))
    L.append("\n" + delta_table("D.4 P4+no-rerank+P7 vs E3（正常重排）", e3, nr_p7, "E3", "E3noRR"))
    L.append("\n" + delta_table("D.5 P4+no-rerank+P0 vs E2（正常重排）", e2, nr_p0, "E2", "E2noRR"))
    L.append("\n" + delta_table("D.6 G_raw+no-rerank+P0 vs E0", p0, nr_g0, "E0", "E0noRR"))
    L.append("\n" + delta_table("D.7 G_raw+no-rerank+P7 vs E1", p7, nr_g7, "E1", "E1noRR"))

    # ---------------- Appendix E: relations ----------------
    L.append("\n## 附录 E：所有 Comprehension 关系清单\n")
    def rel_table(title, relations, note=""):
        L.append(f"### {title}\n")
        if note:
            L.append(note + "\n")
        rows = []
        for i, r in enumerate(relations):
            rows.append([str(i), esc(r["subject"]), esc(r["predicate"]), esc(r["object"]),
                         esc(r.get("relation_type")), esc(r.get("derivation")),
                         str(r.get("source_sessions")), esc(r.get("confidence")),
                         esc(r.get("source_dia_ids"), 60), esc(r.get("evidence"), 90)])
        L.append(table(["#", "subject", "predicate", "object", "type", "deriv", "sessions",
                        "conf", "dia_ids", "evidence"], rows))
        L.append("")
    c3 = json.loads((PATHS["comprehension"] / "conv-30" / "cluster_hybrid_0.5_b8192_comp.json").read_text())
    c6 = json.loads((PATHS["comprehension"] / "conv-30" / "full_conv_comp.json").read_text())
    rel_table("E.1 COMP-P0 / C3（原版，26 条）", c3["relations"])
    rel_table("E.2 C6（全文原版，18 条）", c6["relations"])
    labels = {"COMP-P1": "E.3", "COMP-P2": "E.4", "COMP-P3": "E.5",
              "COMP-P4": "E.6", "COMP-P5": "E.7"}
    for pid, lab in labels.items():
        p = OUT / "comprehension" / "extractions" / f"{pid}.json"
        if p.exists():
            d = json.loads(p.read_text())
            rel_table(f"{lab} {pid} — {d['name']}（{d['n_relations']} 条）", d["relations"],
                      note=f"原始候选/合法/最终：{[(g['group_id'], g.get('n_raw'), len(g['relations']), len(g['rejects'])) for g in d['groups']]}")
    p = OUT / "comprehension" / "extractions" / "COMP-P4-c10.json"
    if p.exists():
        d = json.loads(p.read_text())
        rel_table("E.8 COMP-P4-c10（数量控制，10 条）", d["relations"])

    # ---------------- Appendix F: cases ----------------
    L.append("\n## 附录 F：20 题完整案例\n")
    L.append(demote((OUT / "cases" / "cases.md").read_text(encoding="utf-8")))

    # ---------------- Appendix G: error attribution ----------------
    L.append("\n## 附录 G：105 题完整错误归因（G_raw / P0）\n")
    errs = load_jsonl(OUT / "audit" / "error_analysis_conv30.jsonl")
    rows = []
    for r in errs:
        rows.append([f"q{r['qidx']}", str(r["category"]), esc(r["question"], 80),
                     esc(r["gold_answer"], 60), esc(r["pred_answer"], 70),
                     f"{r['project_em']:.0f}", f"{r['project_f1']:.2f}", f"{r['official_f1']:.2f}",
                     f"{r['evidence_full']:.0f}/{r['evidence_partial']:.2f}",
                     str(r["main_error"] or "-"), ",".join(r["secondary_errors"]) or "-",
                     esc(r["rationale"], 70)])
    L.append(table(["qid", "cat", "question", "gold", "pred", "EM", "F1", "offF1",
                    "Ev full/partial", "main", "secondary", "rationale"], rows))
    L.append("\n标签：A 检索错误；B 证据不完整；C 时间推理；D 多事实整合；"
             "E 表达/部分正确；F 过度拒答；G 错误推断/幻觉；H 评测器/语义等价。\n")

    # ---------------- Appendix H: diagnostics ----------------
    L.append("\n## 附录 H：诊断实验明细\n")
    oracle = json.loads((OUT / "reader" / "oracle" / "summary.json").read_text())["summaries"]
    rows = []
    for k, s in oracle.items():
        rows.append([k, str(s["n"]), f"{s['project_em']:.3f}", f"{s['project_f1']:.3f}",
                     f"{s['official_f1']:.3f}", f"{s['refusal_rate_answerable']:.3f}",
                     f"{s.get('mean_context_tokens', 0):.0f}"])
    L.append("### H.1 Oracle / Full-Context 汇总\n")
    L.append(table(["condition", "n", "EM", "F1", "offF1", "refusal", "ctx tokens"], rows) + "\n")

    rep = json.loads((OUT / "reader" / "repeats" / "summary.json").read_text())["summaries"]
    L.append("### H.2 重复运行（temperature=0，禁用缓存）\n")
    for pid, s in rep.items():
        L.append(f"**{pid}**：词面不稳定 {s['wording_unstable_rate']:.1%}，"
                 f"得分不稳定 {s['scored_unstable_rate']:.1%}，Jaccard {s['mean_pair_jaccard']:.3f}，"
                 f"每次 EM {[round(x,3) for x in s['em_per_run']]}，多数票 {s['majority_vote_em']:.3f}\n")
        rows = []
        for r in s["per_question"]:
            rows.append([f"q{r['qidx']}", str(r["category"]), esc(r["gold"], 50),
                         " / ".join(esc(a, 70) for a in r["answers"]),
                         str([int(e) for e in r["ems"]]), f"{r['mean_pair_jaccard']:.3f}",
                         str(int(r["majority_em"]))])
        L.append(table(["qid", "cat", "gold", "3 次答案", "3 次 EM", "Jaccard", "多数票 EM"], rows) + "\n")

    adv = json.loads((OUT / "reader" / "official_adversarial" / "summary.json").read_text())["summaries"]
    L.append("### H.3 官方 cat-5 多选协议（24 题）\n")
    rows = [[k, f"{v['accuracy']:.3f}", str(v["n"]), esc(v["parse_breakdown"])] for k, v in adv.items()]
    L.append(table(["prompt", "accuracy", "n", "parse"], rows) + "\n")
    for pid in ("OFFICIAL", "P0", "P7"):
        p = OUT / "reader" / "official_adversarial" / f"{pid}.jsonl"
        if not p.exists():
            continue
        rows = []
        for r in load_jsonl(p):
            rows.append([f"q{r['qidx']}", esc(r["question"], 80), esc(r["adversarial_answer"], 50),
                         esc(r["a_text"], 40), esc(r["b_text"], 40),
                         esc(r["raw"], 50), str(r["correct"])])
        L.append(f"**{pid} 逐题**\n")
        L.append(table(["qid", "question", "adversarial answer", "(a)", "(b)", "raw output", "correct"], rows) + "\n")

    ts = json.loads((OUT / "reader" / "truncation_sensitivity.json").read_text())
    L.append("### H.4 max_tokens=256 截断敏感性（512 重跑）\n")
    rows = [[esc(r["file"]), f"q{r['qidx']}", str(r["category"]), esc(r["gold"], 45),
             f"{r['em_256']:.0f}/{r['f1_256']:.2f}", f"{r['em_512']:.0f}/{r['f1_512']:.2f}",
             esc(r["finish_512"])] for r in ts]
    L.append(table(["file", "qid", "cat", "gold", "256 EM/F1", "512 EM/F1", "finish"], rows) + "\n")

    man = json.loads((OUT / "reports" / "run_manifest.json").read_text())
    L.append("### H.5 HY4 调用统计\n")
    rows = [[k, str(v["lines"]), str(v["ok"]), str(v["errors"])]
            for k, v in man["hy4_logs"].items() if any(x in k for x in ("callspo", "callsco", "callsor", "callsoa", "callsrp", "callsjo"))]
    L.append(table(["log", "lines", "ok", "errors"], rows) + "\n")
    L.append("注意：Comprehension 图的官方 LLM fact rerank 调用走 `outputs/raw_graphs/conv-30/llm_cache`，"
             "不在上表；无重排消融的检索调用为 0。\n")

    # ---------------- Appendix I: experiment log ----------------
    L.append("\n## 附录 I：完整实验日志\n")
    L.append((OUT / "reports" / "EXPERIMENT_LOG.md").read_text(encoding="utf-8"))

    doc = "\n".join(L)
    p = OUT / "reports" / "COMPREHENSIVE_REPORT.md"
    p.write_text(doc, encoding="utf-8")
    print(f"wrote {p} ({len(doc.splitlines())} lines, {len(doc)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
