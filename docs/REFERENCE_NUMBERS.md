# 外部工作参考数字（REFERENCE_NUMBERS）

> 用途：给每个数据集提供"别人做到多高"的量级参考。
> **我们的数字和这些数字基本都不能直接比较**：模型（HY4 `no_think`）、题目子集、候选池、评测口径都不同。每张表下列出了口径差异。
> 数字来自各论文报告值，以及我们在 `docs/plans/` 和各阶段报告里整理的调研记录。引用到论文前请再核对原文。

| 数据集 | 主指标 | 外部最好（量级） | 我们的参考点 | 能否直接比 |
|---|---|---|---|---|
| MuSiQue | passage R@5 | HippoRAG2 74.7 | 无全库数字（封闭池 pilot R@5 .771） | ✗ |
| LoCoMo | LLM-Judge J | Full-context 72.9 / Mem0ᵍ 68.4 | 项目 F1 .491–.494；conv-30 官方 F1 .569 | ✗（不同指标） |
| QASPER | Answer F1 | LED-base 33.6；人类 60.9 | Dense 46.4；Oracle 48.7 | △（同指标，不同模型和子集） |
| DetectiveQA | Acc（四选一） | ComoRAG 68.2 | q_graph .683 / recent .805（41 题） | ✗（子集和 prefix 协议不同） |

---

## 1. MuSiQue（及其它多跳 QA）

HippoRAG2 论文设置：1,000 题子集，11,656 passage 语料，NV-Embed-v2 + Llama-3.3-70B-Instruct。

| 方法 | MuSiQue R@5 | 2Wiki R@5 | HotpotQA R@5 |
|---|---:|---:|---:|
| **HippoRAG2** | **74.7** | **90.4** | **96.3** |
| NV-Embed-v2（dense） | 69.7 | 76.5 | — |
| CogitoRAG（我们此前的工作） | 70.30 | — | — |

**口径差异**
- 我们在 MuSiQue 阶段没有完整的全库 HippoRAG2 + 理解层结果。
- 本地 pilot 用 1,816 篇封闭池（8 题）。
- v5 用字符串种子 PPR（100 道 dev 题的 R@5 为 .401）。
- 两者都不能和 74.7 比较。

## 2. LoCoMo

Mem0 论文协议：GPT-4o-mini 作答和作裁，类别 1–4（不含 adversarial），共 1,540 题，主指标 LLM-as-a-Judge（J）。

| 系统 | J（overall） |
|---|---:|
| Full-context | 72.9 |
| Mem0ᵍ（图记忆） | 68.4 |
| Mem0 | 66.9 |
| Zep | 66.0 |
| 最好的 RAG 配置 | 61.0 |
| A-Mem | 48.4 |

**口径差异**
- 我们报告两种口径，都是字符串匹配，不是 LLM 作裁：
  - SQuAD 风格 EM/F1（类别 1–4）。
  - 复刻的官方 LoCoMo F1（带词干）。
- 字符串匹配对"顺序、and、词形、日期粒度"很脆，同一答案可以是 0 / 1 / 0.667。
- 我们一致的观察是 **Full-context ≈ 最好的图**（test F1 .493 vs .492），和 Mem0 论文中 Full-context 最高的结论方向相同。
- LoCoMo 原论文报告的 F1 采用官方 F1 口径（不同模型差异很大），我们复刻了该口径的代码（`src/prompt_optimization/official_metrics.py`），但只在 conv-30 上系统报告。

## 3. QASPER

QASPER v0.3 原论文（Dasigi et al., 2021）：

| 方法 | Answer F1 | Evidence F1 | 设置 |
|---|---:|---:|---|
| LED-base（test） | 33.63 | 29.85 | 全文微调 |
| LED-large（evidence） | — | 39.37 | 全文微调 |
| Human lower bound | 60.92 | 71.62 | 标注者互评 |

**口径差异**
- 我们是 HY4 zero-shot，4096 token 上下文，dev 100 篇 / 372 题样本。
- 我们的 Evidence F1 按 Reader 上下文里的段落算，4096 token 会拉低 precision。检索质量以 Any/All@k 为准。
- 我们的 Answer F1（Dense .464 / Full paper .476 / Oracle .487）已高于微调 LED-base，但远低于人类下界；Oracle 也只有 .487，说明 Reader 是上限。

## 4. DetectiveQA

ComoRAG（AAAI'26）报告，GPT-4o-mini + BGE-M3，全文检索：

| 方法 | Acc |
|---|---:|
| **ComoRAG** | **68.18** |
| RAPTOR + IRCoT | 64.77 |
| HippoRAGv2 + IRCoT | 62.50 |

**口径差异**
- 我们只有 41 道 human 题（6 部小说）。
- 我们用 prefix-safe 检索：只看 `answer_position` 之前的文本。
- Reader 是 HY4，而且可以看到选项。
- 我们发现 `recent`（cutoff 前 4096 token）就有 .805，参数记忆基线有 .463。这说明该数据集上的绝对准确率很大程度上反映的是局部性和剧透，而不是长程检索能力。详见 [04_detectiveqa](../experiments/04_detectiveqa/README.md)。

---

## 5. 拿外部数字时的注意事项

1. **先对齐指标**：J ≠ F1 ≠ EM；官方 LoCoMo F1 比 SQuAD F1 高 3–4pp。
2. **先对齐池**：MuSiQue 全库 11,656 和封闭池 1,816 不是同一个问题；LoCoMo 每题只有 19–32 个候选。
3. **先对齐协议**：DetectiveQA 是否做 prefix 截断、是否给选项，结果可以差很多。
4. **报必报基线**：Full-context（LoCoMo / QASPER）、recent 和 memory（DetectiveQA）应该出现在每张表里。
