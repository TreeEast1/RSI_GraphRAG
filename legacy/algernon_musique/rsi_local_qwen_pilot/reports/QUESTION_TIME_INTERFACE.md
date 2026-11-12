# Question-Time Evolution 接口草案

本草案只定义后续接口，不实现 Question-Time Agent。它接在本次 Read-Time 快照之后，并与 `Comprehension_Test_0929/Question-Time-Skill` 已有的 skill store 对齐：skill 仍是可版本化的方法，而不是把题目实体抄进提示。

## 位置

Question-Time 不改原始语料，也不在线重写正在服务的图。每一轮候选更新先进入隔离快照，重放原始 RAG 后再决定提交或回滚。

## 输入

`QuestionTimeCase`

| 字段 | 含义 |
|---|---|
| `qid` | 问题 ID |
| `query` | 原始问题 |
| `rollout_answer` | 当前 pipeline 的回答 |
| `gold_answer` | 只供离线评分，不进入模型的修订提示 |
| `retrieved_triples` | 本次检索到的三元组 |
| `retrieved_chunks` | 原始段落与 comprehension chunk |
| `gold_support_ids` | 支撑文档 ID，只供诊断对照 |
| `retrieval_trace` | query、top-k、融合分数、命中的 triple/chunk ID |
| `graph_snapshot_id` | Read-Time 已提交的图版本 |

## 诊断

同一道题可以同时给出两类 patch。

`ComprehensionError`：图没有把证据表达出来。允许的图操作与 Read-Time 相同。

- `ADD`
- `MODIFY`
- `DEACTIVATE`
- 新增 comprehension chunk

`StrategyError`：图里已有可用证据，但检索或作答策略没有用上。分成两支 skill。

- Retrieval Skill：query decomposition、检索路径、跳数、该看 triple 还是 passage。
- Answer Skill：证据取舍、多跳整合、答案格式。

不允许用模型自评作为接收条件。沿用现有 skill gate 的原则：只有隔离重放提高了目标题的 Recall 或 EM/F1，且历史题没有退化，才提交。

## Patch

`GraphPatch` 与 Read-Time 共用。

```json
{
  "patch_id": "qp-0001",
  "operation": "ADD",
  "target_id": "",
  "before": null,
  "after": {"head": "", "relation": "", "tail": ""},
  "comprehension_text": "",
  "source_document_ids": ["doc-..."],
  "evidence": "verbatim span",
  "validation_status": "pending"
}
```

`SkillPatch` 对应现有 `SkillStore.add / refine` 的字段。

```json
{
  "skill_id": "S0007",
  "skill_kind": "retrieval",
  "operation": "ADD",
  "trigger": "when the question needs a temporal bridge between two people",
  "retrieval_strategy": ["decompose into entity lookup then time constraint"],
  "reasoning_strategy": [],
  "answer_strategy": [],
  "provenance": {"qid": "", "graph_snapshot_id": ""},
  "validation_status": "pending"
}
```

`skill_kind=answer` 时只填 `answer_strategy`。一条 skill 不得同时塞入本题的实体、标题或金答案。这个约束与现在的 `leaks_question` 检查一致。

## 验证

固定预算，禁止无限递归。

1. 从当前已提交快照克隆隔离版本。
2. 在隔离版本上应用候选 Graph Patch 与 Skill Patch。证据不足的图更新直接拒绝，不进入重放。
3. 用原始 RAG 接口重答目标题：同一 embedding、同一 top-k、同一 reader。Skill 只改变 decomposition / 检索路径 / reader 可使用的方法提示。
4. 比较更新前后的 Recall@k、EM、F1 和检索 trace。
5. 在一小段冻结的历史题上重放，检查退化。
6. 目标题改善且历史题没有超过事先写明的退化阈值时，提交并记录版本。否则回滚隔离版本。
7. 每道题最多 `max_rounds` 轮（建议 2）。一轮可以同时提交已经通过的 graph patch 和 skill patch，但第二轮必须基于新快照重新检索，不能在同一轮里用模型输出再次调用自己。

## 版本

```text
snapshot_id = readtime 基线 + 已提交 question-time 轮次
graph_version += 1 仅当有 graph patch 提交
skill_version += 1 仅当有 skill patch 提交
```

失败的候选保留在 `rejected.jsonl`，不进入检索索引。

## 建议的第一段实现

先做 Retrieval Skill 的隔离重放，不做自由形式的多轮 agent。原因是 Read-Time 已经负责补图；Question-Time 最容易验证、也最容易伤害指标的变量，是检索路径而不是再生成一批三元组。Answer Skill 放在检索技能之后，避免两个技能同时改写答案导致无法归因。
