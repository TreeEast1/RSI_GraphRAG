# Final Experiment Results

这里只写已经落盘的结果。没有跑完的对照保持“未运行”，不用历史 gpt-4o-mini 图代替 Luna 图。

## 完成情况

- 语料冻结: 完成
- 历史结构审计: 完成
- 切分试点: 完成
- Luna 探针: 完成
- 正式前缀 OpenIE: 有记录
- 机制池 OpenIE: 有记录
- Read-time: 有记录
- 检索对照: 有记录
- Question-time: 有记录

## 成本

```json
{
  "max_fresh_calls": 80,
  "fresh_calls": 53,
  "cached_calls": 1,
  "errors": 0,
  "by_stage": {
    "openie_ner": {
      "fresh_calls": 24,
      "cached_calls": 0,
      "errors": 0,
      "prompt_tokens": 6288.0,
      "completion_tokens": 2595.0,
      "reasoning_tokens": 975.0
    },
    "openie_triple": {
      "fresh_calls": 24,
      "cached_calls": 0,
      "errors": 0,
      "prompt_tokens": 14805.0,
      "completion_tokens": 6978.0,
      "reasoning_tokens": 893.0
    },
    "read_time_comprehension": {
      "fresh_calls": 2,
      "cached_calls": 0,
      "errors": 0,
      "prompt_tokens": 9229.0,
      "completion_tokens": 5358.0,
      "reasoning_tokens": 1765.0
    },
    "qa_reader": {
      "fresh_calls": 2,
      "cached_calls": 1,
      "errors": 0,
      "prompt_tokens": 3766.0,
      "completion_tokens": 215.0,
      "reasoning_tokens": 97.0
    },
    "question_time_analyzer": {
      "fresh_calls": 1,
      "cached_calls": 0,
      "errors": 0,
      "prompt_tokens": 2799.0,
      "completion_tokens": 132.0,
      "reasoning_tokens": 44.0
    }
  },
  "response_model_ids": {
    "gpt-5.6-luna-2026-07-09": 53
  }
}
```
全量投影见 cost_estimate，自动运行上限 80 次新鲜调用。

## 研究问题

RQ1. 历史图上的切分可以检查阅读单元是否覆盖 core、是否保留边界。Luna 理解节点是否高质量，只以 comprehension_pilot 里接受/拒绝的真实条数为准。
RQ2. 只有 R0 与 R1 都写出了 Recall / EM / F1，并且记录了 top-10 里的理解节点，才能说理解节点改变了检索。否则本题没有答案。
RQ3. 历史图切分试点比较了 0-hop、1-hop、2-hop 的边界保留率。这是结构结果，不是 QA 收益。
RQ4. Question-time 只有在 verification 里出现真实 RAG 重跑后的 before/after 才算改善。诊断文本本身不算。
RQ5. Retrieval skill 和 answer skill 必须各自经过隔离重跑。没有这两组 before/after，就不能声称它们有泛化收益。
RQ6. 400 道 held-out 的全量评测本次如果没有覆盖索引，就明确写成未运行，不能从 adaptation 试点外推。

## 还没有做的正式实验

- 11,656 passage 的 Luna OpenIE 和带 synonymy 的全量 G_raw。
- 全量 R0/R1/R2/R3。
- 600 道 adaptation 的 Question-time，以及从同一份 G_read 出发的 E2–E5。
- 400 道 held-out 的最终榜。

续跑同一命令会跳过已存在的语料、切分试点和已缓存的 OpenIE 文件。提高 `configs/default.json` 里的 `max_fresh_llm_calls` 后才会继续消耗 Luna。
