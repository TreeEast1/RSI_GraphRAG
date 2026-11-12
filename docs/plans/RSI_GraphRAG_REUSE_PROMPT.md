# RSI GraphRAG 项目接入 Prompt

用途：把本文件内容附加到其他项目的任务中，让后续 Codex / LLM 快速理解
RSI GraphRAG 的位置、实验目标和需要保持的约束。

## 1. 访问方式

远程工作区：

```text
/group/40150/licochen/Memory/RSI_GraphRAG
```

当前 code-server 入口：

<https://launcher-fsuxaqe7fp-codeserver.ide.taiji.woa.com/?folder=/group/40150/licochen/Memory/RSI_GraphRAG>

SSH：

```bash
ssh -p 36000 root@licochen-cpu.devcloud.woa.com
```

2026-11-11 检查时，主机名曾解析到 `121.91.149.198`，但本机直连
`36000` 端口超时。若 SSH 不可达，优先使用 code-server，或从公司网络重新建立
SSH 连接。

本地同步副本：

```text
/Users/licochen/Desktop/Research/RSI GraphRAG/remote_mirror
```

## 2. 读取到的项目定位

RSI GraphRAG / CogitoRAG 研究 **Read-Time Comprehension Evolution ×
Question-Time Retrieval Skill**，底层使用官方 HippoRAG2。项目在 LoCoMo 和
QASPER 上研究三个问题：

1. 阅读阶段跨文档联合抽取和关系理解，能否改善知识图谱与检索；
2. 问题到来时的拆解、检索技能和答案技能，能否补足静态图谱遗漏；
3. 不更新模型权重的可版本化 Skill Memory（RSI 闭环）能否泛化。

项目最终结论强调：LoCoMo 上跨 session 联合抽取能改善部分检索指标，但 QA
收益很小；证据进入上下文后仍答错是主要瓶颈，因此 Reader Prompt 的覆盖度和
回答格式比继续增加少量关系更重要。

## 3. 可直接复用的 Prompt

```text
你正在协助 RSI GraphRAG / CogitoRAG 项目。项目位于：
/group/40150/licochen/Memory/RSI_GraphRAG

研究框架是 Read-Time Comprehension Evolution × Question-Time Retrieval
Skill，底座为 OSU-NLP-Group/HippoRAG @ d5c8329（HippoRAG2 v2.0.0a5）。
所有方法共用 HY4（TokenLink，reasoning_effort=no_think，temperature=0）
和 nvidia/NV-Embed-v2（fp16，4096 维）。

开始工作前先阅读：
- README.md
- EXPERIMENT_PLAN.md
- RSI_GraphRAG_FULL_METHOD_AND_EXPERIMENTS.md
- src/comprehension/prompts.py
- src/prompt_optimization/reader_prompts.py
- src/prompt_optimization/comprehension_prompts.py
- src/retrieval/reader.py
- src/qasper/prompts.py
- outputs/reports/START_HERE.md（如果存在）

必须保留以下实验约束：
1. 官方 HippoRAG 的 index()、retrieve()、PPR 和同义边逻辑不得随意修改。
   自研能力通过适配器、扩展点或独立变体注入。
2. 图谱变体不得直接改写 G_raw。复制原始 embedding store 和 OpenIE state 后，
   在独立目录中以 force_index_from_scratch=True 重跑官方 index()。
3. 方法比较必须共用相同 reader、top-20 截断和 token 预算；改变其中一项时，
   实验必须重新分组，不能直接与旧结果比较。
4. LoCoMo 固定 split seed 为 20240501。dev 为 conv-30、conv-44、conv-48，
   test 为其余 7 个 conversation。不要在看到 test 结果后调 prompt 或 skill。
5. 防止泄漏：LoCoMo 的 session_summary、observation、event_summary、gold
   answer 和 gold evidence 不得进入索引或生成路径。QASPER 的 question、
   gold answer 和 gold evidence 不得进入 read-time 抽取 prompt。
6. 抽取事实必须可追溯，保留 source ids、verbatim evidence，并拒绝重复、
   无支持、含糊或把相关性写成因果性的关系。
7. Reader 默认遵循项目接口：只输出一行 `Answer: <answer>`；无法回答时输出
   `Answer: No answer`。如果修改格式，必须同步 parser 和评测口径。
8. HY4 是共享配额。保留跨进程限速、内容寻址缓存和调用日志；重试风暴会浪费
   配额。正式实验应记录模型、数据 hash、配置、代码版本和 token 成本。

回答任务时先说明改动的实验阶段、预期隔离变量和评测口径。只报告真实运行过的
结果；区分开发集、测试集和 Oracle 设置。结论需要给出绝对指标、变化值、样本量
以及是否已经被噪声解释。
```

## 4. Prompt 资产索引

| 文件 | 用途 | 核心输入输出 |
|---|---|---|
| `src/comprehension/prompts.py` | LoCoMo 原子事实抽取、跨 session 理解、关系验证 | 对话片段和已有事实 → triples / relations → JSON verdicts |
| `src/prompt_optimization/comprehension_prompts.py` | COMP-P0 至 COMP-P5 理解候选、验证 prompt | 保持相同关系 JSON schema，便于替换和消融 |
| `src/retrieval/reader.py` | 所有 LoCoMo 方法共用的 QA reader | retrieved context + question → 单行 `Answer:` |
| `src/prompt_optimization/reader_prompts.py` | P0 至 P8 Reader 候选 | 基线、跨 session 汇总、时间推理、拒答校准、短答案、前提检查 |
| `src/qasper/prompts.py` | QASPER 原子事实、跨段理解、验证和摘要 | paper paragraphs → facts / insights / verdicts / summaries |
| `src/question_skill/skills.py` | 可版本化检索技能 | trigger、failure pattern、retrieval strategy |

## 5. 关键 Prompt 原则

- **原子抽取**：一条事实只连接一个 subject、predicate、object；证据必须来自
  明确文本，不能把多个段落合成一条原子事实。
- **跨文档理解**：新增关系至少引用两个不同 session/paragraph 的证据，且不能
  只是单段事实的改写；因果关系必须有文本支持。
- **验证**：检查 source id 是否存在、证据是否逐字支持、关系是否跨文档、
  是否重复已有事实、是否引入来源中不存在的实体或数字。
- **Reader**：完整回答优先于只追求短；日期题只给支持日期；列表题列全；在证据
  不足或问题前提不成立时明确返回 `No answer`。

## 6. 远程运行提醒

```bash
cd /group/40150/licochen/Memory/RSI_GraphRAG
source .venv/bin/activate
export TIKTOKEN_CACHE_DIR=$PWD/cache/tiktoken
export HF_HUB_OFFLINE=1
```

项目盘为 Ceph，小文件和顺序读较慢。`.venv` 和 NV-Embed-v2 使用本地磁盘路径；
GitHub / HuggingFace 访问使用项目 README 指定的公司代理。
