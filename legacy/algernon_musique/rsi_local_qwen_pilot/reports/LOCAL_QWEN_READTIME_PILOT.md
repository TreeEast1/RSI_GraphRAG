# 本地 Qwen 与 Read-Time Evolution 试点

实验目录：`TAG-experiment/rsi_local_qwen_pilot/`。没有覆盖 2Wiki / Luna 的历史输出。数字来自本目录里的 `results/summary.json`、`predictions/capability/qwen3.5-9b/` 和 `predictions/triples.jsonl`。

## 1. 本地 Qwen 部署

机器不是 4 张 RTX 4090。`nvidia-smi` 看到的是 4 张 RTX 3090，每张 24576 MiB。驱动 `530.30.02`，驱动报告的 CUDA 版本是 12.1，系统 `nvcc` 仍是 9.1。已有任务只动过空闲卡：GPU 1 上的 inkcam / concept-retrieval 没有被结束。

| GPU | 试点开始时 | 用途 |
|---|---|---|
| 0 | 空闲或仅有约 2.7GB 的 ink demo | NV-Embed-v2 |
| 1 | 约 9.6GB，两个已有进程 | 未使用 |
| 2 | 空闲或约 2.7GB ink demo | 试过 vLLM，未能生成 |
| 3 | 空闲 | Qwen3.5-9B 文本服务，约 19–22GB |

独立环境在本目录 `.venv`，没有升级其他项目的全局包。安装结果是 vLLM `0.21.0`、PyTorch `2.11.0+cu128`。vLLM 0.21 的扩展链接 `libcudart.so.13`，需要借用已有的 CUDA 13 运行库才能 import。权重可以加载：Qwen3.5-9B、BF16、text-only、max length 4096、权重约 16.8 GiB。随后 FlashAttention 报 `CUDA driver version is insufficient for CUDA runtime version`。`--enforce-eager` 也过不了这一步。因此 vLLM 没有成为可用服务。

实际在跑的服务是 transformers 的 OpenAI 兼容接口，模型 `Qwen/Qwen3.5-9B`，BF16，非思考模式，GPU 3，`http://127.0.0.1:18181/v1`，服务名 `qwen3.5-9b`。256 token 的测速大约 11 token/s。整段抽取 67454 个 completion token 用了 5558 秒，约 12 token/s。配置在 `configs/model.json`：model、base URL、api key、thinking、temperature、max output tokens、timeout。

`Qwen/Qwen3.8-27B-FP8` 已完整下载到 `/data2-HDD-SATA-20T/nzq/huggingface_model/Qwen3.8-27B-FP8`，81 个文件，30,890,049,597 字节，约 29GB。单张 24GB 3090 放不下；两张卡的 vLLM TP=2 又依赖上面那套跑不起来的 kernel。INT4 checkpoint 没有形成可加载权重。所以今晚没有 27B 的生成结果，也不能把 9B 的数字说成 27B 的结果。

调用方式：

```bash
curl --noproxy '*' http://127.0.0.1:18181/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-9b","temperature":0,"max_tokens":256,
       "chat_template_kwargs":{"enable_thinking":false},
       "messages":[{"role":"user","content":"hello"}]}'
```

Embedding 仍是本地 NV-Embed-v2，维度 4096，路径与 Reading-Time-loop 相同。

## 2. 模型能力

只跑了 Qwen3.5-9B，非思考模式。样本是冻结的 MuSiQue 池的前 24 题，不是额外编的题。金答案只用于打分。有一条池外记录被排除，下面按这 24 题去重。

| 任务 | 结果 |
|---|---|
| A 传统抽取，JSON 有效率 | 0.875 |
| A 三元组端点能在原文中找到 | 0.945（11 条无端点支持） |
| B 补全，JSON 有效率 | 0.958 |
| B 新增三元组端点支持率 | 1.0 |
| B 注入的假三元组被指出 | 0.917 |
| B 提出的 comprehension chunk | 29 |
| C 金证据给定时 EM / F1 | 0.583 / 0.704 |
| C 金证据给定时 JSON | 1.0 |
| C 只有干扰段落时弃答 | 1.0 |

24 题能力测试大约 1147 秒，41201 prompt tokens，19341 completion tokens。

9B 适合做格式稳定的抽取和“证据不够就说不知道”。金证据放在面前时 EM 只有 0.58，多跳整合还会把答案写成整句。它还不能单独替代 GPT 做图谱修订：下面的 Read-Time 里，16 个窗口全部返回空更新。

没有 27B 的同题结果，也没有重打 Luna API。分工只能写成：抽取和弃答可以用本地 9B；图修订和复杂多跳阅读仍没有本地 27B 证据。

## 3. MuSiQue 数据

来源是 `CogitoRAG-openreview/code/reproduce/dataset/musique.json`，1000 道可答题。随机种子 `20261027`，冻结前 100 题，名单在 `data/questions.json`。

实际执行的是这 100 题里的前 8 题。原因是服务大约 12 token/s，20 题对应大约 391 篇文档，来不及抽完。8 题对应 160 篇文档，160/160 都有抽取记录，其中 144 篇 JSON 有效。

检索语料是这 100 题自带段落的并集，1816 篇，含题内干扰段落。不是全库 11656 篇。因此下面的 Recall 是封闭池试点，不能和论文里 MuSiQue Recall@5 70.30% 比较。金答案和 `is_supporting` 没有进入抽取或 Read-Time prompt。Baseline 和 Evolved 用同一批文档、同一个问题集合。

## 4. Read-Time Evolution

传统建图：每篇文档一次抽取，输出实体和三元组。端点不能在该文档里找到的三元组丢掉。160 篇得到 1456 条通过的三元组，另有 18 条因端点不在原文中被拒绝。Query decomposition 固定为原问题本身。检索是 NV-Embed 的 passage / triple 融合，top-5 原始段落交给同一个 reader。

Read-Time：每 10 篇一个窗口，共 16 个窗口。模型看到窗口原文和已有三元组，可以提出 ADD / MODIFY / DEACTIVATE 和 comprehension chunk。引用必须落在窗口内，证据要能在原文中对上。16 个窗口 JSON 都有效，提出的 patch 是 0，拒绝的候选是 0，通过的 chunk 是 0。模型选择不改图，不是门控把提案杀掉。Read-Time 用了 70.4 秒、147217 prompt tokens、240 completion tokens。

|  | Recall@1 | Recall@3 | Recall@5 | Hit@5 | EM | F1 |
|---|---|---|---|---|---|---|
| Baseline | 0.375 | 0.625 | 0.771 | 0.875 | 0.375 | 0.552 |
| Evolved | 0.375 | 0.625 | 0.771 | 0.875 | 0.375 | 0.552 |

两组检索和答案逐题相同。QA 侧 baseline 10.1 秒 / 6733 prompt tokens / 123 completion tokens，evolved 9.2 秒 / 同样 6733 / 123。

8 题里的情况：

- 检索和答案都对：Karen Fairchild；Oriole Records；Autumn Leaves 的作者死于 13 August 1896。这三题传统图已经够用。
- 证据在 top-5 里，答案形式不对：高速公路题金答案是 `140 mi`，模型答 `140`，F1 0.67。Bourne 题金答案是小说名，模型把整句电影介绍都写上了，F1 0.50。
- 证据没找全：编剧的兄弟姐妹只召回一半，答案 `unknown`。Frozen Run 的死刑恢复时间召回 0.67，答案仍是 `unknown`。
- 四跳日光节约时题 Recall@5 为 0。模型答了 `last Sunday of October`，金答案是 `the first Sunday in April`。这是没检索到证据时的错误回答，不是图修订造成的。

没有因为错误新三元组导致下降的题，因为没有新三元组。也没有被接受的 chunk 帮到检索。传统抽取里被丢掉的 18 条是无端点支持的关系，没有进入索引。

## 5. 三个问题

1. Qwen3.8-27B 能不能承担图谱理解和进化，今晚没有答案。权重在盘上，服务没有跑起来。能下结论的是 Qwen3.5-9B：抽取和“证据不足就弃答”可用，金证据下的多跳 EM 是 0.58，完整传统图上的 Read-Time 没有产生任何更新。现在不能把 GPT 全部换成本地模型。

2. 先传统建图、再理解修图，在这 8 题的完整抽取上没有收益。Baseline 和 Evolved 的 Recall、EM、F1 相同。9B 在非思考模式下把窗口判成“不用改”。传统 OpenIE 已经留下 1456 条三元组，模型没有再补隐含关系。

3. Question-Time 应先做 Answer Skill，而不是再做一轮自由的图生成。8 题里 5 题 Recall@5 已经是 1，EM 只有 3 题。失败主要是答案太长、单位丢失、证据到了但仍答 unknown。接口草案在 `reports/QUESTION_TIME_INTERFACE.md`：同一题可以同时出 Graph Patch 和 Skill Patch，隔离快照里重放原来的 RAG，最多两轮，历史题退化就回滚。Retrieval Skill 排在 Answer Skill 后面，因为这批评测里检索不是主要缺口。图修订要等 27B 或思考模式真的能提出非空 patch，再单独测。

## 复现

```bash
cd TAG-experiment/rsi_local_qwen_pilot
# 服务需已在 127.0.0.1:18181 提供 qwen3.5-9b
/home/nzq/anaconda3/bin/python -u src/prepare_musique.py
/home/nzq/anaconda3/bin/python -u src/capability_eval.py
/home/nzq/anaconda3/bin/python -u src/run_readtime.py extract
CUDA_VISIBLE_DEVICES=0 /data2-HDD-SATA-20T/nzq/conda_envs/echorag/bin/python -u src/embed_nv.py --what triples
/home/nzq/anaconda3/bin/python -u src/run_readtime.py baseline
/home/nzq/anaconda3/bin/python -u src/run_readtime.py readtime
/home/nzq/anaconda3/bin/python -u src/run_readtime.py evolved
/home/nzq/anaconda3/bin/python -u src/run_readtime.py summary
```

种子、题号和文档 ID 写在 `data/`。完整对照表在 `results/summary.json`。
