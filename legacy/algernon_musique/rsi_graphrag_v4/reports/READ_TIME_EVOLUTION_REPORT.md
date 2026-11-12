# Read-Time Evolution

机制图 hash `7cf3ea5138959b6eb303480d5c19b46f5850c254b245bf01971d11d0e9ed20c6`，synonymy 状态 `skipped_no_embedding`。

- G_raw hash 未变: True。
- 活跃理解节点: 7，理解边: 12。
- 接受的操作: 7，拒绝的操作: 1。

## 检索评测

覆盖的 update 题: ['2hop__77134_88165']
Embedding: {'device': 'cpu', 'model_path': '/data2-HDD-SATA-20T/nzq/huggingface_model/hub/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef', 'loaded': True}
### R0

```json
{
  "n": 1.0,
  "em": 1.0,
  "f1": 1.0,
  "Recall@1": 0.0,
  "CompleteSupportRecall@1": 0.0,
  "Recall@3": 1.0,
  "CompleteSupportRecall@3": 1.0,
  "Recall@5": 1.0,
  "CompleteSupportRecall@5": 1.0,
  "Recall@10": 1.0,
  "CompleteSupportRecall@10": 1.0,
  "by_hop": {
    "2": {
      "em": 1.0,
      "f1": 1.0,
      "Recall@1": 0.0,
      "Recall@3": 1.0,
      "Recall@5": 1.0,
      "Recall@10": 1.0
    }
  }
}
```
### R1

```json
{
  "n": 1.0,
  "em": 1.0,
  "f1": 1.0,
  "Recall@1": 0.0,
  "CompleteSupportRecall@1": 0.0,
  "Recall@3": 0.5,
  "CompleteSupportRecall@3": 0.0,
  "Recall@5": 1.0,
  "CompleteSupportRecall@5": 1.0,
  "Recall@10": 1.0,
  "CompleteSupportRecall@10": 1.0,
  "by_hop": {
    "2": {
      "em": 1.0,
      "f1": 1.0,
      "Recall@1": 0.0,
      "Recall@3": 0.5,
      "Recall@5": 1.0,
      "Recall@10": 1.0
    }
  }
}
```
R1 top-10 中出现理解节点的题数: 1
Held-out 覆盖评测: {"status": "not_run", "reason": "No held-out question has its full gold passage set inside the mechanism index."}

历史图上的 no-overlap / 1-hop / 2-hop 结构对比见 Graph Construction Audit，不把它写成 Luna 阅读收益。
