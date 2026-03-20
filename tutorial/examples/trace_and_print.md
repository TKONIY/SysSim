# `trace_and_print.py` — 多算子模型追踪与逐节点耗时打印示例

## 文件概述

`examples/trace_and_print.py` 是一个端到端的演示脚本，展示了 SysSim 的核心用法。它定义了一个包含多种算子类型（GEMM、Attention、Conv、BatchNorm、LayerNorm、Dropout 等）的自定义模型 `DiverseModel`，然后分别在训练模式、推理 Prefill 模式和推理 Decode 模式下进行追踪，并以表格形式打印每个算子的名称、类型、耗时、开始/结束时间及所在流。

这是理解 SysSim 工作流程的最佳入门示例。

## 关键代码解析

### 1. 自定义模型 `DiverseModel`

```python
class DiverseModel(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        # Conv + BatchNorm block (COMPUTE)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(embed_dim)

        # Attention projections (GEMM)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Feed-forward (GEMM + COMPUTE)
        self.ln = nn.LayerNorm(embed_dim)
        self.ff1 = nn.Linear(embed_dim, embed_dim * 4)
        self.ff2 = nn.Linear(embed_dim * 4, embed_dim)

        # Final (COMPUTE)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.1)
```

该模型特意覆盖了多种算子类型，使追踪结果能体现 SysSim 对不同算子的分类和耗时估算能力：

| 模块 | 对应算子类型 | 说明 |
|------|-------------|------|
| `Conv1d` | COMPUTE | 一维卷积 |
| `BatchNorm1d` | COMPUTE | 批量归一化 |
| `Linear` (qkv_proj, out_proj, ff1, ff2) | GEMM | 矩阵乘法 |
| `scaled_dot_product_attention` | ATTENTION | 注意力计算 |
| `LayerNorm` | COMPUTE | 层归一化 |
| `GELU` | COMPUTE | 激活函数 |
| `Dropout` | COMPUTE | 随机丢弃 |
| `Softmax` | COMPUTE | 概率归一化 |

### 2. 前向传播逻辑

```python
def forward(self, x):
    b, s, d = x.shape

    # Conv + BatchNorm (COMPUTE ops)
    h = self.conv(x.transpose(1, 2)).transpose(1, 2)
    h = self.bn(h.transpose(1, 2)).transpose(1, 2)
    h = F.gelu(h)
    x = x + h

    # Multi-head attention (GEMM + ATTENTION)
    qkv = self.qkv_proj(x)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
    attn_out = F.scaled_dot_product_attention(q, k, v)
    attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, d)
    x = x + self.out_proj(attn_out)

    # Feed-forward (GEMM + COMPUTE)
    h = self.ln(x)
    h = self.ff1(h)
    h = F.gelu(h)
    h = self.dropout(h)
    h = self.ff2(h)
    x = x + h

    return self.softmax(x)
```

前向传播依次经过 Conv+BN 块、多头注意力块和前馈网络块，体现了典型 Transformer 变体的结构。注意使用了 PyTorch 原生的 `F.scaled_dot_product_attention` 而非手动实现注意力。

### 3. 表格打印函数 `print_table`

```python
def print_table(graph, title):
    graph.compute_critical_path()

    w_name = max(len(op.name) for op in graph.operators.values())
    w_name = max(w_name, 4)
    header = (
        f"{'Name':<{w_name}}  {'Type':<10}  {'Time (ms)':>12}  "
        f"{'Start (ms)':>12}  {'Finish (ms)':>12}  {'Stream':>6}"
    )
    ...
    for name in graph.topological_sort():
        op = graph.operators[name]
        print(
            f"{op.name:<{w_name}}  {op.op_type.value:<10}  "
            f"{op.estimated_time_ms:>12.4e}  {op.earliest_start:>12.4e}  "
            f"{op.earliest_finish:>12.4e}  {op.stream_id:>6}"
        )
```

该函数展示了 `OperatorGraph` 的几个关键 API：

- `compute_critical_path()`：计算关键路径，填充每个算子的 `earliest_start` 和 `earliest_finish`。
- `topological_sort()`：按拓扑序遍历算子。
- `operators`：字典，存储所有 `OperatorNode`。
- 每个 `OperatorNode` 的属性：`name`、`op_type`、`estimated_time_ms`、`earliest_start`、`earliest_finish`、`stream_id`。

### 4. 三种追踪场景

```python
def main():
    hw = HardwareInfo(
        peak_tflops_mm=989.0,
        peak_tflops_math=989.0,
        peak_memory_bandwidth_gbps=3350.0,
    )

    # 训练模式（前向 + 反向）
    config = SimulatorConfig(hw_info=hw)
    x = torch.randn(4, 64, 128).cuda()
    model.train()
    graph_train = trace_model_for_training(model, x, config)

    # 推理 Prefill 模式（处理完整序列）
    model.eval()
    graph = trace_model_for_inference(model, x, config, mode="prefill")

    # 推理 Decode 模式（带 KV Cache，单 token 生成）
    config_decode = SimulatorConfig(hw_info=hw, cache_seq_len=2048)
    x_dec = torch.randn(1, 1, 128).cuda()
    graph_decode = trace_model_for_inference(model, x_dec, config_decode, mode="decode")
```

三种场景覆盖了大模型典型的使用模式：

| 场景 | API | 关键参数 | 说明 |
|------|-----|---------|------|
| 训练 | `trace_model_for_training` | - | 包含前向 + 反向传播 |
| Prefill | `trace_model_for_inference` | `mode="prefill"` | 推理时处理完整 prompt |
| Decode | `trace_model_for_inference` | `mode="decode"`, `cache_seq_len=2048` | 逐 token 生成，使用 KV Cache |

## 核心类/函数表

| 名称 | 类型 | 用途 |
|------|------|------|
| `DiverseModel` | 类 (nn.Module) | 包含多种算子类型的演示模型 |
| `print_table` | 函数 | 以表格形式打印 OperatorGraph 中每个算子的详细信息 |
| `main` | 函数 | 入口函数，依次执行训练、Prefill、Decode 三种追踪 |

## 与其他模块的关系

- **依赖 `syssim.api`**：使用 `trace_model_for_inference` 和 `trace_model_for_training`。
- **依赖 `syssim.config`**：使用 `HardwareInfo` 定义硬件参数，`SimulatorConfig` 传递配置。
- **间接使用 `syssim.operator_graph`**：通过返回的 `OperatorGraph` 对象调用 `compute_critical_path()`、`topological_sort()`、`summary()` 等方法。
- **不依赖 integrations 层**：直接使用底层 API，展示了不借助框架集成层的通用用法。

## 小结

本示例是学习 SysSim 最推荐的起点。它展示了完整的工作流：定义模型 -> 配置硬件参数 -> 追踪算子图 -> 分析结果。通过 `DiverseModel` 覆盖的多种算子类型，用户可以直观地看到 SysSim 如何对不同类型的操作进行分类和耗时估算。同时，三种追踪场景（训练/Prefill/Decode）的对比也帮助用户理解不同工作负载下的性能特征差异。
