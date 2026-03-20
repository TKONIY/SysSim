# `train_qwen3_8b_single.py` — 单 GPU 模拟 Qwen3-8B 训练

## 文件概述

`examples/huggingface/train_qwen3_8b_single.py` 演示了如何使用 SysSim 模拟 Qwen3-8B 模型在单 GPU 上的训练性能。该脚本基于 Qwen3-8B 的公开架构参数（36 层、hidden_size=4096、GQA 8 KV heads 等），使用 Hugging Face Transformers 的 `Qwen2Config` + `AutoModelForCausalLM` 在 `meta` 设备上构建模型（不分配实际显存），然后通过 SysSim 的 HF 集成接口追踪一个完整训练步骤的算子图并输出性能报告。

这是一个面向真实大模型架构的性能模拟示例，展示了 SysSim 在不需要实际 GPU 显存的情况下估算训练耗时的能力。

## 关键代码解析

### 1. Qwen3-8B 架构配置

```python
QWEN3_8B_CONFIG = dict(
    num_hidden_layers=36,
    hidden_size=4096,
    intermediate_size=22016,
    num_attention_heads=32,
    num_key_value_heads=8,       # GQA: 4x 压缩
    vocab_size=152064,
    max_position_embeddings=32768,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    hidden_act="silu",
)
```

这些参数完全对应 Qwen3-8B 的公开规格。几个关键点：
- **GQA (Grouped Query Attention)**：`num_key_value_heads=8` 而 `num_attention_heads=32`，即每 4 个 Query Head 共享一组 KV Head，大幅减少 KV Cache 的显存开销。
- **SwiGLU 激活**：`hidden_act="silu"` 配合 `intermediate_size=22016` 实现 SwiGLU FFN。
- **RoPE 位置编码**：`rope_theta=1000000.0` 支持长上下文。

### 2. Meta 设备构建模型

```python
model_cfg = Qwen2Config(**QWEN3_8B_CONFIG)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(model_cfg, torch_dtype=torch.bfloat16)
model.train()
```

使用 `torch.device("meta")` 上下文管理器构建模型，所有参数只记录元数据（形状、dtype）而不分配实际显存。这是 SysSim 的关键特性之一：**即使没有足够的 GPU 显存，也能追踪 8B 甚至更大的模型**。SysSim 的追踪器会在内部将这些 meta tensor 转换为 FakeTensor 进行追踪。

### 3. 合成输入构造

```python
BATCH_SIZE = 1
SEQ_LEN = 2048

input_ids = torch.randint(0, model_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN))
inputs = {"input_ids": input_ids, "labels": input_ids.clone()}
```

创建形状为 `(1, 2048)` 的随机 token 输入，并将 `input_ids` 的副本作为 `labels`（自回归语言模型的标准做法）。注意输入是 CPU 张量，追踪器会自动处理设备转换。

### 4. 硬件自动检测与追踪

```python
hw, hw_name = get_hardware_info()
sim_cfg = SimulatorConfig(hw_info=hw)
graph = trace_hf_model_for_training(model, inputs, sim_cfg)
```

- `get_hardware_info()` 自动检测当前 GPU 型号并返回对应的硬件参数（TFLOPS、内存带宽等）。
- `trace_hf_model_for_training` 是 HF 集成接口，自动处理 loss 函数推断（因为 `inputs` 包含 `labels`，会使用模型内置 loss）。

### 5. 结果分析

```python
type_counts: dict[OperatorType, int] = {}
for op in graph.operators.values():
    type_counts[op.op_type] = type_counts.get(op.op_type, 0) + 1

print("Operator counts by type:")
for op_type in OperatorType:
    count = type_counts.get(op_type, 0)
    if count:
        print(f"  {op_type.name:<12}: {count}")

critical_path_ms = graph.compute_critical_path()
print(f"Critical path time : {critical_path_ms:.2f} ms")
print(graph.summary())
```

按算子类型统计数量，然后计算关键路径时间。`graph.summary()` 会输出更详细的汇总信息，包括各类型算子的总耗时占比。

## 核心类/函数表

| 名称 | 类型 | 用途 |
|------|------|------|
| `QWEN3_8B_CONFIG` | 字典 | Qwen3-8B 的架构超参数 |
| `param_count` | 函数 | 统计模型总参数量 |
| `main` | 函数 | 入口：构建模型 -> 追踪 -> 打印报告 |

## 与其他模块的关系

- **依赖 `syssim.integrations.huggingface`**：使用 `trace_hf_model_for_training` 进行追踪。
- **依赖 `syssim.config`**：使用 `SimulatorConfig` 和 `get_hardware_info`。
- **依赖 `syssim.operator_graph`**：使用 `OperatorType` 枚举进行算子分类统计。
- **外部依赖 `transformers`**：使用 `Qwen2Config` 和 `AutoModelForCausalLM` 构建模型。
- **不需要实际 GPU 显存**：通过 meta 设备 + FakeTensor 实现零显存追踪。

## 小结

本示例展示了 SysSim 在真实大模型场景下的应用：用 Qwen3-8B 的真实架构参数构建模型，在不消耗 GPU 显存的情况下完成训练步骤的性能模拟。这对于在实际部署前评估不同硬件上的训练效率、优化批大小和序列长度等超参数非常有价值。脚本的运行命令为：

```bash
srun -N 1 --gpus 1 python examples/huggingface/train_qwen3_8b_single.py
```
