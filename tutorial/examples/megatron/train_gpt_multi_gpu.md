# `train_gpt_multi_gpu.py` — Megatron-Core GPT-3 1.3B 多 GPU 张量并行训练模拟

## 文件概述

`examples/megatron/train_gpt_multi_gpu.py` 演示了如何使用 SysSim 模拟基于 Megatron-Core 的 GPT-3 1.3B 模型在多 GPU 张量并行 (Tensor Parallelism, TP) 下的训练性能。脚本通过 `torch.multiprocessing.spawn` 启动多个进程模拟不同 TP rank，每个 rank 独立构建其分片模型并追踪前向 + 反向传播，最终由 rank 0 汇总输出性能报告。

这是 SysSim 项目中最复杂的示例，涉及分布式初始化、Megatron 并行状态管理、模型分片构建、以及多进程追踪协调。

## 关键代码解析

### 1. 模型配置与 Vocab 对齐

```python
_VOCAB_SIZE_BASE = 50257
NUM_LAYERS = 24
HIDDEN_SIZE = 2048
NUM_ATTENTION_HEADS = 16
FFN_HIDDEN_SIZE = 8192
SEQ_LEN = 2048

def _vocab_size_for_tp(tp_size: int) -> int:
    """Round vocab size up to the nearest multiple of tp_size."""
    return math.ceil(_VOCAB_SIZE_BASE / tp_size) * tp_size
```

GPT-3 1.3B 的架构参数。`_vocab_size_for_tp` 将词表大小向上对齐到 TP size 的整数倍，这是 Megatron TP 的要求——Embedding 层和输出头的词表维度需要在 TP rank 间均匀切分。

### 2. 模型构建

```python
def build_model(tp_size: int) -> GPTModel:
    vocab_size = _vocab_size_for_tp(tp_size)
    transformer_config = TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        ffn_hidden_size=FFN_HIDDEN_SIZE,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        bf16=True,
        attention_softmax_in_fp32=True,
    )
    layer_spec = get_gpt_layer_local_spec()

    with torch.device("meta"):
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=MAX_SEQ_LEN,
            pre_process=True,
            post_process=True,
            parallel_output=True,
        )
    return model.train()
```

关键点：
- **`TransformerConfig`**：Megatron-Core 的配置类，`tensor_model_parallel_size=tp_size` 指定 TP 度。
- **`get_gpt_layer_local_spec()`**：获取 GPT 层的本地规格（非 Transformer Engine 版本）。
- **`torch.device("meta")`**：与 Qwen 示例类似，在 meta 设备上构建模型以避免显存分配。
- **`pre_process=True, post_process=True`**：因为 Pipeline Parallelism=1，所有 rank 都包含 Embedding 和输出头。
- **`parallel_output=True`**：输出 logits 保持按 TP 分片，不做 all-gather。

### 3. 输入与 Loss 函数

```python
def make_inputs(vocab_size: int) -> tuple:
    input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN))
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH_SIZE, -1)
    attention_mask = None  # Megatron 内部处理因果 mask
    return (input_ids, position_ids, attention_mask)

def make_loss_fn(vocab_size: int):
    def loss_fn(logits: torch.Tensor) -> torch.Tensor:
        labels = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device="cuda")
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
    return loss_fn
```

注意事项：
- Megatron GPTModel 的输入是 tuple `(input_ids, position_ids, attention_mask)`，而非字典——这就是为什么此示例使用底层 `trace_model_for_training` 而非 HF 集成接口。
- Loss 函数在 FakeTensorMode 内部执行，所以 `device="cuda"` 会创建 FakeTensor 而非真实 CUDA 张量。
- 由于 `parallel_output=True`，logits 的 vocab 维度是 `V/TP`，loss 仅在本地分片上计算。

### 4. 分布式初始化与追踪流程

```python
def _trace_rank(rank: int, world_size: int, port: int, tp_size: int) -> None:
    # 1. 分布式初始化
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(0)  # 所有 rank 共享 cuda:0（SysSim 使用 FakeTensor）

    # 2. Megatron 并行状态初始化
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        create_gloo_process_groups=False,
    )
    model_parallel_cuda_manual_seed(42)

    # 3. 硬件检测
    hw_info, hw_name = get_hardware_info()
    sim_cfg = SimulatorConfig(hw_info=hw_info)

    # 4. 构建模型
    model = build_model(tp_size)

    # 5. 追踪前向 + 反向
    inputs = make_inputs(_vocab_size_for_tp(tp_size))
    graph = trace_model_for_training(model, inputs, sim_cfg, loss_fn=make_loss_fn(vocab_size))

    # 6. 打印结果（仅 rank 0）
    print_results(graph, hw_name, rank, tp_size)
    dist.destroy_process_group()
```

分布式初始化要点：
- **使用 `gloo` 后端**：因为 SysSim 基于 FakeTensor，不需要真实的 NCCL 通信。
- **`torch.cuda.set_device(0)`**：所有 rank 共享同一个 CUDA 设备（实际不分配显存）。
- **`model_parallel_cuda_manual_seed(42)`**：初始化 TP 相关的 RNG 状态，这是 Megatron Attention 中 Dropout 所需要的。

### 5. 多进程启动

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=4, choices=[1, 2, 4, 8, 16])
    args = parser.parse_args()

    port = _find_free_port()
    mp.spawn(_trace_rank, args=(tp_size, port, tp_size), nprocs=tp_size, join=True)
```

使用 `torch.multiprocessing.spawn` 启动 `tp_size` 个进程，每个进程执行 `_trace_rank`。`_find_free_port()` 自动找到一个空闲端口用于进程间通信。

### 6. 结果输出

```python
def print_results(graph, hw_name: str, rank: int, tp_size: int) -> None:
    if rank != 0:
        return
    # 打印硬件信息、算子统计、关键路径时间和详细摘要
    ...
    critical_path_ms = graph.compute_critical_path()
    print(f"  Critical path : {critical_path_ms:.3f} ms")
    print(graph.summary())
```

只有 rank 0 输出结果。在 TP 模式下，每个 rank 的算子图结构相同（只是各维度按 TP 切分），所以只需查看一个 rank 的结果。

## 核心类/函数表

| 名称 | 类型 | 用途 |
|------|------|------|
| `_vocab_size_for_tp` | 函数 | 将词表大小对齐到 TP size 的整数倍 |
| `_find_free_port` | 函数 | 查找空闲 TCP 端口用于分布式初始化 |
| `build_model` | 函数 | 构建 Megatron GPT-3 1.3B 模型（meta 设备） |
| `make_inputs` | 函数 | 创建合成训练输入 (input_ids, position_ids, attention_mask) |
| `make_loss_fn` | 函数 | 创建交叉熵 loss 函数（在 FakeTensorMode 内运行） |
| `print_results` | 函数 | 打印算子统计和关键路径时间（仅 rank 0） |
| `_trace_rank` | 函数 | 单个 rank 的完整追踪流程（分布式初始化 -> 建模 -> 追踪 -> 报告） |
| `main` | 函数 | 入口：解析参数 + 启动多进程 |

## 与其他模块的关系

- **依赖 `syssim.api`**：使用底层 `trace_model_for_training`（而非 HF 集成接口，因为 Megatron 模型的输入格式不同）。
- **依赖 `syssim.config`**：使用 `HardwareInfo`、`SimulatorConfig`、`get_hardware_info`。
- **依赖 `syssim.operator_graph`**：使用 `OperatorType` 枚举进行算子分类。
- **外部依赖 `megatron-core`**：使用 `GPTModel`、`TransformerConfig`、`parallel_state` 等 Megatron-Core 组件。
- **外部依赖 `torch.distributed`**：用于多进程分布式初始化。

## 小结

本示例是 SysSim 项目中最具实战价值的演示，展示了如何在不消耗真实 GPU 显存的情况下模拟多 GPU 张量并行训练。通过 `--tp-size` 参数可以方便地对比不同 TP 度下的性能差异（TP=1/2/4/8/16），帮助用户在实际部署前做出最优的并行策略决策。运行命令：

```bash
srun -N 1 --gpus 1 python examples/megatron/train_gpt_multi_gpu.py --tp-size 4
```
