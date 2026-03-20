# `api.py` — 公共追踪 API 入口

## 文件概述

`api.py` 是用户与 SysSim 交互的主要入口，提供三个高层函数：`trace_model_for_training`（训练追踪）、`trace_model_for_inference`（推理追踪）和 `set_efficiency_model_dir`（效率模型配置）。这些函数封装了底层追踪引擎的复杂性，用户只需传入 PyTorch 模型和配置即可获得算子图。

## 关键代码解析

### 训练追踪：trace_model_for_training

```python
def trace_model_for_training(
    model: nn.Module,
    example_inputs: Any,
    config: SimulatorConfig,
    loss_fn: Any = None,
) -> OperatorGraph:
    tracer = OperatorGraphTracer(
        hw_info=config.hw_info,
        execution_mode=ExecutionMode.TRAINING,
        cache_seq_len=0,
    )
    return tracer.trace(model, example_inputs, forward_backward=True, loss_fn=loss_fn)
```

**逐步解析：**

1. 接收 PyTorch 模型 `model`、示例输入 `example_inputs`、模拟器配置 `config`，以及可选的损失函数 `loss_fn`。
2. 创建 `OperatorGraphTracer` 实例，执行模式固定为 `TRAINING`，KV cache 长度为 0（训练不使用 KV cache）。
3. 调用 `tracer.trace()` 时设置 `forward_backward=True`，表示追踪前向和反向传播两个阶段。
4. 如果用户未提供 `loss_fn`，追踪器内部会默认使用 `lambda out: out.sum()`。

### 推理追踪：trace_model_for_inference

```python
def trace_model_for_inference(
    model: nn.Module,
    example_inputs: Any,
    config: SimulatorConfig,
    mode: str = "prefill",
) -> OperatorGraph:
    mode_map = {
        "prefill": ExecutionMode.PREFILL,
        "decode": ExecutionMode.DECODE,
    }
    if mode not in mode_map:
        raise ValueError(f"Invalid inference mode '{mode}', expected 'prefill' or 'decode'")
    execution_mode = mode_map[mode]
    cache_seq_len = config.cache_seq_len if execution_mode == ExecutionMode.DECODE else 0

    tracer = OperatorGraphTracer(
        hw_info=config.hw_info,
        execution_mode=execution_mode,
        cache_seq_len=cache_seq_len,
    )
    return tracer.trace(model, example_inputs, forward_backward=False, loss_fn=None)
```

**逐步解析：**

1. 通过 `mode` 参数区分两种推理场景：
   - `"prefill"`：预填充阶段，处理完整输入序列，`cache_seq_len=0`。
   - `"decode"`：解码阶段，每次只处理一个 token，需要从 `config.cache_seq_len` 读取 KV cache 长度。
2. 对无效的 `mode` 值抛出 `ValueError`，提供清晰的错误信息。
3. 设置 `forward_backward=False`，推理只追踪前向传播。

### 效率模型配置：set_efficiency_model_dir

```python
def set_efficiency_model_dir(model_dir: str) -> None:
    from .compute.efficiency_models import set_backend_dir
    set_backend_dir(model_dir)
```

**解析：**

使用延迟导入（lazy import）避免在不需要效率模型时加载额外依赖。该函数设置训练好的效率模型文件（`.pth`）所在目录，后续追踪时会自动加载这些模型来提升耗时估算精度。

## 核心类/函数表

| 函数 | 参数 | 返回值 | 用途 |
|------|------|--------|------|
| `trace_model_for_training` | `model`, `example_inputs`, `config`, `loss_fn=None` | `OperatorGraph` | 追踪模型的前向+反向传播，生成完整训练算子图 |
| `trace_model_for_inference` | `model`, `example_inputs`, `config`, `mode="prefill"` | `OperatorGraph` | 追踪模型的前向传播，支持预填充和解码两种推理模式 |
| `set_efficiency_model_dir` | `model_dir` | `None` | 配置效率模型目录路径 |

## 与其他模块的关系

```
api.py
  |
  +---> config.py        读取 ExecutionMode, SimulatorConfig
  +---> tracer.py         使用 OperatorGraphTracer 执行追踪
  +---> operator_graph.py 返回 OperatorGraph 类型
  +---> compute/          set_efficiency_model_dir 延迟导入效率模型后端
```

- `api.py` 是 `tracer.py` 的薄封装层，将追踪引擎的参数从用户友好的 `SimulatorConfig` 转换为追踪器所需的 `hw_info`、`execution_mode` 等内部参数。
- 返回类型 `OperatorGraph` 定义在 `operator_graph.py` 中，用户拿到图后可调用 `summary()`、`compute_critical_path()` 等方法进行分析。

## 小结

`api.py` 是 SysSim 的"前台"，将复杂的追踪流程简化为两个函数调用。它的设计遵循"薄 API 层"原则：不包含核心逻辑，只负责参数校验、模式映射和追踪器调度，让用户以最少的代码完成模型性能分析。
