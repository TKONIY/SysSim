# `huggingface.py` — Hugging Face Transformers 训练追踪封装

## 文件概述

`syssim/integrations/huggingface.py` 为 Hugging Face Transformers 库提供便捷的追踪接口。用户只需传入一个 `PreTrainedModel` 及其输入，即可自动完成前向 + 反向传播的算子追踪，生成 `OperatorGraph`，从而估算训练耗时。

该文件是 SysSim 框架集成层的核心实现，屏蔽了手动构造 loss 函数、处理 `BatchEncoding` 类型转换等细节，让用户可以用最少的代码完成模型性能模拟。

## 关键代码解析

### 1. 可选依赖的安全导入

```python
try:
    from transformers import PreTrainedModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    PreTrainedModel = Any  # Type stub for when transformers not installed
```

通过 `try/except` 实现对 `transformers` 包的可选依赖。当该包未安装时，`HF_AVAILABLE` 标记为 `False`，后续函数调用时会抛出明确的 `ImportError` 提示用户安装。同时用 `Any` 作为类型占位符，避免类型注解报错。

### 2. `trace_hf_model_for_training` — 主追踪函数

```python
def trace_hf_model_for_training(
    model: PreTrainedModel,
    inputs: dict[str, torch.Tensor] | Any,
    config: SimulatorConfig,
    loss_fn: Callable | None = None,
    labels: torch.Tensor | None = None,
) -> OperatorGraph:
```

**关键步骤**：

1. **依赖检查**：若 `HF_AVAILABLE` 为 `False`，立即报错。
2. **输入格式标准化**：
   ```python
   if hasattr(inputs, "data"):
       inputs = dict(inputs.data)
   ```
   Hugging Face 的 `BatchEncoding` 对象带有 `.data` 属性，这里将其转为标准字典。
3. **标签注入**：如果用户通过 `labels` 参数单独传入标签，会将其合并到 `inputs` 字典中。
4. **Loss 函数自动推断**：
   ```python
   if loss_fn is None:
       if "labels" in inputs:
           loss_fn = lambda out: out.loss if hasattr(out, "loss") else out[0]
       else:
           loss_fn = _create_lm_loss_fn(inputs["input_ids"])
   ```
   - 如果 `inputs` 包含 `labels`，使用模型内置 loss（大多数 HF 模型在传入 labels 时会自动计算 loss）。
   - 否则，创建一个标准的语言模型 loss 函数（shift logits/labels + CrossEntropy）。
5. **调用核心追踪 API**：最终委托给 `syssim.api.trace_model_for_training`，该函数会将模型参数和输入转换为 FakeTensor 并在 CUDA 上追踪。

### 3. `trace_hf_training_step` — 高层封装

```python
def trace_hf_training_step(
    model: PreTrainedModel,
    batch: dict[str, torch.Tensor],
    config: SimulatorConfig,
    use_mixed_precision: bool = False,
) -> OperatorGraph:
```

在 `trace_hf_model_for_training` 基础上增加了混合精度支持：

```python
if use_mixed_precision:
    batch = {
        k: v.half() if isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v
        for k, v in batch.items()
    }
```

将 float32 张量转为 float16，模拟 AMP 训练场景。

### 4. `_create_lm_loss_fn` — 内部辅助函数

```python
def _create_lm_loss_fn(input_ids: torch.Tensor) -> Callable:
    def lm_loss(outputs):
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return loss
    return lm_loss
```

实现标准的自回归语言模型 loss 计算：将 logits 左移一位使得第 N 个 token 预测第 N+1 个 token，然后计算交叉熵损失。这是 GPT 系列模型训练的经典做法。

## 核心类/函数表

| 名称 | 类型 | 用途 |
|------|------|------|
| `trace_hf_model_for_training` | 函数 | 追踪 HF 模型前向+反向传播，返回 OperatorGraph |
| `trace_hf_training_step` | 函数 | 高层封装，增加混合精度支持 |
| `_create_lm_loss_fn` | 内部函数 | 创建标准语言模型 loss（shift + CrossEntropy） |
| `HF_AVAILABLE` | 常量 | 标记 transformers 包是否可用 |

## 与其他模块的关系

- **依赖 `syssim.api`**：核心追踪逻辑由 `trace_model_for_training` 完成，本模块只是适配层。
- **依赖 `syssim.config`**：使用 `SimulatorConfig` 传递硬件配置。
- **依赖 `syssim.operator_graph`**：返回的 `OperatorGraph` 是整个模拟器的核心数据结构。
- **外部依赖 `transformers`**：可选依赖，通过安全导入处理缺失情况。
- **被 `syssim/__init__.py` 导入**：用户可直接 `from syssim import trace_hf_model_for_training`。

## 小结

本文件是 SysSim 与 Hugging Face 生态对接的桥梁。它通过自动处理输入格式、loss 函数推断和混合精度转换，将复杂的追踪流程简化为一次函数调用。对于使用 HF Transformers 训练大模型的用户来说，这是最推荐的入口。
