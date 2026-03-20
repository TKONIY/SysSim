# integrations 框架集成模块

## 模块概述

`syssim/integrations` 是 SysSim 的框架集成层，负责将底层追踪 API 适配到主流深度学习框架，降低用户使用门槛。当前已实现 Hugging Face Transformers 集成，用户只需传入 `PreTrainedModel` 即可一键完成训练追踪。

## 架构图

```
                        用户代码
                          |
                          v
              +-------------------------+
              |   syssim (顶层包)       |
              |  from syssim import     |
              |    trace_hf_model_...   |
              +-----------|-------------+
                          |
                          v
              +-------------------------+
              | integrations/__init__.py|  <-- 汇总导出
              |  - trace_hf_model_...   |
              |  - trace_hf_training_.. |
              +-----------|-------------+
                          |
                          v
              +-------------------------+
              | integrations/           |
              |   huggingface.py        |  <-- HF 适配逻辑
              |   - 输入格式转换        |
              |   - Loss 函数推断       |
              |   - 混合精度处理        |
              +-----------|-------------+
                          |
                          v
              +-------------------------+
              |   syssim.api            |  <-- 核心追踪引擎
              | trace_model_for_training|
              +-----------|-------------+
                          |
                          v
              +-------------------------+
              | syssim.operator_graph   |  <-- 算子图数据结构
              |   OperatorGraph         |
              |   OperatorNode          |
              +-------------------------+
```

## 文件说明

| 文件 | 说明 | 详细教程 |
|------|------|---------|
| [`__init__.py`](__init__.md) | 包初始化，汇总导出 HF 集成的两个公开函数 | [查看](__init__.md) |
| [`huggingface.py`](huggingface.md) | Hugging Face Transformers 集成实现，提供 `trace_hf_model_for_training` 和 `trace_hf_training_step` | [查看](huggingface.md) |

## 公开 API

| 函数 | 用途 | 典型用法 |
|------|------|---------|
| `trace_hf_model_for_training(model, inputs, config, ...)` | 追踪 HF 模型的完整训练步骤 | 传入 PreTrainedModel + tokenizer 输出即可 |
| `trace_hf_training_step(model, batch, config, ...)` | 高层封装，额外支持混合精度 | 传入训练 batch，可选 FP16 模式 |

## 设计理念

1. **可选依赖**：`transformers` 包通过 `try/except` 导入，未安装时不影响 SysSim 核心功能。
2. **薄适配层**：集成模块不重复实现追踪逻辑，仅做输入/输出格式适配，核心追踪委托给 `syssim.api`。
3. **可扩展**：如需集成其他框架（如 DeepSpeed、Megatron-Core 原生支持），可在此目录下新增子模块并在 `__init__.py` 中注册。

## 使用示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from syssim import SimulatorConfig, get_hardware_info, trace_hf_model_for_training

# 准备模型和输入
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer("Hello world", return_tensors="pt")
inputs["labels"] = inputs["input_ids"].clone()

# 配置硬件并追踪
hw, _ = get_hardware_info()
config = SimulatorConfig(hw_info=hw)
graph = trace_hf_model_for_training(model, inputs, config)

# 查看结果
print(f"关键路径耗时: {graph.compute_critical_path():.2f} ms")
print(graph.summary())
```
