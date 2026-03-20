# `__init__.py` — integrations 包初始化模块

## 文件概述

`syssim/integrations/__init__.py` 是 `syssim.integrations` 包的入口文件。它的职责非常简单：从子模块 `huggingface` 中导入公开 API，并通过 `__all__` 声明对外暴露的符号列表，使用户可以直接通过 `from syssim.integrations import ...` 来使用集成功能。

## 关键代码解析

```python
"""Integration modules for popular frameworks."""

from .huggingface import (
    trace_hf_model_for_training,
    trace_hf_training_step,
)

__all__ = [
    "trace_hf_model_for_training",
    "trace_hf_training_step",
]
```

- **模块文档字符串**：说明该包用于对接主流深度学习框架。
- **相对导入**：从同级子模块 `huggingface` 导入两个核心函数：
  - `trace_hf_model_for_training` —— 追踪 Hugging Face 模型的完整训练步骤（前向 + 反向）。
  - `trace_hf_training_step` —— 更高层的封装，支持混合精度等选项。
- **`__all__`**：显式声明公开 API，确保 `from syssim.integrations import *` 时只暴露这两个函数。

## 核心类/函数表

| 名称 | 来源 | 用途 |
|------|------|------|
| `trace_hf_model_for_training` | `huggingface.py` | 追踪 HF 模型训练过程，生成 OperatorGraph |
| `trace_hf_training_step` | `huggingface.py` | 封装单步训练追踪，支持混合精度 |

## 与其他模块的关系

- **向下依赖**：直接从 `syssim.integrations.huggingface` 导入。
- **向上暴露**：`syssim/__init__.py` 会进一步从本模块导入这两个函数，使用户可以直接通过 `from syssim import trace_hf_model_for_training` 使用。

## 小结

这是一个标准的 Python 包初始化文件，起到"汇总 + 转发"的作用。当前仅集成了 Hugging Face Transformers 框架；如果将来需要增加对其他框架（如 Megatron-Core、DeepSpeed 等）的原生集成，可在此处添加新的导入。
