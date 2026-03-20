# examples 使用示例

## 概述

`examples/` 目录包含 SysSim 的完整使用示例，从简单的单模型追踪到复杂的多 GPU 张量并行模拟，覆盖了该工具的主要使用场景。示例按框架分组，帮助用户快速找到与自己工作流匹配的参考代码。

## 架构图

```
examples/
│
├── trace_and_print.py              通用入门示例（无框架依赖）
│   │
│   ├── 自定义 DiverseModel (多算子类型)
│   ├── 训练追踪 (forward + backward)
│   ├── 推理 Prefill 追踪
│   └── 推理 Decode 追踪 (KV Cache)
│
├── huggingface/                    Hugging Face Transformers 示例
│   │
│   └── train_qwen3_8b_single.py   单 GPU Qwen3-8B 训练模拟
│       ├── Meta 设备零显存构建 8B 模型
│       ├── 使用 syssim.integrations.huggingface
│       └── 自动硬件检测 + 算子分析
│
└── megatron/                       Megatron-Core 示例
    │
    ├── __init__.py                 包标记
    └── train_gpt_multi_gpu.py      多 GPU GPT-3 1.3B TP 训练模拟
        ├── torch.multiprocessing 多进程
        ├── Megatron parallel_state 初始化
        ├── 可配置 TP size (1/2/4/8/16)
        └── 使用底层 syssim.api

                    依赖关系
        ┌──────────────────────────────────┐
        │          syssim (核心库)          │
        │                                  │
        │  ┌──────────┐  ┌──────────────┐  │
        │  │  api.py   │  │ integrations │  │
        │  │ (底层API) │  │  (HF 封装)   │  │
        │  └─────┬─────┘  └──────┬───────┘  │
        │        │               │          │
        └────────┼───────────────┼──────────┘
                 │               │
        ┌────────┘        ┌──────┘
        │                 │
   trace_and_print.py    train_qwen3_8b_single.py
   train_gpt_multi_gpu.py
```

## 文件说明

| 文件 | 框架依赖 | 场景 | 难度 | 详细教程 |
|------|---------|------|------|---------|
| [`trace_and_print.py`](trace_and_print.md) | 无（纯 PyTorch） | 多算子类型追踪 + 表格打印 | 入门 | [查看](trace_and_print.md) |
| [`huggingface/train_qwen3_8b_single.py`](huggingface/train_qwen3_8b_single.md) | transformers | 单 GPU 大模型训练模拟 | 中级 | [查看](huggingface/train_qwen3_8b_single.md) |
| [`megatron/__init__.py`](megatron/__init__.md) | - | 包标记文件 | - | [查看](megatron/__init__.md) |
| [`megatron/train_gpt_multi_gpu.py`](megatron/train_gpt_multi_gpu.md) | megatron-core | 多 GPU 张量并行训练模拟 | 高级 | [查看](megatron/train_gpt_multi_gpu.md) |

## 学习路径建议

### 第一步：理解基础流程
从 `trace_and_print.py` 开始，了解 SysSim 的基本三步工作流：
1. 定义 `HardwareInfo` + `SimulatorConfig`
2. 调用 `trace_model_for_training` 或 `trace_model_for_inference`
3. 通过 `OperatorGraph` 分析结果

### 第二步：使用框架集成
查看 `train_qwen3_8b_single.py`，学习如何：
- 利用 `torch.device("meta")` 零显存构建大模型
- 使用 HF 集成接口 `trace_hf_model_for_training` 简化追踪
- 使用 `get_hardware_info()` 自动检测硬件

### 第三步：多 GPU 并行模拟
研究 `train_gpt_multi_gpu.py`，掌握：
- 多进程分布式环境下的追踪方法
- Megatron-Core 并行状态管理
- 不同 TP 度下的性能对比分析

## 运行环境要求

| 示例 | Python 包 | GPU 要求 |
|------|-----------|---------|
| `trace_and_print.py` | `torch`, `syssim` | 需要 CUDA 设备 |
| `train_qwen3_8b_single.py` | `torch`, `transformers`, `syssim` | 需要 CUDA 设备（不需要大显存） |
| `train_gpt_multi_gpu.py` | `torch`, `megatron-core`, `syssim` | 需要 CUDA 设备（不需要大显存） |

> **注意**：SysSim 使用 FakeTensor 进行追踪，实际 GPU 显存消耗极小。即使模拟 8B 甚至更大的模型，也不需要与模型参数量匹配的 GPU 显存。
