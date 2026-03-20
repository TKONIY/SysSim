# 阅读指南

本文档为 SysSim (LLM Performance Simulator) 代码教程的阅读指南，帮助不同背景和目标的读者找到适合自己的学习路径。

---

## 前置知识

阅读本教程前，建议具备以下基础：

- **Python 基础**：熟悉 Python 类、装饰器、dataclass 等语法
- **PyTorch 基础**：了解 `torch.nn.Module`、张量操作、`torch.fx` 的基本概念
- **深度学习基础**：理解前向/反向传播、常见算子（MatMul、LayerNorm 等）
- **分布式训练概念**（网络模块需要）：了解数据并行、张量并行、AllReduce 等集合通信原语
- **性能分析概念**（可选）：了解 Roofline 模型、算术强度、计算/访存瓶颈等概念

## 阅读建议

1. **先跑通示例再读源码**：建议先运行 `examples/` 中的示例，建立直观感受后再深入模块实现
2. **从上层 API 向底层实现读**：先理解 `api.py` 暴露的接口，再追踪其调用的内部模块
3. **结合源码阅读**：每篇教程文档对应一个源文件，建议对照 `syssim/` 下的源码一起阅读
4. **关注数据流**：SysSim 的核心流程是 **追踪 -> 建图 -> 计算代价 -> 仿真**，理解数据在模块间的流动比逐行读代码更重要

---

## 阅读路径

### 路径一：快速入门（约 2 小时）

适合希望快速了解项目全貌、能够使用基本功能的读者。只需阅读 5 篇核心文档：

| 顺序 | 文档 | 目的 |
|------|------|------|
| 1 | [syssim/\_\_init\_\_.md](syssim/__init__.md) | 了解包的整体结构和对外导出 |
| 2 | [syssim/config.md](syssim/config.md) | 理解配置体系（硬件参数、仿真选项） |
| 3 | [syssim/api.md](syssim/api.md) | 掌握核心 API，这是使用 SysSim 的主入口 |
| 4 | [syssim/operator\_graph.md](syssim/operator_graph.md) | 理解算子图——SysSim 的核心数据结构 |
| 5 | [examples/trace\_and\_print.md](examples/trace_and_print.md) | 通过最简示例串联以上知识 |

### 路径二：完整学习

适合希望全面掌握 SysSim 设计与实现的读者。按以下顺序依次阅读全部 28 篇文档。

#### 第一阶段：核心框架（建立全局认知）

| 顺序 | 文档 | 阅读理由 |
|------|------|----------|
| 1 | [syssim/\_\_init\_\_.md](syssim/__init__.md) | 包入口，了解模块组织和公共接口 |
| 2 | [syssim/config.md](syssim/config.md) | 配置是所有模块的共同依赖，需优先理解 |
| 3 | [syssim/api.md](syssim/api.md) | 顶层 API，串联追踪、建图、仿真的完整流程 |
| 4 | [syssim/operator\_graph.md](syssim/operator_graph.md) | 算子图是追踪和仿真之间的桥梁数据结构 |
| 5 | [syssim/tracer.md](syssim/tracer.md) | 理解如何从 PyTorch 模型提取算子图 |

#### 第二阶段：计算代价建模

| 顺序 | 文档 | 阅读理由 |
|------|------|----------|
| 6 | [syssim/compute/\_\_init\_\_.md](syssim/compute/__init__.md) | 计算子包入口，了解模块划分 |
| 7 | [syssim/compute/flop\_counter.md](syssim/compute/flop_counter.md) | FLOP 计数是代价预测的基础输入 |
| 8 | [syssim/compute/efficiency\_models.md](syssim/compute/efficiency_models.md) | 效率模型将理论 FLOP 转化为实际耗时 |
| 9 | [syssim/compute/compute\_cost\_predictor.md](syssim/compute/compute_cost_predictor.md) | 基于模型的代价预测器（Roofline 等） |
| 10 | [syssim/compute/compute\_cost\_profiler.md](syssim/compute/compute_cost_profiler.md) | 基于实测的代价采集器 |
| 11 | [syssim/compute/validate\_profiler.md](syssim/compute/validate_profiler.md) | 预测与实测的对比验证 |

#### 第三阶段：网络通信仿真

| 顺序 | 文档 | 阅读理由 |
|------|------|----------|
| 12 | [syssim/network/\_\_init\_\_.md](syssim/network/__init__.md) | 网络子包入口 |
| 13 | [syssim/network/topology.md](syssim/network/topology.md) | 拓扑是网络仿真的基础，定义设备间连接关系 |
| 14 | [syssim/network/device\_mesh.md](syssim/network/device_mesh.md) | 设备网格抽象，描述多维并行的设备布局 |
| 15 | [syssim/network/loggp.md](syssim/network/loggp.md) | LogGP 通信模型，点对点通信的代价估算 |
| 16 | [syssim/network/collectives.md](syssim/network/collectives.md) | 集合通信原语的建模（AllReduce 等） |
| 17 | [syssim/network/protocol\_detector.md](syssim/network/protocol_detector.md) | 通信协议自动检测 |
| 18 | [syssim/network/dag\_builder.md](syssim/network/dag_builder.md) | 将通信操作构建为 DAG 用于仿真 |
| 19 | [syssim/network/simulator.md](syssim/network/simulator.md) | 网络仿真器核心，执行 DAG 调度 |
| 20 | [syssim/network/model\_loader.md](syssim/network/model_loader.md) | 网络参数模型的加载 |
| 21 | [syssim/network/profiler.md](syssim/network/profiler.md) | 网络性能实测采集 |
| 22 | [syssim/network/validation.md](syssim/network/validation.md) | 网络仿真结果验证 |

#### 第四阶段：集成与示例

| 顺序 | 文档 | 阅读理由 |
|------|------|----------|
| 23 | [syssim/integrations/\_\_init\_\_.md](syssim/integrations/__init__.md) | 集成子包入口 |
| 24 | [syssim/integrations/huggingface.md](syssim/integrations/huggingface.md) | HuggingFace 生态集成 |
| 25 | [examples/trace\_and\_print.md](examples/trace_and_print.md) | 最简单的端到端示例 |
| 26 | [examples/huggingface/train\_qwen3\_8b\_single.md](examples/huggingface/train_qwen3_8b_single.md) | 单卡 HuggingFace 训练仿真 |
| 27 | [examples/megatron/\_\_init\_\_.md](examples/megatron/__init__.md) | Megatron 集成入口 |
| 28 | [examples/megatron/train\_gpt\_multi\_gpu.md](examples/megatron/train_gpt_multi_gpu.md) | 多卡 Megatron 训练仿真——最完整的使用场景 |

### 路径三：专题学习

根据关注方向，选择性阅读以下专题。每个专题均可独立阅读，但建议先完成「快速入门」路径。

#### 专题 A：计算代价建模

适合关注单算子/单设备性能建模的读者。

1. [syssim/compute/\_\_init\_\_.md](syssim/compute/__init__.md)
2. [syssim/compute/flop\_counter.md](syssim/compute/flop_counter.md) — FLOP 计数原理
3. [syssim/compute/efficiency\_models.md](syssim/compute/efficiency_models.md) — 效率模型（Roofline 等）
4. [syssim/compute/compute\_cost\_predictor.md](syssim/compute/compute_cost_predictor.md) — 代价预测
5. [syssim/compute/compute\_cost\_profiler.md](syssim/compute/compute_cost_profiler.md) — 实测对比
6. [syssim/compute/validate\_profiler.md](syssim/compute/validate_profiler.md) — 验证流程

#### 专题 B：网络通信仿真

适合关注分布式训练通信开销建模的读者。

1. [syssim/network/\_\_init\_\_.md](syssim/network/__init__.md)
2. [syssim/network/topology.md](syssim/network/topology.md) — 网络拓扑
3. [syssim/network/device\_mesh.md](syssim/network/device_mesh.md) — 设备网格
4. [syssim/network/loggp.md](syssim/network/loggp.md) — LogGP 通信模型
5. [syssim/network/collectives.md](syssim/network/collectives.md) — 集合通信建模
6. [syssim/network/protocol\_detector.md](syssim/network/protocol_detector.md) — 协议检测
7. [syssim/network/dag\_builder.md](syssim/network/dag_builder.md) — DAG 构建
8. [syssim/network/simulator.md](syssim/network/simulator.md) — 仿真执行

#### 专题 C：框架集成与实战

适合希望将 SysSim 集成到自己训练流程中的读者。

1. [syssim/integrations/\_\_init\_\_.md](syssim/integrations/__init__.md)
2. [syssim/integrations/huggingface.md](syssim/integrations/huggingface.md) — HuggingFace 集成
3. [examples/trace\_and\_print.md](examples/trace_and_print.md) — 基础示例
4. [examples/huggingface/train\_qwen3\_8b\_single.md](examples/huggingface/train_qwen3_8b_single.md) — 单卡训练仿真
5. [examples/megatron/\_\_init\_\_.md](examples/megatron/__init__.md) — Megatron 集成
6. [examples/megatron/train\_gpt\_multi\_gpu.md](examples/megatron/train_gpt_multi_gpu.md) — 多卡训练仿真

---

## 推荐模块阅读顺序

无论选择哪条路径，模块层面的阅读顺序建议如下：

```
syssim (核心) → syssim/compute (计算) → syssim/network (网络) → syssim/integrations (集成) → examples (示例)
```

**理由：**

- **核心模块优先**：`config`、`api`、`operator_graph`、`tracer` 定义了整个系统的骨架，是后续所有模块的基础
- **计算先于网络**：单设备计算建模比多设备网络仿真更简单，且网络仿真依赖计算代价作为输入
- **网络模块内部**：拓扑/设备网格（静态结构） → LogGP/集合通信（代价模型） → DAG/仿真器（执行引擎）
- **集成与示例最后**：在理解底层机制后，集成层和示例才能真正看懂其设计意图

---

## 完整文档索引

### syssim/（核心模块）

| 文件 | 文档链接 | 说明 |
|------|----------|------|
| `syssim/__init__.py` | [syssim/\_\_init\_\_.md](syssim/__init__.md) | 包入口与公共导出 |
| `syssim/api.py` | [syssim/api.md](syssim/api.md) | 顶层 API |
| `syssim/config.py` | [syssim/config.md](syssim/config.md) | 配置与硬件参数 |
| `syssim/operator_graph.py` | [syssim/operator\_graph.md](syssim/operator_graph.md) | 算子图数据结构 |
| `syssim/tracer.py` | [syssim/tracer.md](syssim/tracer.md) | 模型追踪器 |

### syssim/compute/（计算代价模块）

| 文件 | 文档链接 | 说明 |
|------|----------|------|
| `syssim/compute/__init__.py` | [syssim/compute/\_\_init\_\_.md](syssim/compute/__init__.md) | 子包入口 |
| `syssim/compute/compute_cost_predictor.py` | [syssim/compute/compute\_cost\_predictor.md](syssim/compute/compute_cost_predictor.md) | 代价预测器 |
| `syssim/compute/compute_cost_profiler.py` | [syssim/compute/compute\_cost\_profiler.md](syssim/compute/compute_cost_profiler.md) | 代价采集器 |
| `syssim/compute/efficiency_models.py` | [syssim/compute/efficiency\_models.md](syssim/compute/efficiency_models.md) | 效率模型 |
| `syssim/compute/flop_counter.py` | [syssim/compute/flop\_counter.md](syssim/compute/flop_counter.md) | FLOP 计数器 |
| `syssim/compute/validate_profiler.py` | [syssim/compute/validate\_profiler.md](syssim/compute/validate_profiler.md) | 预测验证 |

### syssim/network/（网络仿真模块）

| 文件 | 文档链接 | 说明 |
|------|----------|------|
| `syssim/network/__init__.py` | [syssim/network/\_\_init\_\_.md](syssim/network/__init__.md) | 子包入口 |
| `syssim/network/collectives.py` | [syssim/network/collectives.md](syssim/network/collectives.md) | 集合通信 |
| `syssim/network/dag_builder.py` | [syssim/network/dag\_builder.md](syssim/network/dag_builder.md) | DAG 构建器 |
| `syssim/network/device_mesh.py` | [syssim/network/device\_mesh.md](syssim/network/device_mesh.md) | 设备网格 |
| `syssim/network/loggp.py` | [syssim/network/loggp.md](syssim/network/loggp.md) | LogGP 模型 |
| `syssim/network/model_loader.py` | [syssim/network/model\_loader.md](syssim/network/model_loader.md) | 模型加载 |
| `syssim/network/profiler.py` | [syssim/network/profiler.md](syssim/network/profiler.md) | 网络性能采集 |
| `syssim/network/protocol_detector.py` | [syssim/network/protocol\_detector.md](syssim/network/protocol_detector.md) | 协议检测 |
| `syssim/network/simulator.py` | [syssim/network/simulator.md](syssim/network/simulator.md) | 网络仿真器 |
| `syssim/network/topology.py` | [syssim/network/topology.md](syssim/network/topology.md) | 网络拓扑 |
| `syssim/network/validation.py` | [syssim/network/validation.md](syssim/network/validation.md) | 仿真验证 |

### syssim/integrations/（集成模块）

| 文件 | 文档链接 | 说明 |
|------|----------|------|
| `syssim/integrations/__init__.py` | [syssim/integrations/\_\_init\_\_.md](syssim/integrations/__init__.md) | 子包入口 |
| `syssim/integrations/huggingface.py` | [syssim/integrations/huggingface.md](syssim/integrations/huggingface.md) | HuggingFace 集成 |

### examples/（示例）

| 文件 | 文档链接 | 说明 |
|------|----------|------|
| `examples/trace_and_print.py` | [examples/trace\_and\_print.md](examples/trace_and_print.md) | 基础追踪示例 |
| `examples/huggingface/train_qwen3_8b_single.py` | [examples/huggingface/train\_qwen3\_8b\_single.md](examples/huggingface/train_qwen3_8b_single.md) | Qwen3-8B 单卡训练 |
| `examples/megatron/__init__.py` | [examples/megatron/\_\_init\_\_.md](examples/megatron/__init__.md) | Megatron 集成入口 |
| `examples/megatron/train_gpt_multi_gpu.py` | [examples/megatron/train\_gpt\_multi\_gpu.md](examples/megatron/train_gpt_multi_gpu.md) | GPT 多卡训练 |
