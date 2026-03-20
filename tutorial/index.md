# SysSim 代码教程

欢迎阅读 **SysSim** (LLM Performance Simulator) 的源代码教程。本教程将逐文件、逐模块地为您详解 SysSim 的每一行关键代码，帮助您从零开始理解这个 LLM 性能模拟器的设计与实现。

## 项目简介

SysSim 是一个 **LLM 性能模拟器**，它通过追踪神经网络的执行过程来构建计算图（DAG），并利用 **Roofline 模型** 和 **ML 效率预测器** 来估计运行时间。无需实际执行 GPU 计算，即可预测模型在不同硬件上的性能表现。

---

## 系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户 API 层                               │
│  trace_model_for_training()    trace_model_for_inference()       │
│                        (api.py)                                  │
└───────────────┬─────────────────────────────┬───────────────────┘
                │                             │
                ▼                             ▼
┌───────────────────────────┐   ┌─────────────────────────────────┐
│       追踪器 (Tracer)      │   │      硬件配置 (Config)           │
│   TorchDispatchMode 拦截   │   │  HardwareInfo / SimulatorConfig │
│   TensorStorageTracker    │   │  ExecutionMode / NetworkParams  │
│   CUDAEventTracker        │   │         (config.py)             │
│       (tracer.py)         │   └─────────────────────────────────┘
└───────────┬───────────────┘
            │ 生成
            ▼
┌───────────────────────────────────────────────────────────────┐
│                  算子图 (OperatorGraph)                         │
│   OperatorNode (GEMM/ATTN/MATH/COLLECTIVE/MEMORY/BARRIER)     │
│   DAG 结构 + 关键路径分析 + 多流依赖                              │
│                   (operator_graph.py)                          │
└───────────┬───────────────────────────────┬───────────────────┘
            │                               │
            ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────────┐
│    计算成本模块 (compute)   │   │   网络通信模块 (network)        │
│                           │   │                               │
│  ┌─────────────────────┐  │   │  ┌─────────────────────────┐  │
│  │   FLOP Counter      │  │   │  │  Collective Algorithms  │  │
│  │  (flop_counter.py)  │  │   │  │   (collectives.py)      │  │
│  └────────┬────────────┘  │   │  └────────┬────────────────┘  │
│           ▼               │   │           ▼                   │
│  ┌─────────────────────┐  │   │  ┌─────────────────────────┐  │
│  │   Roofline Model    │  │   │  │   DAG Builder           │  │
│  │ (compute_cost_      │  │   │  │   (dag_builder.py)      │  │
│  │  predictor.py)      │  │   │  └────────┬────────────────┘  │
│  └────────┬────────────┘  │   │           ▼                   │
│           ▼               │   │  ┌─────────────────────────┐  │
│  ┌─────────────────────┐  │   │  │  Network Simulator      │  │
│  │  ML Efficiency      │  │   │  │  LogGP + Topology       │  │
│  │  (efficiency_       │  │   │  │  (simulator.py)         │  │
│  │   models.py)        │  │   │  └─────────────────────────┘  │
│  └─────────────────────┘  │   │                               │
└───────────────────────────┘   └───────────────────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
                   ┌─────────────────┐
                   │  性能预测结果     │
                   │  (时间估计/ms)   │
                   └─────────────────┘
```

---

## 主要工作流程

以下展示 SysSim 模拟一次训练过程的完整流水线：

```
步骤1          步骤2           步骤3          步骤4          步骤5
用户调用   →   追踪执行    →   构建算子图  →  成本估计    →  关键路径
API            (Tracer)       (DAG)          (Compute/     分析
                                              Network)

┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│model │    │Dispatch  │    │Operator  │    │Per-op    │    │Critical  │
│+config│──▶│Mode拦截  │──▶│Graph     │──▶│time est. │──▶│path      │
│+input │    │记录算子  │    │(节点+边) │    │(Roofline │    │(总时间)  │
└──────┘    │+依赖关系 │    └──────────┘    │+ML+LogGP)│    └──────────┘
            └──────────┘                    └──────────┘
```

### 步骤详解

| 步骤 | 描述 | 源文件 | 教程文档 |
|------|------|--------|----------|
| 1. API 调用 | 用户传入模型、硬件配置和输入数据 | `syssim/api.py` | [api.py 教程](syssim/api.md) |
| 2. 追踪执行 | TorchDispatchMode 拦截 PyTorch 算子调度 | `syssim/tracer.py` | [tracer.py 教程](syssim/tracer.md) |
| 3. 构建算子图 | 将追踪结果组织为 DAG（有向无环图） | `syssim/operator_graph.py` | [operator_graph.py 教程](syssim/operator_graph.md) |
| 4a. 计算成本 | FLOP 计数 → Roofline 模型 → ML 效率校正 | `syssim/compute/` | [compute 模块](syssim/compute/index.md) |
| 4b. 通信成本 | 集合通信算法 → LogGP 模型 → 网络模拟 | `syssim/network/` | [network 模块](syssim/network/index.md) |
| 5. 关键路径 | 在 DAG 上求最长路径，得到总执行时间 | `syssim/operator_graph.py` | [operator_graph.py 教程](syssim/operator_graph.md) |

---

## 模块索引

| 模块 | 文件数 | 说明 | 教程入口 |
|------|--------|------|----------|
| **syssim (核心)** | 5 | 公共 API、配置、追踪器、算子图 IR | [核心模块概览](syssim/index.md) |
| **syssim/compute** | 6 | FLOP 计数、Roofline 模型、ML 效率预测、性能分析 | [计算成本模块](syssim/compute/index.md) |
| **syssim/network** | 11 | 集合通信、拓扑、LogGP、网络模拟、设备网格 | [网络通信模块](syssim/network/index.md) |
| **syssim/integrations** | 2 | HuggingFace Transformers 集成 | [框架集成模块](syssim/integrations/index.md) |
| **examples** | 4 | 基础追踪、HF 单卡训练、Megatron 多卡训练 | [使用示例](examples/index.md) |

---

## 快速开始

- **零基础读者**：先阅读 [背景知识](00_background_knowledge.md)，再按 [阅读指南](00_reading_guide.md) 的快速入门路径学习
- **有经验的开发者**：直接从 [api.py](syssim/api.md) 和 [tracer.py](syssim/tracer.md) 入手
- **只关心特定主题**：参考 [阅读指南](00_reading_guide.md) 中的专题学习路径

---

## 基础文档

- [背景知识](00_background_knowledge.md) — 性能模拟、Roofline 模型、集合通信等核心概念入门
- [阅读指南](00_reading_guide.md) — 三种阅读路径（快速入门 / 完整学习 / 专题学习）
