# compute 计算成本模块

## 模块概述

`compute` 模块是 SysSim（LLM 性能模拟器）的核心子系统之一，负责 **算子级别的运行时间预测**。它基于 Roofline 性能模型，结合机器学习效率校正，为每个 PyTorch 算子估算在目标硬件上的执行时间。

## 模块架构图

```
                        ┌──────────────────────────────────┐
                        │         estimate_runtime()       │
                        │    (compute_cost_predictor.py)    │
                        └──────────┬───────────┬───────────┘
                                   │           │
                    ┌──────────────▼──┐   ┌────▼──────────────┐
                    │  Roofline 模型   │   │  效率校正模型 (ML)  │
                    │ roofline_estimate│   │ efficiency_estimate│
                    └──────┬──────────┘   └────┬──────────────┘
                           │                   │
              ┌────────────▼────┐    ┌─────────▼──────────┐
              │  FLOP 计数器     │    │  效率模型管理       │
              │ (flop_counter.py)│    │(efficiency_models.py)│
              └────────┬────────┘    └─────────┬──────────┘
                       │                       │
                       │              ┌────────▼────────┐
                       │              │ MLP / XGBoost   │
                       │              │ 预训练权重 (.pth) │
                       │              └────────┬────────┘
                       │                       │
              ┌────────▼───────────────────────▼────────┐
              │            Profiler (数据采集+训练)        │
              │       (compute_cost_profiler.py)          │
              │  - 参数网格采样                            │
              │  - GPU 算子 benchmark                     │
              │  - MLP / XGBoost 交叉验证训练              │
              └──────────────────────────────────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │        Validator (验证脚本)               │
              │       (validate_profiler.py)             │
              └─────────────────────────────────────────┘
```

## 文件说明

| 文件 | 职责 |
|------|------|
| `__init__.py` | 模块包初始化（空文件） |
| `compute_cost_predictor.py` | 核心预测器：Roofline 模型 + 效率校正，提供 `estimate_runtime()` 入口 |
| `flop_counter.py` | FLOP 计数注册表：为每种 PyTorch 算子（mm、bmm、sdpa 等）注册浮点运算量计算公式 |
| `efficiency_models.py` | ML 效率模型的管理与推理：`MLPEfficiencyModel`、`XGBoostEfficiencyModel`、`BackendManager` |
| `compute_cost_profiler.py` | CLI 工具：GPU 算子 profiling、参数网格采样、效率模型训练 |
| `validate_profiler.py` | 验证脚本：检测硬件、特征提取、数据增强、模型架构兼容性 |

## 数据流

整个模块的数据流分为 **离线训练** 和 **在线推理** 两条路径：

### 离线训练路径

```
参数网格 (COMPUTE_GRIDS)
       │
       ▼
GPU Profiling (实测时间 t_measured_ms)
       │
       ▼
Roofline 计算 (t_roofline_ms)
       │
       ▼
效率标签 (efficiency = t_roofline / t_measured)
       │
       ▼
特征工程 (shape + roofline envelope features)
       │
       ▼
K-Fold 交叉验证训练 (MLP 或 XGBoost)
       │
       ▼
保存模型权重 (.pth)
```

### 在线推理路径

```
PyTorch 算子 (func_packet, args, kwargs, out)
       │
       ├──► flop_counter.py ──► FLOP 计数
       │
       ▼
roofline_estimate()
       │
       ├── T_compute = FLOPs / peak_FLOP_s
       ├── T_memory  = bytes / peak_bandwidth
       └── T_roofline = max(T_compute, T_memory)
       │
       ▼
efficiency_estimate()
       │
       ├── 提取特征 (EfficiencyFeatures)
       └── ML 模型预测 η_hat
       │
       ▼
estimate_runtime()
       │
       └── T_predicted = T_roofline / η_hat  (单位: ms)
```

### 核心公式

Roofline 模型的基本思想：

$$T_{roofline} = \max(T_{compute}, T_{memory})$$

其中：

$$T_{compute} = \frac{FLOPs}{peak\_FLOP/s}$$

$$T_{memory} = \frac{bytes}{peak\_bandwidth}$$

最终预测加入效率校正：

$$T_{predicted} = \frac{T_{roofline}}{\hat{\eta}}$$

其中 \(\hat{\eta} \in (0, 1]\) 是 ML 模型预测的硬件利用效率。
