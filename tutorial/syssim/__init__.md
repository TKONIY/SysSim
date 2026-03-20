# `__init__.py` — 包导出入口

## 文件概述

`__init__.py` 是 `syssim` 包的入口文件，负责将各子模块的关键类和函数汇聚到顶层命名空间。用户只需 `import syssim` 即可直接使用所有核心功能，无需记忆各子模块路径。

## 关键代码解析

### 核心模块导出

```python
from .config import ExecutionMode, HardwareInfo, SimulatorConfig, NetworkParams, get_hardware_info
from .operator_graph import OperatorType, OperatorNode, OperatorGraph, TensorMeta
from .api import trace_model_for_training, trace_model_for_inference, set_efficiency_model_dir
```

这三行将核心模块的所有公共符号提升到 `syssim` 命名空间。导出内容按职责分为三组：
- **配置类**：`ExecutionMode`、`HardwareInfo`、`SimulatorConfig`、`NetworkParams`、`get_hardware_info`
- **算子图 IR**：`OperatorType`、`OperatorNode`、`OperatorGraph`、`TensorMeta`
- **公共 API**：`trace_model_for_training`、`trace_model_for_inference`、`set_efficiency_model_dir`

### Hugging Face 集成导出

```python
from .integrations.huggingface import (
    trace_hf_model_for_training,
    trace_hf_training_step,
)
```

导出 Hugging Face 模型的专用追踪函数，方便用户直接对 HF Transformers 模型进行性能分析。

### 网络模拟器导出

```python
from .network import (
    LogGPParams, Topology, Resource, Op, Step, SimulationResult,
    FullyConnectedTopology, RingTopology, SwitchTopology,
    NVLinkMeshTopology, HierarchicalTopology,
    allreduce, broadcast, reduce, reduce_scatter, allgather,
    alltoall, scatter, gather,
    simulate, build_dag,
)
```

网络模拟器相关的导出最为丰富，包含三大类：
- **核心类型**：`LogGPParams`、`Topology`、`Resource` 等基础数据结构
- **拓扑结构**：五种网络拓扑（全连接、环形、交换机、NVLink Mesh、层次化）
- **集合通信与模拟**：八种集合通信原语及 `simulate`、`build_dag` 模拟函数

## 核心类/函数表

| 导出符号 | 来源模块 | 用途 |
|----------|----------|------|
| `ExecutionMode` | `config` | 执行模式枚举（训练/预填充/解码） |
| `HardwareInfo` | `config` | 硬件规格配置 |
| `SimulatorConfig` | `config` | 模拟器顶层配置 |
| `NetworkParams` | `config` | 网络硬件参数 |
| `get_hardware_info` | `config` | 自动检测当前 GPU 硬件 |
| `OperatorType` | `operator_graph` | 算子类型枚举 |
| `OperatorNode` | `operator_graph` | 算子节点数据结构 |
| `OperatorGraph` | `operator_graph` | 算子图 DAG |
| `TensorMeta` | `operator_graph` | 张量元数据 |
| `trace_model_for_training` | `api` | 追踪训练过程 |
| `trace_model_for_inference` | `api` | 追踪推理过程 |
| `set_efficiency_model_dir` | `api` | 配置效率模型目录 |
| `trace_hf_model_for_training` | `integrations.huggingface` | 追踪 HF 模型训练 |
| `trace_hf_training_step` | `integrations.huggingface` | 追踪 HF 训练步骤 |
| `simulate` | `network` | 运行网络通信模拟 |

## 与其他模块的关系

`__init__.py` 本身不包含任何逻辑，纯粹是一个"汇聚层"。它从以下子模块导入：
- `syssim.config` — 配置相关
- `syssim.operator_graph` — 算子图 IR
- `syssim.api` — 公共 API
- `syssim.integrations.huggingface` — HF 集成
- `syssim.network` — 网络模拟器

## 小结

`__init__.py` 通过集中导出机制，让用户以 `syssim.XXX` 的扁平方式访问分布在多个子模块中的功能，是项目对外接口的"总目录"。
