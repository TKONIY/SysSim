# `__init__.py` -- 网络模块入口与公共 API 导出

## 文件概述

`__init__.py` 是 `syssim.network` 包的入口文件，负责将网络通信模块的所有核心组件统一导出，使用户可以通过简洁的 `from syssim.network import ...` 语法访问全部功能。

该文件本身不包含业务逻辑，仅完成以下两件事：

1. 从各子模块导入关键类和函数
2. 通过 `__all__` 列表声明公共 API

## 关键代码解析

```python
# Core abstractions
from .loggp import LogGPParams
from .topology import (
    Topology, Resource,
    FullyConnectedTopology, RingTopology, SwitchTopology,
    NVLinkMeshTopology, HierarchicalTopology
)
from .dag_builder import Op, Step, build_dag

# Collectives
from .collectives import (
    allreduce, broadcast, reduce, reduce_scatter, allgather,
    alltoall, scatter, gather,
)

# Simulation engine
from .simulator import simulate, SimulationResult

# Model loader (LogGP profiler)
from .model_loader import (
    load_loggp_params, load_all_protocols, get_protocol_for_size,
    load_hierarchical_loggp, is_hierarchical_model, get_layer_params
)
```

导入分为四组：

| 分组 | 导入内容 | 说明 |
|------|---------|------|
| Core abstractions | `LogGPParams`, `Topology`, `Resource`, 5 种拓扑, `Op`, `Step`, `build_dag` | 性能模型、拓扑抽象、DAG 构建 |
| Collectives | 8 种集合通信原语 | allreduce, broadcast, reduce 等 |
| Simulation engine | `simulate`, `SimulationResult` | 事件驱动仿真器 |
| Model loader | 6 个加载函数 | 从 JSON 文件加载 LogGP 参数 |

## 核心类/函数表

本文件不定义新的类或函数，仅做重导出。完整 API 列表见 `__all__`，共 22 个符号。

## 与其他模块的关系

作为包入口，`__init__.py` 连接所有子模块：`loggp.py`、`topology.py`、`dag_builder.py`、`collectives.py`、`simulator.py`、`model_loader.py`。

## 小结

`__init__.py` 是网络模块的统一门面，用户只需 `from syssim.network import ...` 即可使用全部功能，无需关心内部模块划分。
