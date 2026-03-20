# network 网络通信模块

## 模块概述

`syssim.network` 是 SysSim（LLM 性能模拟器）的网络通信仿真模块，提供可组合的集合通信原语、自动依赖推断和拥塞感知仿真能力。它覆盖了从性能参数校准（profiling）到仿真执行的完整工作流，支持单节点和多节点集群场景。

**核心特性：**
- 8 种集合通信算法（allreduce, broadcast, reduce, reduce_scatter, allgather, alltoall, scatter, gather）
- 5 种网络拓扑模型（全连接、环形、交换机、NVLink 网格、分层）
- 基于两条物理规则的自动依赖推断
- 事件驱动拥塞仿真，Max-Min Fair 带宽共享
- 多节点支持，层特定 LogGP 参数

## 模块架构图

```
+------------------------------------------------------------------+
|                      syssim.network                              |
|                                                                  |
|  +------------------+     +-------------------+                  |
|  |  collectives.py  |     |  dag_builder.py   |                  |
|  |  ~~~~~~~~~~~~~~~~|     |  ~~~~~~~~~~~~~~~~~|                  |
|  | allreduce        |---->| Step -> Op DAG    |                  |
|  | broadcast        |     | (Rule 1: data dep)|                  |
|  | reduce           |     | (Rule 2: NIC ser.)|                  |
|  | reduce_scatter   |     +--------+----------+                  |
|  | allgather        |              |                             |
|  | alltoall         |              | Op list                     |
|  | scatter          |              v                             |
|  | gather           |     +-------------------+                  |
|  +------------------+     |   simulator.py    |                  |
|                           |   ~~~~~~~~~~~~~~~~|                  |
|  +------------------+     | Event-driven loop |                  |
|  |   topology.py    |---->| Max-Min Fair BW   |                  |
|  |   ~~~~~~~~~~~~~~~ |     | sharing           |                  |
|  | Resource          |     +--------+----------+                  |
|  | Topology (ABC)    |              |                             |
|  |   FullyConnected  |              | SimulationResult            |
|  |   Ring            |              v                             |
|  |   Switch          |     +-------------------+                  |
|  |   NVLinkMesh      |     |  validation.py   |                  |
|  |   Hierarchical ---|--+  |  ~~~~~~~~~~~~~~~~|                  |
|  +------------------+ |   | Analytical checks|                  |
|                       |   +-------------------+                  |
|  +------------------+ |                                          |
|  |    loggp.py      | |                                          |
|  |    ~~~~~~~~~~~~~ | |  LogGPParams (L, o, g, G)                |
|  |    alpha, G      |-+---> simulator / topology / validation    |
|  +------------------+ |                                          |
|                       |                                          |
|  +------------------+ |   +-------------------+                  |
|  | model_loader.py  |<+  | device_mesh.py    |                  |
|  | ~~~~~~~~~~~~~~~~ |     | ~~~~~~~~~~~~~~~~~ |                  |
|  | load_loggp_params|     | DeviceMesh        |                  |
|  | load_hierarchical|     | rank <-> coords   |                  |
|  | get_protocol_for |     | ranks_in_slice    |                  |
|  +--------+---------+     +--------+----------+                  |
|           |                         |                            |
|           v                         v                            |
|  +------------------+     +-------------------+                  |
|  | profiler.py       |<---| protocol_detector |                  |
|  | ~~~~~~~~~~~~~~~~~|     |   .py             |                  |
|  | NCCLBackend      |     | ~~~~~~~~~~~~~~~~ |                  |
|  | PRTT measurement |     | Lookahead detect |                  |
|  | Parameter extract|     | Least-squares fit|                  |
|  | CLI entry point  |     +-------------------+                  |
|  +------------------+                                            |
|                                                                  |
|  +------------------+                                            |
|  |  __init__.py     |  <-- Public API exports (22 symbols)      |
|  +------------------+                                            |
+------------------------------------------------------------------+
```

## 文件说明

| 文件 | 说明 |
|------|------|
| [`__init__.py`](./__init__.md) | 包入口，统一导出 22 个公共 API 符号 |
| [`loggp.py`](./loggp.md) | LogGP 性能模型参数定义（L, o, g, G），支持 3 参数和 4 参数两种变体 |
| [`topology.py`](./topology.md) | 5 种网络拓扑抽象与实现，核心接口 `resolve_path(src, dst) -> list[Resource]` |
| [`dag_builder.py`](./dag_builder.md) | 通信步骤到依赖 DAG 的自动转换，基于两条物理规则推断依赖 |
| [`collectives.py`](./collectives.md) | 8 种集合通信算法构建器，生成步骤序列交由 `build_dag` 处理 |
| [`simulator.py`](./simulator.md) | 事件驱动拥塞仿真引擎，Max-Min Fair 带宽共享 |
| [`validation.py`](./validation.md) | 解析公式验证器，用于校验仿真器在无竞争拓扑上的正确性 |
| [`device_mesh.py`](./device_mesh.md) | 多维设备网格抽象，支持坐标-rank 映射和切片查询 |
| [`model_loader.py`](./model_loader.md) | LogGP 参数加载器，支持单层和分层模型的 JSON 反序列化 |
| [`profiler.py`](./profiler.md) | NCCL ping-pong 基准测试的 CLI 工具，通过 PRTT 方法自动校准 LogGP 参数 |
| [`protocol_detector.py`](./protocol_detector.md) | 通信协议变化检测，Hoefler lookahead 算法识别 eager/rendezvous 切换点 |

## 典型仿真数据流

以一个 8-GPU 单节点 allreduce 仿真为例，数据流如下：

```
1. 参数准备
   profiler.py (NCCL ping-pong) --> JSON 文件
   model_loader.py (JSON 文件) --> LogGPParams

2. 拓扑构建
   topology.py: topo = FullyConnectedTopology(num_ranks=8, link_bandwidth=25e9)

3. 通信模式生成
   collectives.py: steps = allreduce 的 2*(8-1)=14 步，每步 8 个 send
                   |
                   v
   dag_builder.py: build_dag(steps) --> 112 个 Op，自动推断依赖

4. 仿真执行
   simulator.py: simulate(ops, topo, loggp)
     - 初始化: 112 个 op, remaining_bytes = size
     - 事件循环:
       * 注入无依赖的 op
       * 计算资源使用量和瓶颈带宽
       * 推进时间到下一事件
       * 消耗字节，完成传输
       * 更新依赖，注入新 op
     - 输出: SimulationResult(makespan, per_rank_finish)

5. 结果验证（可选）
   validation.py: validate_allreduce(P=8, M=1e9, loggp, result.makespan)
     - 解析公式: T = 2*(8-1) * (alpha + (1e9/8 - 1) * G)
     - 对比仿真结果, 相对误差 < 1e-5
```

**分层场景**（多节点）的数据流增加了以下步骤：
- `device_mesh.py` 定义 mesh 形状和维度
- `profiler.py` 按维度分层 profiling
- `topology.py` 使用 `HierarchicalTopology`，自动路由节点内/节点间流量
- `simulator.py` 使用 `get_loggp()` 获取层特定参数

## 设计亮点

1. **算法与调度分离**：collectives 只描述通信模式，simulator 负责资源调度和拥塞建模
2. **两条规则自动推断依赖**：无需手动指定 Op 间的依赖关系
3. **拓扑可扩展**：只需实现 `resolve_path()` 即可添加新拓扑
4. **profiling 闭环**：从 NCCL 实测到 LogGP 参数到仿真结果的完整流水线
5. **层次化支持**：DeviceMesh + HierarchicalTopology + 分层 profiling 无缝配合
