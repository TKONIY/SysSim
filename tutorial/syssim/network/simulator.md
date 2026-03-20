# `simulator.py` -- 事件驱动拥塞仿真引擎

## 文件概述

`simulator.py` 是网络模块的核心仿真引擎，实现了基于 **事件驱动** 的通信仿真，支持 **max-min fair** 带宽共享。它接收一个通信操作 DAG、一个网络拓扑和 LogGP 参数，输出每个操作的开始/完成时间以及总执行时间（makespan）。

## 关键代码解析

### 1. SimulationResult 数据类

```python
@dataclass
class SimulationResult:
    ops: list[Op]              # 带时间信息的 Op 列表
    makespan: float            # 总执行时间（秒）
    per_rank_finish: dict[int, float]  # 每个 rank 的最晚完成时间
```

### 2. simulate() 函数 -- 仿真入口

```python
def simulate(ops: list[Op], topology: Topology, loggp: Optional[LogGPParams] = None) -> SimulationResult:
```

函数签名接受三个参数：Op DAG、拓扑模型和（可选的）全局 LogGP 参数。对于 `HierarchicalTopology`，LogGP 参数可以从拓扑的 `get_loggp()` 方法按层获取。

### 3. 初始化阶段

```python
# 初始化 op 状态
for op in ops:
    op.remaining_bytes = op.size
    op.start_time = 0.0
    op.finish_time = 0.0

# 为分层拓扑填充层特定 LogGP 参数
has_layer_loggp = hasattr(topology, 'get_loggp') and callable(getattr(topology, 'get_loggp'))
if has_layer_loggp:
    for op in ops:
        if op.loggp is None:
            op.loggp = topology.get_loggp(op.src, op.dst)

# 构建路径缓存
path_cache: dict[tuple[int, int], list[Resource]] = {}
for op in ops:
    key = (op.src, op.dst)
    if key not in path_cache:
        path_cache[key] = topology.resolve_path(op.src, op.dst)
```

**设计要点**：
- 路径缓存避免重复查询拓扑（`resolve_path` 可能较慢）
- 使用 op 索引而非 Op 对象作为字典键，避免 hashability 问题
- 分层拓扑的 LogGP 参数通过 duck typing（检查 `get_loggp` 方法）自动填充

### 4. 依赖跟踪与就绪队列

```python
# 依赖计数器
num_unsatisfied: list[int] = [len(op.deps) for op in ops]

# 反向依赖映射：op_idx -> 依赖它的 op 列表
dependents: dict[int, list[int]] = {}
for i, op in enumerate(ops):
    for dep in op.deps:
        dep_idx = op_to_idx[id(dep)]
        dependents.setdefault(dep_idx, []).append(i)

# 就绪队列（最小堆，按就绪时间排序）
eligible: list[tuple[float, int, int]] = []  # (ready_time, counter, op_idx)

# 无依赖的 op 立即就绪
for i, op in enumerate(ops):
    if num_unsatisfied[i] == 0:
        heapq.heappush(eligible, (0.0, counter, i))
```

### 5. 主事件循环 -- Max-Min Fair 带宽共享

```python
while active or eligible:
    # (a) 注入就绪 op
    while eligible and eligible[0][0] <= current_time + FLOAT_TOL:
        _, _, op_idx = heapq.heappop(eligible)
        op = ops[op_idx]
        op.start_time = current_time
        active.add(op_idx)

    # (b) 计算资源使用量
    resource_usage: dict[str, int] = {}
    for op_idx in active:
        op = ops[op_idx]
        for res in path_cache[(op.src, op.dst)]:
            resource_usage[res.name] = resource_usage.get(res.name, 0) + 1

    # (c) 计算每个 op 的瓶颈带宽
    op_bandwidth: dict[int, float] = {}
    for op_idx in active:
        op = ops[op_idx]
        path = path_cache[(op.src, op.dst)]
        min_bw = float('inf')
        for res in path:
            effective_bw = res.bandwidth / resource_usage[res.name]
            min_bw = min(min_bw, effective_bw)
        op_bandwidth[op_idx] = min_bw

    # (d) 计算下一事件时间
    time_to_completion = min(op.remaining_bytes / bw for op in active_ops)
    time_to_injection = next_eligible_time - current_time
    dt = min(time_to_completion, time_to_injection)
    current_time += dt

    # (e) 消耗字节
    for op_idx in active:
        op.remaining_bytes -= op_bandwidth[op_idx] * dt

    # (f) 完成传输的 op
    for completed_op_idx in completed:
        op.finish_time = current_time + op_loggp.alpha
        # 更新依赖计数，将新就绪的 op 加入队列
        for dep_op_idx in dependents[completed_op_idx]:
            num_unsatisfied[dep_op_idx] -= 1
            if num_unsatisfied[dep_op_idx] == 0:
                heapq.heappush(eligible, (op.finish_time, counter, dep_op_idx))
```

**Max-Min Fair 带宽共享原理**：

如果 $N$ 个 op 共享带宽为 $B$ 的资源，每个 op 分得 $B/N$。每个 op 的有效带宽取其路径上所有资源中的最小值：

$$\text{BW}_{\text{eff}}(op) = \min_{r \in \text{path}(op)} \frac{B_r}{n_r}$$

其中 $B_r$ 是资源 $r$ 的带宽，$n_r$ 是共享该资源的活跃 op 数。

**事件推进策略**：

时间推进到下一个事件发生点，取两者的最小值：
- **op 完成事件**：某个 op 的 `remaining_bytes` 归零
- **op 注入事件**：某个 op 的所有依赖完成

### 6. LogGP 开销的处理

```python
# 完成时添加 LogGP 固定开销
op.finish_time = current_time + op_loggp.alpha
```

注意：LogGP 的固定开销 $\alpha = L + 2o + g$ 在 op **完成时**添加到 `finish_time`，而不是在开始时。这是因为仿真器模拟的是数据传输阶段的带宽共享，$\alpha$ 代表传输前后的固定开销。

### 7. 浮点容差

```python
FLOAT_TOL = 1e-9  # 1 纳秒
```

用于浮点比较，避免因精度问题导致仿真卡死（如 `remaining_bytes` 极接近 0 但不为 0）。

## 核心类/函数表

| 名称 | 类型 | 说明 |
|------|------|------|
| `FLOAT_TOL` | 常量 | 浮点容差（$10^{-9}$ 秒） |
| `SimulationResult` | dataclass | 仿真结果，包含 ops、makespan、per_rank_finish |
| `simulate()` | 函数 | 事件驱动仿真入口，支持拥塞建模和分层 LogGP |

## 与其他模块的关系

- **消费 `dag_builder.py`**：读取 Op 的 `deps`、`src`、`dst`、`size` 驱动调度
- **消费 `topology.py`**：调用 `resolve_path()` 获取路径资源，调用 `get_loggp()` 获取分层参数
- **消费 `loggp.py`**：使用 `alpha` 属性计算 Op 完成时间
- **被 `collectives.py` 间接依赖**：collectives 生成的 Op DAG 最终由 simulate 执行
- **被 `validation.py` 校验**：仿真结果与解析公式对比验证

## 小结

`simulator.py` 是网络模块的执行引擎，通过事件驱动循环和 max-min fair 带宽共享，精确模拟了多操作并发时的带宽竞争。它支持全局和分层 LogGP 参数，使用路径缓存和堆数据结构保证性能。仿真器不需要显式的接收端序列化约束——带宽竞争由共享资源的公平分配自然建模。
