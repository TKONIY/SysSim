# `collectives.py` -- 集合通信算法构建器

## 文件概述

`collectives.py` 提供了 8 种集合通信原语的算法实现，是网络模块的核心组件之一。每个函数接收参与通信的 rank 列表和数据大小，返回一个 Op（操作）DAG，描述了通信的完整模式。

这些构建器只描述"做什么"（通信模式），不涉及"怎么做"（资源分配和时序调度）。拓扑和 LogGP 参数在仿真阶段才被引入。

该文件参考了以下经典算法：
- Thakur et al., 2005: "Optimization of Collective Communication Operations in MPICH"
- NCCL 算法文档

## 关键代码解析

### 1. Ring Allreduce（环形全归约）

Allreduce 是分布式训练中最常用的集合通信操作，采用经典的 **Ring 算法**，分为两个阶段：

```python
def allreduce(ranks: list[int], total_size: float, tag_prefix: str = "") -> list[Op]:
    P = len(ranks)
    if P < 2:
        raise ValueError(f"allreduce requires at least 2 ranks, got {P}")

    chunk_size = total_size / P
    steps: list[Step] = []

    # Reduce-scatter phase: P-1 steps
    for step in range(P - 1):
        step_ops = []
        for i in range(P):
            src = ranks[i]
            dst = ranks[(i + 1) % P]
            step_ops.append((src, dst, chunk_size))
        steps.append(step_ops)

    # Allgather phase: P-1 steps
    for step in range(P - 1):
        step_ops = []
        for i in range(P):
            src = ranks[i]
            dst = ranks[(i + 1) % P]
            step_ops.append((src, dst, chunk_size))
        steps.append(step_ops)

    return build_dag(steps, tag_prefix or "allreduce")
```

**算法分析：**

- 数据被分为 $P$ 个 chunk，每个大小为 $M/P$
- **Reduce-scatter 阶段**：$P-1$ 步，每步每个 rank 沿环向下一个 rank 发送一个 chunk
- **Allgather 阶段**：$P-1$ 步，结构与 reduce-scatter 相同
- 总共 $2(P-1)$ 步，每步传输 $M/P$ 字节

**时间复杂度**：

$$T = 2(P-1) \cdot \left(\alpha + \frac{M}{P} \cdot G\right)$$

其中 $\alpha = L + 2o$ 为固定流水线开销，$G = 1/\text{bandwidth}$ 为每字节传输时间。

### 2. Binomial Tree Broadcast（二项树广播）

```python
def broadcast(ranks: list[int], total_size: float, root: int = 0, tag_prefix: str = "") -> list[Op]:
    P = len(ranks)
    # Reorder ranks so root is at index 0
    root_idx = ranks.index(root)
    reordered = ranks[root_idx:] + ranks[:root_idx]

    steps: list[Step] = []
    num_steps = math.ceil(math.log2(P))

    # Binomial tree: at step k, ranks [0, 2^k) send to [2^k, 2^(k+1))
    for step in range(num_steps):
        step_ops = []
        stride = 1 << step  # 2^step
        for i in range(min(stride, P)):
            dst_idx = i + stride
            if dst_idx < P:
                src = reordered[i]
                dst = reordered[dst_idx]
                step_ops.append((src, dst, total_size))
        if step_ops:
            steps.append(step_ops)

    return build_dag(steps, tag_prefix or "broadcast")
```

**算法原理：**

二项树广播在第 $k$ 步（$k=0,1,...$）中，已有数据的 rank 将数据发送给距离 $2^k$ 的新 rank。例如对于 4 个 rank：
- 第 0 步：`rank 0 -> rank 1`
- 第 1 步：`rank 0 -> rank 2`, `rank 1 -> rank 3`（并行）

**时间复杂度**：

$$T = \lceil \log_2 P \rceil \cdot (\alpha + (M-1) \cdot G)$$

### 3. Binomial Tree Reduce（二项树归约）

Reduce 是 broadcast 的镜像操作。数据从叶子节点汇聚到根节点：

```python
for step in range(num_steps):
    step_ops = []
    stride = 1 << step
    pair_distance = 1 << (step + 1)
    i = 0
    while i < P:
        src_idx = i + stride
        if src_idx < P:
            src = reordered[src_idx]
            dst = reordered[i]
            step_ops.append((src, dst, total_size))
        i += pair_distance
    if step_ops:
        steps.append(step_ops)
```

第 $k$ 步中，间距为 $2^k$ 的 rank 对进行通信，接收方位于 $2^{k+1}$ 的倍数位置。

### 4. Alltoall（全交换）-- 交错配对

```python
def alltoall(ranks: list[int], total_size: float, tag_prefix: str = "") -> list[Op]:
    P = len(ranks)
    chunk_size = total_size / P
    steps: list[Step] = []
    # P-1 steps, staggered pairings
    for step in range(1, P):
        step_ops = []
        for i in range(P):
            src = ranks[i]
            dst = ranks[(i + step) % P]
            step_ops.append((src, dst, chunk_size))
        steps.append(step_ops)
    return build_dag(steps, tag_prefix or "alltoall")
```

**关键设计**：使用交错配对（staggered pairings），第 $k$ 步每个 rank $i$ 向 rank $(i+k) \bmod P$ 发送数据，避免通信冲突。

### 5. Scatter 与 Gather -- 扁平树

```python
# Scatter: root 逐一发送（串行，由 Rule 2 保证）
for rank in ranks:
    if rank != root:
        steps.append([(root, rank, chunk_size)])

# Gather: 所有非 root 并行发送到 root（单步）
step_ops = []
for rank in ranks:
    if rank != root:
        step_ops.append((rank, root, chunk_size))
return build_dag([step_ops], tag_prefix or "gather")
```

Scatter 每个发送占一步（串行化），Gather 所有发送在同一步（并行，由仿真器处理 root 端的带宽竞争）。

## 核心类/函数表

| 函数 | 算法 | 步数 | 每步数据量 | 时间复杂度 |
|------|------|------|-----------|-----------|
| `allreduce` | Ring | $2(P-1)$ | $M/P$ | $2(P-1)(\alpha + M/P \cdot G)$ |
| `broadcast` | Binomial tree | $\lceil \log_2 P \rceil$ | $M$ | $\lceil \log_2 P \rceil(\alpha + (M-1) G)$ |
| `reduce` | Binomial tree | $\lceil \log_2 P \rceil$ | $M$ | $\lceil \log_2 P \rceil(\alpha + (M-1) G)$ |
| `reduce_scatter` | Ring | $P-1$ | $M/P$ | $(P-1)(\alpha + M/P \cdot G)$ |
| `allgather` | Ring | $P-1$ | $M/P$ | $(P-1)(\alpha + M/P \cdot G)$ |
| `alltoall` | Direct staggered | $P-1$ | $M/P$ | $(P-1)(\alpha + M/P \cdot G)$ |
| `scatter` | Flat tree (串行) | $P-1$ | $M/P$ | $(P-1)(\alpha + M/P \cdot G)$ |
| `gather` | Flat tree (并行) | $1$ | $M/P$ | $\alpha + (M/P - 1) G$（无竞争时） |

## 与其他模块的关系

- **依赖 `dag_builder.py`**：所有函数最终调用 `build_dag(steps, tag_prefix)` 将步骤转换为带依赖关系的 Op DAG
- **被 `simulator.py` 消费**：生成的 Op 列表传递给 `simulate()` 进行事件驱动仿真
- **被 `validation.py` 验证**：解析公式用于校验仿真结果的正确性

## 小结

`collectives.py` 是网络模块中算法层的核心，将 8 种集合通信操作抽象为"步骤序列"，再由 `build_dag()` 自动推断依赖关系。这种分层设计使算法描述与底层资源调度完全解耦，便于扩展新算法和拓扑。
