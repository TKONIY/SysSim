# `validation.py` -- 解析公式验证器

## 文件概述

`validation.py` 为每种集合通信算法提供了闭式（closed-form）性能公式，用于验证仿真器的正确性。所有公式假设在理想化拓扑（`FullyConnectedTopology`，无竞争）上运行，使用 LogGP 模型计算预期时间，与仿真结果对比。

## 关键代码解析

### 1. 验证函数的统一模式

所有验证函数遵循相同的签名和返回模式：

```python
def validate_xxx(num_ranks, total_size, loggp, simulated_time, tolerance=1e-5) -> tuple[bool, float, float]:
    # 1. 计算解析时间
    analytical_time = ...
    # 2. 计算相对误差
    relative_error = abs(simulated_time - analytical_time) / analytical_time
    # 3. 判断是否通过
    is_valid = relative_error <= tolerance
    return is_valid, analytical_time, relative_error
```

返回三元组：`(是否通过, 解析时间, 相对误差)`。

### 2. Ring Allreduce 验证

```python
def validate_allreduce(num_ranks, total_size, loggp, simulated_time, tolerance=1e-5):
    P = num_ranks
    chunk_size = total_size / P
    num_steps = 2 * (P - 1)
    step_time = loggp.alpha + (chunk_size - 1) * loggp.G
    analytical_time = num_steps * step_time
    # ...
```

Ring Allreduce 解析公式：

$$T = 2(P-1) \cdot \left(\alpha + \left(\frac{M}{P} - 1\right) \cdot G\right)$$

其中 $\alpha = L + 2o + g$，$M$ 为总数据量，$P$ 为 rank 数。

### 3. Binomial Tree Broadcast / Reduce 验证

```python
def validate_broadcast(num_ranks, total_size, loggp, simulated_time, tolerance=1e-5):
    P = num_ranks
    num_steps = math.ceil(math.log2(P))
    step_time = loggp.alpha + (total_size - 1) * loggp.G
    analytical_time = num_steps * step_time
    # ...
```

二项树广播/归约公式：

$$T = \lceil \log_2 P \rceil \cdot (\alpha + (M - 1) \cdot G)$$

注意：reduce 与 broadcast 使用相同公式（`validate_reduce` 直接调用 `validate_broadcast`）。

### 4. Ring Reduce-Scatter / Allgather / Alltoall 验证

三者在无竞争拓扑上公式相同：

$$T = (P-1) \cdot \left(\alpha + \left(\frac{M}{P} - 1\right) \cdot G\right)$$

代码中 `validate_allgather` 和 `validate_alltoall` 直接委托给 `validate_reduce_scatter`。

### 5. Scatter 验证

Scatter 的发送被串行化（Rule 2），公式与 reduce-scatter 相同：

$$T = (P-1) \cdot \left(\alpha + \left(\frac{M}{P} - 1\right) \cdot G\right)$$

### 6. Gather 验证（特殊情况）

```python
def validate_gather(num_ranks, total_size, loggp, bandwidth, simulated_time, tolerance=1e-5):
    P = num_ranks
    chunk_size = total_size / P
    # FullyConnectedTopology 上无竞争，所有发送并行完成
    analytical_time = loggp.alpha + (chunk_size - 1) * loggp.G
    # ...
```

Gather 在 `FullyConnectedTopology` 上所有非 root rank 并行发送到 root，无带宽竞争，因此时间等于**单条消息**的传输时间：

$$T = \alpha + \left(\frac{M}{P} - 1\right) \cdot G$$

注意：该函数多了一个 `bandwidth` 参数（仅用于文档说明，不参与计算）。在有竞争的拓扑上（如 SwitchTopology），实际时间会更长。

### 7. 公式汇总

| 集合通信 | 解析公式 | 步数 |
|---------|---------|------|
| allreduce | $2(P-1) \cdot (\alpha + (M/P - 1)G)$ | $2(P-1)$ |
| broadcast | $\lceil \log_2 P \rceil \cdot (\alpha + (M-1)G)$ | $\lceil \log_2 P \rceil$ |
| reduce | $\lceil \log_2 P \rceil \cdot (\alpha + (M-1)G)$ | $\lceil \log_2 P \rceil$ |
| reduce_scatter | $(P-1) \cdot (\alpha + (M/P - 1)G)$ | $P-1$ |
| allgather | $(P-1) \cdot (\alpha + (M/P - 1)G)$ | $P-1$ |
| alltoall | $(P-1) \cdot (\alpha + (M/P - 1)G)$ | $P-1$ |
| scatter | $(P-1) \cdot (\alpha + (M/P - 1)G)$ | $P-1$ |
| gather | $\alpha + (M/P - 1)G$ | 1（并行） |

## 核心类/函数表

| 函数 | 说明 |
|------|------|
| `validate_allreduce()` | 验证 ring allreduce |
| `validate_broadcast()` | 验证 binomial tree broadcast |
| `validate_reduce()` | 验证 binomial tree reduce（委托 broadcast） |
| `validate_reduce_scatter()` | 验证 ring reduce-scatter |
| `validate_allgather()` | 验证 ring allgather（委托 reduce_scatter） |
| `validate_alltoall()` | 验证 direct alltoall（委托 reduce_scatter） |
| `validate_scatter()` | 验证 flat scatter（委托 reduce_scatter） |
| `validate_gather()` | 验证 flat gather（无竞争并行） |

## 与其他模块的关系

- **消费 `loggp.py`**：使用 `LogGPParams.alpha` 和 `LogGPParams.G` 计算解析公式
- **验证 `simulator.py`**：将仿真器输出的 `makespan` 与解析结果比较
- **验证 `collectives.py`**：间接验证集合通信算法的正确性
- **假设 `topology.py`**：所有公式基于 `FullyConnectedTopology`（无竞争）

## 小结

`validation.py` 是网络模块的"测试基准"，为 8 种集合通信提供了解析公式。通过将仿真结果与解析结果比较（默认容差 $10^{-5}$），可以高效验证仿真器和算法实现的正确性。所有公式基于 LogGP 模型和无竞争拓扑假设，构成了从理论到实验的可信闭环。
