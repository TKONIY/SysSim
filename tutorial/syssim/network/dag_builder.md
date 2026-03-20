# `dag_builder.py` -- DAG 构建与自动依赖推断

## 文件概述

`dag_builder.py` 负责将高层通信模式（步骤列表）转换为依赖 DAG（有向无环图），供仿真器执行。它是连接集合通信算法（collectives）与仿真引擎（simulator）的桥梁。

核心设计思想：通过 **两条物理规则** 自动推断操作间的依赖关系，无需手动指定。

## 关键代码解析

### 1. 核心数据结构

#### Step 类型别名

```python
Step = list[tuple[int, int, float]]
```

一个 Step 表示一个算法阶段中可以并发执行的所有发送操作，每个元素为 `(src, dst, size_bytes)` 三元组。

#### Op 数据类

```python
@dataclass
class Op:
    src: int
    dst: int
    size: float  # bytes
    deps: list['Op'] = field(default_factory=list)
    tag: str = ""
    loggp: Optional['LogGPParams'] = None

    # Simulation state (populated by engine)
    remaining_bytes: float = 0.0
    start_time: float = 0.0
    finish_time: float = 0.0
```

Op 是通信 DAG 中的基本单元，表示一次点对点发送。字段含义：

| 字段 | 类型 | 说明 |
|------|------|------|
| `src`, `dst` | `int` | 源/目标 rank |
| `size` | `float` | 消息大小（字节） |
| `deps` | `list[Op]` | 前置依赖列表 |
| `tag` | `str` | 调试标签，如 `"allreduce_step_0"` |
| `loggp` | `Optional[LogGPParams]` | 层特定的 LogGP 参数（用于分层拓扑） |
| `remaining_bytes` | `float` | 仿真状态：剩余待传输字节数 |
| `start_time` | `float` | 仿真状态：传输开始时间 |
| `finish_time` | `float` | 仿真状态：传输完成时间 |

### 2. `build_dag()` -- 两条规则的依赖推断

```python
def build_dag(steps: list[Step], tag_prefix: str = "") -> list[Op]:
    all_ops: list[Op] = []

    # Track last send FROM each rank (for Rule 2: send serialization)
    last_send_from: dict[int, Op] = {}

    # Track ops that send TO each rank in previous step (for Rule 1: data dependency)
    prev_sends_to: dict[int, list[Op]] = {}

    for step_idx, step in enumerate(steps):
        curr_sends_to: dict[int, list[Op]] = {}

        for src, dst, size in step:
            tag = f"{tag_prefix}_step_{step_idx}" if tag_prefix else f"step_{step_idx}"
            op = Op(src=src, dst=dst, size=size, tag=tag)

            # Rule 1: Data dependency
            if src in prev_sends_to:
                op.deps.extend(prev_sends_to[src])

            # Rule 2: Send serialization
            if src in last_send_from:
                if last_send_from[src] not in op.deps:
                    op.deps.append(last_send_from[src])

            # Update tracking
            last_send_from[src] = op
            if dst not in curr_sends_to:
                curr_sends_to[dst] = []
            curr_sends_to[dst].append(op)

            all_ops.append(op)

        prev_sends_to = curr_sends_to

    return all_ops
```

**两条物理规则详解：**

**规则 1：数据依赖（receive-before-send）**

> 在第 $s$ 步从 rank $r$ 发送的操作，依赖于第 $s-1$ 步中所有发送**到** rank $r$ 的操作。

直觉：rank 不能发送它还没有收到的数据。例如在 ring reduce-scatter 中，rank 0 在第 1 步发送数据之前，必须先收到第 0 步中 rank 2 发来的数据。

**规则 2：发送序列化（single-threaded NIC）**

> 从 rank $r$ 发送的操作，依赖于该 rank 上一次发送操作。

直觉：NIC 一次只能发送一条消息。rank 0 必须完成第一条消息的发送后，才能开始第二条。

**为什么不需要接收端序列化？**

接收端的带宽竞争由仿真器的 **max-min fair** 带宽共享机制自动处理。显式添加接收端依赖会过于保守，导致不必要的串行化。

### 3. 依赖推断示例

以 3-rank ring reduce-scatter 为例：

```
Step 0: (0->1, 100B), (1->2, 100B), (2->0, 100B)
Step 1: (0->1, 100B), (1->2, 100B), (2->0, 100B)
```

Step 1 中 `0->1` 的依赖：
- **Rule 1**：Step 0 中发送到 rank 0 的操作 = `(2->0)`
- **Rule 2**：rank 0 上一次发送 = Step 0 中的 `(0->1)`

因此 `Step1(0->1).deps = [Step0(2->0), Step0(0->1)]`。

## 核心类/函数表

| 名称 | 类型 | 说明 |
|------|------|------|
| `Step` | 类型别名 | `list[tuple[int, int, float]]`，一步中的并发发送列表 |
| `Op` | dataclass | 通信 DAG 中的单次点对点发送操作 |
| `build_dag()` | 函数 | 将步骤列表转换为带依赖的 Op DAG |

## 与其他模块的关系

- **被 `collectives.py` 调用**：所有 8 种集合通信函数最终调用 `build_dag()` 生成 Op 列表
- **被 `simulator.py` 消费**：仿真引擎读取 Op 的 `deps`、`src`、`dst`、`size` 进行事件驱动调度
- **`Op.loggp` 被 `topology.py` 的 `HierarchicalTopology` 填充**：分层拓扑为不同层的操作设置不同的 LogGP 参数

## 小结

`dag_builder.py` 实现了从"通信模式"到"依赖 DAG"的自动转换。两条物理规则（数据依赖 + 发送序列化）既保证了正确性，又避免了过度保守的约束。这种设计使得上层算法只需描述步骤，而依赖关系完全由物理约束自动推导。
