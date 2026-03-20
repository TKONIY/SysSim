# `operator_graph.py` — 算子图中间表示（IR）

## 文件概述

`operator_graph.py` 定义了 SysSim 的核心数据结构——算子图（OperatorGraph）。这是一个有向无环图（DAG），其中每个节点代表一个算子操作（如矩阵乘法、注意力计算、集合通信等），边表示数据依赖或流同步依赖。该文件提供了完整的图构建、验证、拓扑排序、关键路径分析和多种导出格式。

## 关键代码解析

### OperatorType 枚举

```python
class OperatorType(Enum):
    # Math/Compute
    GEMM = "gemm"
    ATTN = "attn"
    MATH = "math"

    # Communication
    COLLECTIVE = "collective"

    # Memory
    MEMORY = "memory"

    # Sync
    BARRIER = "barrier"
    STREAM_SYNC = "stream_sync"
```

算子类型分为四大类：

| 类别 | 类型 | 说明 |
|------|------|------|
| 计算 | GEMM | 矩阵乘法（mm, addmm, bmm 等） |
| 计算 | ATTN | 注意力计算（scaled_dot_product_attention） |
| 计算 | MATH | 其他数学运算（激活函数、归一化等） |
| 通信 | COLLECTIVE | 集合通信（all_reduce, all_gather 等） |
| 内存 | MEMORY | 跨设备数据拷贝 |
| 同步 | BARRIER | 全局屏障同步 |
| 同步 | STREAM_SYNC | CUDA 流间同步 |

文件还定义了四个冻结集合用于快速分类查询：

```python
_MATH_TYPES = frozenset({OperatorType.GEMM, OperatorType.ATTN, OperatorType.MATH})
_COLLECTIVE_TYPES = frozenset({OperatorType.COLLECTIVE})
_MEMORY_TYPES = frozenset({OperatorType.MEMORY})
_SYNC_TYPES = frozenset({OperatorType.BARRIER, OperatorType.STREAM_SYNC})
```

### TensorMeta 不可变数据类

```python
@dataclass(frozen=True)
class TensorMeta:
    shape: tuple[int, ...]
    dtype: str
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "dtype": self.dtype, "device": self.device}
```

`TensorMeta` 用 `frozen=True` 修饰，保证不可变性，可安全用作字典键或集合元素。它记录张量的形状、数据类型和设备信息，是算子节点输入/输出的元数据描述。

### OperatorNode 数据类

```python
@dataclass
class OperatorNode:
    # Identity
    name: str
    op_type: OperatorType

    # Configuration
    config: dict[str, Any] = field(default_factory=dict)

    # Dependencies
    data_deps: list[str] = field(default_factory=list)   # 数据依赖
    stream_deps: list[str] = field(default_factory=list)  # 流同步依赖

    # Execution context
    stream_id: int = 0
    device_id: int = 0

    # Tensor metadata
    inputs: list[TensorMeta] = field(default_factory=list)
    outputs: list[TensorMeta] = field(default_factory=list)

    # Performance
    estimated_time_ms: float = 0.0

    # Critical path state
    earliest_start: float = 0.0
    earliest_finish: float = 0.0
```

每个节点包含：
- **标识**：唯一名称 `name` 和算子类型 `op_type`。
- **配置**：算子参数字典（如 GEMM 的 M/N/K、ATTN 的 batch/heads/seq_len）。
- **依赖关系**：`data_deps` 表示数据流依赖（算子 A 的输出是算子 B 的输入），`stream_deps` 表示同一 CUDA 流上的执行顺序依赖。
- **执行上下文**：CUDA 流 ID 和设备 ID。
- **性能信息**：估算的执行时间（毫秒），以及关键路径分析中计算的最早开始/结束时间。

### OperatorGraph 图结构

#### 添加算子与验证

```python
def add_operator(self, node: OperatorNode) -> None:
    if node.name in self.operators:
        raise ValueError(f"Duplicate operator name: {node.name}")
    self.operators[node.name] = node
    self.streams.add(node.stream_id)
    self._topo_cache = None  # 使拓扑排序缓存失效
```

添加算子时会检查名称唯一性，并在图被修改后清除拓扑排序缓存。

```python
def validate(self) -> None:
    # 1. 引用完整性检查
    for name, op in self.operators.items():
        for dep in op.data_deps + op.stream_deps:
            if dep not in self.operators:
                raise ValueError(...)

    # 2. DFS 染色法环检测
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {name: WHITE for name in self.operators}

    def dfs(u: str) -> None:
        color[u] = GRAY
        for v in node.data_deps + node.stream_deps:
            if color[v] == GRAY:      # 回边 -> 存在环
                raise ValueError(...)
            if color[v] == WHITE:
                dfs(v)
        color[u] = BLACK
```

验证包含两步：首先检查所有依赖引用的算子是否存在（引用完整性），然后使用经典的 DFS 三色标记法检测环。

#### 拓扑排序（Kahn 算法）

```python
def topological_sort(self) -> list[str]:
    if self._topo_cache is not None:
        return self._topo_cache

    # 计算入度：每个节点的入度等于其依赖数
    in_degree = {name: 0 for name in self.operators}
    for name, op in self.operators.items():
        in_degree[name] = len(set(op.data_deps + op.stream_deps))

    # 构建后继表：如果 name 依赖 dep，则 dep 的后继包含 name
    successors = {name: [] for name in self.operators}
    for name, op in self.operators.items():
        for dep in set(op.data_deps + op.stream_deps):
            if dep in successors:
                successors[dep].append(name)

    # BFS：从入度为 0 的节点开始
    queue = deque(name for name, deg in in_degree.items() if deg == 0)
    result = []
    while queue:
        u = queue.popleft()
        result.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
```

使用 Kahn 算法进行拓扑排序，结果被缓存到 `_topo_cache` 中。如果排序结果的长度小于节点总数，说明图中存在环。

#### 关键路径分析

```python
def compute_critical_path(self) -> float:
    topo = self.topological_sort()
    stream_times: dict[int, float] = {s: 0.0 for s in self.streams}

    for name in topo:
        op = self.operators[name]
        start = stream_times.get(op.stream_id, 0.0)

        # 数据和流依赖约束
        for dep_name in op.data_deps + op.stream_deps:
            dep = self.operators[dep_name]
            start = max(start, dep.earliest_finish)

        # BARRIER: 等待所有流完成
        if op.op_type == OperatorType.BARRIER:
            start = max(start, max(stream_times.values()) if stream_times else 0.0)
        # STREAM_SYNC: 等待指定流完成
        elif op.op_type == OperatorType.STREAM_SYNC:
            target_stream = op.config.get("target_stream")
            if target_stream is not None and target_stream in stream_times:
                start = max(start, stream_times[target_stream])

        op.earliest_start = start
        op.earliest_finish = start + op.estimated_time_ms
        stream_times[op.stream_id] = op.earliest_finish

    return max(op.earliest_finish for op in self.operators.values())
```

关键路径分析是性能模拟的核心。算法按拓扑序遍历每个算子，计算其最早开始时间（受三种约束：同流前序算子、数据依赖、同步操作），然后加上估算执行时间得到最早结束时间。最终返回所有算子中最大的结束时间，即为模型执行的总耗时估算。

特别注意两种同步语义：
- `BARRIER`：等待 **所有** 流的最新算子完成。
- `STREAM_SYNC`：等待 **指定** 流的最新算子完成。

#### 导出功能

```python
def to_dot(self) -> str:
    """生成 Graphviz DOT 格式，按算子类型着色"""
    color_map = {
        **{t: "lightblue" for t in _MATH_TYPES},       # 计算类: 浅蓝
        **{t: "lightyellow" for t in _COLLECTIVE_TYPES}, # 通信类: 浅黄
        **{t: "lightgreen" for t in _MEMORY_TYPES},      # 内存类: 浅绿
        **{t: "lightgray" for t in _SYNC_TYPES},         # 同步类: 浅灰
    }
```

`to_dot()` 生成 Graphviz DOT 格式的图描述，可用 `dot` 命令渲染为可视化图片。数据依赖用实线表示，流同步依赖用虚线表示。

`to_json()` 将整个图序列化为 JSON，便于保存和跨工具共享。

`summary()` 生成人类可读的摘要，包含算子统计、总时间、关键路径时间和并行度。

## 核心类/函数表

| 名称 | 类型 | 用途 |
|------|------|------|
| `OperatorType` | Enum | 算子类型枚举（7 种） |
| `TensorMeta` | frozen dataclass | 不可变张量元数据（shape, dtype, device） |
| `OperatorNode` | dataclass | 算子节点，包含配置、依赖、性能信息 |
| `OperatorGraph` | class | 算子图 DAG 容器 |
| `OperatorGraph.add_operator` | method | 添加算子节点 |
| `OperatorGraph.validate` | method | 验证引用完整性和无环性 |
| `OperatorGraph.topological_sort` | method | Kahn 算法拓扑排序（带缓存） |
| `OperatorGraph.compute_critical_path` | method | 多流感知的关键路径 DP 分析 |
| `OperatorGraph.to_dot` | method | 导出 Graphviz DOT 格式 |
| `OperatorGraph.to_json` | method | 导出 JSON 格式 |
| `OperatorGraph.summary` | method | 生成人类可读的统计摘要 |

## 与其他模块的关系

- **tracer.py** 是 `OperatorGraph` 的主要生产者：追踪器在拦截每个 PyTorch 算子时，创建 `OperatorNode` 并添加到 `OperatorGraph`。
- **api.py** 返回 `OperatorGraph` 给用户。
- **config.py** 的 `HardwareInfo.get_peak_tflops()` 引用了 `OperatorType` 来区分算子类别。
- **compute 子包** 在估算耗时时会使用 `OperatorType` 来选择对应的 roofline 参数。

## 小结

`operator_graph.py` 是 SysSim 的数据核心，定义了从追踪到分析的完整 IR 体系。其设计亮点包括：双层依赖关系（数据依赖 + 流同步依赖）精确建模 GPU 执行语义、多流感知的关键路径分析、以及丰富的导出格式（DOT 可视化、JSON 序列化、文字摘要）。这个 IR 层将追踪阶段和分析阶段解耦，是整个模拟器架构的枢纽。
