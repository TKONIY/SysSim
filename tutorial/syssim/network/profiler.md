# `profiler.py` -- LogGP 参数自动校准工具

## 文件概述

`profiler.py` 是一个完整的 CLI 工具，用于通过 NCCL ping-pong 基准测试自动校准 LogGP 参数。它实现了 Hoefler et al. (2009) 提出的 **PRTT（Parametrized Round Trip Time）** 方法，能够：

1. 在多种消息大小上执行 ping-pong 微基准测试
2. 自动检测协议变化点（eager -> rendezvous）
3. 提取硬件专属的 L, o, g, G 参数
4. 将结果保存为 JSON 文件

支持 **单层 profiling**（如纯 NVLink）和 **分层 profiling**（如 NVLink + InfiniBand）两种模式。

## 关键代码解析

### 1. 核心数据类

```python
@dataclass
class ProfilingResult:
    topology: str
    protocols: List[Dict[str, Any]]
    primary: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class LayerConfig:
    topology_type: str
    scope: Dict[str, Any]  # {"vary_dims": [...], "fix_dims": {...}}
    description: str = ""
    expected_bandwidth_gbs: Optional[float] = None

@dataclass
class HierarchyConfig:
    topology_name: str
    mesh: Dict[str, Any]  # {"shape": [...], "dimension_names": [...], "topology_types": [...]}
    profiling_params: Dict[str, Any]
```

| 数据类 | 说明 |
|--------|------|
| `ProfilingResult` | 单层 profiling 的完整结果 |
| `LayerConfig` | 单个网络层的配置（基于 mesh scope） |
| `HierarchyConfig` | 分层拓扑的完整配置 |
| `LayerProfilingResult` | 单层 profiling 结果 |
| `HierarchicalProfilingResult` | 分层 profiling 的聚合结果 |

### 2. 通信后端抽象与 NCCL 实现

```python
class CommBackend(ABC):
    @abstractmethod
    def ping_pong(self, n, delay, size, peer_rank=None) -> float: ...
    @abstractmethod
    def is_client(self) -> bool: ...
    @abstractmethod
    def is_server(self) -> bool: ...
    @abstractmethod
    def barrier(self): ...
    @abstractmethod
    def cleanup(self): ...

class NCCLBackend(CommBackend):
    def __init__(self):
        import torch, torch.distributed as dist
        dist.init_process_group(backend="nccl")
        self.rank = dist.get_rank()
        # ...
```

`CommBackend` 定义了通信后端的抽象接口，`NCCLBackend` 是基于 PyTorch NCCL 的具体实现。ping-pong 的核心逻辑：

```python
def ping_pong(self, n, delay, size, peer_rank=None) -> float:
    buf = self.torch.zeros(size, dtype=self.torch.uint8, device=self.device)

    if is_client:
        start_event = self.torch.cuda.Event(enable_timing=True)
        end_event = self.torch.cuda.Event(enable_timing=True)
        start_event.record()

        for i in range(n):
            self.dist.send(buf, dst=peer_rank)
            self.dist.recv(buf, src=peer_rank)
            if delay > 0:
                self.torch.cuda.synchronize()
                time.sleep(delay)

        end_event.record()
        self.torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / 1000.0
    elif is_server:
        for i in range(n):
            self.dist.recv(buf, src=peer_rank)
            self.dist.send(buf, dst=peer_rank)
        return 0.0
```

客户端发送-接收 $n$ 次，使用 CUDA Event 精确计时；服务端镜像执行接收-发送。

### 3. PRTT 测量与消息大小扫描

```python
def measure_prtt(backend, n, delay, size, num_runs=10, peer_rank=None) -> float:
    times = []
    for _ in range(num_runs):
        backend.barrier()
        elapsed = backend.ping_pong(n, delay, size, peer_rank=peer_rank)
        times.append(elapsed)
    return float(np.median(times))

def sweep_message_sizes(backend, min_size=1, max_size=65536, n=10, num_runs=10, peer_rank=None):
    # 指数扫描：1, 2, 4, 8, ..., max_size
    sizes = [1, 2, 4, ..., max_size]
    for size in sizes:
        prtt_1_0 = measure_prtt(backend, n=1, delay=0.0, size=size, ...)
        prtt_n_0 = measure_prtt(backend, n=n, delay=0.0, size=size, ...)
        prtt_n_dG = measure_prtt(backend, n=n, delay=prtt_1_0, size=size, ...)
        measurements.append(PRTTMeasurement(size, prtt_1_0, prtt_n_0, prtt_n_dG))
```

对每个消息大小 $s$，测量三个 PRTT 值：

| 测量 | 含义 | 用途 |
|------|------|------|
| $\text{PRTT}(1, 0, s)$ | 单次往返时间 | 提取 $L$，计算延迟 $d_G$ |
| $\text{PRTT}(n, 0, s)$ | $n$ 次无延迟往返 | 计算 $G_{\text{all}}(s) = g + (s-1)G$ |
| $\text{PRTT}(n, d_G, s)$ | $n$ 次带延迟往返 | 提取 CPU overhead $o$ |

### 4. LogGP 参数提取

```python
def extract_loggp_parameters(measurements, protocol, n=10):
    g = protocol.g   # 来自协议检测的拟合值
    G = protocol.G

    # 计算 o
    os = []
    for m in protocol_measurements:
        dG = m.prtt_1_0
        o_s = (m.prtt_n_dG - m.prtt_1_0) / (n - 1) - dG
        os.append(o_s)
    o = float(np.median(os))

    # 计算 L：PRTT(1,0,s) = 2*(L + 2*o + g + (s-1)*G)
    Ls = []
    for m in protocol_measurements:
        L_s = (m.prtt_1_0 / 2.0) - 2*o - g - (m.size - 1)*G
        Ls.append(L_s)
    L = float(np.median(Ls))

    return L, o, g, G
```

**参数提取公式**（Hoefler 方法）：

1. $g$ 和 $G$ 由协议检测阶段的最小二乘拟合得到
2. CPU overhead:
$$o_s = \frac{\text{PRTT}(n, d_G, s) - \text{PRTT}(1, 0, s)}{n - 1} - d_G$$
3. 网络延迟（由 $\text{PRTT}(1, 0, s) = 2(L + 2o + g + (s-1)G)$ 推导）：
$$L_s = \frac{\text{PRTT}(1, 0, s)}{2} - 2o - g - (s-1)G$$

使用 **中位数** 而非均值，增强对异常值的鲁棒性。

### 5. 分层 Profiling

```python
class HierarchyConfig:
    def get_auto_layers(self) -> Dict[str, LayerConfig]:
        mesh = self.get_device_mesh()
        layers = {}
        for dim_idx, dim_name in enumerate(mesh.dimension_names):
            topology_type = mesh.topology_types[dim_idx]
            vary_dims = [dim_name]
            fix_dims = {other: 0 for i, other in enumerate(mesh.dimension_names) if i != dim_idx}
            layers[dim_name] = LayerConfig(topology_type=topology_type, scope={"vary_dims": vary_dims, "fix_dims": fix_dims})
        return layers
```

自动从 mesh 维度生成层配置：每个维度对应一个网络层，变化该维度、固定其余维度为 0。例如 `shape=[4,4], dimension_names=["node","gpu"]` 生成：
- `node` 层：vary `node`，fix `gpu=0`（节点间通信）
- `gpu` 层：vary `gpu`，fix `node=0`（节点内通信）

### 6. CLI 入口

```bash
# 单层 profiling
torchrun --nproc_per_node=2 -m syssim.network.profiler --topology nvlink

# 分层 profiling
torchrun --nproc_per_node=2 -m syssim.network.profiler --hierarchy-config config.json
```

## 核心类/函数表

| 名称 | 类型 | 说明 |
|------|------|------|
| `CommBackend` | ABC | 通信后端抽象接口 |
| `NCCLBackend` | 类 | NCCL GPU-to-GPU 通信后端 |
| `measure_prtt()` | 函数 | 统计采样的 PRTT 测量 |
| `sweep_message_sizes()` | 函数 | 消息大小扫描 |
| `extract_loggp_parameters()` | 函数 | 从 PRTT 提取 L, o, g, G |
| `ProfilingResult` | dataclass | 单层 profiling 结果 |
| `LayerConfig` | dataclass | 网络层配置 |
| `HierarchyConfig` | dataclass | 分层拓扑配置 |
| `profile_single_layer()` | 函数 | 单层 profiling 流程 |
| `profile_hierarchy()` | 函数 | 分层 profiling 流程 |
| `run_profiling()` | 函数 | 单层 profiling 入口 |
| `load_hierarchy_config()` | 函数 | 加载分层配置 JSON |
| `main()` | 函数 | CLI 入口 |

## 与其他模块的关系

- **依赖 `protocol_detector.py`**：使用 `detect_protocol_changes()` 和 `compute_gall()` 进行协议变化检测
- **依赖 `device_mesh.py`**：分层 profiling 使用 `DeviceMesh` 进行 rank 选取
- **产出被 `model_loader.py` 消费**：profiling 结果保存为 JSON，由 model_loader 加载
- **产出被 `topology.py` 消费**：`HierarchicalTopology.from_profiled_model()` 直接使用 profiling 产出的 JSON

## 小结

`profiler.py` 是一个完整的端到端 profiling 工具，从 NCCL ping-pong 基准测试到 LogGP 参数提取全自动化。它支持单层和分层两种模式，使用 PRTT 方法提取的参数可直接用于网络仿真，是性能建模从"理论估算"到"实测校准"的关键环节。
