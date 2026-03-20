# `topology.py` -- 网络拓扑抽象与实现

## 文件概述

`topology.py` 定义了网络拓扑的抽象接口和五种具体实现，是仿真器进行带宽竞争建模的基础。核心抽象是 `resolve_path(src, dst) -> list[Resource]`：给定源和目标 rank，返回消息经过的共享网络资源序列。

五种拓扑从简单到复杂：

1. **FullyConnectedTopology** -- 全连接，无竞争（验证基准）
2. **RingTopology** -- 双向环，最短路径路由
3. **SwitchTopology** -- 星型，共享交换机
4. **NVLinkMeshTopology** -- NVLink 全连接网格（DGX 节点内）
5. **HierarchicalTopology** -- 多节点分层（NVLink + InfiniBand）

## 关键代码解析

### 1. Resource -- 方向性网络资源

```python
@dataclass(frozen=True)
class Resource:
    name: str           # 唯一标识符
    bandwidth: float    # 最大带宽（bytes/second）
```

Resource 是不可变的（`frozen=True`），表示一个**方向性**的网络组件。全双工链路 = 2 个 Resource 对象（正向+反向）。

### 2. Topology 抽象基类

```python
class Topology(ABC):
    @abstractmethod
    def resolve_path(self, src: int, dst: int) -> list[Resource]: ...

    @abstractmethod
    def all_resources(self) -> list[Resource]: ...

    def get_bandwidth(self, src: int, dst: int) -> float:
        path = self.resolve_path(src, dst)
        return min(res.bandwidth for res in path)
```

仿真器只依赖 `resolve_path()` 方法。`get_bandwidth()` 提供便捷的带宽查询（取路径上最小带宽）。

### 3. FullyConnectedTopology -- 全连接拓扑

```python
class FullyConnectedTopology(Topology):
    def __init__(self, num_ranks: int, link_bandwidth: float):
        self._resources: dict[tuple[int, int], Resource] = {}
        for src in range(num_ranks):
            for dst in range(num_ranks):
                if src != dst:
                    self._resources[(src, dst)] = Resource(
                        name=f"link_{src}->{dst}", bandwidth=link_bandwidth)

    def resolve_path(self, src, dst) -> list[Resource]:
        if src == dst: return []
        return [self._resources[(src, dst)]]
```

每对 (src, dst) 有独立链路，$P(P-1)$ 条方向性链路。**不同 (src, dst) 对之间零竞争**，适合验证集合通信算法的解析公式。

### 4. RingTopology -- 环形拓扑

```python
class RingTopology(Topology):
    def resolve_path(self, src, dst) -> list[Resource]:
        forward_dist = (dst - src) % self.num_ranks
        backward_dist = (src - dst) % self.num_ranks

        if forward_dist <= backward_dist:
            # 正向路径: src -> src+1 -> ... -> dst
            path = []
            current = src
            for _ in range(forward_dist):
                next_rank = (current + 1) % self.num_ranks
                path.append(self._resources[(current, next_rank)])
                current = next_rank
            return path
        else:
            # 反向路径（更短）
            # ...
```

**最短路径路由**：比较正向距离和反向距离，选择较短的一条。路径上的每条链路都是共享资源，多消息经过同一链路时会产生带宽竞争。

例如 8 节点环中 `0 -> 7`：正向距离 7，反向距离 1，选择反向路径 `0 -> 7`（1 跳）。

### 5. SwitchTopology -- 星型交换机拓扑

```python
class SwitchTopology(Topology):
    def __init__(self, num_ranks, link_bandwidth, switch_bandwidth):
        self.uplinks = [Resource(f"uplink_{i}", link_bandwidth) for i in range(num_ranks)]
        self.switch_fabric = Resource("switch_fabric", switch_bandwidth)
        self.downlinks = [Resource(f"downlink_{i}", link_bandwidth) for i in range(num_ranks)]

    def resolve_path(self, src, dst) -> list[Resource]:
        return [self.uplinks[src], self.switch_fabric, self.downlinks[dst]]
```

三级资源路径：`uplink[src] -> switch_fabric -> downlink[dst]`

- **uplink/downlink**：每个 rank 独立的 NIC 链路
- **switch_fabric**：所有流量共享的交换矩阵

该模型适合 ToR（Top-of-Rack）交换机场景，switch_fabric 带宽限制了二分带宽。

### 6. NVLinkMeshTopology -- NVLink 全连接网格

```python
class NVLinkMeshTopology(Topology):
    def __init__(self, num_gpus, nvlink_bandwidth, links_per_pair):
        aggregate_bw = nvlink_bandwidth * links_per_pair
        for src in range(num_gpus):
            for dst in range(num_gpus):
                if src != dst:
                    self._resources[(src, dst)] = Resource(
                        name=f"nvlink_{src}->{dst}", bandwidth=aggregate_bw)
```

模拟 DGX A100/H100 等系统的节点内 NVLink 网格。多条 NVLink 的带宽聚合为单一资源：

$$\text{aggregate\_bw} = \text{nvlink\_bandwidth} \times \text{links\_per\_pair}$$

例如 DGX A100：8 GPU，每对 12 条 NVLink @ 25 GB/s = 300 GB/s 聚合带宽。

### 7. HierarchicalTopology -- 分层拓扑

```python
class HierarchicalTopology(Topology):
    def resolve_path(self, src, dst) -> list[Resource]:
        src_node = src // self.gpus_per_node
        dst_node = dst // self.gpus_per_node

        if src_node == dst_node:
            # 节点内: NVLink mesh
            return self.nvlink_meshes[src_node].resolve_path(src_local, dst_local)
        else:
            # 节点间: IB uplink -> fabric -> IB downlink
            return [self.ib_uplinks[src_node], self.ib_fabric, self.ib_downlinks[dst_node]]

    def get_loggp(self, src, dst):
        if src_node == dst_node:
            return self.loggp_nvlink  # 节点内 LogGP
        else:
            return self.loggp_ib     # 节点间 LogGP
```

**分层路由**：根据 src 和 dst 是否在同一节点，选择不同的网络层：

| 场景 | 路径 | LogGP 参数 |
|------|------|-----------|
| 同节点 | NVLink mesh（高带宽、低延迟） | `loggp_nvlink` |
| 跨节点 | IB uplink -> fabric -> IB downlink（低带宽、高延迟） | `loggp_ib` |

**工厂方法** `from_profiled_model` 可以直接从 profiler 产出的 JSON 创建拓扑：

```python
topo = HierarchicalTopology.from_profiled_model(
    "perlmutter", num_ranks=32, ranks_per_node=4
)
```

## 核心类/函数表

| 名称 | 类型 | 说明 |
|------|------|------|
| `Resource` | frozen dataclass | 方向性网络资源（名称 + 带宽） |
| `Topology` | ABC | 拓扑抽象基类，定义 `resolve_path` 接口 |
| `FullyConnectedTopology` | 类 | 全连接，$P(P-1)$ 独立链路 |
| `RingTopology` | 类 | 双向环，最短路径路由 |
| `SwitchTopology` | 类 | 星型交换机（uplink + fabric + downlink） |
| `NVLinkMeshTopology` | 类 | NVLink 全连接网格，多链路聚合 |
| `HierarchicalTopology` | 类 | 分层：节点内 NVLink + 节点间 InfiniBand |

## 与其他模块的关系

- **被 `simulator.py` 消费**：仿真器调用 `resolve_path()` 确定每个 op 的资源路径，调用 `get_loggp()` 获取分层参数
- **使用 `loggp.py`**：`HierarchicalTopology` 存储每层的 `LogGPParams`
- **使用 `model_loader.py`**：`from_profiled_model()` 调用 `load_hierarchical_loggp()` 加载参数
- **被 `validation.py` 间接关联**：验证函数的解析公式假设 `FullyConnectedTopology`（无竞争）

## 小结

`topology.py` 通过 `resolve_path` 抽象将物理网络拓扑与仿真引擎解耦。五种拓扑从无竞争的全连接（验证用）到有层次化带宽竞争的 HierarchicalTopology（生产用），覆盖了从单节点到多节点集群的各种场景。Resource 的方向性和不可变性设计确保了仿真的正确性和安全性。
