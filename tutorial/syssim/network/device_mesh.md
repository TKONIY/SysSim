# `device_mesh.py` -- 设备网格抽象

## 文件概述

`device_mesh.py` 提供了多维设备网格（Device Mesh）的逻辑抽象，用于层次化网络性能分析（hierarchical profiling）。它将物理的 GPU 集群映射为多维逻辑坐标系统（如 `[node, gpu_in_node]`），使得指定网络层的 rank 范围变得直观。

例如，一个 4 节点 x 4 GPU/节点的集群可以表示为 `shape=[4, 4]`，通过固定一个维度、变化另一个维度来选取节点内或节点间的 rank 子集。

## 关键代码解析

### 1. DeviceMesh 数据类

```python
@dataclass
class DeviceMesh:
    shape: Tuple[int, ...]
    dimension_names: List[str]
    topology_types: List[str] = None
    ranks_order: str = 'C'  # 'C' (row-major) or 'F' (column-major)
```

| 属性 | 说明 | 示例 |
|------|------|------|
| `shape` | 每个维度的大小 | `(4, 4)` 表示 4 节点 x 4 GPU |
| `dimension_names` | 维度命名 | `["node", "gpu"]` |
| `topology_types` | 每个维度对应的网络类型 | `["infiniband", "nvlink"]` |
| `ranks_order` | rank 编号顺序 | `'C'`（行优先）或 `'F'`（列优先） |

`__post_init__` 中进行了严格的参数校验：shape 与 dimension_names 长度一致、维度名唯一、shape 值正数等。

### 2. 坐标与 rank 的相互映射

```python
def rank_at(self, coords: List[int]) -> int:
    return int(np.ravel_multi_index(coords, self.shape, order=self.ranks_order))

def coords_of(self, rank: int) -> List[int]:
    return list(np.unravel_index(rank, self.shape, order=self.ranks_order))
```

这两个方法基于 NumPy 的 `ravel_multi_index` / `unravel_index` 实现多维坐标与一维 rank 编号的互相转换。以 `shape=(4, 4)`, row-major 为例：

| 坐标 `[node, gpu]` | rank |
|-------|------|
| `[0, 0]` | 0 |
| `[0, 2]` | 2 |
| `[1, 2]` | 6 |
| `[3, 3]` | 15 |

### 3. 切片查询 -- `ranks_in_slice`

```python
def ranks_in_slice(self, fix_dims: Dict[str, int], vary_dims: List[str]) -> List[int]:
    ranks = []
    for idx in range(self.total_ranks):
        coords = self.coords_of(idx)
        match = True
        for dim_name, dim_value in fix_dims.items():
            dim_idx = self.dimension_names.index(dim_name)
            if coords[dim_idx] != dim_value:
                match = False
                break
        if match:
            ranks.append(idx)
    return sorted(ranks)
```

该方法根据固定维度和变化维度筛选 rank 子集。例如：

```python
mesh = DeviceMesh([4, 4], ["node", "gpu"])

# 获取 node 0 上的所有 GPU
mesh.ranks_in_slice({"node": 0}, ["gpu"])
# 结果: [0, 1, 2, 3]

# 获取所有节点上的 GPU 0
mesh.ranks_in_slice({"gpu": 0}, ["node"])
# 结果: [0, 4, 8, 12]
```

这在分层性能分析中非常有用：固定 `node=0` 变化 `gpu` 得到节点内通信的 rank 组，固定 `gpu=0` 变化 `node` 得到节点间通信的 rank 组。

### 4. 获取代表性 rank 对 -- `get_representative_pairs`

```python
def get_representative_pairs(self, fix_dims, vary_dims, num_pairs=1) -> List[Tuple[int, int]]:
    ranks = self.ranks_in_slice(fix_dims, vary_dims)
    pairs = []
    for i in range(1, min(num_pairs + 1, len(ranks))):
        pairs.append((ranks[0], ranks[i]))
    return pairs
```

从切片中选取对角线 rank 对用于 profiling。例如 `ranks=[0,1,2,3]` 时返回 `[(0,1)]` 或 `[(0,1),(0,2)]`，避免全排列开销。

### 5. 维度一致性校验 -- `validate_dimension_scope`

```python
def validate_dimension_scope(self, fix_dims, vary_dims) -> None:
    overlap = set(fix_dims.keys()) & set(vary_dims)
    if overlap:
        raise ValueError(f"fix_dims and vary_dims cannot overlap, found: {overlap}")
    # ...检查维度名是否在 mesh 中存在
```

确保 fix_dims 和 vary_dims 不重叠，且所有维度名都合法。

## 核心类/函数表

| 名称 | 类型 | 说明 |
|------|------|------|
| `DeviceMesh` | dataclass | 多维设备网格，管理坐标-rank 映射和切片查询 |
| `rank_at()` | 方法 | 坐标 -> rank |
| `coords_of()` | 方法 | rank -> 坐标 |
| `ranks_in_slice()` | 方法 | 按 fix/vary 维度筛选 rank 子集 |
| `get_representative_pairs()` | 方法 | 从切片中选取代表性 rank 对 |
| `validate_dimension_scope()` | 方法 | 校验 fix_dims 与 vary_dims 的一致性 |

## 与其他模块的关系

- **被 `profiler.py` 使用**：`LayerConfig.get_rank_pairs()` 和 `HierarchyConfig.get_device_mesh()` 依赖 DeviceMesh 进行分层 profiling 时的 rank 选取
- **被 `model_loader.py` 间接关联**：加载的分层模型参数通过 DeviceMesh 确定各层对应的 rank 范围
- **与 `topology.py` 互补**：DeviceMesh 描述逻辑拓扑（坐标 -> rank），Topology 描述物理拓扑（rank -> 资源路径）

## 小结

`device_mesh.py` 提供了直观的多维设备网格抽象，通过 `ranks_in_slice` 和 `get_representative_pairs` 方法，使得层次化网络 profiling 只需指定"固定哪些维度、变化哪些维度"，无需手动枚举 rank 列表。这是分层拓扑自动化 profiling 的基础设施。
