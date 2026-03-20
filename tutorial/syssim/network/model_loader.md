# `model_loader.py` -- LogGP 模型加载器

## 文件概述

`model_loader.py` 提供从 JSON 文件加载 LogGP 参数的工具函数集合。它支持两种模型格式：

1. **单层模型**：一个拓扑类型，可含多个协议（eager/rendezvous），对应 `load_loggp_params()` 和 `load_all_protocols()`
2. **分层模型**：多个网络层（如 NVLink + InfiniBand），对应 `load_hierarchical_loggp()`

加载方式支持 **拓扑名称自动解析**（如 `"nvlink"` -> `data/network_models/nvlink_loggp.json`）和 **显式文件路径** 两种。

## 关键代码解析

### 1. 单层模型加载 -- `load_loggp_params`

```python
def load_loggp_params(topology: Union[str, Path]) -> LogGPParams:
    # 路径解析
    if isinstance(topology, str) and not topology.endswith(".json"):
        project_root = Path(__file__).parent.parent.parent
        path = project_root / "data" / "network_models" / f"{topology}_loggp.json"
    else:
        path = Path(topology)

    # 加载 JSON 并提取 primary 协议
    with open(path) as f:
        data = json.load(f)

    primary = data["primary"]
    return LogGPParams(
        L=primary["L"], o=primary["o"], G=primary["G"],
        g=primary.get("g", 0.0)  # Backward compatibility
    )
```

**路径解析逻辑**：如果传入的是不带 `.json` 后缀的字符串（如 `"nvlink"`），则自动在 `data/network_models/` 目录下查找 `{name}_loggp.json`。

**JSON 文件结构**（单层）：

```json
{
  "primary": {"L": 1.5e-6, "o": 7e-6, "G": 4e-11, "g": 2e-6},
  "protocols": [
    {"size_range": [1, 12288], "L": ..., "o": ..., "G": ..., "g": ...},
    {"size_range": [12289, 65536], "L": ..., "o": ..., "G": ..., "g": ...}
  ]
}
```

### 2. 多协议加载 -- `load_all_protocols` 和 `get_protocol_for_size`

```python
def load_all_protocols(topology) -> List[Tuple[Tuple[int, int], LogGPParams]]:
    # ...加载 JSON...
    result = []
    for protocol in data["protocols"]:
        size_range = protocol["size_range"]
        min_size, max_size = size_range
        params = LogGPParams(L=..., o=..., G=..., g=...)
        result.append(((min_size, max_size), params))
    return result

def get_protocol_for_size(protocols, size: int) -> LogGPParams:
    for (min_size, max_size), params in protocols:
        if min_size <= size <= max_size:
            return params
    raise ValueError(f"No protocol found for size {size} bytes.")
```

**使用场景**：消息大小不同时可能使用不同的通信协议（如小消息用 eager 协议，大消息用 rendezvous 协议），每种协议有不同的 LogGP 参数。

```python
protocols = load_all_protocols("nvlink")
params_small = get_protocol_for_size(protocols, 1024)   # eager 协议
params_large = get_protocol_for_size(protocols, 32768)   # rendezvous 协议
```

### 3. 分层模型判断与加载

```python
def is_hierarchical_model(topology) -> bool:
    # ...加载 JSON...
    return "layers" in data and isinstance(data["layers"], dict)

def load_hierarchical_loggp(topology) -> Dict[str, LogGPParams]:
    # ...加载 JSON...
    result = {}
    for layer_name, layer_data in data["layers"].items():
        primary = layer_data["primary"]
        result[layer_name] = LogGPParams(L=..., o=..., G=..., g=...)
    return result
```

**分层 JSON 文件结构**：

```json
{
  "layers": {
    "intra_node_nvlink": {
      "primary": {"L": 1e-6, "o": 5e-6, "G": 1.59e-11}
    },
    "inter_node_ib": {
      "primary": {"L": 5e-6, "o": 10e-6, "G": 7.78e-10}
    }
  }
}
```

### 4. 层参数路由 -- `get_layer_params`

```python
def get_layer_params(
    hierarchical_params: Dict[str, LogGPParams],
    src_rank: int, dst_rank: int,
    topology_map: Dict[str, Callable[[int, int], bool]]
) -> LogGPParams:
    for layer_name, predicate in topology_map.items():
        if predicate(src_rank, dst_rank):
            return hierarchical_params[layer_name]
    raise ValueError(f"No layer found for ranks {src_rank} -> {dst_rank}.")
```

通过 **谓词函数** 判断两个 rank 之间的通信属于哪个网络层：

```python
params = load_hierarchical_loggp("perlmutter")
topology_map = {
    "intra_node_nvlink": lambda s, d: s // 4 == d // 4,  # 同节点
    "inter_node_ib": lambda s, d: s // 4 != d // 4       # 跨节点
}
loggp = get_layer_params(params, 0, 1, topology_map)  # NVLink 参数
loggp = get_layer_params(params, 0, 4, topology_map)  # InfiniBand 参数
```

## 核心类/函数表

| 函数 | 说明 |
|------|------|
| `load_loggp_params(topology)` | 加载单层模型的主协议 LogGP 参数 |
| `load_all_protocols(topology)` | 加载所有协议范围及参数 |
| `get_protocol_for_size(protocols, size)` | 根据消息大小选择合适的协议参数 |
| `is_hierarchical_model(topology)` | 判断模型是否为分层格式 |
| `load_hierarchical_loggp(topology)` | 加载分层模型（每层一套参数） |
| `get_layer_params(params, src, dst, map)` | 根据 rank 对选择对应层的 LogGP 参数 |

## 与其他模块的关系

- **消费 `loggp.py`**：所有加载函数返回 `LogGPParams` 实例
- **被 `topology.py` 调用**：`HierarchicalTopology.from_profiled_model()` 使用 `load_hierarchical_loggp()` 从 JSON 创建拓扑
- **与 `profiler.py` 对接**：profiler 将测量结果保存为 JSON，本模块负责反向加载
- **被 `__init__.py` 导出**：6 个函数全部作为公共 API 导出

## 小结

`model_loader.py` 是 LogGP 参数的序列化/反序列化层，支持单层和分层两种模型格式，提供灵活的路径解析和协议选择机制。它是连接 profiler（生产参数）和 simulator（消费参数）的桥梁。
