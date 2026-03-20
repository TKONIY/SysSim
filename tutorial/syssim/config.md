# `config.py` — 硬件配置与模拟器参数

## 文件概述

`config.py` 定义了 SysSim 的所有配置数据结构，是整个项目的"参数中心"。它包含四个核心组件：

1. **ExecutionMode** — 执行模式枚举（训练/预填充/解码）
2. **NetworkParams** — 网络硬件参数（NVLink、InfiniBand、LogGP 模型参数）
3. **HardwareInfo** — GPU 硬件规格（峰值算力、显存带宽）
4. **SimulatorConfig** — 模拟器顶层配置

此外还提供 `get_hardware_info()` 函数用于自动检测当前 GPU 型号并返回预设参数。

## 关键代码解析

### ExecutionMode 枚举

```python
class ExecutionMode(Enum):
    TRAINING = "training"    # Forward + backward, standard shapes
    PREFILL = "prefill"      # Forward only, full sequence length
    DECODE = "decode"        # Forward only, seq_len=1, KV cache read
```

三种模式对应 LLM 的三种典型工作负载：
- `TRAINING`：前向 + 反向传播，标准形状。
- `PREFILL`：推理预填充阶段，处理完整序列。
- `DECODE`：推理解码阶段，每次仅处理 1 个 token，涉及 KV cache 读取。

这个枚举贯穿整个追踪链路，影响追踪器的行为（是否执行反向传播）和耗时估算（不同模式下算子的计算量不同）。

### NetworkParams 数据类

```python
@dataclass
class NetworkParams:
    # Single-node parameters
    nvlink_bandwidth: float = 25e9  # bytes/second per NVLink
    nvlink_count: int = 12          # NVLinks per GPU pair (DGX A100)
    ib_bandwidth: float = 25e9      # bytes/second (200 Gb/s = 25 GB/s)

    # Multi-node parameters
    num_nodes: int = 1
    gpus_per_node: int = 8

    # LogGP parameters - NVLink (intra-node)
    loggp_nvlink_L: float = 1e-6    # latency (seconds)
    loggp_nvlink_o: float = 5e-6    # CPU overhead (seconds)

    # LogGP parameters - InfiniBand (inter-node)
    loggp_ib_L: float = 5e-6        # latency (seconds)
    loggp_ib_o: float = 10e-6       # CPU overhead (seconds)
```

网络参数分为三层：
- **单节点参数**：NVLink 带宽/链路数、IB 带宽。
- **多节点参数**：节点数量、每节点 GPU 数。
- **LogGP 模型参数**：分别为 NVLink（节点内）和 InfiniBand（节点间）定义延迟 `L` 和 CPU 开销 `o`。

默认值基于 DGX A100 配置（12 条 NVLink、8 GPU/节点）。

### HardwareInfo 类

```python
class HardwareInfo:
    def __init__(
        self,
        peak_tflops_mm: float,
        peak_tflops_math: float,
        peak_memory_bandwidth_gbps: float,
        peak_tflops_mm_conservative: float | None = None,
        network: Optional[NetworkParams] = None,
    ):
        self.peak_tflops_mm = peak_tflops_mm
        self.peak_tflops_math = peak_tflops_math
        self.peak_memory_bandwidth_gbps = peak_memory_bandwidth_gbps
        self.peak_tflops_mm_conservative = (
            peak_tflops_mm_conservative if peak_tflops_mm_conservative is not None else peak_tflops_mm
        )
        self.network = network if network is not None else NetworkParams()
```

**参数含义：**
- `peak_tflops_mm`：矩阵乘法（Tensor Core）峰值算力，单位 TFLOP/s。用于 GEMM 和 ATTN 算子。
- `peak_tflops_math`：向量运算峰值算力，用于非 GEMM/ATTN 的数学运算。
- `peak_memory_bandwidth_gbps`：显存峰值带宽，单位 GB/s，用于 roofline 模型中内存受限场景。
- `peak_tflops_mm_conservative`：保守峰值，用于小矩阵运算（启动开销占主导时），默认等于 `peak_tflops_mm`。

**关键方法 get_peak_tflops：**

```python
def get_peak_tflops(self, op_type, dtype: torch.dtype, is_large_op: bool = False) -> float:
    from .operator_graph import OperatorType
    if op_type in (OperatorType.GEMM, OperatorType.ATTN):
        return self.peak_tflops_mm if is_large_op else self.peak_tflops_mm_conservative
    else:
        return self.peak_tflops_math
```

这个方法体现了 roofline 模型的核心思想：根据算子类型和规模选择合适的峰值算力。
- GEMM/ATTN 大算子：使用 Tensor Core 峰值。
- GEMM/ATTN 小算子：使用保守峰值（考虑 kernel 启动开销）。
- 其他算子：使用向量运算峰值。

注意此处使用函数内 `import` 来避免与 `operator_graph.py` 的循环依赖。

### get_hardware_info 自动检测函数

```python
def get_hardware_info() -> tuple[HardwareInfo, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot auto-detect hardware.")

    device_name = torch.cuda.get_device_name(0).lower()

    hw_database = [
        ("gh200", "gh200", 989.0, 989.0, 3350.0),
        ("h100", "h100", 1979.0, 989.0, 3350.0),
        ("a100", "a100", 312.0, 156.0, 1935.0),
        ("v100", "v100", 125.0, 62.5, 900.0),
        # ... 更多硬件
    ]

    for pattern, hw_name, peak_mm, peak_math, peak_bw in hw_database:
        if pattern in device_name:
            hw_info = HardwareInfo(
                peak_tflops_mm=peak_mm,
                peak_tflops_math=peak_math,
                peak_memory_bandwidth_gbps=peak_bw,
            )
            return hw_info, hw_name
```

通过 `torch.cuda.get_device_name()` 获取 GPU 名称，然后在内置硬件数据库中匹配。支持的硬件包括：

| GPU | 矩阵峰值 (TFLOP/s) | 数学峰值 (TFLOP/s) | 带宽 (GB/s) |
|-----|---------------------|---------------------|-------------|
| GH200 | 989.0 | 989.0 | 3350.0 |
| H100 | 1979.0 | 989.0 | 3350.0 |
| A100 | 312.0 | 156.0 | 1935.0 |
| V100 | 125.0 | 62.5 | 900.0 |
| RTX 4090 | 330.0 | 165.0 | 1008.0 |
| MI250 | 362.0 | 181.0 | 1600.0 |
| MI300 | 653.0 | 326.5 | 5200.0 |

### SimulatorConfig 数据类

```python
@dataclass
class SimulatorConfig:
    hw_info: HardwareInfo
    cache_seq_len: int = 0
```

顶层配置非常简洁，只包含硬件信息和 KV cache 序列长度（仅在 decode 模式下使用）。

## 核心类/函数表

| 名称 | 类型 | 用途 |
|------|------|------|
| `ExecutionMode` | Enum | 执行模式：TRAINING / PREFILL / DECODE |
| `NetworkParams` | dataclass | 网络硬件参数（NVLink、IB、LogGP） |
| `HardwareInfo` | class | GPU 硬件规格，支持 roofline 峰值查询 |
| `HardwareInfo.get_peak_tflops` | method | 根据算子类型和规模返回对应的峰值算力 |
| `HardwareInfo.get_peak_memory_bandwidth_gbps` | method | 返回显存峰值带宽 |
| `get_hardware_info` | function | 自动检测 GPU 型号并返回预设参数 |
| `SimulatorConfig` | dataclass | 模拟器顶层配置容器 |

## 与其他模块的关系

- **api.py** 接收 `SimulatorConfig` 作为参数，从中提取 `hw_info` 和 `cache_seq_len` 传递给追踪器。
- **tracer.py** 使用 `ExecutionMode` 决定是否执行反向传播，使用 `HardwareInfo` 进行耗时估算。
- **operator_graph.py** 的 `OperatorType` 被 `HardwareInfo.get_peak_tflops()` 引用，用于区分算子类型选择不同峰值。
- **network 子模块** 使用 `NetworkParams` 进行集合通信性能建模。

## 小结

`config.py` 是 SysSim 的参数核心，通过清晰的数据类层次（`SimulatorConfig` > `HardwareInfo` > `NetworkParams`）组织所有配置。它的设计兼顾了易用性（`get_hardware_info` 自动检测）和灵活性（用户可手动指定任意硬件参数），同时内置的硬件数据库涵盖了主流 NVIDIA 和 AMD GPU。
