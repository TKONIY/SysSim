# `compute_cost_predictor.py` -- 基于 Roofline 模型的算子运行时间预测器

## 文件概述

`compute_cost_predictor.py` 是 `compute` 模块的核心文件，实现了 **多维 Roofline 性能模型** 和 **混合 Roofline + ML 效率校正** 的运行时间预测。它是整个 SysSim 项目中将算子（operator）映射到执行时间的关键入口。

主要功能：
1. 定义算子分类集合（GEMM、ATTN、VIEW、CREATE 等）
2. 计算 Roofline 上界时间（计算受限 vs 内存受限）
3. 通过 ML 效率模型校正 Roofline 预测
4. 提供统一的 `estimate_runtime()` 入口函数

### 单位系统

文件开头明确定义了单位体系，避免"魔法数字"：

| 量纲 | 单位 | 常量 |
|------|------|------|
| 算力 | TFLOP/s (10^12 FLOP/s) | `TERA_TO_UNIT = 1e12` |
| 带宽 | GB/s (10^9 bytes/s) | `GIGA_TO_UNIT = 1e9` |
| 内部时间 | 纳秒 (ns) | `SECONDS_TO_NS = 1e9` |
| 外部时间 | 毫秒 (ms) | `NS_TO_MS = 1e6` |

## 关键代码解析

### 1. 算子分类集合

```python
_GEMM_OPS = OrderedSet([aten.mm, aten.addmm, aten.bmm, aten.matmul, aten.linear])

_ATTN_OPS = frozenset({
    aten._scaled_dot_product_efficient_attention,
    aten._scaled_dot_product_flash_attention,
    aten._scaled_dot_product_flash_attention_for_cpu,
    aten._scaled_dot_product_cudnn_attention,
    aten._flash_attention_forward,
    aten._efficient_attention_forward,
})

_VIEW_OPS = OrderedSet([aten.t, aten.transpose, aten.view, ...])
_CREATE_OPS = OrderedSet([aten.randint, aten.randn, ...])
_IGNORE_OPS = _VIEW_OPS | _CREATE_OPS
```

算子被分为以下几类：
- **GEMM 算子**：矩阵乘法类（使用 Tensor Core 峰值算力）
- **ATTN 算子**：注意力机制类（Flash Attention / Efficient Attention）
- **VIEW 算子**：视图操作，零开销，直接跳过
- **CREATE 算子**：张量创建操作，不参与计时
- **IGNORE 算子**：VIEW + CREATE 的并集

### 2. Roofline 数据结构

```python
@dataclass
class ConstraintTime:
    """单个 Roofline 约束。"""
    work_type: str    # "math" 或 "memory"
    unit_level: str   # "device"
    time_ms: float    # 该约束下的预估时间
    work_amount: float  # 工作量 W_k(x)（FLOPs 或 bytes）
    capacity: float     # 硬件能力 C_{k,l}(h)（FLOP/s 或 bytes/s）

@dataclass
class RooflineResult:
    """多维 Roofline 结果。"""
    t_roofline_ms: float               # 所有约束中的最大值
    constraints: list[ConstraintTime]   # 约束列表
    dominant_constraint: tuple[str, str] # 主导约束（瓶颈）
```

`RooflineResult` 封装了完整的 Roofline 分析结果，包括哪个约束是瓶颈（compute-bound 还是 memory-bound）。

### 3. 大 Tensor Core 操作检测

```python
LARGE_GEMM_THRESHOLD = 512

def _is_large_tensor_core_op(func_packet, args, op_type: OperatorType) -> bool:
    if op_type == OperatorType.GEMM:
        if func_packet == aten.mm and len(args) >= 2:
            a, b = args[0], args[1]
            m, k = a.shape
            k2, n = b.shape
            return m >= LARGE_GEMM_THRESHOLD and n >= LARGE_GEMM_THRESHOLD and k >= LARGE_GEMM_THRESHOLD
    ...
```

这是一个 **两级峰值选择** 机制：
- **大操作**（所有维度 >= 512）：使用 Tensor Core 峰值（如 GH200 上 1979 TFLOP/s）
- **小操作**（任一维度 < 512）：使用保守峰值（如 535 TFLOP/s），因为 kernel launch 开销约 7 us 会占主导

### 4. 计算受限时间估算

```python
def get_roofline_compute_time(
    func_packet, args, kwargs, out, out_dtypes, hw_info, op_type
) -> float:
    if func_packet in flop_registry:
        dtype = out_dtypes.pop()
        is_large_op = _is_large_tensor_core_op(func_packet, args, op_type)
        peak_tflops = hw_info.get_peak_tflops(op_type, dtype, is_large_op)
        peak_gpu_flops = peak_tflops * TFLOPS_TO_FLOPS  # x 1e12
        flop_count_func = flop_registry[func_packet]
        flop_count = flop_count_func(*args, **kwargs, out_val=out)
        compute_time_ns = (flop_count / peak_gpu_flops) * SECONDS_TO_NS
        return compute_time_ns
    return 0.0
```

核心公式：

$$T_{compute} = \frac{FLOPs}{peak\_FLOP/s} \times 10^9 \text{ (ns)}$$

例如，2048x2048x8192 FP16 GEMM 在 GH200 上：
- FLOPs = \(2 \times 2048 \times 2048 \times 8192 = 68.7 \times 10^9\)
- Peak = 1979 TFLOP/s = \(1979 \times 10^{12}\) FLOP/s
- \(T_{compute} = 68.7 \times 10^9 / (1979 \times 10^{12}) \times 10^9 = 34.7 \mu s\)

### 5. 内存受限时间估算

```python
def get_roofline_transfer_time(flat_args_kwargs, flat_outs, hw_info) -> float:
    read_bytes = sum(get_num_bytes(t) for t in flat_args_kwargs if isinstance(t, torch.Tensor))
    write_bytes = sum(get_num_bytes(t) for t in flat_outs if isinstance(t, torch.Tensor))
    counted_bytes = read_bytes + write_bytes
    transfer_time_ns = counted_bytes / hw_info.get_peak_memory_bandwidth_gbps()
    return transfer_time_ns
```

巧妙的单位简化：`bytes / (GB/s 的数值)` 直接得到纳秒：

$$T_{memory} = \frac{bytes}{bw_{GB/s}} \text{ (ns)}$$

因为 \(\frac{bytes}{GB/s} = \frac{bytes \times s}{10^9 \times bytes} = \frac{s}{10^9} = ns\)

### 6. Roofline 估算主函数

```python
def roofline_estimate(func_packet, args, kwargs, out, hw_info, op_type,
                      execution_mode=None, cache_seq_len=0) -> RooflineResult:
    ...
    max_constraint = max(constraints, key=lambda c: c.time_ms)
    t_roofline_ms = max_constraint.time_ms
    ...
```

Roofline 模型的核心思想：

$$T_{roofline} = \max(T_{compute}, T_{memory})$$

函数还支持 **Decode 模式**下的 KV Cache 感知估算：在解码阶段，K/V 的序列长度被替换为实际的 cache 长度，而不是 trace 时的形状。

### 7. 效率校正

```python
def efficiency_estimate(func_packet, args, kwargs, out, hw_info, op_type,
                        roofline_result, ...) -> float:
    model_manager = get_backend_manager()
    model = model_manager.get_model(op_type)
    if model is None:
        return 1.0  # 无模型时退化为纯 Roofline
    ...
    eta_hat = model.predict(features)
    return eta_hat
```

效率模型预测 \(\hat{\eta} \in (0, 1]\)，代表硬件实际利用率。无模型时返回 1.0（等价于纯 Roofline）。

### 8. 最终运行时间入口

```python
def estimate_runtime(func_packet, args, kwargs, out, hw_info, op_type, ...) -> float:
    roofline_result = roofline_estimate(...)
    efficiency = efficiency_estimate(...)
    return roofline_result.t_roofline_ms / efficiency
```

最终公式：

$$T_{predicted} = \frac{T_{roofline}}{\hat{\eta}}$$

由于 \(\hat{\eta} \leq 1\)，预测时间总是 >= Roofline 上界，符合物理直觉（实际执行不可能快于理论上界）。

## 核心类/函数表

| 名称 | 类型 | 说明 |
|------|------|------|
| `ConstraintTime` | dataclass | 单个 Roofline 约束（计算或内存） |
| `RooflineResult` | dataclass | Roofline 分析完整结果 |
| `get_num_bytes(t)` | function | 计算张量实际占用字节数（考虑 CUDA 内存对齐） |
| `_is_large_tensor_core_op()` | function | 判断算子是否足够大以充分利用 Tensor Core |
| `get_roofline_compute_time()` | function | 计算受限时间（FLOPs / peak_FLOP/s） |
| `get_roofline_transfer_time()` | function | 内存受限时间（bytes / peak_bandwidth） |
| `_decode_attention_compute_ns()` | function | Decode 模式下的 ATTN 计算时间（KV Cache 感知） |
| `_decode_attention_transfer_ns()` | function | Decode 模式下的 ATTN 传输时间（KV Cache 感知） |
| `roofline_estimate()` | function | 多维 Roofline 估算主函数 |
| `_extract_operator_params()` | function | 提取对数缩放的算子参数（ML 特征） |
| `_extract_hardware_params()` | function | 提取硬件描述符（ML 特征） |
| `efficiency_estimate()` | function | ML 效率预测 |
| `estimate_runtime()` | function | 统一入口：Roofline + 效率校正 |

## 与其他模块的关系

- **flop_counter.py**：通过 `flop_registry` 获取各算子的 FLOP 计数函数
- **efficiency_models.py**：通过 `get_backend_manager()` 获取训练好的 ML 效率模型
- **config.py（上层）**：依赖 `HardwareInfo` 获取硬件参数（peak TFLOP/s、带宽等）
- **operator_graph.py（上层）**：使用 `OperatorType` 枚举区分算子类型
- **compute_cost_profiler.py**：profiler 调用 `roofline_estimate()` 生成训练标签

## 小结

`compute_cost_predictor.py` 是计算成本模块的大脑。它实现了经典的 Roofline 性能模型，并创新性地引入了两级峰值选择（大/小 GEMM）、Decode 模式 KV Cache 感知、以及 ML 效率校正机制，使预测精度从纯 Roofline 的理论上界逼近实际执行时间。所有单位换算都通过命名常量显式完成，避免了维度分析错误。
