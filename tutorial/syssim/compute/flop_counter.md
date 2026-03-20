# `flop_counter.py` -- FLOP 计数注册表与计数器

## 文件概述

`flop_counter.py` 提供了一套 **PyTorch 算子级别的浮点运算量（FLOP）计数系统**。它是 Roofline 模型的基础设施之一 -- 要计算 \(T_{compute} = FLOPs / peak\_FLOP/s\)，首先需要知道每个算子的 FLOPs 是多少。

该文件包含两大部分：
1. **FLOP 注册表**（`flop_registry`）：一个字典，将 PyTorch ATen 算子映射到其 FLOP 计算公式
2. **FlopCounterMode**：一个 `TorchDispatchMode` 上下文管理器，可以在模型前向/反向传播时自动统计所有算子的 FLOPs

支持的算子族包括：矩阵乘法（mm, bmm, addmm）、卷积（convolution 正向+反向）、缩放点积注意力（SDPA: Flash Attention, Efficient Attention）等。

## 关键代码解析

### 1. FLOP 注册表与装饰器

```python
flop_registry: dict[Any, Any] = {}

def register_flop_formula(targets, get_raw=False):
    def register_fun(flop_formula):
        if not get_raw:
            flop_formula = shape_wrapper(flop_formula)

        def register(target):
            flop_registry[target] = flop_formula

        torch.utils._pytree.tree_map_(register, targets)
        return flop_formula
    return register_fun
```

注册机制的设计要点：
- `flop_registry` 是一个全局字典：`{aten_op -> flop_count_function}`
- `@register_flop_formula` 装饰器支持同时注册多个算子（传入列表）
- `shape_wrapper` 自动将张量参数转换为形状（shape），使 FLOP 公式只需处理维度信息
- `get_raw=True` 时跳过 shape_wrapper，直接接收原始张量（用于需要访问张量属性的复杂算子）

### 2. 矩阵乘法 FLOP 公式

```python
@register_flop_formula(aten.mm)
def mm_flop(a_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
    m, k = a_shape
    k2, n = b_shape
    return m * n * 2 * k
```

标准矩阵乘法 \(C = A \times B\)，其中 \(A \in \mathbb{R}^{M \times K}\)，\(B \in \mathbb{R}^{K \times N}\)：

$$FLOPs_{mm} = 2 \times M \times N \times K$$

每个输出元素需要 K 次乘法和 K-1 次加法，近似为 \(2K\) 次浮点操作。

```python
@register_flop_formula(aten.bmm)
def bmm_flop(a_shape, b_shape, out_shape=None, **kwargs) -> int:
    b, m, k = a_shape
    b2, k2, n = b_shape
    return b * m * n * 2 * k
```

批量矩阵乘法额外乘以 batch 维度 B：

$$FLOPs_{bmm} = B \times 2 \times M \times N \times K$$

### 3. 缩放点积注意力 FLOP 公式

```python
def sdpa_flop_count(query_shape, key_shape, value_shape):
    b, h, s_q, d_q = query_shape
    _b2, _h2, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    total_flops = 0
    # Q @ K^T -> scores: [b*h, s_q, s_k]
    total_flops += bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))
    # scores @ V -> output: [b*h, s_q, d_v]
    total_flops += bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_v))
    return total_flops
```

自注意力分为两步矩阵乘法：

1. **Score 计算**：\(S = Q \times K^T\)，FLOPs = \(2 \times B \times H \times S_q \times S_k \times D_q\)
2. **Value 加权**：\(O = S \times V\)，FLOPs = \(2 \times B \times H \times S_q \times S_k \times D_v\)

总计：

$$FLOPs_{SDPA} = 2BH \cdot S_q \cdot D_q \cdot S_k + 2BH \cdot S_q \cdot S_k \cdot D_v$$

注意：此公式支持 **GQA（Grouped Query Attention）**，因为 Q 和 K/V 的 head 数可以不同。FLOPs 按 Q 的 head 数 H 计算（K/V 会被广播）。

### 4. SDPA 反向传播 FLOP 公式

```python
def sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape):
    # Step 1: 重计算 scores
    total_flops += bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))
    # Step 2: gradOut @ V^T -> gradScores
    total_flops += bmm_flop((b * h, s_q, d_v), (b * h, d_v, s_k))
    #           scores^T @ gradOut -> gradV
    total_flops += bmm_flop((b * h, s_k, s_q), (b * h, s_q, d_v))
    # Step 3: gradScores @ K -> gradQ
    total_flops += bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_q))
    #           Q^T @ gradScores -> gradK
    total_flops += bmm_flop((b * h, d_q, s_q), (b * h, s_q, s_k))
    return total_flops
```

反向传播需要 5 次矩阵乘法（1 次重计算 + 4 次梯度计算），因此 FLOPs 约为正向的 2.5 倍。

### 5. 卷积 FLOP 公式

```python
def conv_flop_count(x_shape, w_shape, out_shape, transposed=False):
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    c_out, c_in, *filter_size = w_shape
    flop = prod(conv_shape) * prod(filter_size) * batch_size * c_out * c_in * 2
    return flop
```

卷积 FLOPs：

$$FLOPs_{conv} = 2 \times B \times C_{out} \times C_{in} \times \prod(spatial) \times \prod(filter)$$

### 6. NestedTensor / 变长序列支持

```python
def _unpack_flash_attention_nested_shapes(*, query, key, value, cum_seq_q, cum_seq_k, ...):
    if cum_seq_q is not None:
        # NestedTensor: 输入形状为 (sum(seq_lens), heads, dim)
        # 转换为 (1, heads, seq_len_i, dim) 逐 batch 计算
        seq_q_lengths = _offsets_to_lengths(cum_seq_q, max_q)
        for seq_q_len, seq_k_len in zip(seq_q_lengths, seq_k_lengths):
            yield (1, h_q, seq_q_len, d_q), (1, h_k, seq_k_len, d_k), ...
```

Flash Attention 和 Efficient Attention 支持 NestedTensor（变长序列），此函数将偏移量格式 (`cum_seq_q`) 解包为逐 batch 的形状，分别计算每个 batch 的 FLOPs 后求和。

### 7. FlopCounterMode 上下文管理器

```python
class FlopCounterMode:
    def __init__(self, mods=None, depth=2, display=True, custom_mapping=None):
        self.flop_counts: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.flop_registry = {**flop_registry, **custom_mapping}
        self.mod_tracker = ModuleTracker()

    def __enter__(self):
        self.mod_tracker.__enter__()
        self.mode = _FlopCounterMode(self)
        self.mode.__enter__()
        return self

    def _count_flops(self, func_packet, out, args, kwargs):
        if func_packet in self.flop_registry:
            flop_count = self.flop_registry[func_packet](*args, **kwargs, out_val=out)
            for par in set(self.mod_tracker.parents):
                self.flop_counts[par][func_packet] += flop_count
```

使用方式：

```python
mod = MyModel()
with FlopCounterMode(mod) as flop_counter:
    mod(input).sum().backward()
print(flop_counter.get_table())
```

`FlopCounterMode` 通过 `TorchDispatchMode` 拦截所有算子调用，查表计算 FLOPs，并按模块层级聚合统计。`ModuleTracker` 追踪当前在哪个 `nn.Module` 内部执行，从而支持层级化的 FLOP 报表。

### 8. 高阶算子处理

```python
class _FlopCounterMode(TorchDispatchMode):
    def _handle_higher_order_ops(self, func, types, args, kwargs):
        if func is torch.ops.higher_order.cond:
            # cond 算子：取 true/false 分支的 FLOPs 上界
            true_out, true_flop_counts = self._execute_with_isolated_flop_counting(true_branch, operands)
            false_out, false_flop_counts = self._execute_with_isolated_flop_counting(false_branch, operands)
            # 对每个算子取 max(true_flops, false_flops)
            ...
```

对于条件分支（`torch.cond`），采取保守策略：分别统计两个分支的 FLOPs，取上界。对于 Triton kernel，通过特殊的 JIT 注册表查找 FLOP 公式。

## 核心类/函数表

| 名称 | 类型 | 说明 |
|------|------|------|
| `flop_registry` | dict | 全局 FLOP 注册表：`{aten_op -> flop_func}` |
| `register_flop_formula()` | decorator | 将 FLOP 公式注册到指定算子 |
| `shape_wrapper()` | function | 将张量参数自动转换为 shape |
| `mm_flop()` | function | `aten.mm` 的 FLOP 公式：\(2MNK\) |
| `bmm_flop()` | function | `aten.bmm` 的 FLOP 公式：\(2BMNK\) |
| `addmm_flop()` | function | `aten.addmm` 的 FLOP 公式（委托给 mm_flop） |
| `sdpa_flop_count()` | function | SDPA 前向 FLOP：2 次 bmm |
| `sdpa_backward_flop_count()` | function | SDPA 反向 FLOP：5 次 bmm |
| `conv_flop_count()` | function | 卷积 FLOP 公式 |
| `FlopCounterMode` | class | FLOP 计数上下文管理器 |
| `_FlopCounterMode` | class | 内部 TorchDispatchMode 实现 |

## 与其他模块的关系

- **compute_cost_predictor.py**：直接引用 `flop_registry` 和 `sdpa_flop_count`，在 `get_roofline_compute_time()` 中查表获取 FLOP 计数函数
- **compute_cost_profiler.py**：间接使用（通过 `roofline_estimate()` 调用链）
- **PyTorch 内部**：基于 `TorchDispatchMode` 和 `ModuleTracker`，深度集成 PyTorch dispatch 机制

## 小结

`flop_counter.py` 是整个 Roofline 模型的计算基础。它通过注册表模式为每种 PyTorch ATen 算子提供精确的 FLOP 计数公式，覆盖了矩阵乘法、卷积、注意力机制（含前向和反向）等核心操作。`FlopCounterMode` 提供了开箱即用的模型级 FLOP 统计能力，而 `flop_registry` 则被 `compute_cost_predictor.py` 直接用于逐算子的 Roofline 时间估算。
