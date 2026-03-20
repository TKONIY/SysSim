# `protocol_detector.py` -- 通信协议变化检测

## 文件概述

`protocol_detector.py` 实现了 Hoefler et al. (2009) 论文中的 **lookahead 协议检测算法**，用于从 PRTT 测量数据中自动识别通信协议的切换点（如从 eager 协议切换到 rendezvous 协议）。

核心思想：当协议发生变化时，消息大小与间隔时间 $G_{\text{all}}(s)$ 之间的线性关系会出现偏差。通过最小二乘拟合的误差变化来检测这种偏差。

## 关键代码解析

### 1. 数据类

```python
@dataclass
class PRTTMeasurement:
    size: int        # 消息大小（字节）
    prtt_1_0: float  # PRTT(1, 0, size) -- 单次往返
    prtt_n_0: float  # PRTT(n, 0, size) -- n 次无延迟
    prtt_n_dG: float # PRTT(n, dG, size) -- n 次带延迟

@dataclass
class ProtocolRange:
    start_idx: int    # 起始测量点索引
    end_idx: int      # 结束测量点索引（含）
    sizes: List[int]  # 消息大小列表
    g: float          # 拟合的基础间隔
    G: float          # 拟合的每字节间隔
    fit_error: float  # 拟合均方误差
```

### 2. 计算 $G_{\text{all}}$ -- 聚合间隔

```python
def compute_gall(measurements: List[PRTTMeasurement], n: int = 10) -> List[float]:
    gall = []
    for m in measurements:
        g_val = (m.prtt_n_0 - m.prtt_1_0) / (n - 1)
        gall.append(g_val)
    return gall
```

$G_{\text{all}}(s)$ 的物理含义是在给定消息大小 $s$ 下的**聚合间隔**：

$$G_{\text{all}}(s) = \frac{\text{PRTT}(n, 0, s) - \text{PRTT}(1, 0, s)}{n - 1}$$

在同一协议内，$G_{\text{all}}(s)$ 与 $s$ 呈线性关系：

$$G_{\text{all}}(s) = g + (s-1) \cdot G$$

### 3. 最小二乘拟合

```python
def least_squares_fit(sizes: List[int], gall: List[float]) -> Tuple[float, float, float]:
    sizes_arr = np.array(sizes, dtype=np.float64)
    gall_arr = np.array(gall, dtype=np.float64)

    # 设计矩阵: [1, s-1]
    X = np.column_stack([np.ones(len(sizes)), sizes_arr - 1])
    y = gall_arr

    # 求解: [g, G] = argmin ||X*[g, G]^T - y||^2
    params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    g = params[0]
    G = params[1]

    predictions = X @ params
    mse = np.mean((y - predictions) ** 2)

    return g, G, mse
```

对模型 $G_{\text{all}}(s) = g \cdot 1 + G \cdot (s-1)$ 进行最小二乘拟合：

$$\min_{g, G} \sum_i \left( G_{\text{all}}(s_i) - g - (s_i - 1) \cdot G \right)^2$$

返回拟合的 $g$、$G$ 以及均方误差 MSE。

### 4. Lookahead 协议检测算法

```python
def detect_protocol_changes(measurements, n=10, lookahead=3, pfact=2.0) -> List[ProtocolRange]:
    gall = compute_gall(measurements, n)
    ranges = []
    last_change = 0

    i = 0
    while i < len(measurements):
        sizes_current = [m.size for m in measurements[last_change:i+1]]
        gall_current = gall[last_change:i+1]

        if len(sizes_current) < 2:
            i += 1
            continue

        g_current, G_current, mse_current = least_squares_fit(sizes_current, gall_current)

        # 检查后续 lookahead 个点是否都使拟合误差恶化
        protocol_changed = False
        if i + lookahead < len(measurements):
            all_worse = True
            for offset in range(1, lookahead + 1):
                sizes_next = [m.size for m in measurements[last_change:i+1+offset]]
                gall_next = gall[last_change:i+1+offset]
                _, _, mse_next = least_squares_fit(sizes_next, gall_next)
                if mse_next <= pfact * mse_current:
                    all_worse = False
                    break
            if all_worse:
                protocol_changed = True

        if protocol_changed:
            ranges.append(ProtocolRange(
                start_idx=last_change, end_idx=i,
                sizes=[m.size for m in measurements[last_change:i+1]],
                g=g_current, G=G_current, fit_error=mse_current
            ))
            last_change = i + 1

        i += 1

    # 保存最后一段协议
    # ...
    return ranges
```

**算法流程**：

1. 从第一个测量点开始，维护当前协议的起始位置 `last_change`
2. 将 `[last_change, i]` 范围内的数据拟合为 $G_{\text{all}}(s) = g + (s-1)G$，得到 MSE
3. **Lookahead 检测**：检查接下来的 `lookahead` 个点，如果**所有**点加入后的 MSE 都超过当前 MSE 的 `pfact` 倍，则判定发生了协议切换
4. 记录当前协议范围，从切换点重新开始

**参数说明**：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `lookahead` | 前瞻窗口大小 | 3 |
| `pfact` | 敏感度因子（误差放大倍数阈值） | 2.0 |

`lookahead` 越大，抗噪能力越强但响应越慢；`pfact` 越大，对协议变化越不敏感。

**检测原理示意**：

```
MSE
 ^
 |        *  *  *  (rendezvous: 新协议，MSE 剧增)
 |   * *
 | * *              (eager: 当前协议，MSE 稳定)
 +-----|-------> message size
    协议切换点
```

## 核心类/函数表

| 名称 | 类型 | 说明 |
|------|------|------|
| `PRTTMeasurement` | dataclass | 单个消息大小的 PRTT 测量数据 |
| `ProtocolRange` | dataclass | 一段使用相同协议的消息大小范围 |
| `compute_gall()` | 函数 | 从 PRTT 测量计算聚合间隔 $G_{\text{all}}$ |
| `least_squares_fit()` | 函数 | 对 $G_{\text{all}}(s) = g + (s-1)G$ 进行最小二乘拟合 |
| `detect_protocol_changes()` | 函数 | Lookahead 算法检测协议变化点 |

## 与其他模块的关系

- **被 `profiler.py` 调用**：profiler 在测量完成后调用 `detect_protocol_changes()` 检测协议边界，再用 `compute_gall()` 的结果进行参数提取
- **间接影响 `model_loader.py`**：检测到的协议范围决定了保存的 JSON 文件中 `protocols` 数组的结构

## 小结

`protocol_detector.py` 通过最小二乘拟合和 lookahead 前瞻算法，自动从 PRTT 测量数据中识别通信协议的切换点。这使得 profiler 能够为不同协议段（如 eager 和 rendezvous）分别提取准确的 LogGP 参数，提升性能模型的精度。
