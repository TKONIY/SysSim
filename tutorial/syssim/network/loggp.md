# `loggp.py` -- LogGP 性能模型参数

## 文件概述

`loggp.py` 定义了 LogGP 性能模型的参数数据类 `LogGPParams`，这是整个网络仿真模块的性能建模基础。LogGP 模型用四个参数刻画点对点通信的时间开销，广泛应用于高性能计算领域。

本文件支持两种模型变体：
1. **简化 3 参数模型**（$g=0$）：向后兼容默认值
2. **Hoefler 4 参数模型**（$g>0$）：通过 PRTT 方法实测得到，更精确

## 关键代码解析

### 1. LogGPParams 数据类

```python
@dataclass(frozen=True)
class LogGPParams:
    L: float  # latency (seconds)
    o: float  # per-message CPU overhead (seconds)
    G: float  # gap per byte (seconds/byte)
    g: float = 0.0  # base gap (seconds)
```

使用 `frozen=True` 使实例不可变，确保参数在仿真过程中不会被意外修改。

**四个参数的物理含义：**

| 参数 | 含义 | 单位 | 典型值（NVLink） |
|------|------|------|-----------------|
| $L$ | 网络延迟（发送最小消息的时间） | 秒 | $\sim 1 \mu s$ |
| $o$ | 每次 send/recv 的 CPU 开销 | 秒 | $\sim 5 \mu s$ |
| $G$ | 每字节间隔 $= 1/\text{bandwidth}$ | 秒/字节 | $\sim 4 \times 10^{-11}$（25 GB/s） |
| $g$ | 连续发送间的基础间隔 | 秒 | $\sim 2 \mu s$ |

### 2. alpha 属性 -- 固定流水线开销

```python
@property
def alpha(self) -> float:
    return self.L + 2 * self.o + self.g
```

$\alpha$ 表示每条消息的固定开销，与消息大小无关：

$$\alpha = L + 2o + g$$

乘以 2 的原因是每条消息的 CPU 开销包含发送端和接收端各一次 $o$。

- 简化模型（$g=0$）：$\alpha = L + 2o$
- Hoefler 模型（$g>0$）：$\alpha = L + 2o + g$

### 3. message_time -- 消息传输时间

```python
def message_time(self, size_bytes: float) -> float:
    if size_bytes <= 0:
        return 0.0
    return self.alpha + (size_bytes - 1) * self.G
```

单条消息的传输时间公式：

$$T(m) = \alpha + (m - 1) \cdot G$$

其中 $m$ 为消息大小（字节）。$(m-1)$ 而非 $m$ 是因为**流水线效应**：第一个字节承担完整的 $\alpha$ 开销，之后的 $m-1$ 个字节只需各承担 $G$ 的传输时间。

**计算示例**：

```python
# 25 GB/s NVLink, 1μs latency, 5μs overhead
loggp = LogGPParams(L=1e-6, o=5e-6, G=1/(25e9))

# 发送 1 MB 消息
T = loggp.message_time(1e6)
# = (1e-6 + 2*5e-6) + (1e6 - 1) * 4e-11
# = 11e-6 + 39.99996e-6
# ≈ 51 μs
```

### 4. 两种模型变体对比

**简化 3 参数模型**（默认 $g=0$）：

$$T = L + 2o + (m-1) \cdot G$$

适用于 $g$ 可忽略或已被其他参数吸收的场景。

**Hoefler 4 参数模型**（$g>0$）：

$$T = L + 2o + g + (m-1) \cdot G$$

区分了基础间隔 $g$（协议开销）和每字节间隔 $G$（带宽限制），对 eager/rendezvous 协议切换的建模更准确。

## 核心类/函数表

| 名称 | 类型 | 说明 |
|------|------|------|
| `LogGPParams` | frozen dataclass | LogGP 模型的 4 个参数容器 |
| `alpha` | 属性 | 固定流水线开销 $L + 2o + g$ |
| `message_time()` | 方法 | 计算给定大小消息的传输时间 |

## 与其他模块的关系

- **被 `simulator.py` 使用**：仿真引擎在计算 Op 完成时间时使用 `alpha` 属性
- **被 `topology.py` 存储**：`HierarchicalTopology` 为每层（NVLink/InfiniBand）分别持有不同的 `LogGPParams`
- **由 `profiler.py` 生产**：profiler 通过 PRTT 测量自动校准 L, o, g, G 参数
- **由 `model_loader.py` 加载**：从 JSON 文件反序列化为 `LogGPParams` 实例
- **被 `validation.py` 消费**：解析验证公式中直接使用 `alpha` 和 `G`
- **被 `collectives.py` 间接关联**：集合通信的时间复杂度公式均以 $\alpha$ 和 $G$ 表达

## 小结

`loggp.py` 是网络模块的性能建模基石。它用 4 个参数（L, o, g, G）简洁地刻画了点对点通信的时间特性，支持简化模型和 Hoefler 扩展模型两种变体。`frozen=True` 设计保证了参数的不可变性，适合在仿真过程中安全共享。
