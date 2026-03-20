# syssim 核心模块

## 模块架构图

```
                        +------------------+
                        |  __init__.py     |
                        |  (统一导出入口)   |
                        +--------+---------+
                                 |
              +------------------+------------------+
              |                  |                  |
     +--------v--------+ +------v-------+ +--------v--------+
     |    api.py        | | config.py    | | operator_graph.py|
     | (公共追踪API)    | | (硬件配置与  | | (算子图IR数据    |
     |                  | |  模拟器参数)  | |  结构与分析)     |
     +--------+---------+ +------+-------+ +--------+--------+
              |                  |                  |
              +--------+---------+------------------+
                       |
              +--------v--------+
              |   tracer.py     |
              | (PyTorch模型    |
              |  追踪引擎)      |
              +-----------------+
```

## 各文件简介

| 文件 | 职责 |
|------|------|
| `__init__.py` | 包的统一导出入口，将核心模块、Hugging Face 集成、网络模拟器的关键符号汇聚到顶层命名空间 |
| `api.py` | 面向用户的公共 API，提供训练追踪、推理追踪、效率模型配置三个高层函数 |
| `config.py` | 定义执行模式（训练/预填充/解码）、硬件规格（HardwareInfo）、网络参数（NetworkParams）和模拟器配置（SimulatorConfig） |
| `operator_graph.py` | 算子图中间表示（IR），包含算子类型枚举、张量元数据、算子节点、以及支持拓扑排序、关键路径分析和 DOT/JSON 导出的有向无环图 |
| `tracer.py` | 追踪引擎核心，利用 PyTorch 的 TorchDispatchMode + FakeTensorMode 拦截模型执行中的所有算子调用，构建 OperatorGraph |

## 文件间关系

1. **api.py** 是用户的主入口，它依赖 **config.py** 获取硬件配置和执行模式，依赖 **tracer.py** 的 `OperatorGraphTracer` 执行追踪，追踪结果是 **operator_graph.py** 中定义的 `OperatorGraph`。

2. **tracer.py** 是最重的模块，它同时依赖 **config.py**（读取 `ExecutionMode`）和 **operator_graph.py**（构建 `OperatorNode` 并填充 `OperatorGraph`）。追踪过程中还会调用 `compute` 子包进行算子耗时估算。

3. **config.py** 的 `HardwareInfo.get_peak_tflops()` 方法反向引用了 **operator_graph.py** 的 `OperatorType`，形成一个轻量的循环依赖（通过函数内 import 延迟加载解决）。

4. **__init__.py** 聚合所有模块的公共符号，使用户只需 `import syssim` 即可访问全部功能。

## 典型使用流程

```python
import syssim

# 1. 配置硬件参数
hw = syssim.HardwareInfo(
    peak_tflops_mm=1979.0,
    peak_tflops_math=989.0,
    peak_memory_bandwidth_gbps=3350.0,
)
config = syssim.SimulatorConfig(hw_info=hw)

# 2. 追踪模型
graph = syssim.trace_model_for_training(model, example_inputs, config)

# 3. 分析结果
print(graph.summary())          # 文字摘要
print(graph.compute_critical_path())  # 关键路径耗时 (ms)
```
