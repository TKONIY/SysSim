# `__init__.py` -- compute 模块包初始化

## 文件概述

`__init__.py` 是 `syssim.compute` 包的初始化文件，当前为空文件（仅包含一个空行）。它的存在使 `syssim/compute/` 目录被 Python 识别为一个合法的包（package），从而允许其他模块通过如下方式导入：

```python
from syssim.compute.compute_cost_predictor import estimate_runtime
from syssim.compute.flop_counter import flop_registry
from syssim.compute.efficiency_models import get_backend_manager
```

## 核心类/函数表

| 名称 | 说明 |
|------|------|
| （无） | 当前文件为空，不导出任何符号 |

## 与其他模块的关系

- 作为包入口，使 `compute_cost_predictor`、`flop_counter`、`efficiency_models`、`compute_cost_profiler` 等子模块可被外部引用。
- 未来如果需要统一对外暴露 API，可以在此文件中添加 `__all__` 列表或重导出关键函数。

## 小结

`__init__.py` 目前仅起到 Python 包标识的作用，不包含任何逻辑。所有功能实现分布在同级的各个子模块中。
