# `efficiency_models.py` -- ML 效率模型管理与推理

## 文件概述

`efficiency_models.py` 负责 **效率模型的加载、管理和推理**。它定义了统一的模型抽象接口，实现了 MLP 和 XGBoost 两种后端，并通过 `BackendManager` 单例管理所有已训练模型的生命周期。

这个文件是训练阶段（`compute_cost_profiler.py`）和推理阶段（`compute_cost_predictor.py`）之间的桥梁。

## 关键代码解析

### 1. 特征数据结构

```python
@dataclass
class EfficiencyFeatures:
    """ML 效率模型的输入特征。"""
    constraint_times: dict[tuple[str, str], float]   # T_{k,l}：各约束时间
    constraint_ratios: dict[tuple[str, str], float]   # r_{k,l}：约束比率
    dominant_constraint: tuple[str, str]               # 主导约束（瓶颈）
    op_params: dict[str, float]                        # 算子参数（对数缩放）
    hw_params: dict[str, float]                        # 硬件描述符

    def to_array(self, feature_order: list[str]) -> np.ndarray:
        """按照指定顺序转换为一维数组，供 ML 模型输入。"""
        feature_dict = {}
        # 约束时间: T_math_device, T_memory_device
        for key, val in self.constraint_times.items():
            feature_dict[f"T_{key[0]}_{key[1]}"] = val
        # 约束比率: r_math_device, r_memory_device
        for key, val in self.constraint_ratios.items():
            feature_dict[f"r_{key[0]}_{key[1]}"] = val
        # 主导约束 one-hot: dom_math_device 或 dom_memory_device
        feature_dict[f"dom_{self.dominant_constraint[0]}_{self.dominant_constraint[1]}"] = 1.0
        # 算子参数 + 硬件参数
        feature_dict.update(self.op_params)
        feature_dict.update(self.hw_params)
        # 按 feature_order 排列，缺失值填 0.0
        values = [feature_dict.get(name, 0.0) for name in feature_order]
        return np.array(values, dtype=np.float32)
```

`EfficiencyFeatures` 将来自不同来源的特征统一封装：
- **Roofline envelope 特征**：约束时间和比率（来自 `roofline_estimate()`）
- **算子参数**：如 log_M, log_N, log_K（来自 `_extract_operator_params()`）
- **硬件描述符**：如 flop_ratio, log_peak_bw（来自 `_extract_hardware_params()`）

`to_array()` 方法按训练时保存的 `feature_order` 严格排列特征，确保训练和推理时特征顺序一致。

### 2. 模型抽象基类

```python
class EfficiencyModel(ABC):
    """效率模型基类。"""
    @abstractmethod
    def predict(self, features: EfficiencyFeatures) -> float:
        """预测效率值。"""
        pass
```

使用抽象基类定义统一接口，所有后端都必须实现 `predict()` 方法。

### 3. MLP 效率模型

```python
class MLPEfficiencyModel(EfficiencyModel):
    def __init__(self, model_path: str, feature_order: list[str]):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model = self._build_model(checkpoint["input_dim"], checkpoint["hidden_dims"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _build_model(self, input_dim: int, hidden_dims: list[int]) -> nn.Module:
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # 输出约束在 (0, 1)
        return nn.Sequential(*layers)

    def predict(self, features: EfficiencyFeatures) -> float:
        x = features.to_array(self.feature_order)
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            eta = self.model(x_tensor).item()
        return float(np.clip(eta, 0.01, 1.0))
```

注意推理时的架构与训练时略有不同：推理版本不包含 `BatchNorm1d`（训练版本中有），但保留了 `Dropout`（在 `eval()` 模式下自动关闭）。Sigmoid 输出层保证 \(\hat{\eta} \in (0, 1)\)，外加 clip 到 [0.01, 1.0] 确保数值稳定。

### 4. XGBoost 效率模型

```python
class XGBoostEfficiencyModel(EfficiencyModel):
    def __init__(self, model_path: str, feature_order: list[str]):
        import pickle
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model_bytes = checkpoint["model_state_dict"]
        self.model = pickle.loads(model_bytes)

    def predict(self, features: EfficiencyFeatures) -> float:
        x = features.to_array(self.feature_order)
        prediction = self.model.predict(x.reshape(1, -1))[0]
        return float(np.clip(prediction, 0.01, 1.0))
```

XGBoost 模型通过 pickle 序列化存储在 `.pth` 文件中。因为 XGBoost 的回归预测可能超出 [0, 1] 范围，所以需要 clip 操作。

注意：加载 XGBoost 模型时使用 `weights_only=False`，因为需要反序列化 pickle 对象。

### 5. BackendManager -- 模型管理器

```python
class BackendManager:
    """管理各算子类型的效率模型。"""

    def __init__(self, model_dir: Optional[str] = None):
        self._models: dict[Any, Optional[EfficiencyModel]] = {}
        if model_dir is not None:
            self._load_models()

    def _load_models(self):
        _, hw_name = get_hardware_info()
        for op_type in OperatorType:
            # 优先加载 XGBoost
            xgb_name = f"{op_type.value}_{hw_name}_xgb.pth"
            if os.path.exists(xgb_path):
                model = XGBoostEfficiencyModel(xgb_path, feature_order)
                self._models[op_type] = model
                continue  # 有 XGBoost 就跳过 MLP
            # 回退到 MLP
            mlp_name = f"{op_type.value}_{hw_name}_mlp.pth"
            if os.path.exists(mlp_path):
                model = MLPEfficiencyModel(mlp_path, feature_order)
                self._models[op_type] = model
```

模型加载策略：
1. 根据硬件名称（如 `GH200`）查找对应模型文件
2. **优先加载 XGBoost**，不存在时回退到 MLP
3. 文件命名规则：`{operator}_{hardware}_{backend}.pth`，如 `gemm_GH200_xgb.pth`

### 6. 全局单例

```python
_backend_manager: Optional[BackendManager] = None

def get_backend_manager() -> BackendManager:
    """获取全局模型管理器实例。"""
    global _backend_manager
    if _backend_manager is None:
        model_dir = os.environ.get("RLSYSIM_MODEL_DIR")
        _backend_manager = BackendManager(model_dir)
    return _backend_manager

def set_backend_dir(model_dir: str):
    """设置模型目录并重新加载。"""
    global _backend_manager
    _backend_manager = BackendManager(model_dir)
```

使用懒加载单例模式，通过环境变量 `RLSYSIM_MODEL_DIR` 指定模型目录。`set_backend_dir()` 允许运行时切换模型目录。

## 核心类/函数表

| 名称 | 类型 | 说明 |
|------|------|------|
| `EfficiencyFeatures` | dataclass | ML 模型输入特征的统一封装 |
| `EfficiencyModel` | ABC | 效率模型抽象基类，定义 `predict()` 接口 |
| `MLPEfficiencyModel` | class | MLP 后端：PyTorch Sequential + Sigmoid |
| `XGBoostEfficiencyModel` | class | XGBoost 后端：梯度提升树回归 |
| `BackendManager` | class | 模型管理器：加载/缓存/分发各算子类型的模型 |
| `get_backend_manager()` | function | 获取全局 BackendManager 单例 |
| `set_backend_dir()` | function | 设置模型目录并重新加载 |

## 与其他模块的关系

- **compute_cost_predictor.py**：`efficiency_estimate()` 函数通过 `get_backend_manager()` 获取模型，调用 `model.predict(features)` 进行推理
- **compute_cost_profiler.py**：训练好的模型以 `.pth` 格式保存，包含 `model_state_dict`、`feature_order`、`input_dim` 等元数据
- **config.py（上层）**：`_load_models()` 调用 `get_hardware_info()` 确定硬件名称，匹配正确的模型文件

## 小结

`efficiency_models.py` 实现了一个干净的策略模式：通过抽象基类 `EfficiencyModel` 统一 MLP 和 XGBoost 的接口，由 `BackendManager` 根据文件系统中的可用模型自动选择最优后端。特征序列化通过 `EfficiencyFeatures.to_array(feature_order)` 保证训练与推理的一致性。整个设计使得添加新的 ML 后端（如 LightGBM）只需实现 `EfficiencyModel` 接口即可。
