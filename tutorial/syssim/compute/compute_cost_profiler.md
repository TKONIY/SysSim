# `compute_cost_profiler.py` -- 算子 Profiling 与效率模型训练工具

## 文件概述

`compute_cost_profiler.py` 是一个 CLI 工具模块，负责 **离线数据采集** 和 **效率模型训练** 两大任务。它在 GPU 上实际运行各种算子配置，测量真实执行时间，然后训练 MLP 或 XGBoost 模型来预测 Roofline 效率。

核心工作流：
1. **Profiling 阶段**：遍历参数网格，在 GPU 上 benchmark 每个配置
2. **特征工程阶段**：提取形状特征 + Roofline envelope 特征
3. **训练阶段**：K-Fold 交叉验证训练 MLP 或 XGBoost

支持的算子类型：GEMM、Attention、RMSNorm、SiLU、LayerNorm (math)

## 关键代码解析

### 1. 比例桶采样算法

```python
def _generate_proportional_samples(
    start: int, end: int, total_samples: int = 64, seed: int = 42,
) -> list[int]:
    # Step 1: 找到范围内所有 2 的幂次作为锚点
    powers_of_two = [2**i for i in range(first_exp, last_exp + 1) if start <= 2**i <= end]
    P = len(powers_of_two)

    # Step 2: 计算填充预算
    F = total_samples - P

    # Step 3: 按区间宽度比例分配采样数
    for i in range(len(powers_of_two) - 1):
        lower = powers_of_two[i]
        upper = powers_of_two[i + 1]
        interval_width = upper - lower
        k_i = round(F * interval_width / W)  # 比例分配
        fill_samples = rng.sample(available, num_to_sample)
        ...
```

这是一个精心设计的采样策略，核心思想是：
- **锚点**：每个 2 的幂次值（2, 4, 8, ..., 131072）必选
- **填充**：在相邻锚点之间按区间宽度比例分配随机采样点
- **效果**：大区间（如 [65536, 131072]）获得更多采样点，小区间（如 [2, 4]）获得较少采样点

这比均匀采样更合理，因为 LLM 的参数空间跨越多个数量级。

### 2. 参数网格构建

```python
COMPUTE_GRIDS = {
    "gemm": construct_dataset_gemm(GEMM_SAMPLES_PER_DIM, RANDOM_SEED),
    "attn": construct_dataset_attn(ATTN_SAMPLES_SEQ, RANDOM_SEED),
    "rmsnorm": construct_dataset_rmsnorm(MATH_SAMPLES_PER_DIM, RANDOM_SEED),
    "silu": construct_dataset_silu(MATH_SAMPLES_PER_DIM, RANDOM_SEED),
}
```

各算子的参数范围：

| 算子 | 维度 | 范围 | 采样策略 |
|------|------|------|----------|
| GEMM | M | [2, 131072] | 比例采样 64 |
| GEMM | N | [256, 65536] | 比例采样 64 |
| GEMM | K | [256, 16384] | 比例采样 64 |
| ATTN | bs | [1, 16] | 仅 2 的幂 |
| ATTN | seq | [1, 131072] | 比例采样 64 |
| ATTN | nh | [2, 128] | 仅 2 的幂 |
| ATTN | nkv | [1, 8] | 仅 2 的幂 |
| ATTN | hd | {64, 128} | 离散值 |

Attention 的网格大小约为 5 x 64 x 7 x 4 x 2 = 17,920 个配置。

### 3. 单算子 Profiling

```python
def _profile_gemm(m: int, n: int, k: int, num_runs: int = 100) -> float:
    device = torch.device("cuda")
    a = torch.randn(m, k, device=device, dtype=torch.float16)
    b = torch.randn(k, n, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(5):
        torch.mm(a, b)
    torch.cuda.synchronize()

    # Profile
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        torch.mm(a, b)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return float(np.median(times))
```

标准的 GPU benchmark 流程：
1. **Warmup**（5-10 次）：消除 JIT 编译、缓存预热等冷启动效应
2. **Synchronize**：`torch.cuda.synchronize()` 确保 GPU kernel 执行完毕
3. **取中位数**：比均值更鲁棒，避免异常值影响

### 4. Roofline 特征计算（使用 FakeTensor）

```python
def _add_roofline_and_efficiency(df, hw_info, operator):
    from torch._subclasses.fake_tensor import FakeTensorMode
    fake_mode = FakeTensorMode()

    for idx, row in df.iterrows():
        if operator == "gemm":
            with fake_mode:
                a = torch.empty(m, k, dtype=torch.float16, device='cuda')
                b = torch.empty(k, n, dtype=torch.float16, device='cuda')
                out = torch.empty(m, n, dtype=torch.float16, device='cuda')
            result = roofline_estimate(aten.mm, (a, b), {}, out, hw_info, OperatorType.GEMM)
            ...
```

关键设计：使用 PyTorch 的 `FakeTensorMode` 创建 **零内存开销** 的假张量。Roofline 计算只需要张量的元数据（shape、dtype、device），不需要实际数据。这意味着即使维度很大（如 M=131072），也不会分配 GPU 显存。

效率标签的计算：

$$\eta = \frac{T_{roofline}}{T_{measured}}$$

### 5. GEMM 数据增强（转置对称性）

```python
def _augment_gemm_data(df):
    # 交换 M 和 N 创建转置版本
    df_transposed = df.copy()
    df_transposed[["M", "N"]] = df[["N", "M"]].values
    df_augmented = pd.concat([df, df_transposed], ignore_index=True)
    df_augmented = df_augmented.drop_duplicates(subset=["M", "N", "K"], keep="first")
    return df_augmented
```

利用矩阵乘法的性质：\(C = A \times B\) 和 \(C^T = B^T \times A^T\) 具有相同的算术强度和相似的效率。交换 M 和 N 即等价于转置操作，可以将数据集大小翻倍。

### 6. 通用形状特征提取

```python
OPERATOR_SHAPE_CONFIGS = {
    "gemm": {
        "inputs": [["M", "K"], ["K", "N"]],
        "output": ["M", "N"]
    },
    "attn": {
        "inputs": [["bs", "nh", "seq", "hd"], ["bs", "nkv", "seq", "hd"], ["bs", "nkv", "seq", "hd"]],
        "output": ["bs", "nh", "seq", "hd"]
    },
    ...
}

def _extract_shape_features(df, operator, base_cols):
    config = OPERATOR_SHAPE_CONFIGS[operator]
    # 拼接所有 input + output 维度，取对数
    for input_shape_cols in config["inputs"]:
        for col_name in input_shape_cols:
            features.append(np.log(dim_vals + 1))
    ...
```

配置驱动的特征提取：添加新算子只需在 `OPERATOR_SHAPE_CONFIGS` 中注册，无需修改代码逻辑。所有维度取对数缩放，这对 MLP 至关重要（使不同数量级的维度值处于相似范围）。

### 7. MLP 模型架构与训练

```python
def _build_mlp_model(input_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(64, 1),          nn.Sigmoid(),
    )
```

架构要点：
- 3 层隐藏层（256-128-64），逐层缩小
- BatchNorm 加速收敛并稳定训练
- Dropout(0.1) 防过拟合
- **Sigmoid 输出层**：将预测约束在 (0, 1) 范围，符合效率的物理意义

损失函数使用 MAPE（Mean Absolute Percentage Error）：

$$MAPE = \frac{1}{N}\sum_{i=1}^{N}\frac{|y_{true}^{(i)} - y_{pred}^{(i)}|}{|y_{true}^{(i)}| + \epsilon} \times 100$$

训练使用 K-Fold 交叉验证（默认 5 折），配合 Early Stopping 和 ReduceLROnPlateau 学习率调度。

### 8. XGBoost 训练

```python
default_params = {
    "tree_method": "hist",        # 快速直方图算法
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 50,
    ...
}
```

XGBoost 作为替代后端，使用梯度提升树进行回归。预测结果会被 clip 到 [0.01, 1.0] 范围。

### 9. CLI 入口

```bash
# Profiling 模式（先采集数据）
python -m syssim.compute.compute_cost_profiler \
    --operator gemm --output models/gemm_mlp.pth

# Training 模式（使用已有数据训练模型）
python -m syssim.compute.compute_cost_profiler \
    --operator gemm --data-path data/gemm_GH200_data.csv \
    --output models/gemm_GH200_xgb.pth --backend xgboost
```

两种模式：
- **无 `--data-path`**：进入 Profiling 模式，在 GPU 上采集数据
- **有 `--data-path`**：进入 Training 模式，跳过 profiling 直接训练

## 核心类/函数表

| 名称 | 类型 | 说明 |
|------|------|------|
| `_generate_proportional_samples()` | function | 比例桶采样算法 |
| `_generate_power_of_two_range()` | function | 生成范围内的 2 的幂次序列 |
| `construct_dataset_gemm()` | function | 构建 GEMM 参数网格 |
| `construct_dataset_attn()` | function | 构建 Attention 参数网格 |
| `construct_dataset_rmsnorm()` | function | 构建 RMSNorm 参数网格 |
| `construct_dataset_silu()` | function | 构建 SiLU 参数网格 |
| `_profile_gemm()` | function | 单配置 GEMM benchmark |
| `_profile_attention()` | function | 单配置 Attention benchmark（支持 GQA） |
| `_profile_rmsnorm()` | function | 单配置 RMSNorm benchmark |
| `_profile_silu()` | function | 单配置 SiLU benchmark |
| `_profile_math()` | function | 单配置 LayerNorm benchmark |
| `_profile_gemm_grid()` | function | GEMM 全网格 profiling |
| `_profile_attn_grid()` | function | Attention 全网格 profiling |
| `_add_roofline_and_efficiency()` | function | 为 profiling 数据添加 Roofline 标签 |
| `_augment_gemm_data()` | function | GEMM 转置对称性数据增强 |
| `OPERATOR_SHAPE_CONFIGS` | dict | 算子形状配置（配置驱动特征提取） |
| `_extract_shape_features()` | function | 提取对数缩放的形状特征 |
| `_extract_roofline_features()` | function | 提取 Roofline envelope 特征 |
| `_build_training_features()` | function | 组合形状 + Roofline 特征 |
| `_build_mlp_model()` | function | 构建 MLP 网络（256-128-64 + Sigmoid） |
| `MAPELoss` | class | MAPE 损失函数 |
| `_train_and_validate_mlp()` | function | MLP K-Fold 训练 |
| `_train_and_validate_xgboost()` | function | XGBoost K-Fold 训练 |
| `profile_operator()` | function | 完整 profiling 流程 |
| `train_efficiency_model()` | function | 完整训练流程 |

## 与其他模块的关系

- **compute_cost_predictor.py**：导入 `roofline_estimate` 和 `aten` 用于计算 Roofline 标签
- **efficiency_models.py**：训练好的模型保存为 `.pth` 文件，由 `efficiency_models.py` 在推理时加载
- **config.py（上层）**：使用 `get_hardware_info()` 自动检测 GPU 硬件参数
- **operator_graph.py（上层）**：使用 `OperatorType` 枚举

## 小结

`compute_cost_profiler.py` 是整个效率模型的"数据工厂"。它通过精心设计的参数网格采样、FakeTensor 零开销 Roofline 计算、转置对称性数据增强等技术，高效地生成训练数据。支持 MLP 和 XGBoost 两种后端，通过 K-Fold 交叉验证确保模型泛化能力。整个流程被拆分为 Profiling 和 Training 两个独立阶段，允许在不同硬件上重新训练模型而无需重新 profiling。
