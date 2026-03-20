# `validate_profiler.py` -- Profiler 改进验证脚本

## 文件概述

`validate_profiler.py` 是一个独立的验证脚本，用于测试 `compute_cost_profiler` 和 `efficiency_models` 模块的各项改进是否正确工作。它不参与正常的推理或训练流程，而是作为开发阶段的 **冒烟测试（smoke test）** 工具。

验证内容包括：
1. 硬件检测与单位换算（Phase 0）
2. 增强特征提取（Phase 2.2）
3. 数据增强 -- 转置对称性（Phase 2.3）
4. 模型架构兼容性（Phase 2.1）

运行方式：

```bash
python -m syssim.compute.validate_profiler
```

## 关键代码解析

### 1. 硬件检测验证

```python
if torch.cuda.is_available():
    from syssim.compute.compute_cost_profiler import _get_hardware_info
    hw = _get_hardware_info()
    print(f"  Peak FLOP/s (MM): {hw.peak_tflops_mm:.1f} TFLOP/s")
    # 验证: 现代 GPU 应该 > 50 TFLOP/s FP16
    if hw.peak_tflops_mm < 50.0:
        print(f"WARNING: Peak FLOP/s seems low")
```

检查硬件检测是否返回合理的数值。现代 GPU（V100 以上）的 FP16 算力应该超过 50 TFLOP/s。如果返回值过低，可能说明单位换算有误。

### 2. 特征提取验证

```python
gemm_data = {
    "M": [64, 128, 256, 512],
    "N": [64, 128, 256, 512],
    "K": [64, 128, 256, 512],
    "t_measured_ms": [0.1, 0.5, 2.0, 8.0],
    "t_roofline_ms": [0.05, 0.3, 1.5, 6.0],
    "efficiency": [0.5, 0.6, 0.75, 0.75],
}
df_gemm = pd.DataFrame(gemm_data)
X_gemm, features_gemm = _extract_enhanced_features(df_gemm, "gemm", ["M", "N", "K"])
```

使用小型虚构数据验证特征提取函数的输出形状和特征名称是否与预期一致。

### 3. 数据增强验证

```python
df_original = pd.DataFrame({
    "M": [64, 128, 64],
    "N": [128, 128, 64],  # (64, 64) 是方阵
    "K": [256, 256, 256],
    "efficiency": [0.5, 0.6, 0.7],
})
df_augmented = _augment_gemm_data(df_original)
# 预期: 3 原始 + 1 新转置 = 4（2 个重复被去掉）
if len(df_augmented) == 4:
    print("PASS: 正确增强（3 -> 4，重复已移除）")
```

验证转置对称性增强的正确性：
- 原始 (64, 128) -> 转置 (128, 64)：新增
- 原始 (128, 128) -> 转置 (128, 128)：重复，去除
- 原始 (64, 64) -> 转置 (64, 64)：重复，去除

### 4. 模型架构兼容性验证

```python
# 创建 mock checkpoint
mock_checkpoint = {
    "input_dim": 7,
    "hidden_dims": [128, 128, 64],
    "model_state_dict": model_ref.state_dict(),
    "feature_order": [...],
}

# 保存并重新加载
torch.save(mock_checkpoint, model_path)
loaded_model = MLPEfficiencyModel(model_path, mock_checkpoint["feature_order"])
```

验证 `MLPEfficiencyModel` 能否正确加载保存的 checkpoint，确保训练阶段和推理阶段的模型架构一致。

## 核心类/函数表

| 名称 | 类型 | 说明 |
|------|------|------|
| Test 1: Hardware Detection | 测试 | 验证 GPU 硬件检测和单位换算 |
| Test 2: Enhanced Features | 测试 | 验证增强特征提取的输出维度和名称 |
| Test 3: Data Augmentation | 测试 | 验证 GEMM 转置对称性增强的正确性 |
| Test 4: Model Architecture | 测试 | 验证 MLP 模型 save/load 兼容性 |

## 与其他模块的关系

- **compute_cost_profiler.py**：导入并测试 `_get_hardware_info`、`_extract_enhanced_features`、`_augment_gemm_data` 等内部函数
- **efficiency_models.py**：导入并测试 `MLPEfficiencyModel` 的加载和推理

注意：此脚本引用了一些在当前代码版本中可能已被重命名或重构的函数（如 `_extract_enhanced_features`、`_get_hardware_info`），这暗示它可能对应较早的开发阶段。

## 小结

`validate_profiler.py` 是一个轻量级的开发验证工具，通过构造小型模拟数据快速检查各组件的正确性。它不需要 GPU 即可运行大部分测试（硬件检测除外），适合在开发迭代中作为回归测试使用。脚本末尾还提供了下一步操作建议（运行 profiler、检查 MAPE 指标等），起到开发文档的辅助作用。
