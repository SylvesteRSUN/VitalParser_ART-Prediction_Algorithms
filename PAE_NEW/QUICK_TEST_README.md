# Quick Test Script Usage Guide
# 快速测试脚本使用指南

## Overview / 概述

`quick_test_transfer_learning.py` 是一个快速测试脚本，可以复用已训练好的通用模型，只执行微调和评估步骤，大幅节省时间。

This script reuses pre-trained general models and only performs fine-tuning and evaluation, significantly saving time.

## Time Savings / 时间节省

- **Full pipeline** (main_transfer_learning.py): ~10-20 minutes
  - 完整流程：约10-20分钟
- **Quick test** (quick_test_transfer_learning.py): ~2-3 minutes
  - 快速测试：约2-3分钟
- **Speedup**: 5-10x faster / 加速：5-10倍

## Prerequisites / 前提条件

### First-time Setup / 首次设置

You need to run the full training pipeline **once** to generate and save the general models:

首次使用前需要运行完整训练流程**一次**来生成并保存通用模型：

```bash
python main_transfer_learning.py
```

This will create:
这将创建：

```
saved_models/general_models/
├── general_model_systolic.pkl      # Systolic BP model / 收缩压模型
├── general_model_diastolic.pkl     # Diastolic BP model / 舒张压模型
└── feature_scaler.pkl              # Feature scaler / 特征标准化器
```

## Usage / 使用方法

### Basic Usage / 基本使用

```bash
python quick_test_transfer_learning.py
```

This will:
这将：
1. Load pre-trained general models / 加载预训练通用模型
2. Process test data only / 仅处理测试数据
3. Fine-tune personalized models / 微调个性化模型
4. Generate results and visualizations / 生成结果和可视化

### With Custom Test File / 指定测试文件

```bash
python quick_test_transfer_learning.py --test-file your_test_file.mat
```

## When to Use Quick Test / 何时使用快速测试

Use `quick_test_transfer_learning.py` when:
在以下情况使用快速测试：

✅ **Testing different hyperparameters** in `config_transfer.py`
   - 测试 `config_transfer.py` 中的不同超参数

✅ **Testing different fine-tuning strategies**
   - 测试不同的微调策略

✅ **Testing different calibration sample sizes**
   - 测试不同的校准样本数量

✅ **Testing on different test patients** (without retraining general models)
   - 测试不同的测试病人（无需重新训练通用模型）

## When to Use Full Pipeline / 何时使用完整流程

Use `main_transfer_learning.py` when:
在以下情况使用完整流程：

⚠️ **Changing training data**
   - 更改训练数据时

⚠️ **Changing feature extraction** (adding/removing features)
   - 更改特征提取（增加/删除特征）时

⚠️ **Changing general model hyperparameters** (in `GENERAL_MODEL_CONFIG`)
   - 更改通用模型超参数（在 `GENERAL_MODEL_CONFIG` 中）时

⚠️ **First-time setup**
   - 首次设置时

## Output Files / 输出文件

Results are saved to:
结果保存到：

```
results_transfer_learning/<test_file_name>/
├── plots/
│   ├── quick_test_comparison.png     # Comparison plots / 对比图
│   └── quick_test_improvement.png    # Improvement charts / 改善图表
├── reports/
│   └── quick_test_report.txt         # Detailed report / 详细报告
└── predictions/
    └── quick_test_predictions.csv    # Predictions / 预测结果
```

## Workflow Example / 工作流示例

### Scenario: Testing Different Hyperparameters / 场景：测试不同超参数

```bash
# Step 1: Initial training (once) / 步骤1：初始训练（一次）
python main_transfer_learning.py

# Step 2: Modify hyperparameters in config_transfer.py
# 步骤2：在 config_transfer.py 中修改超参数
# Edit FINE_TUNING_CONFIG parameters

# Step 3: Quick test with new parameters / 步骤3：使用新参数快速测试
python quick_test_transfer_learning.py

# Step 4: Compare results, adjust parameters again
# 步骤4：对比结果，再次调整参数

# Step 5: Repeat Step 3 as needed
# 步骤5：根据需要重复步骤3
python quick_test_transfer_learning.py  # Fast iteration!
```

## Important Notes / 重要说明

### Scaler Consistency / 标准化器一致性

- The scaler is saved during the first full training run
  - 标准化器在首次完整训练时保存

- If you change features, you **must** retrain using `main_transfer_learning.py`
  - 如果更改特征，**必须**使用 `main_transfer_learning.py` 重新训练

- The quick test script will warn if scaler is missing
  - 快速测试脚本会在标准化器缺失时发出警告

### Model Compatibility / 模型兼容性

- Saved models are compatible across different fine-tuning configurations
  - 保存的模型在不同微调配置下兼容

- Changing the base model type (e.g., XGBoost → LightGBM) requires retraining
  - 更改基础模型类型（如 XGBoost → LightGBM）需要重新训练

## Troubleshooting / 故障排除

### Error: "Pre-trained models not found"
### 错误："未找到预训练模型"

**Solution**: Run the full pipeline first:
**解决方案**：首先运行完整流程：

```bash
python main_transfer_learning.py
```

### Warning: "Scaler not found, will need to re-fit on data"
### 警告："未找到标准化器，需要在数据上重新拟合"

**Impact**: Results may be slightly different due to different normalization
**影响**：由于不同的归一化，结果可能略有不同

**Solution**: Update your saved models by running full pipeline with updated code:
**解决方案**：使用更新的代码运行完整流程以更新保存的模型：

```bash
python main_transfer_learning.py
```

### Feature Dimension Mismatch
### 特征维度不匹配

**Cause**: Test data has different features than training data
**原因**：测试数据的特征与训练数据不同

**Solution**: Retrain with consistent feature extraction:
**解决方案**：使用一致的特征提取重新训练：

```bash
python main_transfer_learning.py
```

## Performance Tips / 性能提示

1. **Batch Testing**: Test multiple parameter combinations in succession
   - **批量测试**：连续测试多个参数组合

2. **Keep Training Data Unchanged**: Avoid retraining unless necessary
   - **保持训练数据不变**：除非必要，避免重新训练

3. **Document Parameters**: Keep track of which parameters gave best results
   - **记录参数**：记录哪些参数产生了最佳结果

## Summary / 总结

| Feature | Full Pipeline | Quick Test |
|---------|--------------|------------|
| Training general models | ✅ Yes | ❌ No (reuses) |
| Fine-tuning | ✅ Yes | ✅ Yes |
| Evaluation | ✅ Yes | ✅ Yes |
| Time | ~10-20 min | ~2-3 min |
| Use case | First run, feature changes | Parameter tuning, testing |

---

For questions or issues, refer to the main documentation or contact the development team.
如有疑问或问题，请参阅主文档或联系开发团队。
