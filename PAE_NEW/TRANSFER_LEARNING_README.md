# Transfer Learning Framework for Blood Pressure Prediction
# 血压预测迁移学习框架

## Overview / 概述

This framework implements transfer learning for personalized blood pressure prediction from PPG (PLETH) signals. It significantly improves prediction accuracy by adapting general models to individual patients.

本框架实现了基于PPG(PLETH)信号的个性化血压预测迁移学习。通过将通用模型适配到个体患者,显著提高预测准确性。

### Expected Performance / 预期性能

- **General Model**: MAE ~10-15 mmHg / 通用模型: MAE约10-15 mmHg
- **Simple Calibration**: MAE ~8-12 mmHg (may be worse) / 简单校准: MAE约8-12 mmHg(可能更差)
- **Transfer Learning**: MAE ~3-5 mmHg (60-70% improvement) / 迁移学习: MAE约3-5 mmHg(改善60-70%)

---

## Project Structure / 项目结构

```
PAE_NEW/
├── config_transfer.py          # Transfer learning configuration / 迁移学习配置
├── data_splitter.py            # Patient data splitter / 患者数据分割器
├── transfer_learning.py        # Core transfer learning classes / 核心迁移学习类
├── main_transfer_learning.py   # Main pipeline / 主程序
├── evaluation.py               # Enhanced with TL visualizations / 增强的可视化
└── TRANSFER_LEARNING_README.md # This file / 本文件
```

---

## Quick Start / 快速开始

### 1. Installation / 安装

Make sure you have all dependencies installed:
确保已安装所有依赖:

```bash
pip install -r requirements.txt
```

Required packages:
必需的包:
- numpy
- pandas
- matplotlib
- scikit-learn
- xgboost (recommended) / 推荐
- lightgbm (optional) / 可选

### 2. Configuration / 配置

Edit `config_transfer.py` to customize:
编辑 `config_transfer.py` 来自定义:

```python
# Model type / 模型类型
GENERAL_MODEL_CONFIG['model_type'] = 'xgboost'  # or 'lightgbm', 'gradient_boosting'

# Calibration data amount / 校准数据量
DATA_SPLIT_CONFIG['sample_based']['n_samples'] = 200  # heartbeats

# Fine-tuning parameters / 微调参数
FINE_TUNING_CONFIG['xgboost']['learning_rate'] = 0.01  # reduced for stability
FINE_TUNING_CONFIG['xgboost']['n_estimators'] = 100    # new trees to add
```

### 3. Run Pipeline / 运行流程

Basic usage:
基本用法:

```bash
python main_transfer_learning.py
```

With custom parameters:
使用自定义参数:

```bash
python main_transfer_learning.py \
    --test-file path/to/test.vital \
    --model-type xgboost \
    --calibration-samples 200 \
    --verbose
```

---

## Pipeline Steps / 流程步骤

The transfer learning pipeline consists of 7 steps:
迁移学习流程包含7个步骤:

1. **Load Training Data** / 加载训练数据
   - Load all training patient files / 加载所有训练患者文件
   - Process signals and extract features / 处理信号并提取特征
   - Combine into unified dataset / 合并为统一数据集

2. **Train General Models** / 训练通用模型
   - Train on all training data / 在所有训练数据上训练
   - Separate models for systolic and diastolic BP / 收缩压和舒张压分别建模

3. **Save General Models** / 保存通用模型
   - Save to `saved_models/general_models/` / 保存到通用模型目录

4. **Load Test Data** / 加载测试数据
   - Load and process test patient file / 加载并处理测试患者文件

5. **Split Test Data** / 分割测试数据
   - Calibration set: First 100-300 heartbeats / 校准集: 前100-300个心跳
   - Evaluation set: Remaining heartbeats / 评估集: 剩余心跳

6. **Evaluate General Models** / 评估通用模型
   - Baseline performance on evaluation set / 在评估集上的基线性能

7. **Fine-tune and Evaluate** / 微调并评估
   - Fine-tune using calibration data / 使用校准数据微调
   - Evaluate personalized models / 评估个性化模型
   - Generate comparison visualizations / 生成对比可视化

---

## Output / 输出

Results are saved to `results_transfer_learning/[test_file_name]/`:
结果保存到 `results_transfer_learning/[测试文件名]/`:

### Directory Structure / 目录结构

```
results_transfer_learning/
└── [test_file_name]/
    ├── plots/                                    # Visualizations / 可视化
    │   ├── transfer_learning_comparison_Systolic.png
    │   ├── transfer_learning_comparison_Diastolic.png
    │   └── transfer_learning_improvement_summary.png
    ├── reports/                                  # Text reports / 文本报告
    │   └── transfer_learning_report.txt
    └── predictions/                              # Prediction results / 预测结果
        └── predictions.csv
```

### Visualizations / 可视化

1. **Transfer Learning Comparison** (2 files: Systolic & Diastolic)
   迁移学习对比 (2个文件: 收缩压和舒张压)
   - Time series comparison / 时间序列对比
   - Scatter plot comparison / 散点图对比
   - Error distribution / 误差分布
   - Performance metrics / 性能指标

2. **Improvement Summary**
   改善总结
   - MAE improvement percentage / MAE改善百分比
   - MSE improvement percentage / MSE改善百分比

### Text Report / 文本报告

Includes:
包含:
- Dataset statistics / 数据集统计
- General model performance / 通用模型性能
- Personalized model performance / 个性化模型性能
- Improvement percentages / 改善百分比
- Target achievement / 目标达成情况

---

## Key Configuration Parameters / 关键配置参数

### Model Selection / 模型选择

```python
# config_transfer.py

# Choose model type / 选择模型类型
GENERAL_MODEL_CONFIG['model_type'] = 'xgboost'  # Best performance / 最佳性能

# XGBoost parameters / XGBoost参数
GENERAL_MODEL_CONFIG['xgboost'] = {
    'n_estimators': 200,        # Number of trees / 树的数量
    'max_depth': 10,            # Tree depth / 树深度
    'learning_rate': 0.1,       # Learning rate / 学习率
    'subsample': 0.8,           # Data subsampling / 数据子采样
    'reg_alpha': 0.01,          # L1 regularization / L1正则化
    'reg_lambda': 1.0,          # L2 regularization / L2正则化
}
```

### Fine-tuning Strategy / 微调策略

```python
# Fine-tuning configuration / 微调配置
FINE_TUNING_CONFIG = {
    'strategy': 'incremental',  # or 'correction_model' / 或'校正模型'

    'xgboost': {
        'n_estimators': 100,      # New trees to add / 增量添加的树
        'learning_rate': 0.01,    # Reduced LR / 降低的学习率
        'reg_alpha': 0.05,        # Increased regularization / 增加正则化
    },

    'early_stopping': {
        'enabled': True,
        'rounds': 20,
        'validation_fraction': 0.2
    }
}
```

### Data Splitting / 数据分割

```python
# Choose split method / 选择分割方法
DATA_SPLIT_CONFIG['split_method'] = 'sample_based'  # Recommended / 推荐

# Sample-based configuration / 基于样本的配置
DATA_SPLIT_CONFIG['sample_based'] = {
    'n_samples': 200,      # Calibration heartbeats / 校准心跳数
    'min_samples': 100,    # Minimum required / 最小要求
    'max_samples': 300     # Maximum allowed / 最大允许
}

# Alternative: time-based / 替代方案: 基于时间
DATA_SPLIT_CONFIG['time_based'] = {
    'duration_minutes': 5   # First 5 minutes for calibration / 前5分钟用于校准
}

# Alternative: ratio-based / 替代方案: 基于比例
DATA_SPLIT_CONFIG['ratio_based'] = {
    'calibration_ratio': 0.25   # 25% for calibration / 25%用于校准
}
```

---

## Advanced Usage / 高级用法

### Experiment with Different Calibration Sizes / 实验不同的校准集大小

Enable calibration size experiment in `config_transfer.py`:
在 `config_transfer.py` 中启用校准大小实验:

```python
EXPERIMENT_CONFIG['calibration_size_experiment'] = {
    'enabled': True,
    'sizes': [100, 150, 200, 250, 300]
}
```

This will test multiple calibration sizes and report the best one.
这将测试多个校准大小并报告最佳的。

### Using Different Models / 使用不同模型

The framework supports three model types:
框架支持三种模型类型:

1. **XGBoost** (Recommended / 推荐)
   - Best performance / 最佳性能
   - Supports true incremental training / 支持真正的增量训练
   - Fast and efficient / 快速高效

2. **LightGBM** (Alternative / 替代方案)
   - Similar to XGBoost / 类似XGBoost
   - Faster training on large datasets / 在大数据集上训练更快
   - Good memory efficiency / 良好的内存效率

3. **Gradient Boosting** (Scikit-learn)
   - No external dependencies / 无外部依赖
   - Limited incremental training support / 增量训练支持有限
   - Good baseline / 良好的基线

### Custom Fine-tuning Strategy / 自定义微调策略

Two strategies are available:
提供两种策略:

1. **Incremental Training** (Default / 默认)
   - Continue training from general model / 从通用模型继续训练
   - Add new trees with reduced learning rate / 使用降低的学习率添加新树
   - Best for tree-based models / 最适合基于树的模型

2. **Correction Model** (Alternative / 替代方案)
   - Train small model to correct general predictions / 训练小模型来校正通用预测
   - Final prediction = general + correction / 最终预测 = 通用 + 校正
   - More robust to overfitting / 更能抵抗过拟合

---

## Troubleshooting / 故障排除

### Issue: Poor personalized model performance / 问题: 个性化模型性能差

**Possible causes / 可能原因:**
1. Too few calibration samples / 校准样本太少
2. Learning rate too high / 学习率太高
3. Overfitting on calibration data / 在校准数据上过拟合

**Solutions / 解决方案:**
1. Increase `n_samples` in `DATA_SPLIT_CONFIG` / 增加校准样本数
2. Reduce `learning_rate` in `FINE_TUNING_CONFIG` / 降低学习率
3. Enable early stopping / 启用early stopping
4. Increase regularization (`reg_alpha`, `reg_lambda`) / 增加正则化

### Issue: Models not saving / 问题: 模型未保存

**Solution / 解决方案:**
Check that output directories have write permissions:
检查输出目录有写权限:

```bash
# Create directories manually if needed / 如需要手动创建目录
mkdir -p saved_models/general_models
mkdir -p saved_models/personalized_models
mkdir -p results_transfer_learning
```

### Issue: Memory error with large datasets / 问题: 大数据集内存错误

**Solutions / 解决方案:**
1. Use LightGBM instead of XGBoost (more memory efficient) / 使用LightGBM代替XGBoost
2. Reduce `n_estimators` in model config / 减少模型中的树数量
3. Process training files one at a time / 逐个处理训练文件

---

## Performance Tips / 性能提示

### For Best Results / 获得最佳结果:

1. **Use XGBoost** - Best accuracy and speed / 最佳准确性和速度
2. **200 calibration samples** - Good balance / 良好平衡
3. **Early stopping enabled** - Prevents overfitting / 防止过拟合
4. **Low learning rate (0.01)** - Stable fine-tuning / 稳定的微调

### For Fast Experimentation / 快速实验:

1. **Use fewer training files** - Faster general model training / 更快的通用模型训练
2. **Reduce n_estimators** - Faster training / 更快的训练
3. **Disable visualizations** - Set `save_figures=False` / 禁用可视化

---

## Technical Details / 技术细节

### How Transfer Learning Works / 迁移学习原理

1. **General Model Phase / 通用模型阶段**
   - Train on diverse patient data / 在多样化患者数据上训练
   - Learn general PLETH-BP relationships / 学习通用的PLETH-BP关系
   - Capture physiological patterns / 捕获生理模式

2. **Personalization Phase / 个性化阶段**
   - Use small amount of patient-specific data / 使用少量患者特定数据
   - Fine-tune model parameters / 微调模型参数
   - Adapt to individual characteristics / 适应个体特征

3. **Key Advantages / 关键优势**
   - Preserve general knowledge / 保留通用知识
   - Adapt to individual differences / 适应个体差异
   - Avoid catastrophic forgetting / 避免灾难性遗忘

### Why It Works Better Than Calibration / 为什么比校准效果好

- **Simple calibration**: Only adjusts output offset / 简单校准: 只调整输出偏移
- **Transfer learning**: Adapts entire model / 迁移学习: 适配整个模型
- **Result**: Captures complex individual patterns / 结果: 捕获复杂的个体模式

---

## Citation / 引用

If you use this framework in your research, please cite:
如果在研究中使用此框架,请引用:

```
[Your citation information here]
```

---

## Support / 支持

For issues or questions:
问题或疑问:

1. Check this README / 查看本README
2. Review `transfer_learning.md` for detailed design / 查看详细设计文档
3. Check configuration files / 检查配置文件
4. Review error messages and logs / 查看错误消息和日志

---

## License / 许可

[Your license information]

---

**Happy Predicting! / 预测愉快!** 🩺📊
