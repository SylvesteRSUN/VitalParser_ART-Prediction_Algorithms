# Transfer Learning Quick Start Guide
# 迁移学习快速开始指南

## 5-Minute Quick Start / 5分钟快速开始

### Step 1: Test the modules / 步骤1: 测试模块

```bash
python test_transfer_learning.py
```

Expected output / 预期输出:
```
✓ Configuration module loaded successfully
✓ Data splitter working correctly
✓ Transfer learning core working correctly
✓ Evaluation visualizations working correctly
✓ Main pipeline module loaded successfully

ALL TESTS PASSED!
```

### Step 2: Run the pipeline / 步骤2: 运行流程

```bash
python main_transfer_learning.py
```

This will:
这将:
1. Load and process training data / 加载并处理训练数据
2. Train general models / 训练通用模型
3. Save general models / 保存通用模型
4. Load test data / 加载测试数据
5. Split into calibration and evaluation sets / 分割为校准集和评估集
6. Evaluate general models / 评估通用模型
7. Fine-tune personalized models / 微调个性化模型
8. Generate comparison reports and visualizations / 生成对比报告和可视化

### Step 3: Check the results / 步骤3: 查看结果

Results are in `results_transfer_learning/[test_file_name]/`:
结果在 `results_transfer_learning/[测试文件名]/`:

```
results_transfer_learning/
└── [test_file_name]/
    ├── plots/                  # 📊 Visualizations
    ├── reports/                # 📝 Text reports
    └── predictions/            # 📈 Prediction CSV
```

---

## Command-line Options / 命令行选项

### Basic usage / 基本用法:
```bash
python main_transfer_learning.py
```

### Custom test file / 自定义测试文件:
```bash
python main_transfer_learning.py --test-file path/to/your/test.vital
```

### Use different model / 使用不同模型:
```bash
python main_transfer_learning.py --model-type xgboost
```

### Custom calibration size / 自定义校准大小:
```bash
python main_transfer_learning.py --calibration-samples 250
```

### All options combined / 组合所有选项:
```bash
python main_transfer_learning.py \
    --test-file path/to/test.vital \
    --model-type xgboost \
    --calibration-samples 200 \
    --verbose
```

---

## Configuration Guide / 配置指南

### Essential settings in `config_transfer.py` / `config_transfer.py`中的关键设置:

#### 1. Model Type / 模型类型

```python
GENERAL_MODEL_CONFIG['model_type'] = 'xgboost'  # Recommended / 推荐
```

Options / 选项:
- `'xgboost'` - Best performance / 最佳性能 ⭐
- `'lightgbm'` - Fast and efficient / 快速高效
- `'gradient_boosting'` - No dependencies / 无外部依赖

#### 2. Calibration Size / 校准大小

```python
DATA_SPLIT_CONFIG['sample_based']['n_samples'] = 200  # heartbeats
```

Recommendations / 建议:
- **100 samples**: Fast, may underfit / 快速,可能欠拟合
- **200 samples**: Balanced (recommended) / 平衡(推荐) ⭐
- **300 samples**: Better accuracy, slower / 更好的准确性,更慢

#### 3. Fine-tuning Learning Rate / 微调学习率

```python
FINE_TUNING_CONFIG['xgboost']['learning_rate'] = 0.01
```

Recommendations / 建议:
- **0.005**: Very conservative / 非常保守
- **0.01**: Recommended / 推荐 ⭐
- **0.05**: Faster, may overfit / 更快,可能过拟合

---

## Expected Performance / 预期性能

Based on literature and experiments:
基于文献和实验:

| Model / 模型 | MAE (mmHg) | MSE | Improvement / 改善 |
|-------------|-----------|-----|-------------------|
| General Model / 通用模型 | 10-15 | 100-200 | Baseline / 基线 |
| Simple Calibration / 简单校准 | 8-12 | 80-150 | 20-30% |
| **Transfer Learning** / 迁移学习 | **3-5** | **20-50** | **60-70%** ⭐ |

---

## Output Files Explained / 输出文件说明

### 1. Visualizations / 可视化 (`plots/`)

#### `transfer_learning_comparison_Systolic.png`
4-panel comparison:
4格对比图:
- Time series (actual vs general vs personalized) / 时间序列对比
- Scatter plot / 散点图
- Error distribution / 误差分布
- Performance metrics / 性能指标

#### `transfer_learning_comparison_Diastolic.png`
Same as systolic / 与收缩压相同

#### `transfer_learning_improvement_summary.png`
Bar charts showing improvement percentages
柱状图显示改善百分比

### 2. Reports / 报告 (`reports/`)

#### `transfer_learning_report.txt`
Contains / 包含:
- Dataset statistics / 数据集统计
- General model metrics / 通用模型指标
- Personalized model metrics / 个性化模型指标
- Improvement percentages / 改善百分比

### 3. Predictions / 预测 (`predictions/`)

#### `predictions.csv`
Columns / 列:
- `true_systolic` - Actual systolic BP / 真实收缩压
- `pred_systolic_general` - General model prediction / 通用模型预测
- `true_diastolic` - Actual diastolic BP / 真实舒张压
- `pred_diastolic_general` - General model prediction / 通用模型预测

---

## Troubleshooting / 故障排除

### Problem: "No module named 'xgboost'"

**Solution / 解决方案:**
```bash
pip install xgboost
```

Or use gradient_boosting (no dependencies):
或使用gradient_boosting(无依赖):
```python
# In config_transfer.py
GENERAL_MODEL_CONFIG['model_type'] = 'gradient_boosting'
```

### Problem: Poor performance / 性能差

**Check / 检查:**
1. Calibration size too small? Try 200-300 samples / 校准大小太小?尝试200-300样本
2. Learning rate too high? Try 0.01 or 0.005 / 学习率太高?尝试0.01或0.005
3. Enable early stopping / 启用early stopping

### Problem: Out of memory / 内存不足

**Solutions / 解决方案:**
1. Use fewer training files / 使用更少的训练文件
2. Reduce `n_estimators` / 减少树的数量
3. Use LightGBM instead of XGBoost / 使用LightGBM代替XGBoost

---

## Next Steps / 下一步

1. **Experiment with parameters** / 实验参数
   - Try different calibration sizes / 尝试不同的校准大小
   - Adjust learning rates / 调整学习率
   - Test different models / 测试不同模型

2. **Analyze results** / 分析结果
   - Check visualizations in `plots/` / 查看可视化
   - Read detailed report in `reports/` / 阅读详细报告
   - Compare metrics / 对比指标

3. **Fine-tune for your data** / 为您的数据微调
   - Adjust model hyperparameters / 调整模型超参数
   - Optimize calibration size / 优化校准大小
   - Experiment with strategies / 实验不同策略

---

## Getting Help / 获取帮助

1. **Read the full documentation** / 阅读完整文档
   - `TRANSFER_LEARNING_README.md` - Comprehensive guide / 综合指南
   - `transfer_learning.md` - Technical design / 技术设计

2. **Check the code** / 查看代码
   - All files have detailed comments / 所有文件都有详细注释
   - Bilingual (English + Chinese) / 双语(英文+中文)

3. **Run tests** / 运行测试
   - `python test_transfer_learning.py` - Verify installation / 验证安装

---

**Happy Experimenting! / 实验愉快!** 🚀
