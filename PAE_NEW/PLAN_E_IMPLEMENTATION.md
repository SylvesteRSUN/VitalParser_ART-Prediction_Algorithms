# Plan E Implementation: Weighted Loss for Extreme BP Values
# 方案E实施：极端血压值的加权损失

## Overview / 概述

Plan E implements weighted loss function that assigns higher weights to extreme blood pressure values during model training. This helps the model pay more attention to extreme BP variations, which are critical for ICU patient monitoring.

方案E实施了加权损失函数，在模型训练期间为极端血压值分配更高的权重。这有助于模型更加关注极端血压变化，这对ICU患者监测至关重要。

## Implementation Details / 实施细节

### 1. Sample Weight Calculation / 样本权重计算

The `_calculate_sample_weights()` method assigns weights based on BP value distribution:

```python
weights = np.ones(len(y))  # Default weight: 1.0

# Extreme values (beyond mean ± std)
upper_threshold = mean + std
lower_threshold = mean - std
extreme_mask = (y > upper_threshold) | (y < lower_threshold)
weights[extreme_mask] = weight_extreme_multiplier  # Default: 2.0x

# Very extreme values (beyond mean ± 1.5*std)
very_extreme_mask = (y > mean + 1.5 * std) | (y < mean - 1.5 * std)
weights[very_extreme_mask] = weight_extreme_multiplier * 1.5  # Default: 3.0x
```

### 2. Integration Points / 集成点

Sample weighting is applied at three key training points:

#### A. Incremental Training (`_incremental_training()`)
- Applied to XGBoost, LightGBM, and GradientBoosting
- Weights calculated from calibration BP labels
- Passed via `sample_weight` parameter to `fit()`

#### B. Correction Model Training (`_correction_model()`)
- Applied when training the correction model on residuals
- Uses weights based on true BP values (not residuals)
- Ensures correction model focuses on extreme value corrections

#### C. Early Stopping Validation
- Sample weights applied to training set during early stopping
- Validation set uses default weights (no weighting)

### 3. Configuration Parameters / 配置参数

Added to `config_transfer.py` under `FINE_TUNING_CONFIG`:

```python
'sample_weighting': {
    'enabled': True,              # Enable/disable feature
    'extreme_multiplier': 2.0,    # Weight for extreme values (mean ± std)
}
```

### 4. Modified Files / 修改的文件

1. **transfer_learning.py**
   - Added `use_sample_weights` and `weight_extreme_multiplier` parameters to `__init__()`
   - Created `_calculate_sample_weights()` method (lines 213-248)
   - Modified `_incremental_training()` to apply weights (lines 350-443)
   - Modified `_correction_model()` to apply weights (lines 457-516)
   - Added verbose output for weighting statistics

2. **config_transfer.py**
   - Added `sample_weighting` configuration (lines 73-78)

3. **quick_test_transfer_learning.py**
   - Modified to read sample weighting config (lines 440-443)
   - Pass parameters to PersonalFineTuner (lines 445-451)

4. **main_transfer_learning.py**
   - Modified to read sample weighting config (lines 397-400)
   - Pass parameters to PersonalFineTuner (lines 403-409)

## How It Works / 工作原理

### Weight Distribution Example

For a calibration set with systolic BP:
- Mean = 120 mmHg, Std = 15 mmHg

Weight assignment:
- **Normal values** (105-135 mmHg): weight = 1.0
- **Extreme values** (< 105 or > 135 mmHg): weight = 2.0
- **Very extreme** (< 97.5 or > 142.5 mmHg): weight = 3.0

### Loss Function Impact

With weighted loss, the model's loss function becomes:

```
Loss = Σ (weight_i * (y_true_i - y_pred_i)²)
```

This means:
- Error on extreme values contributes 2-3x more to total loss
- Model learns to prioritize accuracy on extreme values
- Regularization still prevents overfitting on extreme values

## Testing / 测试

To test Plan E with quick test:

```bash
# First ensure general models are trained
python main_transfer_learning.py

# Quick test with Plan E enabled
python quick_test_transfer_learning.py
```

Expected output will include:
```
Fine-tuning systolic BP model...
  Sample Weighting Enabled:
    Extreme values: 150/500
    Weight multiplier: 2.0x
  Correction Model - Sample Weighting Enabled:
    Extreme values: 150/500
    Weight multiplier: 2.0x
```

## Advantages / 优势

1. **Better extreme value tracking** / 更好的极端值跟踪
   - Model pays more attention to clinically critical BP ranges

2. **Balanced learning** / 平衡学习
   - Prevents model from only optimizing for common BP values

3. **Flexible configuration** / 灵活配置
   - Can adjust weight multiplier based on clinical needs
   - Can disable if not needed

4. **Compatible with existing strategies** / 兼容现有策略
   - Works with both incremental and correction_model strategies
   - No conflict with other optimizations (Plan G, etc.)

## Tuning Parameters / 调优参数

You can experiment with different weight multipliers in [config_transfer.py](config_transfer.py):

```python
'sample_weighting': {
    'enabled': True,
    'extreme_multiplier': 2.0,  # Try: 1.5, 2.0, 2.5, 3.0
}
```

Higher multipliers:
- ✅ More sensitivity to extreme values
- ❌ Risk of overfitting on extreme values

Lower multipliers:
- ✅ More stable predictions
- ❌ Less sensitivity to extreme variations

## Comparison with Other Plans / 与其他方案的对比

| Plan | Focus | Implementation |
|------|-------|----------------|
| **Plan E** | Weight extreme values | Sample weighting in loss function |
| **Plan F** | BP state features | Add 18 new features (z-scores, trends) |
| **Plan G** | Reduce smoothing | Lower regularization, higher complexity |

**Recommendation**: Use Plan E + Plan F + Plan G together for best extreme value tracking.
**建议**：同时使用方案E + 方案F + 方案G以获得最佳极端值跟踪效果。

## Notes / 注意事项

1. **Scaler compatibility**: No impact on saved scalers, compatible with existing general models
   - 标准化器兼容性：不影响保存的标准化器，兼容现有通用模型

2. **Quick test ready**: Can test immediately with `quick_test_transfer_learning.py`
   - 快速测试就绪：可以立即使用 `quick_test_transfer_learning.py` 测试

3. **Model-agnostic**: Works with XGBoost, LightGBM, and GradientBoosting
   - 模型无关：适用于XGBoost、LightGBM和GradientBoosting

## Next Steps / 下一步

1. Run quick test to evaluate Plan E performance
2. Compare metrics before/after Plan E
3. Consider combining with Plan F (BP state features) if not already done
4. Fine-tune `extreme_multiplier` based on results

---

**Implementation completed**: 2025-11-26
**实施完成时间**：2025-11-26
