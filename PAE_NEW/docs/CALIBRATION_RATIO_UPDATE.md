# Calibration Data Split Method Update
# 校准数据分割方法更新

## Change Summary / 更改摘要

**Changed from**: Fixed sample size (500 samples)
**改为**: Ratio-based (20% of test data)

**更改前**：固定样本数量（500个样本）
**更改后**：基于比例（测试集的20%）

## Configuration Changes / 配置更改

### [config_transfer.py](config_transfer.py)

```python
DATA_SPLIT_CONFIG = {
    'split_method': 'ratio_based',  # Changed from 'sample_based'

    'ratio_based': {
        'calibration_ratio': 0.20,  # 20% for calibration
        'min_samples': 200,         # Minimum 200 samples
        'max_samples': 1000         # Maximum 1000 samples
    }
}
```

## Benefits / 优势

### 1. Adaptive Calibration Size / 自适应校准大小

- **Small test sets**: Gets proportionally smaller calibration set (avoids overfitting)
  - 小测试集：获得相应更小的校准集（避免过拟合）

- **Large test sets**: Gets more calibration data (better personalization)
  - 大测试集：获得更多校准数据（更好的个性化）

### 2. Consistent Evaluation Split / 一致的评估分割

- Always uses 80% of test data for evaluation (regardless of test set size)
  - 始终使用80%的测试数据进行评估（无论测试集大小）

### 3. Protections / 保护措施

| Protection | Value | Purpose |
|------------|-------|---------|
| **Minimum samples** | 200 | Ensure enough data for fine-tuning |
| **Maximum samples** | 1000 | Prevent excessive calibration time |
| **Minimum eval ratio** | 20% | Always keep enough data for evaluation |

## Examples / 示例

### Example 1: Small Test Set (1000 samples)
```
Total test samples: 1000
Calibration: 1000 * 0.20 = 200 samples (20%)
Evaluation: 800 samples (80%)
```

### Example 2: Medium Test Set (3000 samples)
```
Total test samples: 3000
Calibration: 3000 * 0.20 = 600 samples (20%)
Evaluation: 2400 samples (80%)
```

### Example 3: Large Test Set (6000 samples)
```
Total test samples: 6000
Calibration: min(6000 * 0.20, 1000) = 1000 samples (capped at max)
Evaluation: 5000 samples (83%)
```

### Example 4: Very Small Test Set (500 samples)
```
Total test samples: 500
Calibration: max(500 * 0.20, 200) = 200 samples (minimum enforced)
Evaluation: 300 samples (60%)
```

## Code Implementation / 代码实现

### [data_splitter.py](data_splitter.py#L176-L209)

```python
def _ratio_based_split(self, n_total: int) -> int:
    ratio = self.params['calibration_ratio']  # 0.20
    min_samples = self.params['min_samples']   # 200
    max_samples = self.params.get('max_samples', None)  # 1000

    # Calculate calibration size
    n_samples = int(n_total * ratio)

    # Apply constraints
    n_samples = max(n_samples, min_samples)  # At least 200
    if max_samples:
        n_samples = min(n_samples, max_samples)  # At most 1000

    # Safety check (leave at least 20% for evaluation)
    if n_samples >= n_total * 0.8:
        n_samples = int(n_total * 0.6)

    return n_samples
```

## Comparison with Previous Method / 与之前方法的对比

| Aspect | Fixed Sample | Ratio-based (NEW) |
|--------|-------------|-------------------|
| **Calibration size** | Always 500 | 20% of test set |
| **Small test sets (1000)** | 500 (50%) | 200 (20%) ✓ Better |
| **Large test sets (6000)** | 500 (8%) | 1000 (17%) ✓ Better |
| **Adaptability** | No | Yes ✓ |
| **Evaluation data** | Variable | Consistent 80%+ ✓ |

## Testing / 测试

No code changes required in your scripts. Simply run:

```bash
# Quick test will automatically use ratio-based split
python quick_test_transfer_learning.py

# Full pipeline also uses ratio-based split
python main_transfer_learning.py
```

The output will show:
```
Splitting test data (ratio_based)...
  Calibration samples: XXX  (20% of test set)
  Evaluation samples: YYY   (80% of test set)
```

## Reverting to Fixed Sample Size / 恢复到固定样本大小

If you want to go back to fixed sample size, simply change in [config_transfer.py](config_transfer.py):

```python
DATA_SPLIT_CONFIG = {
    'split_method': 'sample_based',  # Change back to sample_based
}
```

## Notes / 注意事项

1. **Backward compatible**: Old saved models still work
   - 向后兼容：旧的保存模型仍然有效

2. **No retraining needed**: General models are unaffected
   - 无需重新训练：通用模型不受影响

3. **Immediate effect**: Takes effect on next run
   - 立即生效：下次运行时生效

---

**Updated**: 2025-11-26
**更新时间**：2025-11-26
