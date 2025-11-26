"""
Patient Data Splitter for Transfer Learning
用于迁移学习的患者数据分割器

This module splits test patient data into calibration set and evaluation set.
该模块将测试患者数据分割为校准集和评估集。
"""

import numpy as np
from typing import Tuple, Dict, Optional
import warnings


class PatientDataSplitter:
    """
    Split patient data into calibration and evaluation sets.
    将患者数据分割为校准集和评估集。
    """

    def __init__(self, split_method='sample_based', **kwargs):
        """
        Initialize the data splitter.
        初始化数据分割器。

        Args:
            split_method: 'sample_based', 'time_based', or 'ratio_based'
                         分割方法：基于样本、基于时间或基于比例
            **kwargs: Additional parameters for the split method
                     分割方法的附加参数
        """
        self.split_method = split_method
        self.params = kwargs

        # Default parameters / 默认参数
        self._set_default_params()

    def _set_default_params(self):
        """Set default parameters based on split method / 根据分割方法设置默认参数"""
        if self.split_method == 'sample_based':
            self.params.setdefault('n_samples', 200)
            self.params.setdefault('min_samples', 100)
            self.params.setdefault('max_samples', 300)

        elif self.split_method == 'time_based':
            self.params.setdefault('duration_minutes', 5)
            self.params.setdefault('sampling_rate', 100)
            self.params.setdefault('expected_heart_rate', 60)

        elif self.split_method == 'ratio_based':
            self.params.setdefault('calibration_ratio', 0.25)
            self.params.setdefault('min_samples', 100)

        # Quality control / 质量控制
        self.params.setdefault('check_continuity', True)
        self.params.setdefault('remove_outliers', False)

    def split(self, X_test: np.ndarray, y_test_sys: np.ndarray,
              y_test_dia: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray, np.ndarray]:
        """
        Split test data into calibration and evaluation sets.
        将测试数据分割为校准集和评估集。

        Args:
            X_test: Feature matrix (n_samples, n_features) / 特征矩阵
            y_test_sys: Systolic BP labels / 收缩压标签
            y_test_dia: Diastolic BP labels / 舒张压标签

        Returns:
            Tuple of (X_calib, y_calib_sys, y_calib_dia, X_eval, y_eval_sys, y_eval_dia)
            校准集和评估集的元组
        """
        # Get split index / 获取分割索引
        split_idx = self._calculate_split_index(X_test, y_test_sys, y_test_dia)

        # Validate split / 验证分割
        self._validate_split(split_idx, len(X_test))

        # Split data / 分割数据
        X_calib = X_test[:split_idx]
        y_calib_sys = y_test_sys[:split_idx]
        y_calib_dia = y_test_dia[:split_idx]

        X_eval = X_test[split_idx:]
        y_eval_sys = y_test_sys[split_idx:]
        y_eval_dia = y_test_dia[split_idx:]

        # Quality control / 质量控制
        if self.params['remove_outliers']:
            X_calib, y_calib_sys, y_calib_dia = self._remove_outliers(
                X_calib, y_calib_sys, y_calib_dia
            )

        # Print split summary / 打印分割摘要
        self._print_split_summary(split_idx, X_calib, X_eval)

        return X_calib, y_calib_sys, y_calib_dia, X_eval, y_eval_sys, y_eval_dia

    def _calculate_split_index(self, X_test: np.ndarray,
                                y_test_sys: np.ndarray,
                                y_test_dia: np.ndarray) -> int:
        """
        Calculate the split index based on the split method.
        根据分割方法计算分割索引。

        Args:
            X_test: Feature matrix / 特征矩阵
            y_test_sys: Systolic BP / 收缩压
            y_test_dia: Diastolic BP / 舒张压

        Returns:
            Split index / 分割索引
        """
        n_total = len(X_test)

        if self.split_method == 'sample_based':
            split_idx = self._sample_based_split(n_total)

        elif self.split_method == 'time_based':
            split_idx = self._time_based_split()

        elif self.split_method == 'ratio_based':
            split_idx = self._ratio_based_split(n_total)

        else:
            raise ValueError(f"Unknown split method: {self.split_method}")

        return split_idx

    def _sample_based_split(self, n_total: int) -> int:
        """
        Sample-based split: Use fixed number of samples for calibration.
        基于样本的分割：使用固定数量的样本进行校准。

        Args:
            n_total: Total number of samples / 样本总数

        Returns:
            Split index / 分割索引
        """
        n_samples = self.params['n_samples']
        min_samples = self.params['min_samples']
        max_samples = self.params['max_samples']

        # Ensure within bounds / 确保在范围内
        n_samples = max(min_samples, min(n_samples, max_samples))

        # Ensure we have enough evaluation data / 确保有足够的评估数据
        if n_samples >= n_total * 0.8:
            warnings.warn(
                f"Calibration samples ({n_samples}) too large for total samples ({n_total}). "
                f"Reducing to 60% of total.",
                UserWarning
            )
            n_samples = int(n_total * 0.6)

        return n_samples

    def _time_based_split(self) -> int:
        """
        Time-based split: Use data from first N minutes.
        基于时间的分割：使用前N分钟的数据。

        Returns:
            Split index (estimated number of heartbeats) / 分割索引（估计的心跳数）
        """
        duration_min = self.params['duration_minutes']
        heart_rate = self.params['expected_heart_rate']

        # Estimate number of heartbeats / 估计心跳数
        # heartbeats = duration (min) * heart_rate (beats/min)
        n_samples = int(duration_min * heart_rate)

        return n_samples

    def _ratio_based_split(self, n_total: int) -> int:
        """
        Ratio-based split: Use a percentage of data for calibration.
        基于比例的分割：使用一定百分比的数据进行校准。

        Args:
            n_total: Total number of samples / 样本总数

        Returns:
            Split index / 分割索引
        """
        ratio = self.params['calibration_ratio']
        min_samples = self.params['min_samples']
        max_samples = self.params.get('max_samples', None)  # Optional max limit / 可选最大限制

        # Calculate calibration size / 计算校准集大小
        n_samples = int(n_total * ratio)

        # Ensure minimum / 确保最小值
        n_samples = max(n_samples, min_samples)

        # Apply maximum limit if specified / 如果指定则应用最大限制
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)

        # Ensure not too large (leave at least 20% for evaluation) / 确保不太大（至少留20%用于评估）
        if n_samples >= n_total * 0.8:
            warnings.warn(
                f"Calibration size too large ({n_samples}). Reducing to 60% of total ({int(n_total * 0.6)}).",
                UserWarning
            )
            n_samples = int(n_total * 0.6)

        return n_samples

    def _validate_split(self, split_idx: int, n_total: int):
        """
        Validate the split index.
        验证分割索引。

        Args:
            split_idx: Split index / 分割索引
            n_total: Total number of samples / 样本总数

        Raises:
            ValueError if split is invalid / 如果分割无效则引发ValueError
        """
        if split_idx <= 0:
            raise ValueError(f"Calibration set is empty (split_idx={split_idx})")

        if split_idx >= n_total:
            raise ValueError(
                f"Split index ({split_idx}) exceeds total samples ({n_total}). "
                "No evaluation data available."
            )

        # Check minimum evaluation size / 检查最小评估集大小
        n_eval = n_total - split_idx
        if n_eval < 50:
            warnings.warn(
                f"Evaluation set is very small ({n_eval} samples). "
                "Results may not be reliable.",
                UserWarning
            )

    def _remove_outliers(self, X: np.ndarray, y_sys: np.ndarray,
                         y_dia: np.ndarray, threshold: float = 3.0
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove outliers from calibration data using Z-score method.
        使用Z分数方法从校准数据中移除异常值。

        Args:
            X: Feature matrix / 特征矩阵
            y_sys: Systolic BP / 收缩压
            y_dia: Diastolic BP / 舒张压
            threshold: Z-score threshold / Z分数阈值

        Returns:
            Cleaned data / 清理后的数据
        """
        # Calculate Z-scores for BP values / 计算血压值的Z分数
        z_sys = np.abs((y_sys - np.mean(y_sys)) / np.std(y_sys))
        z_dia = np.abs((y_dia - np.mean(y_dia)) / np.std(y_dia))

        # Keep only inliers / 只保留非异常值
        mask = (z_sys < threshold) & (z_dia < threshold)

        n_removed = len(X) - np.sum(mask)
        if n_removed > 0:
            print(f"Removed {n_removed} outliers from calibration set "
                  f"({n_removed / len(X) * 100:.1f}%)")

        return X[mask], y_sys[mask], y_dia[mask]

    def _print_split_summary(self, split_idx: int, X_calib: np.ndarray, X_eval: np.ndarray):
        """
        Print summary of data split.
        打印数据分割摘要。

        Args:
            split_idx: Split index / 分割索引
            X_calib: Calibration features / 校准集特征
            X_eval: Evaluation features / 评估集特征
        """
        n_calib = len(X_calib)
        n_eval = len(X_eval)
        n_total = n_calib + n_eval

        print("\n" + "=" * 60)
        print("Data Split Summary / 数据分割摘要")
        print("=" * 60)
        print(f"Split Method: {self.split_method}")
        print(f"Total Samples: {n_total}")
        print(f"Calibration Set: {n_calib} samples ({n_calib / n_total * 100:.1f}%)")
        print(f"Evaluation Set: {n_eval} samples ({n_eval / n_total * 100:.1f}%)")
        print(f"Split Index: {split_idx}")
        print("=" * 60 + "\n")

    def get_split_info(self, X_test: np.ndarray) -> Dict[str, any]:
        """
        Get information about the split without actually splitting.
        获取分割信息而不实际分割数据。

        Args:
            X_test: Feature matrix / 特征矩阵

        Returns:
            Dictionary with split information / 包含分割信息的字典
        """
        n_total = len(X_test)
        split_idx = self._calculate_split_index(X_test, None, None)

        return {
            'split_method': self.split_method,
            'total_samples': n_total,
            'calibration_samples': split_idx,
            'evaluation_samples': n_total - split_idx,
            'calibration_ratio': split_idx / n_total,
            'split_index': split_idx
        }


class MultiSizeSplitter:
    """
    Create multiple splits with different calibration sizes.
    创建具有不同校准集大小的多个分割。

    Useful for experiments to find optimal calibration size.
    用于实验以找到最佳校准集大小。
    """

    def __init__(self, calibration_sizes: list):
        """
        Initialize multi-size splitter.
        初始化多尺寸分割器。

        Args:
            calibration_sizes: List of calibration sizes to test
                              要测试的校准集大小列表
        """
        self.calibration_sizes = sorted(calibration_sizes)

    def split_all(self, X_test: np.ndarray, y_test_sys: np.ndarray,
                  y_test_dia: np.ndarray) -> Dict[int, Tuple]:
        """
        Create splits for all calibration sizes.
        为所有校准集大小创建分割。

        Args:
            X_test: Feature matrix / 特征矩阵
            y_test_sys: Systolic BP / 收缩压
            y_test_dia: Diastolic BP / 舒张压

        Returns:
            Dictionary mapping calibration size to split data
            将校准集大小映射到分割数据的字典
        """
        splits = {}

        for cal_size in self.calibration_sizes:
            if cal_size >= len(X_test):
                warnings.warn(
                    f"Calibration size {cal_size} exceeds total samples {len(X_test)}. Skipping.",
                    UserWarning
                )
                continue

            # Create splitter / 创建分割器
            splitter = PatientDataSplitter(split_method='sample_based', n_samples=cal_size)

            # Split data / 分割数据
            split_data = splitter.split(X_test, y_test_sys, y_test_dia)

            splits[cal_size] = split_data

        return splits


# ============================================================================
# Utility Functions / 工具函数
# ============================================================================

def validate_split_data(X_calib: np.ndarray, y_calib_sys: np.ndarray,
                        y_calib_dia: np.ndarray, X_eval: np.ndarray,
                        y_eval_sys: np.ndarray, y_eval_dia: np.ndarray) -> bool:
    """
    Validate that split data is properly formatted.
    验证分割数据格式正确。

    Args:
        X_calib, y_calib_sys, y_calib_dia: Calibration data / 校准数据
        X_eval, y_eval_sys, y_eval_dia: Evaluation data / 评估数据

    Returns:
        True if valid / 如果有效则返回True

    Raises:
        ValueError if invalid / 如果无效则引发ValueError
    """
    # Check shapes / 检查形状
    if len(X_calib) != len(y_calib_sys) or len(X_calib) != len(y_calib_dia):
        raise ValueError("Calibration data shapes don't match")

    if len(X_eval) != len(y_eval_sys) or len(X_eval) != len(y_eval_dia):
        raise ValueError("Evaluation data shapes don't match")

    # Check for NaN / 检查NaN
    if np.any(np.isnan(X_calib)) or np.any(np.isnan(X_eval)):
        warnings.warn("NaN values detected in features", UserWarning)

    if np.any(np.isnan(y_calib_sys)) or np.any(np.isnan(y_eval_sys)):
        warnings.warn("NaN values detected in systolic BP", UserWarning)

    if np.any(np.isnan(y_calib_dia)) or np.any(np.isnan(y_eval_dia)):
        warnings.warn("NaN values detected in diastolic BP", UserWarning)

    # Check minimum sizes / 检查最小尺寸
    if len(X_calib) < 50:
        warnings.warn(
            f"Calibration set is very small ({len(X_calib)} samples). "
            "Fine-tuning may be unstable.",
            UserWarning
        )

    if len(X_eval) < 50:
        warnings.warn(
            f"Evaluation set is very small ({len(X_eval)} samples). "
            "Evaluation may not be reliable.",
            UserWarning
        )

    print("Data validation passed.")
    return True


# ============================================================================
# Testing / 测试
# ============================================================================

if __name__ == '__main__':
    # Test data splitter / 测试数据分割器
    print("Testing PatientDataSplitter...\n")

    # Create dummy data / 创建虚拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 17

    X_test = np.random.randn(n_samples, n_features)
    y_test_sys = np.random.uniform(90, 180, n_samples)
    y_test_dia = np.random.uniform(60, 110, n_samples)

    # Test sample-based split / 测试基于样本的分割
    print("1. Testing sample-based split:")
    splitter1 = PatientDataSplitter(split_method='sample_based', n_samples=200)
    X_calib, y_calib_sys, y_calib_dia, X_eval, y_eval_sys, y_eval_dia = splitter1.split(
        X_test, y_test_sys, y_test_dia
    )

    # Test time-based split / 测试基于时间的分割
    print("\n2. Testing time-based split:")
    splitter2 = PatientDataSplitter(split_method='time_based', duration_minutes=5)
    split2 = splitter2.split(X_test, y_test_sys, y_test_dia)

    # Test ratio-based split / 测试基于比例的分割
    print("\n3. Testing ratio-based split:")
    splitter3 = PatientDataSplitter(split_method='ratio_based', calibration_ratio=0.25)
    split3 = splitter3.split(X_test, y_test_sys, y_test_dia)

    # Validate split data / 验证分割数据
    print("\n4. Validating split data:")
    validate_split_data(*split3)

    # Test multi-size splitter / 测试多尺寸分割器
    print("\n5. Testing MultiSizeSplitter:")
    multi_splitter = MultiSizeSplitter(calibration_sizes=[100, 200, 300])
    all_splits = multi_splitter.split_all(X_test, y_test_sys, y_test_dia)
    print(f"Created {len(all_splits)} different splits")
    for size, data in all_splits.items():
        print(f"  Size {size}: {len(data[0])} calibration samples")

    print("\nAll tests passed!")
