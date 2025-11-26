"""
Transfer Learning Configuration
迁移学习配置文件

This file contains all configuration parameters for the transfer learning framework.
本文件包含迁移学习框架的所有配置参数。
"""

import os

# ============================================================================
# General Model Configuration / 通用模型配置
# ============================================================================
GENERAL_MODEL_CONFIG = {
    # Model type / 模型类型
    # Options: 'xgboost', 'lightgbm', 'gradient_boosting'
    'model_type': 'xgboost',

    # XGBoost parameters for general model / XGBoost通用模型参数
    # Optimized for better sensitivity to BP variations (Plan G) / 优化以提高对血压变化的敏感性（方案G）
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 12,          # Increased from 10 to capture finer patterns / 从10增加以捕获更细微的模式
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,    # Reduced from 3 to allow finer splits / 从3降低以允许更细的分裂
        'gamma': 0,               # Removed complexity penalty / 移除复杂度惩罚
        'reg_alpha': 0.005,       # Reduced L1 regularization from 0.01 / 从0.01降低L1正则化
        'reg_lambda': 0.3,        # Significantly reduced L2 from 1.0 for wider prediction range / 从1.0大幅降低L2以获得更宽的预测范围
        'max_delta_step': 2,      # Allow larger prediction steps / 允许更大的预测步长
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1
    },

    # LightGBM parameters for general model / LightGBM通用模型参数
    'lightgbm': {
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.01,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    },

    # Sklearn Gradient Boosting parameters / Sklearn梯度提升参数
    'gradient_boosting': {
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'min_samples_split': 5,
        'min_samples_leaf': 3,
        'random_state': 42
    }
}

# ============================================================================
# Fine-tuning Configuration / 微调配置
# ============================================================================
FINE_TUNING_CONFIG = {
    # Fine-tuning strategy / 微调策略
    # Options: 'incremental', 'correction_model'
    # Changed to correction_model for better variation tracking / 改为校正模型以更好地跟踪变化
    'strategy': 'correction_model',

    # XGBoost fine-tuning parameters / XGBoost微调参数
    # Optimized for aggressive correction of extreme values (Plan G) / 优化以积极校正极端值（方案G）
    'xgboost': {
        'n_estimators': 200,      # Number of new trees to add / 增量添加的树数量
        'learning_rate': 0.1,     # Increased from 0.05 for more aggressive learning / 从0.05提高以更积极地学习
        'max_depth': 10,          # Reduced from 15 to prevent overfitting on small calib set / 从15降低以防止在小校准集上过拟合
        'subsample': 0.9,         # Slight subsampling / 轻微子采样
        'colsample_bytree': 0.8,
        'min_child_weight': 1,    # Allow finer splits / 允许更细的分裂
        'gamma': 0,               # Removed to allow correction model more freedom / 移除以给校正模型更多自由
        'reg_alpha': 0,           # No L1 regularization for correction model / 校正模型不使用L1正则化
        'reg_lambda': 0.1,        # Minimal L2 regularization from 0.5 / 从0.5降低L2正则化至最小
        'max_delta_step': 3,      # Allow larger correction steps / 允许更大的校正步长
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1
    },

    # LightGBM fine-tuning parameters / LightGBM微调参数
    'lightgbm': {
        'n_estimators': 100,
        'learning_rate': 0.01,
        'max_depth': 10,
        'subsample': 1.0,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.05,
        'reg_lambda': 2.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    },

    # Early stopping configuration / Early stopping配置
    'early_stopping': {
        'enabled': True,
        'rounds': 20,                    # Stop if no improvement for N rounds / N轮无改善则停止
        'validation_fraction': 0.2       # Use 20% of calibration data for validation / 使用20%校准数据作为验证集
    },

    # Overfitting prevention / 过拟合防止
    'max_iterations': 150,  # Maximum number of fine-tuning iterations / 最大微调迭代次数
}

# ============================================================================
# Data Splitting Configuration / 数据分割配置
# ============================================================================
DATA_SPLIT_CONFIG = {
    # Calibration data selection method / 校准数据选择方法
    # Options: 'sample_based', 'time_based', 'ratio_based'
    'split_method': 'sample_based',

    # Sample-based split / 基于样本的分割
    # Reduced to prevent over-smoothing (Plan G) / 减少以防止过度平滑（方案G）
    'sample_based': {
        'n_samples': 500,  # Reduced from 600 to avoid over-smoothing / 从600减少以避免过度平滑
        'min_samples': 300,  # Minimum required samples / 最小所需样本数
        'max_samples': 900   # Maximum calibration samples / 最大校准样本数
    },

    # Time-based split / 基于时间的分割
    'time_based': {
        'duration_minutes': 5,      # Duration in minutes / 持续时间(分钟)
        'sampling_rate': 100,       # Hz
        'expected_heart_rate': 60   # bpm (for estimation / 用于估算)
    },

    # Ratio-based split / 基于比例的分割
    'ratio_based': {
        'calibration_ratio': 0.25,  # 25% for calibration / 25%用于校准
        'min_samples': 100          # Minimum calibration samples / 最小校准样本数
    },

    # Quality control / 质量控制
    'quality_control': {
        'check_continuity': True,    # Ensure calibration data is continuous / 确保校准数据连续
        'remove_outliers': False     # Remove outliers from calibration set / 从校准集中移除异常值
    }
}

# ============================================================================
# Evaluation Configuration / 评估配置
# ============================================================================
EVALUATION_CONFIG = {
    # Comparison mode / 对比模式
    'compare_models': True,  # Compare general vs personalized / 对比通用模型vs个性化模型

    # Metrics to compute / 要计算的指标
    'metrics': ['MAE', 'MSE', 'RMSE', 'R2'],

    # Performance targets / 性能目标
    'targets': {
        'systolic': {'MAE': 10, 'MSE': 70},
        'diastolic': {'MAE': 10, 'MSE': 60}
    },

    # Visualization / 可视化
    'visualize': {
        'time_series_comparison': True,   # Plot general vs personalized predictions / 绘制通用vs个性化预测
        'scatter_plot': True,              # Scatter plot for both models / 两种模型的散点图
        'error_distribution': True,        # Error distribution comparison / 误差分布对比
        'improvement_bar_chart': True,     # Show improvement percentage / 显示改善百分比
        'max_points_display': 500          # Maximum points to display in time series / 时间序列中显示的最大点数
    },

    # Output / 输出
    'save_predictions': True,   # Save predictions to CSV / 保存预测到CSV
    'generate_report': True     # Generate text report / 生成文本报告
}

# ============================================================================
# Path Configuration / 路径配置
# ============================================================================
PATH_CONFIG = {
    # Output directory for transfer learning results / 迁移学习结果输出目录
    'output_dir': 'results_transfer_learning',

    # Model save directory / 模型保存目录
    'model_dir': 'saved_models',

    # Subdirectories / 子目录
    'general_model_dir': 'general_models',      # General models / 通用模型
    'personalized_model_dir': 'personalized_models',  # Personalized models / 个性化模型
    'plots_dir': 'plots',                       # Visualization plots / 可视化图表
    'reports_dir': 'reports',                   # Text reports / 文本报告
    'predictions_dir': 'predictions'            # Prediction results / 预测结果
}

# ============================================================================
# Experiment Configuration / 实验配置
# ============================================================================
EXPERIMENT_CONFIG = {
    # Random seed for reproducibility / 可重复性的随机种子
    'random_seed': 42,

    # Logging / 日志
    'verbose': True,            # Print detailed progress / 打印详细进度
    'log_level': 'INFO',        # Logging level / 日志级别

    # Multiple calibration sizes experiment / 多校准大小实验
    'calibration_size_experiment': {
        'enabled': False,        # Run experiment with different calibration sizes / 使用不同校准大小运行实验
        'sizes': [100, 150, 200, 250, 300]  # Different calibration sizes to test / 要测试的不同校准大小
    },

    # Data augmentation / 数据增强
    'data_augmentation': {
        'enabled': False,        # Enable data augmentation for small calibration sets / 为小校准集启用数据增强
        'noise_level': 0.01,     # Gaussian noise std / 高斯噪声标准差
        'augmentation_factor': 2  # How many times to augment / 增强倍数
    }
}

# ============================================================================
# Helper Functions / 辅助函数
# ============================================================================

def get_output_path(test_file_name, subdir=None, suffix=''):
    """
    Generate output path for a specific test file.
    为特定测试文件生成输出路径。

    Args:
        test_file_name: Name of test file / 测试文件名
        subdir: Subdirectory name (plots, reports, etc.) / 子目录名
        suffix: Optional suffix to add to base name (e.g., '_quicktest') / 可选后缀（如'_quicktest'）

    Returns:
        Full output path / 完整输出路径
    """
    # Extract filename without extension / 提取不含扩展名的文件名
    base_name = os.path.splitext(os.path.basename(test_file_name))[0]

    # Add suffix if provided / 如果提供了后缀则添加
    if suffix:
        base_name = base_name + suffix

    # Build path / 构建路径
    path = os.path.join(PATH_CONFIG['output_dir'], base_name)

    if subdir:
        path = os.path.join(path, subdir)

    return path


def get_model_save_path(model_name, test_file_name=None):
    """
    Generate path for saving models.
    生成保存模型的路径。

    Args:
        model_name: 'general' or 'personalized' / '通用'或'个性化'
        test_file_name: For personalized models / 用于个性化模型

    Returns:
        Model save path / 模型保存路径
    """
    if model_name == 'general':
        path = os.path.join(PATH_CONFIG['model_dir'], PATH_CONFIG['general_model_dir'])
    else:
        base_name = os.path.splitext(os.path.basename(test_file_name))[0]
        path = os.path.join(PATH_CONFIG['model_dir'], PATH_CONFIG['personalized_model_dir'], base_name)

    return path


def create_output_directories(test_file_name, suffix=''):
    """
    Create all necessary output directories.
    创建所有必要的输出目录。

    Args:
        test_file_name: Name of test file / 测试文件名
        suffix: Optional suffix to add to directory name (e.g., '_quicktest') / 可选后缀（如'_quicktest'）
    """
    # Main output directory / 主输出目录
    base_path = get_output_path(test_file_name, suffix=suffix)
    os.makedirs(base_path, exist_ok=True)

    # Subdirectories / 子目录
    for subdir in [PATH_CONFIG['plots_dir'], PATH_CONFIG['reports_dir'], PATH_CONFIG['predictions_dir']]:
        os.makedirs(os.path.join(base_path, subdir), exist_ok=True)

    # Model directories / 模型目录
    os.makedirs(get_model_save_path('general'), exist_ok=True)
    os.makedirs(get_model_save_path('personalized', test_file_name), exist_ok=True)

    display_name = os.path.splitext(os.path.basename(test_file_name))[0] + suffix
    print(f"Created output directories for: {display_name}")


# ============================================================================
# Configuration Validation / 配置验证
# ============================================================================

def validate_config():
    """
    Validate configuration parameters.
    验证配置参数。
    """
    # Check model type / 检查模型类型
    assert GENERAL_MODEL_CONFIG['model_type'] in ['xgboost', 'lightgbm', 'gradient_boosting'], \
        "Invalid model type"

    # Check split method / 检查分割方法
    assert DATA_SPLIT_CONFIG['split_method'] in ['sample_based', 'time_based', 'ratio_based'], \
        "Invalid split method"

    # Check fine-tuning strategy / 检查微调策略
    assert FINE_TUNING_CONFIG['strategy'] in ['incremental', 'correction_model'], \
        "Invalid fine-tuning strategy"

    print("Configuration validation passed.")
    return True


if __name__ == '__main__':
    # Test configuration / 测试配置
    validate_config()
    print("\n=== Transfer Learning Configuration ===")
    print(f"Model Type: {GENERAL_MODEL_CONFIG['model_type']}")
    print(f"Split Method: {DATA_SPLIT_CONFIG['split_method']}")
    print(f"Fine-tuning Strategy: {FINE_TUNING_CONFIG['strategy']}")
    print(f"Output Directory: {PATH_CONFIG['output_dir']}")
