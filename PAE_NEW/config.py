"""
配置文件 - 存放所有可调参数
"""
from pathlib import Path
import os
# ==================== 数据配置 ====================
# 尝试自动定位项目根目录（向上三级）
_repo_root = Path(__file__).resolve().parents[2]

def _collect_vital_files(directory: Path):
    if not directory.exists():
        return []
    # 只在该目录及子目录中查找 .vital 文件，按字母序排序
    files = sorted([str(p) for p in directory.rglob("*.vital")])
    return files

# 优先从仓库根目录下的 records/ 和 testset/ 读取
_records_dir = _repo_root / "records"
_testset_dir = _repo_root / "testset"

_train_files_found = _collect_vital_files(_records_dir)
_test_files_found = _collect_vital_files(_testset_dir)

DATA_CONFIG = {
    # 训练集文件路径列表（可以有多个.vital文件）
    'train_file_paths': 
        _train_files_found if len(_train_files_found) > 0 else [
        'F:/Study/KTH/UPC/PAESAV/VitalParser-main/VitalParser-main/PAE_old/VitalDB_data/VitalDB_data/230718/QUI12_230718_000947.vital',
        'F:/Study/KTH/UPC/PAESAV/VitalParser-main/VitalParser-main/PAE_old/VitalDB_data/VitalDB_data/230718/QUI12_230718_113224.vital',
        'F:/Study/KTH/UPC/PAESAV/VitalParser-main/VitalParser-main/PAE_old/VitalDB_data/VitalDB_data/230718/QUI12_230718_173226.vital',
        'F:/Study/KTH/UPC/PAESAV/VitalParser-main/VitalParser-main/PAE_old/VitalDB_data/VitalDB_data/230718/QUI12_230718_183239.vital',
        'F:/Study/KTH/UPC/PAESAV/VitalParser-main/VitalParser-main/PAE_old/VitalDB_data/VitalDB_data/230718/QUI12_230718_203231.vital',
        'F:/Study/KTH/UPC/PAESAV/VitalParser-main/VitalParser-main/PAE_old/VitalDB_data/VitalDB_data/230718/QUI12_230718_213243.vital',
    # 可以继续添加更多文件...
    ],
    
    # 测试集文件路径（单个文件）
    'test_file_path': _test_files_found[0] if len(_test_files_found) > 0 else
                      'F:/Study/KTH/UPC/PAESAV/VitalParser-main/VitalParser-main/PAE_old/VitalDB_data/VitalDB_data/230807/QUI12_230807_163951.vital',
                          
    # 信号名称
    # 信号名称（支持多个候选，按优先级顺序尝试）
    'pleth_signal_candidates': [
        'Intellivue/PLETH',     # 优先级1
        'Demo/PLETH',           # 优先级2
        'SNUADC/PLETH',         # 优先级3
        # 'Solar8000/PLETH',      # 优先级4
    ],
    
    'art_signal_candidates': [
        'Intellivue/ART',       # 优先级1
        'Intellivue/ABP',       # 优先级2
        'Demo/ART',             # 优先级3
        'Demo/ABP',             # 优先级4
        'SNUADC/ART',           # 优先级5
        'SNUADC/ABP',           # 优先级6
        # 'Solar8000/ART',        # 优先级7
        # 'Solar8000/ABP',        # 优先级8
    ],
    
    # 采样率（Hz）
    'sampling_rate': 100,
    
    # 随机种子
    'random_seed': 42,
}

# ==================== 信号处理配置 ====================
SIGNAL_CONFIG = {
    # Savitzky-Golay滤波参数
    'savgol_window': 51,
    'savgol_polyorder': 3,
    
    # Gaussian滤波参数
    'gaussian_sigma': 5,
    
    # PLETH信号参数
    'peak_prominence': 0.2,
    'peak_distance': 30,
    
    # ART信号参数（尖锐波形）
    'art_peak_prominence': 0.1,
    'art_peak_distance': 20,
    
    # ABP信号参数（平滑波形）← 新增
    'abp_peak_prominence': 0.03,  # 更敏感
    'abp_peak_distance': 20,
    
    # 谷值检测
    'valley_prominence': 0.05,  # ABP谷值也需要调低
}

# ==================== 特征提取配置 ====================
FEATURE_CONFIG = {
    # 是否启用各类特征
    'use_peak_valley_features': True,
    'use_pulse_amplitude': True,
    'use_temporal_features': True,
    'use_cycle_integrals': True,
    'use_waveform_shape': True,
    
    # 特征标准化
    'standardize_features': True,
}

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    # 要训练的模型列表
    'models': [
        'linear_regression',
        'random_forest',
        'gradient_boosting',
        'hist_gradient_boosting',
        'stacking_ensemble',  # 集成模型
    ],
    
    # 模型超参数
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1,
    },
    
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'random_state': 42,
    },
    
    'hist_gradient_boosting': {
        'max_iter': 200,
        'max_depth': 20,
        'learning_rate': 0.1,
        'l2_regularization': 0.1,
        'random_state': 42,
    },
}

# ==================== 评估配置 ====================
EVALUATION_CONFIG = {
    # 输出目录
    'output_dir': 'results',
    
    # 性能指标阈值
    'mae_threshold': 10,
    'mse_threshold_systolic': 70,
    'mse_threshold_diastolic': 60,
    
    # 可视化
    'figure_size': (12, 8),
    'dpi': 100,
    'save_figures': True,
}

# ==================== 运行模式配置 ====================
RUN_CONFIG = {
    # 是否显示详细日志
    'verbose': True,
    
    # 是否在训练过程中显示进度
    'show_progress': True,
    
    # 预测目标
    'targets': ['systolic', 'diastolic'],  # 收缩压和舒张压
}