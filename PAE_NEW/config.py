"""
配置文件 - 存放所有可调参数
"""

# ==================== 数据配置 ====================
DATA_CONFIG = {
    # 训练集文件路径列表（可以有多个.vital文件）
    'train_file_paths': [
        'F:/Study/KTH/UPC/PAESAV/VitalParser-main/VitalParser-main/PAE_old/VitalDB_data/VitalDB_data/230718/QUI12_230718_000947.vital',  # ← 训练文件1
        'F:/Study/KTH/UPC/PAESAV/VitalParser-main/VitalParser-main/PAE_old/VitalDB_data/VitalDB_data/230718/QUI12_230718_113224.vital',  # ← 训练文件2
        'F:/Study/KTH/UPC/PAESAV/VitalParser-main/VitalParser-main/PAE_old/VitalDB_data/VitalDB_data/230718/QUI12_230718_173226.vital',  # ← 训练文件3
        'F:/Study/KTH/UPC/PAESAV/VitalParser-main/VitalParser-main/PAE_old/VitalDB_data/VitalDB_data/230718/QUI12_230718_183239.vital',  # ← 训练文件4
        'F:/Study/KTH/UPC/PAESAV/VitalParser-main/VitalParser-main/PAE_old/VitalDB_data/VitalDB_data/230718/QUI12_230718_203231.vital',  # ← 训练文件5
        'F:/Study/KTH/UPC/PAESAV/VitalParser-main/VitalParser-main/PAE_old/VitalDB_data/VitalDB_data/230718/QUI12_230718_213243.vital',  # ← 训练文件6
        # 可以继续添加更多文件...
    ],
    
    # 测试集文件路径（单个文件）
    'test_file_path': 'F:/Study/KTH/UPC/PAESAV/VitalParser-main/VitalParser-main/PAE_old/VitalDB_data/VitalDB_data/230807/QUI12_230807_163951.vital',  # ← 测试文件
    
    # 信号名称
    'pleth_signal': 'Demo/PLETH',
    'art_signal': 'Demo/ART',
    
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
    
    # 峰值检测参数
    'peak_prominence': 0.1,  # 相对突出度
    'peak_distance': 20,     # 最小距离（采样点）
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