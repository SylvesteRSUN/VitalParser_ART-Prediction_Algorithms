"""
feature_extraction.py
增强版特征提取 - 添加有效信号片段检测

核心改进:
1. 检测并提取有效信号片段
2. 只在高质量片段上提取特征
3. 过滤异常值和不合理的血压值
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from config import FEATURE_CONFIG, DATA_CONFIG
from collections import deque


def detect_valid_segments(signal, segment_duration=60, sample_rate=100, 
                          nan_threshold=0.1, std_threshold_percentile=5):
    """
    检测信号中的有效片段
    
    参数:
        signal: 原始信号数组
        segment_duration: 片段时长（秒）
        sample_rate: 采样率（Hz）
        nan_threshold: NaN值比例阈值（超过此比例则认为无效）
        std_threshold_percentile: 标准差百分位阈值（低于此百分位的片段被认为是"平坦"的）
    
    返回:
        valid_segments: 有效片段的列表，每个元素是(start_idx, end_idx)
    """
    segment_length = int(segment_duration * sample_rate)
    n_segments = len(signal) // segment_length
    
    valid_segments = []
    segment_stats = []
    
    print(f"\n检测有效信号片段...")
    print(f"  总信号长度: {len(signal)} 点 ({len(signal)/sample_rate:.1f} 秒)")
    print(f"  片段长度: {segment_length} 点 ({segment_duration} 秒)")
    print(f"  片段数量: {n_segments}")
    
    for i in range(n_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segment = signal[start_idx:end_idx]
        
        # 检查1: NaN值比例
        nan_ratio = np.isnan(segment).sum() / len(segment)
        
        # 检查2: 信号标准差（检测是否是"平坦"信号）
        valid_values = segment[~np.isnan(segment)]
        if len(valid_values) > 0:
            std = np.std(valid_values)
            mean = np.mean(valid_values)
        else:
            std = 0
            mean = 0
        
        # 检查3: 信号范围是否合理（针对ART/PLETH）
        if len(valid_values) > 0:
            signal_range = np.max(valid_values) - np.min(valid_values)
        else:
            signal_range = 0
        
        segment_stats.append({
            'start': start_idx,
            'end': end_idx,
            'nan_ratio': nan_ratio,
            'std': std,
            'mean': mean,
            'range': signal_range
        })
    
    # 计算全局标准差阈值（使用百分位数）
    all_stds = [s['std'] for s in segment_stats if s['nan_ratio'] < nan_threshold]
    if len(all_stds) > 0:
        std_threshold = np.percentile(all_stds, std_threshold_percentile)
    else:
        std_threshold = 0
    
    # 筛选有效片段
    for stats in segment_stats:
        is_valid = (
            stats['nan_ratio'] < nan_threshold and  # NaN比例低
            stats['std'] > std_threshold and         # 标准差足够大（有变化）
            stats['range'] > 0                       # 有信号变化范围
        )
        
        if is_valid:
            valid_segments.append((stats['start'], stats['end']))
    
    print(f"  有效片段: {len(valid_segments)}/{n_segments} ({len(valid_segments)/n_segments*100:.1f}%)")
    
    if len(valid_segments) > 0:
        total_valid_duration = len(valid_segments) * segment_duration
        print(f"  有效信号总时长: {total_valid_duration:.1f} 秒")
    else:
        print(f"  ⚠ 警告: 未检测到任何有效片段！")
    
    return valid_segments


def extract_valid_signal_portions(signal, valid_segments):
    """
    从原始信号中提取有效片段并拼接
    
    参数:
        signal: 原始信号数组
        valid_segments: 有效片段列表[(start, end), ...]
    
    返回:
        valid_signal: 拼接后的有效信号
        segment_boundaries: 每个片段在拼接后信号中的起止位置
    """
    if len(valid_segments) == 0:
        return np.array([]), []
    
    valid_portions = []
    segment_boundaries = []
    current_pos = 0
    
    for start, end in valid_segments:
        portion = signal[start:end]
        valid_portions.append(portion)
        
        segment_boundaries.append((current_pos, current_pos + len(portion)))
        current_pos += len(portion)
    
    valid_signal = np.concatenate(valid_portions)
    
    print(f"\n  提取的有效信号:")
    print(f"    原始长度: {len(signal)} 点")
    print(f"    有效长度: {len(valid_signal)} 点 ({len(valid_signal)/len(signal)*100:.1f}%)")
    
    return valid_signal, segment_boundaries


class CycleBasedFeatureExtractor:
    """
    基于心跳周期的特征提取器（增强版 - 带时序特征和有效片段检测）
    Enhanced cycle-based feature extractor with temporal features
    """

    def __init__(self, enable_segment_detection=True, enable_temporal_features=True):
        """
        初始化特征提取器
        Initialize feature extractor

        参数 / Args:
            enable_segment_detection: 是否启用有效片段检测 / Enable valid segment detection
            enable_temporal_features: 是否启用时序特征 / Enable temporal features
        """
        self.scaler = StandardScaler() if FEATURE_CONFIG['standardize_features'] else None
        self.feature_names = []
        self.enable_segment_detection = enable_segment_detection
        self.enable_temporal_features = enable_temporal_features
        self.sample_rate = DATA_CONFIG['sampling_rate']

        # Temporal feature parameters / 时序特征参数
        self.lag_steps = 3  # Number of previous cycles to use / 使用的历史周期数
        self.window_size = 5  # Sliding window size / 滑动窗口大小

        # Feature history buffer / 特征历史缓冲区
        self.feature_history = deque(maxlen=max(self.lag_steps, self.window_size))

        # BP state-aware feature parameters (Plan F) / 血压状态感知特征参数（方案F）
        self.bp_history = deque(maxlen=10)  # Store recent BP values / 存储最近的血压值
        self.enable_bp_state_features = True  # Enable BP state features / 启用血压状态特征
    
    def extract_single_cycle_features(self, cycle_signal, cycle_peaks, cycle_valleys):
        """
        从单个心跳周期提取特征
        
        参数:
            cycle_signal: 一个周期的信号数组
            cycle_peaks: 周期内峰值的相对位置
            cycle_valleys: 周期内谷值的相对位置
        
        返回:
            features: 特征向量
        """
        features = []
        
        # === 基础统计特征 (4维) ===
        features.append(np.mean(cycle_signal))
        features.append(np.std(cycle_signal))
        features.append(np.max(cycle_signal))
        features.append(np.min(cycle_signal))
        
        # === 峰值特征 (4维) ===
        if len(cycle_peaks) > 0:
            peak_idx = cycle_peaks[0]
            peak_value = cycle_signal[peak_idx]
            peak_position = peak_idx / len(cycle_signal)
            
            features.append(peak_value)
            features.append(peak_position)
            
            # 峰值宽度
            half_max = (peak_value + np.min(cycle_signal)) / 2
            above_half = cycle_signal > half_max
            peak_width = np.sum(above_half)
            features.append(peak_width)
            
            # 峰值斜率
            if peak_idx > 0:
                rise_slope = (cycle_signal[peak_idx] - cycle_signal[0]) / peak_idx
                features.append(rise_slope)
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0, 0])
        
        # === 谷值特征 (2维) ===
        if len(cycle_valleys) > 0:
            valley_idx = cycle_valleys[0]
            valley_value = cycle_signal[valley_idx]
            valley_position = valley_idx / len(cycle_signal)
            
            features.append(valley_value)
            features.append(valley_position)
        else:
            features.extend([0, 0])
        
        # === 脉搏幅度特征 (3维) ===
        if len(cycle_peaks) > 0 and len(cycle_valleys) > 0:
            pulse_amplitude = cycle_signal[cycle_peaks[0]] - cycle_signal[cycle_valleys[0]]
            features.append(pulse_amplitude)
            
            pulse_pressure_ratio = pulse_amplitude / (np.mean(cycle_signal) + 1e-8)
            features.append(pulse_pressure_ratio)
            
            if cycle_peaks[0] > 0:
                asymmetry = cycle_peaks[0] / len(cycle_signal)
                features.append(asymmetry)
            else:
                features.append(0.5)
        else:
            features.extend([0, 0, 0.5])
        
        # === 积分特征 (2维) ===
        integral = np.trapz(cycle_signal)
        features.append(integral)
        features.append(integral / len(cycle_signal))
        
        # === 形状特征 (2维) ===
        mean_val = np.mean(cycle_signal)
        std_val = np.std(cycle_signal) + 1e-8
        skewness = np.mean(((cycle_signal - mean_val) / std_val) ** 3)
        features.append(skewness)
        
        kurtosis = np.mean(((cycle_signal - mean_val) / std_val) ** 4)
        features.append(kurtosis)

        # === Enhanced features (Plan D) / 增强特征 (方案D) ===

        # Derivative features (4维) / 导数特征
        first_derivative = np.diff(cycle_signal)
        if len(first_derivative) > 0:
            features.append(np.max(first_derivative))  # Max rise rate / 最大上升速率
            features.append(np.min(first_derivative))  # Max fall rate / 最大下降速率
            features.append(np.std(first_derivative))  # Derivative variability / 导数变异性
        else:
            features.extend([0, 0, 0])

        second_derivative = np.diff(first_derivative) if len(first_derivative) > 1 else np.array([0])
        features.append(np.std(second_derivative) if len(second_derivative) > 0 else 0)  # Acceleration variability / 加速度变异性

        # Morphology ratios (3维) / 形态比率
        cycle_len = len(cycle_signal)
        if cycle_len > 4:
            # Split cycle into quarters / 将周期分成四份
            q1 = np.mean(cycle_signal[:cycle_len // 4])
            q2 = np.mean(cycle_signal[cycle_len // 4:cycle_len // 2])
            q3 = np.mean(cycle_signal[cycle_len // 2:3 * cycle_len // 4])
            q4 = np.mean(cycle_signal[3 * cycle_len // 4:])

            features.append(q1 / (q3 + 1e-8))  # Rise vs fall ratio / 上升/下降比率
            features.append((q1 + q2) / (q3 + q4 + 1e-8))  # First half vs second half / 前半/后半比率
            features.append(np.abs(q2 - q3) / (std_val + 1e-8))  # Peak area asymmetry / 峰值区域不对称性
        else:
            features.extend([1.0, 1.0, 0])

        # Cycle length feature (1维) / 周期长度特征
        features.append(cycle_len / self.sample_rate)  # Cycle duration in seconds / 周期持续时间(秒)

        return np.array(features)

    def compute_temporal_features(self, current_features, feature_history):
        """
        计算时序特征（方案A）
        Compute temporal features (Plan A)

        Args:
            current_features: 当前周期的基础特征 / Current cycle base features
            feature_history: 历史特征队列 / Feature history queue

        Returns:
            temporal_features: 时序特征向量 / Temporal feature vector
        """
        temporal_features = []
        history_list = list(feature_history)

        # Key feature indices for temporal analysis / 用于时序分析的关键特征索引
        key_indices = [0, 2, 4, 10, 12]  # mean, max, peak_value, pulse_amplitude, asymmetry

        for idx in key_indices:
            current_val = current_features[idx]

            # Lag features (3维) / 滞后特征
            for lag in range(1, self.lag_steps + 1):
                if len(history_list) >= lag:
                    lag_val = history_list[-lag][idx]
                    temporal_features.append(current_val - lag_val)  # Change from lag / 与滞后值的变化
                else:
                    temporal_features.append(0)

            # Sliding window statistics (2维) / 滑动窗口统计
            if len(history_list) >= 2:
                window_vals = [h[idx] for h in history_list[-self.window_size:]] + [current_val]
                temporal_features.append(np.mean(window_vals))  # Window mean / 窗口均值
                temporal_features.append(np.std(window_vals))  # Window std / 窗口标准差
            else:
                temporal_features.extend([current_val, 0])

        # HRV-like features (2维) / 类心率变异性特征
        # Based on cycle duration (last feature in current_features)
        cycle_duration_idx = len(current_features) - 1  # Cycle duration is the last feature
        current_rr = current_features[cycle_duration_idx]

        if len(history_list) >= 2:
            rr_intervals = [h[cycle_duration_idx] for h in history_list[-self.window_size:]] + [current_rr]
            rr_diffs = np.diff(rr_intervals)
            temporal_features.append(np.std(rr_diffs) if len(rr_diffs) > 0 else 0)  # SDNN-like / 类SDNN
            temporal_features.append(np.sqrt(np.mean(np.square(rr_diffs))) if len(rr_diffs) > 0 else 0)  # RMSSD-like / 类RMSSD
        else:
            temporal_features.extend([0, 0])

        return np.array(temporal_features)

    def compute_bp_state_features(self, current_features, bp_history, _unused):
        """
        计算血压状态感知特征（方案F）- 帮助模型识别极端值和趋势
        Compute BP state-aware features (Plan F) - Help model track extreme values and trends

        Args:
            current_features: 当前周期的基础特征 / Current cycle base features
            bp_history: 历史血压值 (systolic, diastolic) 元组列表 / Historical BP values as (sys, dia) tuples
            _unused: 未使用参数（保持接口一致性）/ Unused parameter (for interface consistency)

        Returns:
            bp_state_features: 血压状态特征向量 / BP state feature vector
        """
        bp_state_features = []

        # Extract key predictive features / 提取关键预测特征
        pulse_amplitude = current_features[10]  # Feature index 10: pulse_amplitude
        peak_value = current_features[4]  # Feature index 4: peak_value
        mean_value = current_features[0]  # Feature index 0: mean

        # Extract systolic and diastolic history from tuples / 从元组中提取收缩压和舒张压历史
        if len(bp_history) >= 3:
            sys_history = [bp[0] for bp in bp_history]
            dia_history = [bp[1] for bp in bp_history]
        else:
            sys_history = []
            dia_history = []

        # === Z-score features (6维) / Z分数特征 ===
        # Measure how far current features deviate from recent history
        # 衡量当前特征与近期历史的偏差程度
        if len(sys_history) >= 3:
            # Systolic BP z-score estimation / 收缩压Z分数估计
            sys_mean = np.mean(sys_history)
            sys_std = np.std(sys_history) + 1e-8

            # Estimate current systolic using pulse amplitude / 使用脉搏幅度估计当前收缩压
            # Higher pulse amplitude typically correlates with higher systolic BP
            estimated_sys_deviation = (pulse_amplitude - np.mean([current_features[10] for _ in range(1)]))
            bp_state_features.append(estimated_sys_deviation / sys_std)  # Normalized deviation / 归一化偏差

            # Diastolic BP z-score estimation / 舒张压Z分数估计
            dia_mean = np.mean(dia_history)
            dia_std = np.std(dia_history) + 1e-8

            estimated_dia_deviation = (mean_value - np.mean([current_features[0] for _ in range(1)]))
            bp_state_features.append(estimated_dia_deviation / dia_std)

            # Pulse pressure z-score / 脉压差Z分数
            pulse_pressure_history = [s - d for s, d in zip(sys_history, dia_history)]
            pp_mean = np.mean(pulse_pressure_history)
            pp_std = np.std(pulse_pressure_history) + 1e-8
            estimated_pp = pulse_amplitude * 0.8  # Rough estimation / 粗略估计
            bp_state_features.append((estimated_pp - pp_mean) / pp_std)

            # Feature deviation from population mean / 特征偏离总体均值
            bp_state_features.append((peak_value - sys_mean) / sys_std)  # Peak value deviation
            bp_state_features.append((mean_value - dia_mean) / dia_std)  # Mean value deviation
            bp_state_features.append((pulse_amplitude - pp_mean) / pp_std)  # Pulse amplitude deviation
        else:
            bp_state_features.extend([0, 0, 0, 0, 0, 0])

        # === Binary abnormal range indicators (4维) / 异常范围二元指示器 ===
        # Help model explicitly recognize extreme BP states
        # 帮助模型明确识别极端血压状态
        if len(sys_history) >= 3:

            # High BP indicators / 高血压指示器
            # If recent average is high (>140 systolic or >90 diastolic)
            recent_sys_mean = np.mean(sys_history[-3:])
            recent_dia_mean = np.mean(dia_history[-3:])

            is_high_sys = 1.0 if recent_sys_mean > 140 else 0.0
            is_high_dia = 1.0 if recent_dia_mean > 90 else 0.0

            # Low BP indicators / 低血压指示器
            # If recent average is low (<100 systolic or <60 diastolic)
            is_low_sys = 1.0 if recent_sys_mean < 100 else 0.0
            is_low_dia = 1.0 if recent_dia_mean < 60 else 0.0

            bp_state_features.extend([is_high_sys, is_high_dia, is_low_sys, is_low_dia])
        else:
            bp_state_features.extend([0, 0, 0, 0])

        # === BP trend features (6维) / 血压趋势特征 ===
        # Capture whether BP is rising, falling, or stable
        # 捕获血压是上升、下降还是稳定
        if len(sys_history) >= 5:

            # Recent trend (last 3 values) / 近期趋势（最近3个值）
            recent_sys = sys_history[-3:]
            recent_dia = dia_history[-3:]

            # Systolic trend / 收缩压趋势
            sys_trend = np.mean(np.diff(recent_sys))  # Average change per beat / 每搏平均变化
            bp_state_features.append(sys_trend)
            bp_state_features.append(1.0 if sys_trend > 2 else 0.0)  # Is rising? / 是否上升？
            bp_state_features.append(1.0 if sys_trend < -2 else 0.0)  # Is falling? / 是否下降？

            # Diastolic trend / 舒张压趋势
            dia_trend = np.mean(np.diff(recent_dia))
            bp_state_features.append(dia_trend)
            bp_state_features.append(1.0 if dia_trend > 2 else 0.0)  # Is rising?
            bp_state_features.append(1.0 if dia_trend < -2 else 0.0)  # Is falling?
        else:
            bp_state_features.extend([0, 0, 0, 0, 0, 0])

        # === Variability indicators (2维) / 变异性指示器 ===
        # High variability may indicate unstable BP state
        # 高变异性可能表示不稳定的血压状态
        if len(sys_history) >= 5:

            # Coefficient of variation (CV) = std / mean
            sys_cv = np.std(sys_history) / (np.mean(sys_history) + 1e-8)
            dia_cv = np.std(dia_history) / (np.mean(dia_history) + 1e-8)

            bp_state_features.append(sys_cv)
            bp_state_features.append(dia_cv)
        else:
            bp_state_features.extend([0, 0])

        return np.array(bp_state_features)

    def is_valid_cycle(self, cycle_signal, cycle_length, target_systolic=None, target_diastolic=None):
        """
        检查单个周期是否有效（增强版质量控制）
        
        参数:
            cycle_signal: 周期信号
            cycle_length: 周期长度
            target_systolic: 对应的收缩压（可选）
            target_diastolic: 对应的舒张压（可选）
        
        返回:
            is_valid: 布尔值
        """
        # 检查1: 周期长度合理性 (0.5-2秒，对应心率30-120 bpm)
        min_length = int(0.5 * self.sample_rate)
        max_length = int(2.0 * self.sample_rate)
        if cycle_length < min_length or cycle_length > max_length:
            return False
        
        # 检查2: 信号标准差（检测是否有变化）
        if np.std(cycle_signal) < 0.1:
            return False
        
        # 检查3: 无NaN值
        if np.isnan(cycle_signal).any():
            return False
        
        # 检查4: 血压值合理性（如果提供）
        if target_systolic is not None:
            if not (50 < target_systolic < 200):
                return False
        
        if target_diastolic is not None:
            if not (30 < target_diastolic < 120):
                return False
        
        if target_systolic is not None and target_diastolic is not None:
            if target_systolic <= target_diastolic:
                return False
            # 脉压差应该在合理范围内
            pulse_pressure = target_systolic - target_diastolic
            if pulse_pressure < 20 or pulse_pressure > 100:
                return False
        
        return True
    
    def prepare_cycle_based_dataset(self, pleth_processed, art_processed):
        """
        基于心跳周期准备数据集（增强版 - 带有效片段检测）
        
        参数:
            pleth_processed: PLETH处理后的数据
            art_processed: ART处理后的数据
        
        返回:
            (X, y_systolic, y_diastolic): 特征矩阵和目标向量
        """
        pleth_signal_full = pleth_processed['filtered_signal']
        art_signal_full = art_processed['filtered_signal']
        
        # ========== 步骤1: 有效片段检测 ==========
        if self.enable_segment_detection:
            print("\n" + "="*60)
            print("步骤1: 检测有效信号片段")
            print("="*60)
            
            # 对PLETH和ART分别检测有效片段
            pleth_valid_segments = detect_valid_segments(
                pleth_signal_full, 
                segment_duration=60,
                sample_rate=self.sample_rate,
                nan_threshold=0.1,  # 类似去年学生在abp_hpi.py中的10%阈值
                std_threshold_percentile=5
            )
            
            art_valid_segments = detect_valid_segments(
                art_signal_full,
                segment_duration=60,
                sample_rate=self.sample_rate,
                nan_threshold=0.1,
                std_threshold_percentile=5
            )
            
            # 找到PLETH和ART都有效的重叠片段
            common_valid_segments = []
            for pleth_seg in pleth_valid_segments:
                for art_seg in art_valid_segments:
                    # 计算重叠区域
                    overlap_start = max(pleth_seg[0], art_seg[0])
                    overlap_end = min(pleth_seg[1], art_seg[1])
                    
                    if overlap_end > overlap_start:
                        common_valid_segments.append((overlap_start, overlap_end))
            
            print(f"\n  PLETH和ART共同有效片段: {len(common_valid_segments)}")
            
            if len(common_valid_segments) == 0:
                print("\n  ⚠ 警告: 未找到PLETH和ART共同的有效片段，使用全部信号")
                pleth_signal = pleth_signal_full
                art_signal = art_signal_full
            else:
                # 提取并拼接有效片段
                pleth_signal, _ = extract_valid_signal_portions(pleth_signal_full, common_valid_segments)
                art_signal, _ = extract_valid_signal_portions(art_signal_full, common_valid_segments)
        else:
            pleth_signal = pleth_signal_full
            art_signal = art_signal_full
        
        # ========== 步骤2: 重新检测峰值和谷值 ==========
        print("\n" + "="*60)
        print("步骤2: 在有效信号上重新检测峰值和谷值")
        print("="*60)
        
        from signal_processing import SignalProcessor
        processor = SignalProcessor()
        
        # 对提取的有效信号重新检测
        pleth_peaks, _ = processor.find_peaks(pleth_signal)
        pleth_valleys, _ = processor.find_valleys(pleth_signal)
        art_peaks, _ = processor.find_peaks(art_signal)
        art_valleys, _ = processor.find_valleys(art_signal)
        
        print(f"  PLETH: {len(pleth_peaks)} 峰值, {len(pleth_valleys)} 谷值")
        print(f"  ART: {len(art_peaks)} 峰值, {len(art_valleys)} 谷值")
        
        # ========== 步骤3: 逐周期特征提取 ==========
        print("\n" + "="*60)
        print("步骤3: 逐周期特征提取（带质量控制）")
        print("="*60)
        
        X = []
        y_systolic = []
        y_diastolic = []
        
        total_cycles = len(pleth_valleys) - 1
        valid_cycles = 0
        rejected_reasons = {
            'invalid_length': 0,
            'no_art_match': 0,
            'invalid_bp_values': 0,
            'quality_check_failed': 0
        }
        
        for i in range(total_cycles):
            start = pleth_valleys[i]
            end = pleth_valleys[i + 1]
            cycle_length = end - start
            
            # 提取这个周期的PLETH信号
            cycle_signal = pleth_signal[start:end]
            
            # 找到这个周期内的PLETH峰值和谷值（相对位置）
            cycle_peaks_abs = pleth_peaks[(pleth_peaks > start) & (pleth_peaks < end)]
            cycle_peaks_rel = cycle_peaks_abs - start
            
            cycle_valleys_abs = pleth_valleys[(pleth_valleys >= start) & (pleth_valleys <= end)]
            cycle_valleys_rel = cycle_valleys_abs - start
            
            # 找到对应的ART峰值和谷值
            mid_point = (start + end) // 2
            tolerance = cycle_length
            
            nearby_art_peaks = art_peaks[
                (art_peaks >= mid_point - tolerance) & 
                (art_peaks <= mid_point + tolerance)
            ]
            
            nearby_art_valleys = art_valleys[
                (art_valleys >= mid_point - tolerance) & 
                (art_valleys <= mid_point + tolerance)
            ]
            
            if len(nearby_art_peaks) == 0 or len(nearby_art_valleys) == 0:
                rejected_reasons['no_art_match'] += 1
                continue
            
            # 取最近的ART峰值和谷值
            closest_peak = nearby_art_peaks[np.argmin(np.abs(nearby_art_peaks - mid_point))]
            closest_valley = nearby_art_valleys[np.argmin(np.abs(nearby_art_valleys - mid_point))]
            
            systolic = art_signal[closest_peak]
            diastolic = art_signal[closest_valley]
            
            # 质量检查
            if not self.is_valid_cycle(cycle_signal, cycle_length, systolic, diastolic):
                if not (50 < systolic < 200 and 30 < diastolic < 120):
                    rejected_reasons['invalid_bp_values'] += 1
                else:
                    rejected_reasons['quality_check_failed'] += 1
                continue
            
            # 提取基础特征 / Extract base features
            base_features = self.extract_single_cycle_features(
                cycle_signal, cycle_peaks_rel, cycle_valleys_rel
            )

            # 计算时序特征 / Compute temporal features
            if self.enable_temporal_features:
                temporal_features = self.compute_temporal_features(base_features, self.feature_history)
                # Update history / 更新历史
                self.feature_history.append(base_features)
            else:
                temporal_features = np.array([])

            # 计算血压状态感知特征 (Plan F) / Compute BP state-aware features (Plan F)
            if self.enable_bp_state_features:
                # Use actual BP values from current and history
                bp_state_features = self.compute_bp_state_features(
                    base_features,
                    self.bp_history,  # Will contain (systolic, diastolic) tuples
                    self.bp_history  # Using same history, will extract separately in method
                )
                # Update BP history with current values / 更新血压历史
                self.bp_history.append((systolic, diastolic))
            else:
                bp_state_features = np.array([])

            # Combine all features / 组合所有特征
            if self.enable_temporal_features and self.enable_bp_state_features:
                features = np.concatenate([base_features, temporal_features, bp_state_features])
            elif self.enable_temporal_features:
                features = np.concatenate([base_features, temporal_features])
            elif self.enable_bp_state_features:
                features = np.concatenate([base_features, bp_state_features])
            else:
                features = base_features

            X.append(features)
            y_systolic.append(systolic)
            y_diastolic.append(diastolic)
            valid_cycles += 1
        
        X = np.array(X)
        y_systolic = np.array(y_systolic)
        y_diastolic = np.array(y_diastolic)
        
        # ========== 结果统计 ==========
        print(f"\n✓ 特征提取完成")
        print(f"  总周期数: {total_cycles}")
        print(f"  有效周期数: {valid_cycles} ({valid_cycles/total_cycles*100:.1f}%)")
        print(f"\n  拒绝原因统计:")
        print(f"    周期长度不合理: {rejected_reasons['invalid_length']}")
        print(f"    无匹配的ART信号: {rejected_reasons['no_art_match']}")
        print(f"    血压值不合理: {rejected_reasons['invalid_bp_values']}")
        print(f"    其他质量问题: {rejected_reasons['quality_check_failed']}")
        
        if len(X) > 0:
            print(f"\n  特征矩阵形状: {X.shape}")
            print(f"  收缩压范围: [{y_systolic.min():.1f}, {y_systolic.max():.1f}] mmHg (均值: {y_systolic.mean():.1f})")
            print(f"  舒张压范围: [{y_diastolic.min():.1f}, {y_diastolic.max():.1f}] mmHg (均值: {y_diastolic.mean():.1f})")
        
        # 定义特征名称 / Define feature names
        base_feature_names = [
            'mean', 'std', 'max', 'min',
            'peak_value', 'peak_position', 'peak_width', 'peak_slope',
            'valley_value', 'valley_position',
            'pulse_amplitude', 'pulse_ratio', 'asymmetry',
            'integral', 'integral_norm',
            'skewness', 'kurtosis',
            # Enhanced features (Plan D) / 增强特征
            'max_rise_rate', 'max_fall_rate', 'derivative_std', 'acceleration_std',
            'rise_fall_ratio', 'half_ratio', 'peak_asymmetry',
            'cycle_duration'
        ]

        temporal_feature_names = []
        if self.enable_temporal_features:
            # Temporal feature names / 时序特征名称
            key_features = ['mean', 'max', 'peak_value', 'pulse_amplitude', 'asymmetry']
            for feat in key_features:
                for lag in range(1, self.lag_steps + 1):
                    temporal_feature_names.append(f'{feat}_lag{lag}')
                temporal_feature_names.append(f'{feat}_window_mean')
                temporal_feature_names.append(f'{feat}_window_std')
            temporal_feature_names.extend(['rr_sdnn', 'rr_rmssd'])

        bp_state_feature_names = []
        if self.enable_bp_state_features:
            # BP state-aware feature names (Plan F) / 血压状态感知特征名称（方案F）
            bp_state_feature_names = [
                # Z-score features (6维)
                'sys_zscore', 'dia_zscore', 'pp_zscore',
                'peak_sys_deviation', 'mean_dia_deviation', 'pulse_pp_deviation',
                # Binary abnormal indicators (4维)
                'is_high_sys', 'is_high_dia', 'is_low_sys', 'is_low_dia',
                # BP trend features (6维)
                'sys_trend', 'sys_rising', 'sys_falling',
                'dia_trend', 'dia_rising', 'dia_falling',
                # Variability indicators (2维)
                'sys_cv', 'dia_cv'
            ]

        # Combine feature names / 组合特征名称
        self.feature_names = base_feature_names + temporal_feature_names + bp_state_feature_names
        
        # 标准化特征
        if self.scaler is not None and len(X) > 0:
            print(f"\n  正在标准化特征...")
            X = self.scaler.fit_transform(X)
        
        return X, y_systolic, y_diastolic


if __name__ == '__main__':
    # 测试代码
    print("=" * 60)
    print("Enhanced Cycle-Based Feature Extraction Test")
    print("=" * 60)
    
    # 创建模拟信号（包含有效和无效片段）
    np.random.seed(42)
    sample_rate = 100
    
    # 10分钟信号，前3分钟有效，中间2分钟无效（全是NaN），后5分钟有效
    valid_signal_1 = np.sin(np.linspace(0, 180*np.pi, 3*60*sample_rate)) + np.random.randn(3*60*sample_rate) * 0.1
    invalid_signal = np.full(2*60*sample_rate, np.nan)
    valid_signal_2 = np.sin(np.linspace(0, 300*np.pi, 5*60*sample_rate)) + np.random.randn(5*60*sample_rate) * 0.1
    
    test_signal = np.concatenate([valid_signal_1, invalid_signal, valid_signal_2])
    
    print(f"\n测试信号: {len(test_signal)} 点 ({len(test_signal)/sample_rate:.1f} 秒)")
    
    # 测试有效片段检测
    valid_segments = detect_valid_segments(test_signal, segment_duration=60, sample_rate=sample_rate)
    
    print(f"\n检测到 {len(valid_segments)} 个有效片段")
    
    # 提取有效信号
    valid_signal, boundaries = extract_valid_signal_portions(test_signal, valid_segments)
    
    print(f"提取的有效信号长度: {len(valid_signal)} 点")