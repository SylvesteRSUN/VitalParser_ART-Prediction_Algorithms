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
    """基于心跳周期的特征提取器（增强版 - 带有效片段检测）"""
    
    def __init__(self, enable_segment_detection=True):
        """
        初始化特征提取器
        
        参数:
            enable_segment_detection: 是否启用有效片段检测
        """
        self.scaler = StandardScaler() if FEATURE_CONFIG['standardize_features'] else None
        self.feature_names = []
        self.enable_segment_detection = enable_segment_detection
        self.sample_rate = DATA_CONFIG['sampling_rate']
    
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
        
        return np.array(features)
    
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
            
            # 提取特征
            features = self.extract_single_cycle_features(
                cycle_signal, cycle_peaks_rel, cycle_valleys_rel
            )
            
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
        
        # 定义特征名称
        self.feature_names = [
            'mean', 'std', 'max', 'min',
            'peak_value', 'peak_position', 'peak_width', 'peak_slope',
            'valley_value', 'valley_position',
            'pulse_amplitude', 'pulse_ratio', 'asymmetry',
            'integral', 'integral_norm',
            'skewness', 'kurtosis'
        ]
        
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