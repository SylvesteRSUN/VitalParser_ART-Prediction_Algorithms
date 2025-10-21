"""
信号处理模块 - 滤波、峰值检测等信号处理功能
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from config import SIGNAL_CONFIG

class SignalProcessor:
    """信号处理器类"""
    
    def __init__(self):
        """初始化信号处理器"""
        self.savgol_window = SIGNAL_CONFIG['savgol_window']
        self.savgol_polyorder = SIGNAL_CONFIG['savgol_polyorder']
        self.gaussian_sigma = SIGNAL_CONFIG['gaussian_sigma']
        self.peak_prominence = SIGNAL_CONFIG['peak_prominence']
        self.peak_distance = SIGNAL_CONFIG['peak_distance']
    
    def apply_savgol_filter(self, signal_data):
        """
        应用Savitzky-Golay滤波器
        
        参数:
            signal_data: 原始信号数组
        
        返回:
            filtered_signal: 滤波后的信号
        """
        # 确保窗口长度为奇数且不超过信号长度
        window_length = self.savgol_window
        if len(signal_data) < window_length:
            window_length = len(signal_data) if len(signal_data) % 2 == 1 else len(signal_data) - 1
        
        if window_length < self.savgol_polyorder + 2:
            print(f"警告: 信号太短，无法应用Savitzky-Golay滤波")
            return signal_data
        
        filtered = signal.savgol_filter(signal_data, window_length, self.savgol_polyorder)
        return filtered
    
    def apply_gaussian_filter(self, signal_data):
        """
        应用Gaussian滤波器
        
        参数:
            signal_data: 原始信号数组
        
        返回:
            filtered_signal: 滤波后的信号
        """
        filtered = gaussian_filter1d(signal_data, sigma=self.gaussian_sigma, mode='reflect')
        return filtered
    
    def apply_combined_filter(self, signal_data):
        """
        应用组合滤波器（Savitzky-Golay + Gaussian）
        
        参数:
            signal_data: 原始信号数组
        
        返回:
            filtered_signal: 滤波后的信号
        """
        # 先应用Savitzky-Golay
        signal_sg = self.apply_savgol_filter(signal_data)
        
        # 再应用Gaussian
        signal_filtered = self.apply_gaussian_filter(signal_sg)
        
        return signal_filtered
    
    def find_peaks(self, signal_data):
        """
        检测信号中的峰值（最大值）
        
        参数:
            signal_data: 信号数组
        
        返回:
            peaks: 峰值位置的索引数组
            peak_properties: 峰值的属性字典
        """
        # 归一化信号以计算相对突出度
        signal_norm = (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data) + 1e-8)
        
        peaks, properties = signal.find_peaks(
            signal_norm,
            prominence=self.peak_prominence,
            distance=self.peak_distance
        )
        
        return peaks, properties
    
    def find_valleys(self, signal_data):
        """
        检测信号中的谷值（最小值）
        
        参数:
            signal_data: 信号数组
        
        返回:
            valleys: 谷值位置的索引数组
            valley_properties: 谷值的属性字典
        """
        # 反转信号以检测谷值
        signal_inverted = -signal_data
        
        # 归一化
        signal_norm = (signal_inverted - np.min(signal_inverted)) / (np.max(signal_inverted) - np.min(signal_inverted) + 1e-8)
        
        valleys, properties = signal.find_peaks(
            signal_norm,
            prominence=self.peak_prominence,
            distance=self.peak_distance
        )
        
        return valleys, properties
    
    def calculate_cycle_integrals(self, signal_data, valleys):
        """
        计算周期积分（两个谷值之间的信号积分）
        
        参数:
            signal_data: 信号数组
            valleys: 谷值位置数组
        
        返回:
            integrals: 周期积分数组
            durations: 周期持续时间数组
        """
        integrals = []
        durations = []
        
        for i in range(len(valleys) - 1):
            start_idx = valleys[i]
            end_idx = valleys[i + 1]
            
            # 计算该周期的积分
            cycle_signal = signal_data[start_idx:end_idx]
            integral = np.trapz(cycle_signal)
            duration = end_idx - start_idx
            
            integrals.append(integral)
            durations.append(duration)
        
        return np.array(integrals), np.array(durations)
    
    def process_signal(self, signal_data, signal_name="Signal"):
        """
        完整的信号处理流程
        
        参数:
            signal_data: 原始信号数组
            signal_name: 信号名称（用于日志）
        
        返回:
            processed_data: 包含处理结果的字典
        """
        print(f"\n处理 {signal_name} 信号...")
        
        # 1. 滤波
        filtered_signal = self.apply_combined_filter(signal_data)
        print(f"  ✓ 滤波完成")
        
        # 2. 检测峰值和谷值
        peaks, peak_props = self.find_peaks(filtered_signal)
        valleys, valley_props = self.find_valleys(filtered_signal)
        print(f"  ✓ 检测到 {len(peaks)} 个峰值, {len(valleys)} 个谷值")
        
        # 3. 计算周期积分
        if len(valleys) > 1:
            integrals, durations = self.calculate_cycle_integrals(filtered_signal, valleys)
            print(f"  ✓ 计算了 {len(integrals)} 个周期积分")
        else:
            integrals = np.array([])
            durations = np.array([])
            print(f"  ⚠ 谷值数量不足，无法计算周期积分")
        
        # 组织结果
        processed_data = {
            'raw_signal': signal_data,
            'filtered_signal': filtered_signal,
            'peaks': peaks,
            'peak_values': filtered_signal[peaks] if len(peaks) > 0 else np.array([]),
            'valleys': valleys,
            'valley_values': filtered_signal[valleys] if len(valleys) > 0 else np.array([]),
            'cycle_integrals': integrals,
            'cycle_durations': durations,
        }
        
        return processed_data


def process_pleth_art_signals(pleth_data, art_data):
    """
    便捷函数：同时处理PLETH和ART信号
    
    参数:
        pleth_data: PLETH信号数组
        art_data: ART信号数组
    
    返回:
        (pleth_processed, art_processed): 处理后的数据字典
    """
    processor = SignalProcessor()
    
    print("=" * 60)
    print("信号处理")
    print("=" * 60)
    
    pleth_processed = processor.process_signal(pleth_data, "PLETH")
    art_processed = processor.process_signal(art_data, "ART")
    
    print(f"\n✓ 信号处理完成")
    
    return pleth_processed, art_processed


if __name__ == '__main__':
    # 测试代码
    print("=" * 60)
    print("信号处理模块测试")
    print("=" * 60)
    
    # 生成测试信号
    t = np.linspace(0, 10, 1000)
    test_signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.randn(1000)
    
    # 创建处理器
    processor = SignalProcessor()
    
    # 处理信号
    processed = processor.process_signal(test_signal, "测试信号")
    
    print(f"\n处理结果:")
    print(f"  原始信号长度: {len(processed['raw_signal'])}")
    print(f"  滤波信号长度: {len(processed['filtered_signal'])}")
    print(f"  峰值数量: {len(processed['peaks'])}")
    print(f"  谷值数量: {len(processed['valleys'])}")
    print(f"  周期积分数量: {len(processed['cycle_integrals'])}")