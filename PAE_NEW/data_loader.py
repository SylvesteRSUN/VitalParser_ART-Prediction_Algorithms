"""
数据加载模块 - 读取.vital文件并提取信号数据
支持单个文件或多个文件
支持多个候选信号名称（按优先级尝试）
"""

import numpy as np
import vitaldb
from config import DATA_CONFIG


def try_extract_signal(vital_file, signal_candidates, sampling_rate):
    """
    尝试从多个候选信号名称中提取信号
    
    参数:
        vital_file: VitalFile对象
        signal_candidates: 候选信号名称列表（按优先级排序）
        sampling_rate: 采样率
    
    返回:
        (signal_data, signal_name): 成功提取的信号数据和使用的信号名称
        或 (None, None): 如果所有候选都失败
    """
    interval = 1.0 / sampling_rate
    
    for signal_name in signal_candidates:
        try:
            print(f"  尝试提取: {signal_name}...", end='')
            signal = vital_file.to_numpy(signal_name, interval)
            
            if signal is not None and len(signal) > 0:
                # 检查是否大部分是NaN
                nan_ratio = np.isnan(signal).sum() / len(signal)
                if nan_ratio < 0.99:  # 如果有效数据>1%，认为成功
                    print(f" ✓ 成功 (有效数据: {100-nan_ratio*100:.1f}%)")
                    return signal, signal_name
                else:
                    print(f" ✗ 几乎全是NaN ({nan_ratio*100:.1f}%)")
            else:
                print(f" ✗ 无数据")
        except Exception as e:
            print(f" ✗ 失败 ({str(e)})")
            continue
    
    return None, None


class VitalDataLoader:
    """加载和预处理.vital文件的类"""
    
    def __init__(self, vital_file_path=None):
        """
        初始化数据加载器
        
        参数:
            vital_file_path: .vital文件的路径，如果为None则使用config中的路径
        """
        self.vital_file_path = vital_file_path
        
        # 支持新的候选列表格式和旧的单一信号格式
        self.pleth_signal_candidates = DATA_CONFIG.get('pleth_signal_candidates', 
                                                       [DATA_CONFIG.get('pleth_signal', 'Demo/PLETH')])
        self.art_signal_candidates = DATA_CONFIG.get('art_signal_candidates',
                                                     [DATA_CONFIG.get('art_signal', 'Demo/ART')])
        
        self.sampling_rate = DATA_CONFIG['sampling_rate']
        
        self.vital_file = None
        self.pleth_data = None
        self.art_data = None
        self.pleth_signal_used = None  # 记录实际使用的信号名
        self.art_signal_used = None
        
    def load_vital_file(self):
        """加载.vital文件"""
        print(f"正在加载文件: {self.vital_file_path}")
        try:
            self.vital_file = vitaldb.VitalFile(self.vital_file_path)
            print(f"✓ 文件加载成功")
            return True
        except Exception as e:
            print(f"✗ 文件加载失败: {e}")
            return False
    
    def get_available_tracks(self):
        """获取.vital文件中所有可用的信号轨道"""
        if self.vital_file is None:
            print("请先加载.vital文件")
            return []
        
        tracks = self.vital_file.get_track_names()
        print(f"\n可用的信号轨道 ({len(tracks)}个):")
        for i, track in enumerate(tracks, 1):
            print(f"  {i}. {track}")
        return tracks
    
    def extract_signals(self):
        """提取PLETH和ART信号数据（支持多个候选名称）"""
        if self.vital_file is None:
            print("请先加载.vital文件")
            return False
        
        try:
            # 尝试提取PLETH信号
            print(f"\n正在提取PLETH信号 (尝试{len(self.pleth_signal_candidates)}个候选):")
            pleth_raw, self.pleth_signal_used = try_extract_signal(
                self.vital_file, 
                self.pleth_signal_candidates, 
                self.sampling_rate
            )
            
            if pleth_raw is None:
                print(f"✗ PLETH信号提取失败: 所有候选都不可用")
                print(f"  尝试过的候选: {', '.join(self.pleth_signal_candidates)}")
                return False
            
            # 尝试提取ART信号
            print(f"\n正在提取ART信号 (尝试{len(self.art_signal_candidates)}个候选):")
            art_raw, self.art_signal_used = try_extract_signal(
                self.vital_file,
                self.art_signal_candidates,
                self.sampling_rate
            )
            
            if art_raw is None:
                print(f"✗ ART信号提取失败: 所有候选都不可用")
                print(f"  尝试过的候选: {', '.join(self.art_signal_candidates)}")
                return False
            
            # 移除NaN值
            self.pleth_data, self.art_data = self._remove_nan_values(pleth_raw, art_raw)
            
            print(f"\n✓ 信号提取成功")
            print(f"  使用的PLETH信号: {self.pleth_signal_used}")
            print(f"  使用的ART信号: {self.art_signal_used}")
            print(f"  PLETH数据点数: {len(self.pleth_data):,}")
            print(f"  ART数据点数: {len(self.art_data):,}")
            print(f"  数据时长: {len(self.pleth_data)/self.sampling_rate:.2f} 秒")
            
            return True
            
        except Exception as e:
            print(f"✗ 信号提取失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _remove_nan_values(self, pleth, art):
        """
        移除包含NaN的数据点，保持两个信号同步
        
        参数:
            pleth: PLETH信号数组
            art: ART信号数组
        
        返回:
            (pleth_clean, art_clean): 清理后的信号数组
        """
        # 找出有效数据点的索引
        valid_indices = ~(np.isnan(pleth) | np.isnan(art))
        
        pleth_clean = pleth[valid_indices]
        art_clean = art[valid_indices]
        
        removed_count = len(pleth) - len(pleth_clean)
        if removed_count > 0:
            print(f"  已移除 {removed_count:,} 个无效数据点 ({removed_count/len(pleth)*100:.2f}%)")
        
        return pleth_clean, art_clean
    
    def get_data(self):
        """
        获取提取的信号数据
        
        返回:
            (pleth_data, art_data): PLETH和ART信号数组
        """
        if self.pleth_data is None or self.art_data is None:
            print("请先提取信号数据")
            return None, None
        
        return self.pleth_data, self.art_data
    
    def get_signal_info(self):
        """
        获取实际使用的信号名称信息
        
        返回:
            dict: 包含信号名称和数据统计的字典
        """
        return {
            'pleth_signal': self.pleth_signal_used,
            'art_signal': self.art_signal_used,
            'pleth_length': len(self.pleth_data) if self.pleth_data is not None else 0,
            'art_length': len(self.art_data) if self.art_data is not None else 0,
        }
    
    def load_and_extract(self):
        """
        一键加载文件并提取信号（便捷方法）
        
        返回:
            (pleth_data, art_data): PLETH和ART信号数组，失败则返回(None, None)
        """
        if not self.load_vital_file():
            return None, None
        
        if not self.extract_signals():
            return None, None
        
        return self.get_data()


def load_multiple_files(file_paths, dataset_name="Dataset"):
    """
    加载多个.vital文件并合并数据
    
    参数:
        file_paths: .vital文件路径列表
        dataset_name: 数据集名称（用于日志）
    
    返回:
        (pleth_all, art_all): 合并后的PLETH和ART信号数组
    """
    print(f"\n{'='*60}")
    print(f"加载{dataset_name} - {len(file_paths)}个文件")
    print(f"{'='*60}")
    
    pleth_all = []
    art_all = []
    signal_info_list = []
    
    for i, file_path in enumerate(file_paths, 1):
        print(f"\n[{i}/{len(file_paths)}] 处理文件: {file_path}")
        
        loader = VitalDataLoader(file_path)
        pleth, art = loader.load_and_extract()
        
        if pleth is not None and art is not None:
            pleth_all.append(pleth)
            art_all.append(art)
            signal_info_list.append(loader.get_signal_info())
            print(f"  ✓ 成功加载 {len(pleth):,} 个数据点")
        else:
            print(f"  ✗ 文件加载失败，跳过")
    
    if len(pleth_all) == 0:
        print(f"\n✗ {dataset_name}加载失败：没有成功加载任何文件")
        return None, None
    
    # 合并所有数据
    pleth_combined = np.concatenate(pleth_all)
    art_combined = np.concatenate(art_all)
    
    # 统计使用的信号名称
    pleth_signals_used = set([info['pleth_signal'] for info in signal_info_list])
    art_signals_used = set([info['art_signal'] for info in signal_info_list])
    
    print(f"\n{'='*60}")
    print(f"✓ {dataset_name}加载完成")
    print(f"  成功文件数: {len(pleth_all)}/{len(file_paths)}")
    print(f"  使用的PLETH信号类型: {', '.join(pleth_signals_used)}")
    print(f"  使用的ART信号类型: {', '.join(art_signals_used)}")
    print(f"  总PLETH数据点: {len(pleth_combined):,}")
    print(f"  总ART数据点: {len(art_combined):,}")
    print(f"  总时长: {len(pleth_combined)/DATA_CONFIG['sampling_rate']:.2f} 秒")
    print(f"{'='*60}")
    
    return pleth_combined, art_combined


def load_train_test_data():
    """
    加载训练集和测试集数据
    
    返回:
        (pleth_train, art_train, pleth_test, art_test): 训练和测试数据
    """
    # 加载训练集（多个文件）
    train_paths = DATA_CONFIG['train_file_paths']
    pleth_train, art_train = load_multiple_files(train_paths, "训练集")
    
    # 加载测试集（单个文件）
    test_path = DATA_CONFIG['test_file_path']
    print(f"\n{'='*60}")
    print(f"加载测试集 - 1个文件")
    print(f"{'='*60}")
    
    loader_test = VitalDataLoader(test_path)
    pleth_test, art_test = loader_test.load_and_extract()
    
    if pleth_test is None or art_test is None:
        print("\n✗ 测试集加载失败")
        return None, None, None, None
    
    print(f"\n{'='*60}")
    print(f"✓ 测试集加载完成")
    print(f"  使用的PLETH信号: {loader_test.pleth_signal_used}")
    print(f"  使用的ART信号: {loader_test.art_signal_used}")
    print(f"  PLETH数据点: {len(pleth_test):,}")
    print(f"  ART数据点: {len(art_test):,}")
    print(f"  时长: {len(pleth_test)/DATA_CONFIG['sampling_rate']:.2f} 秒")
    print(f"{'='*60}")
    
    return pleth_train, art_train, pleth_test, art_test


if __name__ == '__main__':
    # 测试代码
    print("=" * 60)
    print("数据加载模块测试")
    print("=" * 60)
    
    # 测试加载训练集和测试集
    pleth_train, art_train, pleth_test, art_test = load_train_test_data()
    
    if pleth_train is not None:
        print(f"\n最终数据:")
        print(f"  训练集: {len(pleth_train):,} 点")
        print(f"  测试集: {len(pleth_test):,} 点")