"""
数据加载模块 - 读取.vital文件并提取信号数据
支持单个文件或多个文件
"""

import numpy as np
import vitaldb
from config import DATA_CONFIG

class VitalDataLoader:
    """加载和预处理.vital文件的类"""
    
    def __init__(self, vital_file_path=None):
        """
        初始化数据加载器
        
        参数:
            vital_file_path: .vital文件的路径，如果为None则使用config中的路径
        """
        self.vital_file_path = vital_file_path
        self.pleth_signal_name = DATA_CONFIG['pleth_signal']
        self.art_signal_name = DATA_CONFIG['art_signal']
        self.sampling_rate = DATA_CONFIG['sampling_rate']
        
        self.vital_file = None
        self.pleth_data = None
        self.art_data = None
        
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
        """提取PLETH和ART信号数据"""
        if self.vital_file is None:
            print("请先加载.vital文件")
            return False
        
        try:
            # 计算采样间隔（秒）
            interval = 1.0 / self.sampling_rate  # 例如100Hz → 0.01秒
            
            # 提取PLETH信号
            print(f"\n正在提取 {self.pleth_signal_name}...")
            pleth_raw = self.vital_file.to_numpy(self.pleth_signal_name, interval)
            
            # 提取ART信号
            print(f"正在提取 {self.art_signal_name}...")
            art_raw = self.vital_file.to_numpy(self.art_signal_name, interval)
            
            # 移除NaN值
            self.pleth_data, self.art_data = self._remove_nan_values(pleth_raw, art_raw)
            
            print(f"✓ 信号提取成功")
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
    
    for i, file_path in enumerate(file_paths, 1):
        print(f"\n[{i}/{len(file_paths)}] 处理文件: {file_path}")
        
        loader = VitalDataLoader(file_path)
        pleth, art = loader.load_and_extract()
        
        if pleth is not None and art is not None:
            pleth_all.append(pleth)
            art_all.append(art)
            print(f"  ✓ 成功加载 {len(pleth):,} 个数据点")
        else:
            print(f"  ✗ 文件加载失败，跳过")
    
    if len(pleth_all) == 0:
        print(f"\n✗ {dataset_name}加载失败：没有成功加载任何文件")
        return None, None
    
    # 合并所有数据
    pleth_combined = np.concatenate(pleth_all)
    art_combined = np.concatenate(art_all)
    
    print(f"\n{'='*60}")
    print(f"✓ {dataset_name}加载完成")
    print(f"  成功文件数: {len(pleth_all)}/{len(file_paths)}")
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