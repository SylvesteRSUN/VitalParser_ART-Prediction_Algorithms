"""
split_vital.py
切分.vital文件为训练集和测试集 (70-30)
"""

import vitaldb
import os
import sys

def split_vital_file(input_path, output_dir='results', train_ratio=0.7):
    """
    切分vital文件为训练集和测试集
    
    参数:
        input_path: 输入.vital文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例 (默认0.7)
    """
    print(f"读取文件: {input_path}")
    
    # 读取vital文件
    vf = vitaldb.VitalFile(input_path)
    
    # 获取所有轨道
    track_names = vf.get_track_names()
    print(f"包含 {len(track_names)} 个轨道: {', '.join(track_names)}")
    
    # 直接从VitalFile对象获取时间范围
    dt_start = vf.dtstart
    dt_end = vf.dtend
    duration = dt_end - dt_start
    
    print(f"\n文件时长: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
    
    # 计算切分点
    split_time = dt_start + duration * train_ratio
    
    print(f"\n切分方案:")
    print(f"  训练集: 0 - {(split_time - dt_start):.2f} 秒 ({train_ratio*100:.0f}%)")
    print(f"  测试集: {(split_time - dt_start):.2f} - {duration:.2f} 秒 ({(1-train_ratio)*100:.0f}%)")
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n创建输出目录: {output_dir}")
    
    # 获取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # 训练集
    print("\n生成训练集...")
    vf_train = vitaldb.VitalFile(input_path)
    vf_train.crop(dtfrom=dt_start, dtend=split_time)
    train_path = os.path.join(output_dir, f"{base_name}_train.vital")
    vf_train.to_vital(train_path)
    
    train_duration = split_time - dt_start
    print(f"✓ 训练集保存: {train_path}")
    print(f"  时长: {train_duration:.2f} 秒 ({train_duration/60:.2f} 分钟)")
    
    # 测试集
    print("\n生成测试集...")
    vf_test = vitaldb.VitalFile(input_path)
    vf_test.crop(dtfrom=split_time, dtend=dt_end)
    test_path = os.path.join(output_dir, f"{base_name}_test.vital")
    vf_test.to_vital(test_path)
    
    test_duration = dt_end - split_time
    print(f"✓ 测试集保存: {test_path}")
    print(f"  时长: {test_duration:.2f} 秒 ({test_duration/60:.2f} 分钟)")
    
    print(f"\n" + "="*50)
    print(f"完成!")
    print(f"训练集: {train_path}")
    print(f"测试集: {test_path}")
    print(f"="*50)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python split_vital.py <input.vital> [output_dir] [train_ratio]")
        print("示例: python split_vital.py data/0002.vital results 0.7")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'results'
    train_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    
    if not os.path.exists(input_path):
        print(f"错误: 文件不存在 - {input_path}")
        sys.exit(1)
    
    if train_ratio <= 0 or train_ratio >= 1:
        print(f"错误: train_ratio 必须在 0 和 1 之间")
        sys.exit(1)
    
    split_vital_file(input_path, output_dir, train_ratio)