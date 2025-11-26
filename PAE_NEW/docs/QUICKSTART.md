# 快速开始指南

这是一份精简的快速开始指南，帮助你在5分钟内运行PLETH-BP预测系统。

## 📦 文件清单

```
pleth_bp_project/
├── config.py                 # ⚙️ 配置参数（需要修改）
├── data_loader.py            # 📂 数据加载
├── signal_processing.py      # 📊 信号处理
├── feature_extraction.py     # 🔬 特征提取
├── models.py                 # 🤖 模型训练
├── evaluation.py             # 📈 评估可视化
├── utils.py                  # 🛠️ 工具函数
├── main.py                   # ▶️ 主程序（运行这个）
├── test_setup.py             # ✅ 环境测试
├── example_usage.py          # 📚 使用示例
├── requirements.txt          # 📋 依赖清单
├── README.md                 # 📖 完整文档
└── QUICKSTART.md             # 🚀 本文件
```

## ⚡ 三步开始

### 步骤 1: 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- numpy, scipy, scikit-learn（核心）
- matplotlib（可视化）
- vitaldb（读取.vital文件）

### 步骤 2: 配置文件路径

打开 `config.py`，修改第6行：

```python
DATA_CONFIG = {
    'vital_file_path': '/path/to/your/data.vital',  # ← 改成你的文件路径
    ...
}
```

### 步骤 3: 运行程序

```bash
python main.py
```

程序会自动完成：
1. 加载数据
2. 信号处理
3. 特征提取
4. 模型训练
5. 生成评估报告

结果保存在 `results/` 目录。

## 🧪 测试环境（推荐）

运行之前先测试环境：

```bash
python test_setup.py
```

这会检查：
- ✅ 所有依赖是否安装
- ✅ 模块是否能正常导入
- ✅ 配置文件是否正确
- ✅ 核心功能是否正常

## 📊 查看结果

训练完成后，查看 `results/` 目录：

```
results/
├── performance_comparison.png      # 模型性能对比
├── prediction_vs_actual_*.png      # 预测准确度
├── residuals_*.png                 # 残差分析
├── timeseries_*.png                # 时序对比
└── evaluation_report_*.txt         # 详细报告
```

## ⚙️ 快速调整

### 调整信号名称

如果你的.vital文件中信号名称不是 `Demo/PLETH` 和 `Demo/ART`：

```python
# config.py
DATA_CONFIG = {
    'pleth_signal': 'YourDevice/PLETH',  # 改成你的PLETH信号名
    'art_signal': 'YourDevice/ART',      # 改成你的ART信号名
}
```

### 减少训练时间

如果训练太慢，减少模型数量：

```python
# config.py
MODEL_CONFIG = {
    'models': [
        'random_forest',        # 只训练这两个
        'gradient_boosting',
    ],
}
```

### 调整窗口大小

如果内存不足：

```python
# main.py 第145行左右
pleth_windows, art_windows = create_sliding_windows(
    pleth_data, art_data,
    window_size=3000,  # 减小窗口（原5000）
    overlap=1000       # 减小重叠（原2500）
)
```

## 🎯 性能目标

| 指标 | 收缩压 | 舒张压 |
|------|--------|--------|
| MAE  | < 10   | < 10   |
| MSE  | < 70   | < 60   |

运行结束后会自动显示是否达标。

## 📚 学习示例

想了解各个模块如何使用？运行示例：

```bash
python example_usage.py
```

提供了6个示例：
1. 数据加载
2. 信号处理
3. 特征提取
4. 模型训练
5. 结果评估
6. 完整流程

## ❓ 常见问题

**Q: 找不到.vital文件？**
```
A: 检查config.py中的路径是否正确，使用绝对路径更保险
```

**Q: ImportError: No module named 'vitaldb'?**
```
A: 运行 pip install vitaldb
```

**Q: 内存不足？**
```
A: 减小window_size或增大overlap步长
```

**Q: 训练太慢？**
```
A: 减少MODEL_CONFIG中的模型数量
```

**Q: 信号名称错误？**
```
A: 先运行data_loader模块查看可用信号：
   from data_loader import VitalDataLoader
   loader = VitalDataLoader('your_file.vital')
   loader.load_vital_file()
   loader.get_available_tracks()
```

## 📧 下一步

- 阅读 `README.md` 了解完整功能
- 查看 `example_usage.py` 学习各模块用法
- 调整 `config.py` 优化性能
- 根据你的数据调整参数

## 🎉 完成！

如果一切顺利，你现在应该有：
- ✅ 训练好的模型
- ✅ 性能评估报告
- ✅ 可视化图表
- ✅ 对系统的基本了解

祝训练成功！🚀