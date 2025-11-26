# 项目结构说明 / Project Structure

本文档说明整理后的项目结构和模块组织方式。

This document describes the reorganized project structure and module organization.

## 📁 目录结构 / Directory Structure

```
PAE_NEW/
├── README.md                          # 项目说明 / Project README
├── requirements.txt                   # 依赖列表 / Dependencies
├── PROJECT_STRUCTURE.md               # 本文档 / This document
│
├── main.py                            # 主程序入口 / Main entry point
├── main_transfer_learning.py          # 迁移学习主程序 / Transfer learning main
├── quick_test_transfer_learning.py    # 快速测试脚本 / Quick test script
│
├── config/                            # 📂 配置模块 / Configuration module
│   ├── __init__.py
│   ├── config.py                      # 基础配置 / Base config
│   └── config_transfer.py             # 迁移学习配置 / Transfer learning config
│
├── core/                              # 📂 核心功能模块 / Core functionality
│   ├── __init__.py
│   ├── data_loader.py                 # 数据加载 / Data loading
│   ├── signal_processing.py           # 信号处理 / Signal processing
│   ├── feature_extraction.py          # 特征提取 / Feature extraction
│   ├── models.py                      # 模型定义 / Model definitions
│   └── utils.py                       # 工具函数 / Utility functions
│
├── transfer_learning_module/          # 📂 迁移学习模块 / Transfer learning
│   ├── __init__.py
│   ├── transfer_learning.py           # 迁移学习核心 / TL core
│   ├── data_splitter.py               # 数据分割 / Data splitting
│   └── evaluation.py                  # 评估工具 / Evaluation tools
│
├── biomarkers/                        # 📂 生物标志物计算 / Biomarkers
│   ├── __init__.py
│   ├── funcion_RR.py                  # RR间期计算 / RR intervals
│   ├── funcion_RSA.py                 # RSA计算 / RSA calculation
│   └── funcion_BRS.py                 # BRS计算 / BRS calculation
│
├── utils/                             # 📂 通用工具 / General utilities
│   ├── __init__.py
│   └── utils.py                       # 工具函数 / Utility functions
│
├── tests/                             # 📂 测试文件 / Test files
│   ├── __init__.py
│   ├── test_setup.py
│   ├── test_transfer_learning.py
│   └── single_file_split.py
│
└── docs/                              # 📂 文档 / Documentation
    ├── QUICKSTART.md
    ├── QUICKSTART_TRANSFER_LEARNING.md
    ├── TRANSFER_LEARNING_README.md
    ├── CALIBRATION_RATIO_UPDATE.md
    ├── PLAN_E_IMPLEMENTATION.md
    ├── QUICK_TEST_README.md
    └── transfer_learning.md
```

## 🔧 导入语句变化 / Import Changes

### 旧的导入方式 / Old Import Style
```python
from config import DATA_CONFIG
from data_loader import load_train_test_data
from signal_processing import SignalProcessor
from feature_extraction import CycleBasedFeatureExtractor
from models import ModelTrainer
from evaluation import ModelEvaluator
```

### 新的导入方式 / New Import Style
```python
from config.config import DATA_CONFIG
from core.data_loader import load_train_test_data
from core.signal_processing import SignalProcessor
from core.feature_extraction import CycleBasedFeatureExtractor
from core.models import ModelTrainer
from transfer_learning_module.evaluation import ModelEvaluator
```

### 简化导入（通过 __init__.py）/ Simplified Imports
```python
# 配置 / Config
from config import DATA_CONFIG, GENERAL_MODEL_CONFIG

# 核心模块 / Core modules
from core import load_train_test_data, SignalProcessor, CycleBasedFeatureExtractor

# 迁移学习 / Transfer learning
from transfer_learning_module import GeneralTrainer, PersonalFineTuner, ModelEvaluator

# 生物标志物 / Biomarkers
from biomarkers import funcion_rr, calcular_rsa, calcular_brs
```

## 📦 模块说明 / Module Descriptions

### config/ - 配置模块
包含所有配置参数，包括数据配置、模型配置、迁移学习配置等。

Contains all configuration parameters including data config, model config, and transfer learning config.

### core/ - 核心功能模块
包含数据处理、信号处理、特征提取和模型训练的核心功能。

Contains core functionality for data processing, signal processing, feature extraction, and model training.

### transfer_learning_module/ - 迁移学习模块
专门用于迁移学习的功能，包括通用模型训练、个性化微调和评估。

Dedicated to transfer learning functionality including general model training, personalized fine-tuning, and evaluation.

### biomarkers/ - 生物标志物计算模块
计算各种生理标志物，如RR间期、RSA（呼吸性窦性心律不齐）和BRS（压力反射敏感性）。

Calculates various physiological biomarkers such as RR intervals, RSA (Respiratory Sinus Arrhythmia), and BRS (Baroreflex Sensitivity).

### tests/ - 测试模块
包含各种测试脚本和验证代码。

Contains various test scripts and validation code.

### docs/ - 文档模块
包含项目文档、快速开始指南和技术说明。

Contains project documentation, quickstart guides, and technical notes.

## 🚀 使用方法 / Usage

### 运行主程序 / Run Main Program
```bash
cd PAE_NEW
python main.py
```

### 运行迁移学习 / Run Transfer Learning
```bash
python main_transfer_learning.py
```

### 运行快速测试 / Run Quick Test
```bash
python quick_test_transfer_learning.py
```

## ⚠️ 注意事项 / Important Notes

1. **工作目录**: 确保在 `PAE_NEW` 目录下运行脚本

   **Working Directory**: Ensure you run scripts from the `PAE_NEW` directory

2. **Python路径**: 如果遇到导入错误，确认当前目录在 Python 路径中

   **Python Path**: If you encounter import errors, verify the current directory is in Python path

3. **依赖安装**: 运行前请确保安装所有依赖

   **Dependencies**: Install all dependencies before running
   ```bash
   pip install -r requirements.txt
   ```

4. **向后兼容**: 旧的导入方式将不再工作，请使用新的导入路径

   **Backward Compatibility**: Old import style will no longer work, use new import paths

## 📝 更新日志 / Changelog

- **2025-11-26**: 项目重组完成
  - 创建模块化文件夹结构
  - 更新所有导入语句
  - 添加 `__init__.py` 文件以支持包导入

- **2025-11-26**: Project reorganization completed
  - Created modular folder structure
  - Updated all import statements
  - Added `__init__.py` files for package imports
