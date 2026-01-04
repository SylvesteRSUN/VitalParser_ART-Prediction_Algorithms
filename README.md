# VitalParser - 血压预测迁移学习系统
# VitalParser - Blood Pressure Prediction with Transfer Learning

<div align="center">

**基于PPG信号的个性化无创血压预测 | Personalized Non-invasive Blood Pressure Prediction from PPG Signals**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.ai/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)

[English](#english) | [中文](#中文)

</div>

---
<a name="english"></a>

# English Documentation

## 📖 Table of Contents

- [Project Overview](#project-overview-en)
- [Key Features](#key-features-en)
- [Quick Start](#quick-start-en)
- [Project Structure](#project-structure-en)
- [Usage Guide](#usage-guide-en)
- [Configuration](#configuration-en)
- [Results](#results-en)
- [FAQ](#faq-en)

---

<a name="project-overview-en"></a>

## 🎯 Project Overview

**VitalParser ART-Prediction** is an intelligent blood pressure prediction system based on transfer learning. The system analyzes Photoplethysmography (PLETH/PPG) signals and uses transfer learning techniques to build personalized blood pressure prediction models for individual patients.

### Why Transfer Learning?

Traditional general machine learning models have significant prediction errors across different patients (MAE typically 10-15 mmHg). **Transfer Learning** significantly improves accuracy through a two-step strategy:

```
Step 1: General Model Training
├─ Train base model using multi-patient data
├─ Learn universal PPG-BP mappings
└─ MAE: 10-15 mmHg (baseline)

Step 2: Personalization
├─ Use small patient-specific data (20%)
├─ Fine-tune model parameters for individual differences
└─ MAE: 3-5 mmHg (60-70% improvement) ✨
```

### Main Applications

- 🏥 **Clinical Research**: Continuous BP monitoring, hemodynamic analysis
- 💊 **Personalized Medicine**: Custom BP prediction models for each patient
- 📊 **Algorithm Development**: Test and validate new transfer learning strategies
- 🎓 **Education**: Learn medical signal processing and transfer learning

---

<a name="key-features-en"></a>

## ✨ Key Features

### 🔬 Advanced Transfer Learning Framework

- **General Model Training**: Supports XGBoost, LightGBM, Gradient Boosting
- **Personalization**: Two strategies available
  - `incremental`: Add new trees incrementally
  - `correction_model`: Train correction model (recommended)
- **Sample Weighting** (Plan E): Increase weight for extreme BP values
- **Training Diagnostics** (Plan C): Auto-detect underfitting

### 📈 High Accuracy

| Metric | General Model | Personalized | Improvement |
|--------|--------------|--------------|-------------|
| **MAE (mmHg)** | 10-15 | **3-5** | **60-70%** ⭐ |
| **RMSE (mmHg)** | 12-18 | **5-8** | **50-60%** |
| **R²** | 0.75-0.85 | **0.90-0.95** | **10-15%** |

### 🚀 Quick Test Mode

`quick_test_transfer_learning.py` provides:
- ⚡ **5-10x faster** than full pipeline
- 💾 **Reuse models**: Auto-load pre-trained general models
- 🔧 **Parameter tuning**: Rapid iteration for fine-tuning parameters

---

<a name="quick-start-en"></a>

## 🚀 Quick Start

### System Requirements

- **Python**: 3.8+
- **OS**: Windows / Linux / macOS
- **RAM**: 8GB recommended
- **Storage**: 2GB minimum

### Installation

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python tests/test_transfer_learning.py
```

### First Run

#### Option A: Full Pipeline

```bash
python main_transfer_learning.py
```

**Estimated time**: 10-20 minutes

#### Option B: Quick Test (Recommended for tuning)

```bash
python quick_test_transfer_learning.py
```

**Advantages**:
- ⚡ Only 2-3 minutes
- 💾 Auto-load saved general models
- 🔧 Skip time-consuming general model training

---

<a name="project-structure-en"></a>

## 📁 Project Structure

```
PAE_NEW/
├── 🚀 Main Programs
│   ├── main_transfer_learning.py           # Full pipeline
│   └── quick_test_transfer_learning.py     # Quick test
│
├── ⚙️ Configuration (config/)
│   ├── config.py                           # Base config
│   └── config_transfer.py                  # TL config
│
├── 🔧 Core Modules (core/)
│   ├── data_loader.py
│   ├── signal_processing.py
│   ├── feature_extraction.py
│   └── models.py
│
├── 🎓 Transfer Learning (transfer_learning_module/)
│   ├── transfer_learning.py
│   ├── data_splitter.py
│   └── evaluation.py
│
└── 📚 Documentation (docs/)
    ├── QUICKSTART_TRANSFER_LEARNING.md
    ├── TRANSFER_LEARNING_README.md
    └── QUICK_TEST_README.md
```

### Key Paths

```python
DATA_CONFIG = {
    'train_data_dir': r'F:\...\records',
    'test_data_dir': r'F:\...\testset',
    'results_dir': r'F:\...\results_transfer_learning',
    'models_dir': r'F:\...\saved_models',
}
```

---

<a name="usage-guide-en"></a>

## 📖 Usage Guide

### Command Line Arguments

```bash
python main_transfer_learning.py \
    --test-file path/to/test.vital \
    --model-type xgboost \
    --calibration-samples 200 \
    --verbose
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--test-file` | Test file path | First .vital in testset/ |
| `--model-type` | Model type | `xgboost` |
| `--calibration-samples` | Calibration samples | 500 |
| `--verbose` | Verbose output | False |

---

<a name="configuration-en"></a>

## ⚙️ Configuration

### General Model Config

```python
GENERAL_MODEL_CONFIG = {
    'model_type': 'xgboost',
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 12,          # Plan G: 10→12
        'learning_rate': 0.1,
        'reg_lambda': 0.3,        # Plan G: 1.0→0.3
        'max_delta_step': 2,      # Plan G: allow larger steps
    }
}
```

### Fine-Tuning Config

```python
FINE_TUNING_CONFIG = {
    'strategy': 'correction_model',  # Recommended
    'sample_weighting': {
        'enabled': True,             # Plan E
        'extreme_multiplier': 2.0,
    },
    'xgboost': {
        'n_estimators': 200,
        'learning_rate': 0.1,        # Plan G: 0.05→0.1
        'max_depth': 10,             # Plan G: 15→10
        'reg_lambda': 0.1,           # Plan G: 0.5→0.1
    }
}
```

### Data Split Config

```python
DATA_SPLIT_CONFIG = {
    'split_method': 'ratio_based',  # Recommended
    'ratio_based': {
        'calibration_ratio': 0.20,  # 20% for calibration
        'min_samples': 200,
        'max_samples': 1000
    }
}
```

---

<a name="results-en"></a>

## 📊 Results

### Output Directory

```
results_transfer_learning/[test_file_name]/
├── plots/
│   ├── transfer_learning_comparison_Systolic.png
│   ├── transfer_learning_comparison_Diastolic.png
│   └── transfer_learning_improvement_summary.png
├── reports/
│   └── transfer_learning_report.txt
└── predictions/
    └── predictions.csv
```
---

<a name="faq-en"></a>

## ❓ FAQ

### Q: Which script should I use?

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| **First run** | `main_transfer_learning.py` | Need to train general model |
| **Parameter tuning** | `quick_test_transfer_learning.py` ⭐ | 5-10x faster |
| **New patient** | `quick_test_transfer_learning.py` | General model ready |

### Q: How to improve accuracy?

1. **Increase calibration samples**
   ```python
   DATA_SPLIT_CONFIG['ratio_based']['calibration_ratio'] = 0.30
   ```

2. **Adjust learning rate**
   ```python
   FINE_TUNING_CONFIG['xgboost']['learning_rate'] = 0.05
   ```

3. **Enable early stopping**
   ```python
   FINE_TUNING_CONFIG['early_stopping']['enabled'] = True
   ```

---

<a name="中文"></a>

## 📖 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [详细使用指南](#详细使用指南)
- [配置说明](#配置说明)
- [输出结果](#输出结果)
- [性能评估](#性能评估)
- [常见问题](#常见问题)
- [技术原理](#技术原理)

---

## 🎯 项目简介

**VitalParser ART-Prediction** 是一个基于迁移学习的智能血压预测系统。该系统通过分析脉搏波（PLETH/PPG）信号，结合迁移学习技术，为每个患者构建个性化的血压预测模型。

### 为什么需要迁移学习？

传统的通用机器学习模型在不同患者之间存在较大的预测误差（MAE通常为10-15 mmHg）。**迁移学习**通过以下两步策略显著提升精度：

```
步骤1: 通用模型训练
├─ 使用多个患者的数据训练基础模型
├─ 学习通用的PPG-BP映射关系
└─ MAE: 10-15 mmHg（基线性能）

步骤2: 个性化微调
├─ 使用少量患者特异性数据（20%）
├─ 微调模型参数以适应个体差异
└─ MAE: 3-5 mmHg（改善60-70%） ✨
```

### 主要应用场景

- 🏥 **临床研究**: 连续血压监测研究、血流动力学参数分析
- 💊 **个性化医疗**: 为每位患者定制专属的血压预测模型
- 📊 **算法开发**: 测试和验证新的迁移学习策略
- 🎓 **教育培训**: 学习医疗信号处理和迁移学习技术

---

## ✨ 核心特性

### 🔬 先进的迁移学习框架

- **通用模型训练**: 支持XGBoost、LightGBM、Gradient Boosting
- **个性化微调**: 两种策略可选
  - `incremental`: 增量添加新树
  - `correction_model`: 训练校正模型（推荐）
- **样本加权** (Plan E): 对极端血压值增加权重，改善边界预测
- **训练诊断** (Plan C): 自动检测欠拟合并提供优化建议

### 📈 高精度预测

| 指标 | 通用模型 | 个性化模型 | 改善幅度 |
|------|----------|------------|---------|
| **MAE (mmHg)** | 10-15 | **3-5** | **60-70%** ⭐ |
| **RMSE (mmHg)** | 12-18 | **5-8** | **50-60%** |
| **R² Score** | 0.75-0.85 | **0.90-0.95** | **10-15%** |

### 🚀 快速测试模式

提供 `quick_test_transfer_learning.py` 脚本：
- ⚡ **速度提升**: 比完整流程快5-10倍
- 💾 **复用模型**: 自动加载预训练的通用模型
- 🔧 **参数调试**: 快速迭代测试不同的微调参数

### 🛠️ 灵活配置

- **多种校准方式**: 样本数量、时间长度、百分比（自适应）
- **参数优化**: 完整的超参数配置（已针对血压预测优化）
- **可视化分析**: 自动生成对比图表和性能报告

---

## 🚀 快速开始

### 系统要求

- **Python**: 3.8 或更高版本
- **操作系统**: Windows / Linux / macOS
- **内存**: 建议 8GB RAM
- **存储**: 至少 2GB 可用空间

### 安装步骤

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**核心依赖**:
```
vitaldb >= 1.7.0      # VitalDB数据加载
numpy >= 1.21.0       # 数值计算
scipy >= 1.7.0        # 信号处理
scikit-learn >= 1.0.0 # 机器学习
xgboost >= 1.5.0      # XGBoost模型（推荐）
matplotlib >= 3.5.0   # 可视化
```

#### 2. 验证安装

```bash
python tests/test_transfer_learning.py
```

**预期输出**:
```
✓ Configuration module loaded successfully
✓ Data splitter working correctly
✓ Transfer learning core working correctly
✓ Evaluation visualizations working correctly
✓ Main pipeline module loaded successfully

ALL TESTS PASSED!
```

### 第一次运行

#### 选项A: 完整流程（首次使用）

```bash
python main_transfer_learning.py
```

**流程说明**:
1. 从 `..\records\` 加载所有训练数据
2. 训练通用模型（XGBoost，收缩压和舒张压）
3. 保存通用模型到 `..\saved_models\general_models\`
4. 从 `..\testset\` 加载测试数据
5. 分割校准集（20%）和评估集（80%）
6. 个性化微调
7. 生成对比报告和可视化
8. 保存结果到 `..\results_transfer_learning\`

**预计时间**: 10-20分钟（取决于数据量和硬件）

#### 选项B: 快速测试（推荐用于参数调试）

```bash
python quick_test_transfer_learning.py
```

**优势**:
- ⚡ 仅需 2-3分钟
- 💾 自动加载已保存的通用模型
- 🔧 跳过耗时的通用模型训练
- 🎯 专注于测试微调效果

**使用场景**:
- 测试不同的校准样本数量
- 调整微调超参数
- 验证新的分割策略

---

## 📁 项目结构

```
PAE_NEW/
│
├── 🚀 主程序
│   ├── main_transfer_learning.py           # 完整迁移学习流程
│   └── quick_test_transfer_learning.py     # 快速测试（复用预训练模型）
│
├── ⚙️ 配置模块 (config/)
│   ├── config.py                           # 基础配置
│   │   ├── DATA_CONFIG                     # 数据路径配置
│   │   ├── SIGNAL_CONFIG                   # 信号处理参数
│   │   └── FEATURE_CONFIG                  # 特征提取配置
│   │
│   └── config_transfer.py                  # 迁移学习专用配置 ⭐
│       ├── GENERAL_MODEL_CONFIG            # 通用模型参数（XGBoost/LightGBM/GradientBoosting）
│       ├── FINE_TUNING_CONFIG              # 微调策略和超参数
│       ├── DATA_SPLIT_CONFIG               # 数据分割方法（样本/时间/比例）
│       ├── EVALUATION_CONFIG               # 评估指标和可视化
│       └── PATH_CONFIG                     # 输出路径配置
│
├── 🔧 核心功能模块 (core/)
│   ├── data_loader.py                      # VitalDB数据加载
│   │   ├── 支持多信号候选（Intellivue/Demo/SNUADC）
│   │   ├── 自动寻找PLETH和ART信号
│   │   └── 采样率统一化（100 Hz）
│   │
│   ├── signal_processing.py                # 信号处理
│   │   ├── Savitzky-Golay滤波
│   │   ├── Gaussian平滑
│   │   ├── 峰值检测（收缩压/舒张压）
│   │   └── 异常值处理
│   │
│   ├── feature_extraction.py               # 特征提取
│   │   ├── 基于心动周期的特征提取
│   │   ├── 峰值-谷值特征（5个）
│   │   ├── 脉动幅度特征（3个）
│   │   ├── 时间特征（4个）
│   │   ├── 周期积分（2个）
│   │   └── 波形形状特征（3个）
│   │   └── 总计: ~17个特征
│   │
│   ├── models.py                           # 模型定义
│   └── utils.py                            # 工具函数
│
├── 🎓 迁移学习模块 (transfer_learning_module/)
│   ├── transfer_learning.py                # 核心迁移学习引擎 ⭐⭐⭐
│   │   ├── GeneralTrainer                  # 通用模型训练器
│   │   │   └── train(X, y_sys, y_dia) → (model_sys, model_dia)
│   │   │
│   │   ├── PersonalFineTuner               # 个性化微调器
│   │   │   ├── fine_tune() → (model_sys, model_dia)
│   │   │   ├── 策略: incremental / correction_model
│   │   │   ├── 样本加权（Plan E）
│   │   │   └── early stopping支持
│   │   │
│   │   └── ModelManager                    # 模型管理器
│   │       ├── save_general_models()
│   │       ├── load_general_models()
│   │       └── save_personalized_models()
│   │
│   ├── data_splitter.py                    # 数据分割器
│   │   ├── PatientDataSplitter
│   │   │   ├── sample_based: 固定样本数（如500个心跳）
│   │   │   ├── time_based: 固定时长（如5分钟）
│   │   │   └── ratio_based: 百分比（如20%，推荐）
│   │   │
│   │   └── MultiSizeSplitter
│   │       └── 多尺寸实验（寻找最优校准集大小）
│   │
│   └── evaluation.py                       # 评估和可视化
│       ├── ModelEvaluator
│       │   ├── plot_transfer_learning_comparison()  # 4格对比图
│       │   ├── plot_improvement_summary()           # 改善百分比
│       │   └── generate_evaluation_report()         # 详细报告
│       │
│       └── 指标计算: MAE, MSE, RMSE, R²
│
├── 📊 测试模块 (tests/)
│   ├── test_transfer_learning.py           # 完整功能测试
│   ├── test_setup.py                       # 环境检查
│   └── single_file_split.py                # 单文件分割测试
│
├── 📚 文档 (docs/)
│   ├── QUICKSTART_TRANSFER_LEARNING.md     # 5分钟快速开始 ⭐
│   ├── TRANSFER_LEARNING_README.md         # 完整技术文档
│   ├── QUICK_TEST_README.md                # 快速测试指南
│   ├── transfer_learning.md                # 技术设计文档
│   ├── PLAN_E_IMPLEMENTATION.md            # 极端值加权方案
│   └── CALIBRATION_RATIO_UPDATE.md         # 校准比例更新说明
│
├── PROJECT_STRUCTURE.md                    # 项目结构说明
├── README.md                               # 本文档
└── requirements.txt                        # 依赖清单
```

---

## 📖 详细使用指南

### 命令行参数

#### main_transfer_learning.py

```bash
python main_transfer_learning.py [选项]
```

**可用参数**:

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--test-file` | 指定测试文件路径 | testset目录中的第一个.vital文件 | `--test-file path/to/test.vital` |
| `--model-type` | 模型类型 | `xgboost` | `--model-type lightgbm` |
| `--calibration-samples` | 校准样本数（sample_based模式） | 500 | `--calibration-samples 300` |
| `--verbose` | 详细输出模式 | False | `--verbose` |


#### quick_test_transfer_learning.py

```bash
python quick_test_transfer_learning.py [选项]
```

**参数说明**: 与 `main_transfer_learning.py` 相同

**区别**:
- 自动加载预训练的通用模型
- 跳过步骤1-4（数据加载、训练、保存）
- 直接从步骤5开始（加载测试数据、分割、微调、评估）

---

### 工作流程详解

#### 完整流程（main_transfer_learning.py）

```
┌─────────────────────────────────────────────────────────────┐
│ 步骤1: 加载并处理训练数据                                   │
├─────────────────────────────────────────────────────────────┤
│ • 从 records/ 目录加载所有.vital文件                        │
│ • 提取PLETH和ART信号（尝试多个候选名称）                   │
│ • 信号处理: Savitzky-Golay滤波 + Gaussian平滑             │
│ • 峰值检测: 定位收缩压峰值                                 │
│ • 特征提取: 17个周期特征                                   │
│ • 特征标准化: StandardScaler                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤2: 训练通用模型                                         │
├─────────────────────────────────────────────────────────────┤
│ • 使用GeneralTrainer类                                      │
│ • 模型类型: XGBoost（推荐）/ LightGBM / GradientBoosting   │
│ • 分别训练收缩压和舒张压模型                               │
│ • 参数: config_transfer.py → GENERAL_MODEL_CONFIG          │
│   - n_estimators: 200                                       │
│   - max_depth: 12                                           │
│   - learning_rate: 0.1                                      │
│   - reg_lambda: 0.3 (Plan G优化)                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤3: 训练诊断 (Plan C)                                    │
├─────────────────────────────────────────────────────────────┤
│ • 在训练集上预测                                           │
│ • 检查预测范围: pred_range / true_range                    │
│ • 如果 < 0.3: 警告可能欠拟合                               │
│ • 检查R²: 如果 < 0.5: 建议使用更复杂模型                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤4: 保存通用模型                                         │
├─────────────────────────────────────────────────────────────┤
│ • 保存到: saved_models/general_models/                      │
│ • 文件:                                                     │
│   - general_model_systolic.pkl                              │
│   - general_model_diastolic.pkl                             │
│   - feature_scaler.pkl                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤5: 加载并处理测试数据                                   │
├─────────────────────────────────────────────────────────────┤
│ • 从 testset/ 目录加载测试文件                             │
│ • 使用相同的信号处理和特征提取流程                         │
│ • 使用保存的StandardScaler进行标准化（重要！）             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤6: 分割测试数据                                         │
├─────────────────────────────────────────────────────────────┤
│ • 使用PatientDataSplitter                                   │
│ • 方法: ratio_based（推荐）                                │
│   - 校准集: 前20%数据                                      │
│   - 评估集: 后80%数据                                      │
│ • 质量控制:                                                │
│   - 检查数据连续性                                         │
│   - 移除离群值（可选）                                     │
│   - 最小样本数: 200                                        │
│   - 最大样本数: 1000                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤7: 评估通用模型                                         │
├─────────────────────────────────────────────────────────────┤
│ • 在评估集上预测                                           │
│ • 计算指标: MAE, MSE, RMSE, R²                             │
│ • 记录基线性能                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤8: 个性化微调                                           │
├─────────────────────────────────────────────────────────────┤
│ • 使用PersonalFineTuner类                                   │
│ • 策略: correction_model（推荐）                            │
│   - 训练校正模型来修正通用模型的误差                       │
│   - 最终预测 = 通用模型预测 + 校正模型预测                 │
│ • 样本加权 (Plan E):                                        │
│   - 对 |BP - mean| > std 的样本加权×2.0                    │
│   - 改善对极端血压值的预测                                 │
│ • Early Stopping:                                           │
│   - 使用20%校准数据作为验证集                              │
│   - 20轮无改善则停止                                       │
│ • 参数: FINE_TUNING_CONFIG                                  │
│   - n_estimators: 200                                       │
│   - learning_rate: 0.1 (Plan G优化)                        │
│   - max_depth: 10                                           │
│   - reg_lambda: 0.1 (降低正则化)                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤9: 评估个性化模型                                       │
├─────────────────────────────────────────────────────────────┤
│ • 在评估集上预测                                           │
│ • 计算指标: MAE, MSE, RMSE, R²                             │
│ • 计算改善百分比                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤10: 生成报告和可视化                                    │
├─────────────────────────────────────────────────────────────┤
│ • 对比图表:                                                │
│   - 时间序列（实际值 vs 通用模型 vs 个性化模型）           │
│   - 散点图（预测值 vs 实际值）                             │
│   - 误差分布直方图                                         │
│   - 性能指标柱状图                                         │
│ • 改善总结:                                                │
│   - 各指标改善百分比                                       │
│ • 文本报告:                                                │
│   - 数据集统计                                             │
│   - 通用模型性能                                           │
│   - 个性化模型性能                                         │
│   - 改善详情                                               │
└─────────────────────────────────────────────────────────────┘
```

#### 快速测试流程（quick_test_transfer_learning.py）

```
┌─────────────────────────────────────────────────────────────┐
│ 步骤1: 加载预训练模型                                       │
├─────────────────────────────────────────────────────────────┤
│ • 从 saved_models/general_models/ 加载                      │
│ • 加载文件:                                                │
│   - general_model_systolic.pkl                              │
│   - general_model_diastolic.pkl                             │
│   - feature_scaler.pkl                                      │
│ • 如果文件不存在: 提示运行完整流程                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
             [直接跳转到步骤5-10]
                            ↓
                    [与完整流程相同]
```

---

## ⚙️ 配置说明

### 核心配置文件: config_transfer.py

#### 1. 通用模型配置 (GENERAL_MODEL_CONFIG)

```python
GENERAL_MODEL_CONFIG = {
    'model_type': 'xgboost',  # 选项: 'xgboost', 'lightgbm', 'gradient_boosting'

    'xgboost': {
        'n_estimators': 200,      # 树的数量
        'max_depth': 12,          # 树的深度（Plan G: 从10增加）
        'learning_rate': 0.1,     # 学习率
        'subsample': 0.8,         # 行采样比例
        'colsample_bytree': 0.8,  # 列采样比例
        'min_child_weight': 1,    # 最小子节点权重（Plan G: 从3降低）
        'gamma': 0,               # 分裂所需最小损失减少（Plan G: 移除惩罚）
        'reg_alpha': 0.005,       # L1正则化（Plan G: 从0.01降低）
        'reg_lambda': 0.3,        # L2正则化（Plan G: 从1.0大幅降低）
        'max_delta_step': 2,      # 最大预测步长（Plan G: 允许更大步长）
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1
    }
}
```

#### 2. 数据分割配置 (DATA_SPLIT_CONFIG)

```python
DATA_SPLIT_CONFIG = {
    'split_method': 'ratio_based',  # 推荐方法

    # 方法1: 基于样本数
    'sample_based': {
        'n_samples': 500,       # 固定使用500个心动周期
        'min_samples': 300,     # 最少300个
        'max_samples': 900      # 最多900个
    },

    # 方法2: 基于时间
    'time_based': {
        'duration_minutes': 5,      # 固定5分钟
        'sampling_rate': 100,       # 100 Hz
        'expected_heart_rate': 60   # 60 bpm（估算样本数）
    },

    # 方法3: 基于比例（自适应，推荐）
    'ratio_based': {
        'calibration_ratio': 0.20,  # 20%用于校准
        'min_samples': 200,         # 最少200个样本
        'max_samples': 1000         # 最多1000个样本
    },

    # 质量控制
    'quality_control': {
        'check_continuity': True,      # 检查数据连续性
        'remove_outliers': False,      # 是否移除离群值
        'outlier_threshold': 3.0       # Z-score阈值
    }
}
```

**分割方法选择建议**:

| 方法 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| `sample_based` | 已知最优样本数 | 精确控制 | 不同患者心率差异大 |
| `time_based` | 需要固定时长的数据 | 时间一致 | 样本数不固定 |
| `ratio_based` ⭐ | 通用场景 | 自适应，公平 | 需设置合理范围 |

#### 3. 评估配置 (EVALUATION_CONFIG)

```python
EVALUATION_CONFIG = {
    # 目标性能
    'targets': {
        'systolic': {
            'MAE': 10,  # 目标MAE < 10 mmHg
            'MSE': 70   # 目标MSE < 70
        },
        'diastolic': {
            'MAE': 10,
            'MSE': 60
        }
    },

    # 可视化选项
    'visualize': {
        'time_series': True,           # 时间序列对比
        'scatter': True,               # 散点图
        'error_distribution': True,    # 误差分布
        'metrics_comparison': True,    # 指标对比
        'improvement_summary': True,   # 改善总结
        'max_points_timeseries': 500   # 时间序列最多显示500点
    },

    # 输出选项
    'save_predictions': True,      # 保存预测值CSV
    'generate_report': True        # 生成文本报告
}
```

---

## 📊 输出结果

所有结果保存在:
```
..\results_transfer_learning\[测试文件名]\
```

### 目录结构

```
results_transfer_learning/
└── [test_file_name]/                    # 例如: patient_001
    ├── plots/                           # 📊 可视化图表
    │   ├── transfer_learning_comparison_Systolic.png
    │   ├── transfer_learning_comparison_Diastolic.png
    │   └── transfer_learning_improvement_summary.png
    │
    ├── reports/                         # 📝 文本报告
    │   └── transfer_learning_report.txt
    │
    └── predictions/                     # 📈 预测值
        └── predictions.csv
```

#### 4. 保存的模型文件

#### 通用模型 (saved_models/general_models/)

```
saved_models/
└── general_models/
    ├── general_model_systolic.pkl        # 收缩压通用模型
    ├── general_model_diastolic.pkl       # 舒张压通用模型
    └── feature_scaler.pkl                # 特征标准化器
```

**用途**:
- 快速测试脚本自动加载
- 部署到其他系统
- 作为新患者的初始模型

#### 个性化模型 (saved_models/personalized_models/)

```
saved_models/
└── personalized_models/
    └── [test_file_name]/
        ├── personalized_model_systolic.pkl
        └── personalized_model_diastolic.pkl
```

**用途**:
- 为特定患者保存定制模型
- 后续预测时直接加载
- 避免重复微调

---

## ❓ 常见问题

### Q1: 我应该使用哪个脚本？

**回答**:

| 场景 | 推荐脚本 | 原因 |
|------|---------|------|
| **第一次运行** | `main_transfer_learning.py` | 需要训练并保存通用模型 |
| **调试参数** | `quick_test_transfer_learning.py` ⭐ | 快5-10倍，专注于微调 |
| **训练数据变化** | `main_transfer_learning.py` | 需要重新训练通用模型 |
| **测试新患者** | `quick_test_transfer_learning.py` | 通用模型已训练好 |

### Q2: 如何提高预测精度？

**策略**:

1. **增加校准样本数**
   ```python
   # 在 config_transfer.py 中
   DATA_SPLIT_CONFIG['ratio_based']['calibration_ratio'] = 0.30  # 从0.20提高到0.30
   ```

2. **调整微调学习率**
   ```python
   # 降低学习率以更细致地学习
   FINE_TUNING_CONFIG['xgboost']['learning_rate'] = 0.05  # 从0.1降低到0.05
   ```

3. **启用early stopping**
   ```python
   FINE_TUNING_CONFIG['early_stopping']['enabled'] = True
   ```

4. **增加通用模型复杂度**
   ```python
   GENERAL_MODEL_CONFIG['xgboost']['n_estimators'] = 300  # 从200增加到300
   GENERAL_MODEL_CONFIG['xgboost']['max_depth'] = 15      # 从12增加到15
   ```

5. **使用样本加权**
   ```python
   FINE_TUNING_CONFIG['sample_weighting']['enabled'] = True
   ```

### 特征工程

**17个特征**:

```python
# 1. 峰值-谷值特征 (5个)
peak_value          # 峰值（收缩压相关）
valley_value        # 谷值（舒张压相关）
peak_to_valley      # 峰谷差（脉压相关）
valley_to_peak_ratio
peak_position       # 峰值位置（归一化）

# 2. 脉动幅度 (3个)
pulse_amplitude     # 脉搏幅度
normalized_amplitude
amplitude_variability

# 3. 时间特征 (4个)
cycle_duration      # 周期时长
heart_rate          # 心率
time_to_peak        # 到达峰值时间
time_after_peak     # 峰值后时间

# 4. 周期积分 (2个)
cycle_area          # 周期下面积
normalized_area

# 5. 波形形状 (3个)
upslope             # 上升斜率
downslope           # 下降斜率
skewness            # 偏度
```

### 信号处理流程

```
原始PLETH信号
      ↓
┌─────────────────┐
│ 1. 移除NaN值     │
└─────────────────┘
      ↓
┌─────────────────┐
│ 2. Savitzky-    │
│    Golay滤波    │  参数: window=51, polyorder=3
│ (平滑噪声)      │
└─────────────────┘
      ↓
┌─────────────────┐
│ 3. Gaussian     │
│    平滑滤波     │  参数: sigma=5
└─────────────────┘
      ↓
┌─────────────────┐
│ 4. 异常值检测   │  方法: Z-score > 3
│    (可选)       │
└─────────────────┘
      ↓
┌─────────────────┐
│ 5. 峰值检测     │  scipy.signal.find_peaks
│                 │  prominence=0.2, distance=30
└─────────────────┘
      ↓
┌─────────────────┐
│ 6. 周期分割     │  每两个相邻峰值之间
└─────────────────┘
      ↓
  处理后的信号
```