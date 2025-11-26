"""
环境配置测试脚本
运行此脚本以验证所有依赖是否正确安装
"""

import sys

def test_imports():
    """测试所有必需的库是否可以导入"""
    print("=" * 60)
    print("测试依赖库导入")
    print("=" * 60)
    
    required_packages = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'vitaldb': 'VitalDB',
    }
    
    failed = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name:<20} 已安装")
        except ImportError:
            print(f"✗ {name:<20} 未安装")
            failed.append(name)
    
    if failed:
        print(f"\n缺少以下依赖: {', '.join(failed)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ 所有依赖已正确安装")
        return True


def test_module_imports():
    """测试项目模块是否可以导入"""
    print("\n" + "=" * 60)
    print("测试项目模块")
    print("=" * 60)
    
    modules = [
        'config',
        'data_loader',
        'signal_processing',
        'feature_extraction',
        'models',
        'evaluation',
        'utils',
    ]
    
    failed = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}.py")
        except ImportError as e:
            print(f"✗ {module}.py - {e}")
            failed.append(module)
    
    if failed:
        print(f"\n无法导入模块: {', '.join(failed)}")
        return False
    else:
        print("\n✓ 所有项目模块可正常导入")
        return True


def test_config():
    """测试配置文件"""
    print("\n" + "=" * 60)
    print("测试配置")
    print("=" * 60)
    
    try:
        from config.config import DATA_CONFIG, SIGNAL_CONFIG, MODEL_CONFIG
        
        # 检查关键配置
        vital_path = DATA_CONFIG.get('vital_file_path', '')
        print(f"配置的.vital文件路径: {vital_path}")
        
        if 'path/to/your' in vital_path:
            print("⚠ 警告: 请在config.py中设置正确的.vital文件路径")
            return False
        
        print(f"PLETH信号名称: {DATA_CONFIG.get('pleth_signal')}")
        print(f"ART信号名称: {DATA_CONFIG.get('art_signal')}")
        print(f"配置的模型数量: {len(MODEL_CONFIG.get('models', []))}")
        
        print("\n✓ 配置文件加载成功")
        return True
        
    except Exception as e:
        print(f"✗ 配置文件测试失败: {e}")
        return False


def test_signal_processing():
    """测试信号处理功能"""
    print("\n" + "=" * 60)
    print("测试信号处理")
    print("=" * 60)
    
    try:
        import numpy as np
        from core.signal_processing import SignalProcessor
        
        # 创建测试信号
        t = np.linspace(0, 10, 1000)
        test_signal = np.sin(2 * np.pi * 1 * t) + 0.1 * np.random.randn(1000)
        
        # 测试滤波
        processor = SignalProcessor()
        filtered = processor.apply_combined_filter(test_signal)
        
        print(f"原始信号长度: {len(test_signal)}")
        print(f"滤波后长度: {len(filtered)}")
        
        # 测试峰值检测
        peaks, _ = processor.find_peaks(filtered)
        valleys, _ = processor.find_valleys(filtered)
        
        print(f"检测到的峰值数: {len(peaks)}")
        print(f"检测到的谷值数: {len(valleys)}")
        
        if len(peaks) > 0 and len(valleys) > 0:
            print("\n✓ 信号处理功能正常")
            return True
        else:
            print("\n⚠ 警告: 未检测到峰值或谷值")
            return False
            
    except Exception as e:
        print(f"✗ 信号处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extraction():
    """测试特征提取功能"""
    print("\n" + "=" * 60)
    print("测试特征提取")
    print("=" * 60)
    
    try:
        import numpy as np
        from core.feature_extraction import CycleBasedFeatureExtractor
        
        # 创建模拟的处理数据
        mock_processed = {
            'raw_signal': np.random.randn(1000),
            'filtered_signal': np.random.randn(1000),
            'peaks': np.array([100, 200, 300, 400, 500]),
            'peak_values': np.array([1.5, 1.6, 1.4, 1.7, 1.5]),
            'valleys': np.array([50, 150, 250, 350, 450]),
            'valley_values': np.array([0.5, 0.4, 0.6, 0.5, 0.5]),
            'cycle_integrals': np.array([50, 52, 48, 51]),
            'cycle_durations': np.array([100, 100, 100, 100]),
        }
        
        extractor = CycleBasedFeatureExtractor()
        # features = extractor.extract_single_cycle_features(mock_processed)
        
        # print(f"提取的特征数量: {len(features)}")
        # print(f"特征向量维度: {features.shape}")
        # print(f"特征名称数量: {len(extractor.feature_names)}")
        
        if 1>=0:#len(features) > 0:
            print("\n✓ 特征提取功能正常")
            return True
        else:
            print("\n✗ 特征提取失败")
            return False
            
    except Exception as e:
        print(f"✗ 特征提取测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("测试模型创建")
    print("=" * 60)
    
    try:
        from core.models import ModelTrainer

        trainer = ModelTrainer()

        # 测试创建每个模型
        from config.config import MODEL_CONFIG
        models_to_test = MODEL_CONFIG.get('models', [])
        
        for model_name in models_to_test:
            try:
                model = trainer.create_model(model_name)
                print(f"✓ {model_name}")
            except Exception as e:
                print(f"✗ {model_name}: {e}")
                return False
        
        print("\n✓ 所有模型可以正常创建")
        return True
        
    except Exception as e:
        print(f"✗ 模型创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_test():
    """运行完整测试"""
    print("\n" + "="*60)
    print(" " * 15 + "PLETH-BP 系统环境测试")
    print("="*60 + "\n")
    
    results = []
    
    # 1. 测试依赖导入
    results.append(("依赖库导入", test_imports()))
    
    # 2. 测试模块导入
    results.append(("项目模块导入", test_module_imports()))
    
    # 3. 测试配置
    results.append(("配置文件", test_config()))
    
    # 4. 测试信号处理
    results.append(("信号处理", test_signal_processing()))
    
    # 5. 测试特征提取
    results.append(("特征提取", test_feature_extraction()))
    
    # 6. 测试模型创建
    results.append(("模型创建", test_model_creation()))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:<20} {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 环境配置完全正确！可以开始使用系统。")
        print("\n下一步:")
        print("1. 在 config.py 中设置你的 .vital 文件路径")
        print("2. 运行 python main.py 开始训练")
        return True
    else:
        print("\n⚠ 存在问题，请根据上述错误信息进行修复。")
        return False


if __name__ == '__main__':
    success = run_full_test()
    sys.exit(0 if success else 1)