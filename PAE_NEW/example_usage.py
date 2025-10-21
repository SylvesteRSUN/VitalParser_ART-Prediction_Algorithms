"""
使用示例脚本
展示如何分步骤使用各个模块
"""

import numpy as np
from config import DATA_CONFIG

def example_1_load_data():
    """示例1: 加载数据"""
    print("=" * 60)
    print("示例 1: 加载 .vital 文件数据")
    print("=" * 60)
    
    from data_loader import VitalDataLoader
    
    # 创建数据加载器
    loader = VitalDataLoader()
    
    # 方式1: 逐步加载
    if loader.load_vital_file():
        # 查看可用的信号轨道
        tracks = loader.get_available_tracks()
        
        # 提取PLETH和ART信号
        if loader.extract_signals():
            pleth, art = loader.get_data()
            loader.get_data_statistics()
            
            print(f"\n✓ 成功加载数据")
            print(f"  PLETH: {len(pleth)} 个数据点")
            print(f"  ART: {len(art)} 个数据点")
            return pleth, art
    
    # 方式2: 一键加载（推荐）
    # pleth, art = loader.load_and_extract()
    
    return None, None


def example_2_process_signal():
    """示例2: 信号处理"""
    print("\n" + "=" * 60)
    print("示例 2: 信号处理和峰值检测")
    print("=" * 60)
    
    from signal_processing import SignalProcessor
    
    # 创建测试信号
    t = np.linspace(0, 10, 1000)
    test_signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.randn(1000)
    
    # 创建信号处理器
    processor = SignalProcessor()
    
    # 处理信号
    processed = processor.process_signal(test_signal, "测试信号")
    
    print(f"\n✓ 信号处理完成")
    print(f"  原始信号: {len(processed['raw_signal'])} 点")
    print(f"  滤波信号: {len(processed['filtered_signal'])} 点")
    print(f"  检测到 {len(processed['peaks'])} 个峰值")
    print(f"  检测到 {len(processed['valleys'])} 个谷值")
    
    return processed


def example_3_extract_features():
    """示例3: 特征提取"""
    print("\n" + "=" * 60)
    print("示例 3: 从PLETH信号提取特征")
    print("=" * 60)
    
    from feature_extraction import FeatureExtractor
    
    # 创建模拟的PLETH处理数据
    pleth_processed = {
        'raw_signal': np.random.randn(1000),
        'filtered_signal': np.random.randn(1000),
        'peaks': np.array([100, 200, 300, 400, 500, 600, 700, 800]),
        'peak_values': np.array([1.5, 1.6, 1.4, 1.7, 1.5, 1.6, 1.5, 1.4]),
        'valleys': np.array([50, 150, 250, 350, 450, 550, 650, 750]),
        'valley_values': np.array([0.5, 0.4, 0.6, 0.5, 0.5, 0.4, 0.6, 0.5]),
        'cycle_integrals': np.array([50, 52, 48, 51, 49, 53, 50]),
        'cycle_durations': np.array([100, 100, 100, 100, 100, 100, 100]),
    }
    
    # 创建特征提取器
    extractor = FeatureExtractor()
    
    # 提取特征
    features = extractor.extract_all_features(pleth_processed)
    
    print(f"\n✓ 特征提取完成")
    print(f"  特征向量维度: {len(features)}")
    print(f"  特征名称: {extractor.feature_names[:5]}... (显示前5个)")
    print(f"\n  部分特征值:")
    for i, (name, value) in enumerate(zip(extractor.feature_names[:5], features[:5])):
        print(f"    {name}: {value:.4f}")
    
    return features, extractor.feature_names


def example_4_train_model():
    """示例4: 训练模型"""
    print("\n" + "=" * 60)
    print("示例 4: 训练预测模型")
    print("=" * 60)
    
    from models import ModelTrainer
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 200
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    # 创建与特征相关的目标值（模拟真实关系）
    y = X[:, 0] * 2 + X[:, 1] * 3 + X[:, 2] * -1 + np.random.randn(n_samples) * 0.5 + 120
    
    print(f"训练数据: {n_samples} 个样本, {n_features} 个特征")
    
    # 创建训练器
    trainer = ModelTrainer()
    
    # 训练所有模型
    results = trainer.train_all_models(X, y, "示例目标")
    
    # 打印对比
    trainer.print_model_comparison(results, "示例目标")
    
    return results


def example_5_evaluate_model():
    """示例5: 评估和可视化"""
    print("\n" + "=" * 60)
    print("示例 5: 模型评估和可视化")
    print("=" * 60)
    
    from evaluation import ModelEvaluator
    
    # 生成模拟的预测结果
    n_test = 100
    y_true = np.random.randn(n_test) * 10 + 120
    y_pred = y_true + np.random.randn(n_test) * 3  # 添加一些误差
    
    # 创建评估器
    evaluator = ModelEvaluator(output_dir='results/examples')
    
    # 模拟结果数据
    mock_result = {
        'model_name': 'example_model',
        'y_test': y_true,
        'predictions_test': y_pred,
    }
    
    # 生成可视化
    print("\n生成可视化图表...")
    evaluator.plot_prediction_vs_actual(y_true, y_pred, 'example_model', 'example_target', 'test')
    evaluator.plot_residuals(y_true, y_pred, 'example_model', 'example_target', 'test')
    
    print(f"\n✓ 可视化完成")
    print(f"  图表保存在: {evaluator.output_dir}/")


def example_6_complete_workflow():
    """示例6: 完整工作流程（使用模拟数据）"""
    print("\n" + "=" * 60)
    print("示例 6: 完整工作流程（模拟数据）")
    print("=" * 60)
    
    from signal_processing import SignalProcessor
    from feature_extraction import FeatureExtractor
    from models import ModelTrainer
    from evaluation import ModelEvaluator
    
    print("\n步骤1: 生成模拟PLETH和ART信号")
    # 生成模拟的生理信号
    t = np.linspace(0, 60, 6000)  # 60秒，100Hz
    
    # PLETH信号（模拟脉搏波形）
    pleth_signal = (
        np.sin(2 * np.pi * 1.2 * t) +  # 心率约72 bpm
        0.3 * np.sin(2 * np.pi * 2.4 * t) +  # 二次谐波
        0.1 * np.random.randn(len(t))  # 噪声
    )
    
    # ART信号（模拟动脉压，与PLETH相关）
    art_signal = (
        120 + 20 * np.sin(2 * np.pi * 1.2 * t) +  # 收缩压约140
        0.5 * np.random.randn(len(t))
    )
    
    print(f"  PLETH信号: {len(pleth_signal)} 点")
    print(f"  ART信号: {len(art_signal)} 点")
    
    print("\n步骤2: 信号处理")
    processor = SignalProcessor()
    pleth_processed = processor.process_signal(pleth_signal, "PLETH")
    art_processed = processor.process_signal(art_signal, "ART")
    
    print("\n步骤3: 特征提取")
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(pleth_processed)
    systolic, diastolic = extractor.extract_targets(art_processed)
    
    print(f"  特征维度: {len(features)}")
    print(f"  目标值 - 收缩压: {systolic:.2f} mmHg")
    print(f"  目标值 - 舒张压: {diastolic:.2f} mmHg")
    
    print("\n步骤4: 创建训练数据集")
    # 创建多个样本（通过滑动窗口）
    n_samples = 50
    X = np.array([features + np.random.randn(len(features)) * 0.1 for _ in range(n_samples)])
    y_sys = np.array([systolic + np.random.randn() * 5 for _ in range(n_samples)])
    y_dia = np.array([diastolic + np.random.randn() * 3 for _ in range(n_samples)])
    
    print(f"  训练样本数: {n_samples}")
    
    print("\n步骤5: 模型训练")
    trainer = ModelTrainer()
    results_sys, results_dia = trainer.train_for_targets(X, y_sys, y_dia)
    
    print("\n步骤6: 评估可视化")
    evaluator = ModelEvaluator(output_dir='results/workflow_example')
    
    # 选择一个模型生成可视化
    if 'random_forest' in results_sys:
        rf_result = results_sys['random_forest'].copy()
        rf_result['y_test'] = results_sys['y_test']
        evaluator.evaluate_single_model(rf_result, "收缩压")
    
    print("\n✓ 完整工作流程演示完成！")


def main_menu():
    """主菜单"""
    print("\n" + "=" * 60)
    print(" " * 15 + "PLETH-BP 使用示例")
    print("=" * 60)
    
    examples = [
        ("加载 .vital 文件数据", example_1_load_data),
        ("信号处理和峰值检测", example_2_process_signal),
        ("特征提取", example_3_extract_features),
        ("训练预测模型", example_4_train_model),
        ("模型评估和可视化", example_5_evaluate_model),
        ("完整工作流程（模拟数据）", example_6_complete_workflow),
    ]
    
    print("\n请选择要运行的示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  0. 运行所有示例")
    print(f"  q. 退出")
    
    choice = input("\n请输入选项: ").strip()
    
    if choice == 'q':
        print("退出程序")
        return
    
    if choice == '0':
        # 运行所有示例
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\n✗ 示例 '{name}' 执行失败: {e}")
                import traceback
                traceback.print_exc()
    else:
        # 运行选定的示例
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                name, func = examples[idx]
                func()
            else:
                print("无效的选项")
        except ValueError:
            print("无效的输入")
        except Exception as e:
            print(f"\n✗ 执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 询问是否继续
    cont = input("\n\n按Enter继续，输入q退出: ").strip()
    if cont.lower() != 'q':
        main_menu()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" " * 10 + "欢迎使用 PLETH-BP 预测系统")
    print(" " * 20 + "使用示例演示")
    print("=" * 60)
    
    print("\n提示:")
    print("- 示例1需要真实的.vital文件")
    print("- 其他示例使用模拟数据，可以直接运行")
    print("- 示例6展示了完整的工作流程")
    
    main_menu()