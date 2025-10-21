"""
主程序 - PLETH-BP预测系统（逐周期方法）
从PLETH信号预测动脉血压(ABP/ART)的机器学习系统
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import DATA_CONFIG, RUN_CONFIG
from data_loader import load_train_test_data
from signal_processing import SignalProcessor
from feature_extraction import CycleBasedFeatureExtractor
from models import ModelTrainer
from evaluation import ModelEvaluator


def main():
    """
    主函数 - 执行完整的训练和评估流程
    """
    print("=" * 70)
    print(" " * 20 + "PLETH-BP Prediction System")
    print(" " * 15 + "Cycle-Based Feature Extraction")
    print("=" * 70)
    
    # ==================== 步骤1: 加载数据 ====================
    print("\n" + "=" * 70)
    print("Step 1/4: Load Training and Test Data")
    print("=" * 70)
    
    pleth_train, art_train, pleth_test, art_test = load_train_test_data()
    
    if pleth_train is None or pleth_test is None:
        print("\n✗ Data loading failed, exiting")
        return
    
    # ==================== 步骤2: 信号处理 ====================
    print("\n" + "=" * 70)
    print("Step 2/4: Signal Processing and Peak Detection")
    print("=" * 70)
    
    processor = SignalProcessor()
    
    print("\n[Training Set]")
    pleth_train_processed = processor.process_signal(pleth_train, "PLETH_train")
    art_train_processed = processor.process_signal(art_train, "ART_train")
    
    print("\n[Test Set]")
    pleth_test_processed = processor.process_signal(pleth_test, "PLETH_test")
    art_test_processed = processor.process_signal(art_test, "ART_test")
    
    # ==================== 步骤3: 逐周期特征提取 ====================
    print("\n" + "=" * 70)
    print("Step 3/4: Cycle-Based Feature Extraction")
    print("=" * 70)
    
    extractor = CycleBasedFeatureExtractor()
    
    # 提取训练集特征
    print("\n[Training Set Feature Extraction]")
    X_train, y_train_systolic, y_train_diastolic = extractor.prepare_cycle_based_dataset(
        pleth_train_processed, art_train_processed
    )
    
    # 提取测试集特征（使用训练集的scaler）
    print("\n[Test Set Feature Extraction]")
    # 保存训练集的scaler
    train_scaler = extractor.scaler
    
    # 创建新的extractor用于测试集
    test_extractor = CycleBasedFeatureExtractor()
    test_extractor.scaler = None  # 先不标准化
    
    X_test_raw, y_test_systolic, y_test_diastolic = test_extractor.prepare_cycle_based_dataset(
        pleth_test_processed, art_test_processed
    )
    
    # 使用训练集的scaler标准化测试集
    if train_scaler is not None:
        X_test = train_scaler.transform(X_test_raw)
    else:
        X_test = X_test_raw
    
    print(f"\n✓ Dataset Ready:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    
    # 检查数据有效性
    if len(X_train) < 10 or len(X_test) < 5:
        print("\n✗ Insufficient samples, exiting")
        print("  Hint: Check if signal quality is good and peak detection is working")
        return
    
    # ==================== 步骤4: 模型训练 ====================
    print("\n" + "=" * 70)
    print("Step 4/4: Model Training and Evaluation")
    print("=" * 70)
    
    trainer = ModelTrainer()
    
    # 训练收缩压模型
    print(f"\n{'='*70}")
    print("Training Systolic BP Models")
    print(f"{'='*70}")
    results_systolic = trainer.train_models_with_split(
        X_train, y_train_systolic, X_test, y_test_systolic, "Systolic"
    )
    
    # 训练舒张压模型
    print(f"\n{'='*70}")
    print("Training Diastolic BP Models")
    print(f"{'='*70}")
    results_diastolic = trainer.train_models_with_split(
        X_train, y_train_diastolic, X_test, y_test_diastolic, "Diastolic"
    )
    
    # 打印模型对比
    print("\n" + "=" * 70)
    print("Model Performance Summary")
    print("=" * 70)
    
    trainer.print_model_comparison(results_systolic, "Systolic")
    trainer.print_model_comparison(results_diastolic, "Diastolic")
    
    # ==================== 评估和可视化 ====================
    print("\n" + "=" * 70)
    print("Generating Evaluation Reports")
    print("=" * 70)
    
    evaluator = ModelEvaluator()
    
    # 为最佳模型生成详细可视化
    best_model = trainer.best_model
    print(f"\nGenerating visualizations for best model: {best_model}")
    
    if best_model in results_systolic:
        # 准备收缩压结果
        sys_result = {
            'model_name': best_model,
            'y_test': results_systolic['y_test'],
            'predictions_test': results_systolic[best_model]['predictions_test'],
            'metrics_test': results_systolic[best_model]['metrics_test'],
        }
        
        # 准备舒张压结果
        dia_result = {
            'model_name': best_model,
            'y_test': results_diastolic['y_test'],
            'predictions_test': results_diastolic[best_model]['predictions_test'],
            'metrics_test': results_diastolic[best_model]['metrics_test'],
        }
        
        # 生成可视化
        evaluator.evaluate_single_model(sys_result, "Systolic")
        evaluator.evaluate_single_model(dia_result, "Diastolic")
    
    # # 同时为线性回归生成可视化（用于对比）
    # if 'linear_regression' in results_systolic:
    #     print(f"\nGenerating visualizations for linear_regression (baseline)")
        
    #     sys_result_lr = {
    #         'model_name': 'linear_regression',
    #         'y_test': results_systolic['y_test'],
    #         'predictions_test': results_systolic['linear_regression']['predictions_test'],
    #         'metrics_test': results_systolic['linear_regression']['metrics_test'],
    #     }
        
    #     dia_result_lr = {
    #         'model_name': 'linear_regression',
    #         'y_test': results_diastolic['y_test'],
    #         'predictions_test': results_diastolic['linear_regression']['predictions_test'],
    #         'metrics_test': results_diastolic['linear_regression']['metrics_test'],
    #     }
        
    #     evaluator.evaluate_single_model(sys_result_lr, "Systolic")
    #     evaluator.evaluate_single_model(dia_result_lr, "Diastolic")

    # 生成综合报告
    evaluator.create_summary_report(results_systolic, results_diastolic)
    
    # ==================== 完成 ====================
    print("\n" + "=" * 70)
    print("✓ Training and Evaluation Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {evaluator.output_dir}/")
    
    # 检查是否达标
    print("\n" + "=" * 70)
    print("Performance Target Achievement")
    print("=" * 70)
    
    best_sys_metrics = results_systolic[best_model]['metrics_test']
    best_dia_metrics = results_diastolic[best_model]['metrics_test']
    
    print(f"\nBest Model: {best_model}")
    print(f"\nSystolic BP:")
    print(f"  MAE: {best_sys_metrics['mae']:.2f} {'✓ PASS' if best_sys_metrics['mae'] < 10 else '✗ FAIL'} (Target: <10)")
    print(f"  MSE: {best_sys_metrics['mse']:.2f} {'✓ PASS' if best_sys_metrics['mse'] < 70 else '✗ FAIL'} (Target: <70)")
    print(f"  R²:  {best_sys_metrics['r2']:.4f}")
    
    print(f"\nDiastolic BP:")
    print(f"  MAE: {best_dia_metrics['mae']:.2f} {'✓ PASS' if best_dia_metrics['mae'] < 10 else '✗ PASS'} (Target: <10)")
    print(f"  MSE: {best_dia_metrics['mse']:.2f} {'✓ PASS' if best_dia_metrics['mse'] < 60 else '✗ FAIL'} (Target: <60)")
    print(f"  R²:  {best_dia_metrics['r2']:.4f}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()