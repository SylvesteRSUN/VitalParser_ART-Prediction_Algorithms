"""
主程序 - LSTM版本
使用时序LSTM模型 + 个体校准进行血压预测
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path

from config import DATA_CONFIG, RUN_CONFIG
from data_loader import load_train_test_data
from signal_processing import SignalProcessor
from feature_extraction import CycleBasedFeatureExtractor
from lstm_model import LSTMBloodPressurePredictor


def main():
    """
    主函数 - LSTM + 个体校准完整流程
    """
    print("=" * 70)
    print(" " * 15 + "PLETH-BP Prediction System (LSTM)")
    print(" " * 18 + "With Individual Calibration")
    print("=" * 70)
    
    # ==================== 步骤0: 设置输出目录 ====================
    test_file_path = DATA_CONFIG['test_file_path']
    test_file_name = Path(test_file_path).stem
    
    output_dir = os.path.join('results', f"{test_file_name}_LSTM")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n✓ Created output directory: {output_dir}")
    else:
        print(f"\n✓ Using output directory: {output_dir}")

    # ==================== 步骤1: 加载数据 ====================
    print("\n" + "=" * 70)
    print("Step 1/6: Load Training and Test Data")
    print("=" * 70)
    
    pleth_train, art_train, pleth_test, art_test = load_train_test_data()
    
    if pleth_train is None or pleth_test is None:
        print("\n✗ Data loading failed, exiting")
        return
    
    # ==================== 步骤2: 信号处理 ====================
    print("\n" + "=" * 70)
    print("Step 2/6: Signal Processing and Peak Detection")
    print("=" * 70)
    
    processor = SignalProcessor()
    
    print("\n[Training Set]")
    pleth_train_processed = processor.process_signal(pleth_train, "PLETH_train")
    art_train_processed = processor.process_signal(art_train, "ART_train")
    
    print("\n[Test Set]")
    pleth_test_processed = processor.process_signal(pleth_test, "PLETH_test")
    art_test_processed = processor.process_signal(art_test, "ART_test")
    
    # ==================== 步骤3: 特征提取 ====================
    print("\n" + "=" * 70)
    print("Step 3/6: Cycle-Based Feature Extraction")
    print("=" * 70)
    
    extractor = CycleBasedFeatureExtractor()
    
    # 提取训练集特征
    print("\n[Training Set Feature Extraction]")
    X_train, y_train_systolic, y_train_diastolic = extractor.prepare_cycle_based_dataset(
        pleth_train_processed, art_train_processed
    )
    
    # 提取测试集特征（不标准化，稍后用训练集的scaler）
    print("\n[Test Set Feature Extraction]")
    test_extractor = CycleBasedFeatureExtractor()
    test_extractor.scaler = None
    
    X_test_raw, y_test_systolic, y_test_diastolic = test_extractor.prepare_cycle_based_dataset(
        pleth_test_processed, art_test_processed
    )
    
    # 检查数据有效性
    if len(X_train) < 100 or len(X_test_raw) < 50:
        print(f"\n✗ Insufficient samples:")
        print(f"  Training: {len(X_train)}, Test: {len(X_test_raw)}")
        print("  Exiting...")
        return
    
    print(f"\n✓ Dataset Ready:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test_raw)}")
    print(f"  Features per cycle: {X_train.shape[1]}")
    
    # ==================== 步骤4: LSTM模型训练 ====================
    print("\n" + "=" * 70)
    print("Step 4/6: LSTM Model Training")
    print("=" * 70)
    
    # 创建LSTM预测器
    lstm_predictor = LSTMBloodPressurePredictor(
        n_past=20,  # 使用过去20个心跳
        n_features=X_train.shape[1]
    )
    
    # 训练模型
    history_sys, history_dia = lstm_predictor.train(
        X_train,
        y_train_systolic,
        y_train_diastolic,
        validation_split=0.15,
        epochs=100,  # 最多100轮，实际会提前停止
        batch_size=32,
        verbose=1
    )
    
    # 保存训练历史图
    lstm_predictor.plot_training_history(
        history_sys, history_dia,
        save_path=os.path.join(output_dir, 'training_history.png')
    )
    
    # ==================== 步骤5: 个体校准 ====================
    print("\n" + "=" * 70)
    print("Step 5/6: Individual Calibration")
    print("=" * 70)
    
    # 使用测试集前5分钟（约300个心跳）进行校准
    calibration_size = min(300, len(X_test_raw) // 3)  # 至多用1/3数据校准
    
    print(f"\n使用测试集前 {calibration_size} 个心跳进行校准")
    print(f"  约 {calibration_size/60:.1f} 分钟的数据")
    
    X_calib = X_test_raw[:calibration_size]
    y_calib_sys = y_test_systolic[:calibration_size]
    y_calib_dia = y_test_diastolic[:calibration_size]
    
    lstm_predictor.calibrate(X_calib, y_calib_sys, y_calib_dia)
    
    # ==================== 步骤6: 预测和评估 ====================
    print("\n" + "=" * 70)
    print("Step 6/6: Prediction and Evaluation")
    print("=" * 70)
    
    # 预测整个测试集（带校准）
    print("\n[预测 - 带个体校准]")
    pred_sys_calib, pred_dia_calib = lstm_predictor.predict(
        X_test_raw, apply_calibration=True
    )
    
    # 预测整个测试集（不带校准，用于对比）
    print("\n[预测 - 不带校准]")
    pred_sys_no_calib, pred_dia_no_calib = lstm_predictor.predict(
        X_test_raw, apply_calibration=False
    )
    
    # 对齐真实值（因为LSTM需要n_past个历史，所以预测结果少了n_past个）
    n_past = lstm_predictor.n_past
    y_test_sys_aligned = y_test_systolic[n_past:]
    y_test_dia_aligned = y_test_diastolic[n_past:]
    
    print(f"\n✓ 预测完成:")
    print(f"  预测样本数: {len(pred_sys_calib)}")
    print(f"  真实值样本数: {len(y_test_sys_aligned)}")
    
    # ==================== 评估结果 ====================
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    def calculate_metrics(y_true, y_pred):
        """计算评估指标"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}
    
    def print_metrics(metrics, label):
        """打印评估指标"""
        print(f"\n{label}:")
        print(f"  MAE:  {metrics['mae']:.2f} mmHg")
        print(f"  MSE:  {metrics['mse']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f} mmHg")
        print(f"  R²:   {metrics['r2']:.4f}")
    
    # 评估带校准的结果
    print("\n" + "="*60)
    print("收缩压预测结果（带校准）")
    print("="*60)
    metrics_sys_calib = calculate_metrics(y_test_sys_aligned, pred_sys_calib)
    print_metrics(metrics_sys_calib, "收缩压（带校准）")
    
    print("\n" + "="*60)
    print("舒张压预测结果（带校准）")
    print("="*60)
    metrics_dia_calib = calculate_metrics(y_test_dia_aligned, pred_dia_calib)
    print_metrics(metrics_dia_calib, "舒张压（带校准）")
    
    # 评估不带校准的结果（对比）
    print("\n" + "="*60)
    print("收缩压预测结果（不带校准）")
    print("="*60)
    metrics_sys_no_calib = calculate_metrics(y_test_sys_aligned, pred_sys_no_calib)
    print_metrics(metrics_sys_no_calib, "收缩压（不带校准）")
    
    print("\n" + "="*60)
    print("舒张压预测结果（不带校准）")
    print("="*60)
    metrics_dia_no_calib = calculate_metrics(y_test_dia_aligned, pred_dia_no_calib)
    print_metrics(metrics_dia_no_calib, "舒张压（不带校准）")
    
    # ==================== 生成可视化 ====================
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    import matplotlib.pyplot as plt
    
    def plot_timeseries(y_true, y_pred, title, save_path):
        """绘制时序对比图"""
        plt.figure(figsize=(15, 5))
        plt.plot(y_true, 'b-', label='Actual', alpha=0.7, linewidth=1.5)
        plt.plot(y_pred, 'r-', label='Predicted', alpha=0.7, linewidth=1.5)
        plt.xlabel('Cardiac Cycle')
        plt.ylabel('Blood Pressure (mmHg)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def plot_scatter(y_true, y_pred, title, save_path):
        """绘制散点图"""
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # 45度参考线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
        
        plt.xlabel('Actual BP (mmHg)')
        plt.ylabel('Predicted BP (mmHg)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    # 收缩压 - 带校准
    plot_timeseries(
        y_test_sys_aligned, pred_sys_calib,
        'Systolic BP - LSTM with Calibration',
        os.path.join(output_dir, 'timeseries_systolic_calibrated.png')
    )
    plot_scatter(
        y_test_sys_aligned, pred_sys_calib,
        'Systolic BP - LSTM with Calibration',
        os.path.join(output_dir, 'scatter_systolic_calibrated.png')
    )
    
    # 舒张压 - 带校准
    plot_timeseries(
        y_test_dia_aligned, pred_dia_calib,
        'Diastolic BP - LSTM with Calibration',
        os.path.join(output_dir, 'timeseries_diastolic_calibrated.png')
    )
    plot_scatter(
        y_test_dia_aligned, pred_dia_calib,
        'Diastolic BP - LSTM with Calibration',
        os.path.join(output_dir, 'scatter_diastolic_calibrated.png')
    )
    
    # ==================== 保存模型 ====================
    model_dir = os.path.join(output_dir, 'saved_model')
    lstm_predictor.save_model(model_dir)
    
    # ==================== 性能总结 ====================
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    
    print("\n【收缩压】")
    print(f"  带校准:   MAE={metrics_sys_calib['mae']:.2f}, R²={metrics_sys_calib['r2']:.4f}")
    print(f"  不带校准: MAE={metrics_sys_no_calib['mae']:.2f}, R²={metrics_sys_no_calib['r2']:.4f}")
    print(f"  改善:     MAE减少 {metrics_sys_no_calib['mae'] - metrics_sys_calib['mae']:.2f} mmHg")
    
    print("\n【舒张压】")
    print(f"  带校准:   MAE={metrics_dia_calib['mae']:.2f}, R²={metrics_dia_calib['r2']:.4f}")
    print(f"  不带校准: MAE={metrics_dia_no_calib['mae']:.2f}, R²={metrics_dia_no_calib['r2']:.4f}")
    print(f"  改善:     MAE减少 {metrics_dia_no_calib['mae'] - metrics_dia_calib['mae']:.2f} mmHg")
    
    # ==================== 达标检查 ====================
    print("\n" + "=" * 70)
    print("Target Achievement Check")
    print("=" * 70)
    
    target_mae = 10
    target_mse_sys = 70
    target_mse_dia = 60
    
    print("\n收缩压:")
    print(f"  MAE: {metrics_sys_calib['mae']:.2f} {'✓' if metrics_sys_calib['mae'] < target_mae else '✗'} (目标: <{target_mae})")
    print(f"  MSE: {metrics_sys_calib['mse']:.2f} {'✓' if metrics_sys_calib['mse'] < target_mse_sys else '✗'} (目标: <{target_mse_sys})")
    print(f"  R²:  {metrics_sys_calib['r2']:.4f}")
    
    print("\n舒张压:")
    print(f"  MAE: {metrics_dia_calib['mae']:.2f} {'✓' if metrics_dia_calib['mae'] < target_mae else '✗'} (目标: <{target_mae})")
    print(f"  MSE: {metrics_dia_calib['mse']:.2f} {'✓' if metrics_dia_calib['mse'] < target_mse_dia else '✗'} (目标: <{target_mse_dia})")
    print(f"  R²:  {metrics_dia_calib['r2']:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ LSTM Training and Evaluation Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()