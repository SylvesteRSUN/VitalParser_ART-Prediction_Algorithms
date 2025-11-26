"""
Quick Test Transfer Learning Script
快速测试迁移学习脚本

This script loads pre-trained general models and only performs fine-tuning and evaluation.
本脚本加载预训练的通用模型，仅执行微调和评估。

Time savings: ~5-10x faster than full pipeline
节省时间：比完整流程快5-10倍
"""

import numpy as np
import os
import sys
from pathlib import Path
import time
from datetime import datetime

# Import existing modules / 导入现有模块
from config import DATA_CONFIG
from data_loader import load_train_test_data
from signal_processing import SignalProcessor
from feature_extraction import CycleBasedFeatureExtractor

# Import transfer learning modules / 导入迁移学习模块
from config_transfer import (
    FINE_TUNING_CONFIG, DATA_SPLIT_CONFIG,
    EVALUATION_CONFIG as TL_EVAL_CONFIG, PATH_CONFIG,
    create_output_directories, get_output_path, get_model_save_path
)
from data_splitter import PatientDataSplitter
from transfer_learning import PersonalFineTuner, ModelManager
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ============================================================================
# Helper Functions / 辅助函数
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    计算评估指标。

    Args:
        y_true: True values / 真实值
        y_pred: Predicted values / 预测值

    Returns:
        Dictionary of metrics / 指标字典
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }


def plot_transfer_learning_comparison(general_results, personalized_results, save_path=None):
    """
    Plot comparison between general and personalized models.
    绘制通用模型和个性化模型的对比。
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, bp_type in enumerate(['systolic', 'diastolic']):
        y_true = general_results[bp_type]['y_true']
        y_pred_gen = general_results[bp_type]['y_pred']
        y_pred_pers = personalized_results[bp_type]['y_pred']

        # Time series (left column)
        ax = axes[idx, 0]
        max_points = min(500, len(y_true))
        indices = np.linspace(0, len(y_true)-1, max_points, dtype=int)

        ax.plot(indices, y_true[indices], 'g-', label='Actual', alpha=0.7, linewidth=2)
        ax.plot(indices, y_pred_gen[indices], 'b--', label='General Model', alpha=0.6, linewidth=1.5)
        ax.plot(indices, y_pred_pers[indices], 'r-', label='Personalized Model', alpha=0.6, linewidth=1.5)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Blood Pressure (mmHg)')
        ax.set_title(f'{bp_type.capitalize()} BP - Time Series Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Scatter plot (right column)
        ax = axes[idx, 1]
        min_val = min(y_true.min(), y_pred_gen.min(), y_pred_pers.min())
        max_val = max(y_true.max(), y_pred_gen.max(), y_pred_pers.max())

        ax.scatter(y_true, y_pred_gen, alpha=0.4, s=20, c='blue', label='General Model')
        ax.scatter(y_true, y_pred_pers, alpha=0.4, s=20, c='red', label='Personalized Model')
        ax.plot([min_val, max_val], [min_val, max_val], 'g--', lw=2, label='Ideal Prediction')
        ax.set_xlabel('Actual Value (mmHg)')
        ax.set_ylabel('Predicted Value (mmHg)')
        ax.set_title(f'{bp_type.capitalize()} BP - Prediction Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot: {save_path}")

    plt.close()


def plot_improvement_summary(general_results, personalized_results, save_path=None):
    """
    Plot improvement summary bar charts.
    绘制改进总结条形图。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = ['MAE', 'MSE', 'RMSE']
    bp_types = ['Systolic', 'Diastolic']

    for idx, bp_type_key in enumerate(['systolic', 'diastolic']):
        ax = axes[idx]

        gen_metrics = general_results[bp_type_key]['metrics']
        pers_metrics = personalized_results[bp_type_key]['metrics']

        x = np.arange(len(metrics))
        width = 0.35

        gen_values = [gen_metrics[m] for m in metrics]
        pers_values = [pers_metrics[m] for m in metrics]

        ax.bar(x - width/2, gen_values, width, label='General Model', color='blue', alpha=0.7)
        ax.bar(x + width/2, pers_values, width, label='Personalized Model', color='red', alpha=0.7)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Error Value')
        ax.set_title(f'{bp_types[idx]} BP - Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (g, p) in enumerate(zip(gen_values, pers_values)):
            ax.text(i - width/2, g, f'{g:.1f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, p, f'{p:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved improvement summary: {save_path}")

    plt.close()


def generate_transfer_learning_report(general_results, personalized_results,
                                      pipeline_results, report_path):
    """
    Generate text report.
    生成文本报告。
    """
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(" QUICK TEST TRANSFER LEARNING REPORT\n")
        f.write(" 快速测试迁移学习报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("=" * 80 + "\n")
        f.write("Dataset Information / 数据集信息\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test samples: {pipeline_results.get('n_test', 'N/A')}\n")
        f.write(f"Calibration samples: {pipeline_results.get('n_calib', 'N/A')}\n")
        f.write(f"Evaluation samples: {pipeline_results.get('n_eval', 'N/A')}\n\n")

        for bp_type in ['systolic', 'diastolic']:
            f.write("=" * 80 + "\n")
            f.write(f"{bp_type.upper()} BLOOD PRESSURE RESULTS / {('收缩压' if bp_type == 'systolic' else '舒张压')}结果\n")
            f.write("=" * 80 + "\n\n")

            gen_m = general_results[bp_type]['metrics']
            pers_m = personalized_results[bp_type]['metrics']

            f.write(f"General Model / 通用模型:\n")
            f.write(f"  MAE:  {gen_m['MAE']:.2f} mmHg\n")
            f.write(f"  MSE:  {gen_m['MSE']:.2f}\n")
            f.write(f"  RMSE: {gen_m['RMSE']:.2f} mmHg\n")
            f.write(f"  R²:   {gen_m['R2']:.4f}\n\n")

            f.write(f"Personalized Model / 个性化模型:\n")
            f.write(f"  MAE:  {pers_m['MAE']:.2f} mmHg\n")
            f.write(f"  MSE:  {pers_m['MSE']:.2f}\n")
            f.write(f"  RMSE: {pers_m['RMSE']:.2f} mmHg\n")
            f.write(f"  R²:   {pers_m['R2']:.4f}\n\n")

            # Improvement
            mae_improvement = (gen_m['MAE'] - pers_m['MAE']) / gen_m['MAE'] * 100
            mse_improvement = (gen_m['MSE'] - pers_m['MSE']) / gen_m['MSE'] * 100

            f.write(f"Improvement / 改进:\n")
            f.write(f"  MAE: {mae_improvement:.1f}%\n")
            f.write(f"  MSE: {mse_improvement:.1f}%\n\n")

            # Target achievement
            target_mae = 10
            target_mse = 70 if bp_type == 'systolic' else 60

            f.write(f"Target Achievement / 目标达成:\n")
            f.write(f"  MAE < {target_mae}: {'✓ Pass' if pers_m['MAE'] < target_mae else '✗ Fail'} "
                   f"(Current: {pers_m['MAE']:.2f})\n")
            f.write(f"  MSE < {target_mse}: {'✓ Pass' if pers_m['MSE'] < target_mse else '✗ Fail'} "
                   f"(Current: {pers_m['MSE']:.2f})\n\n")

    print(f"Report saved: {report_path}")


class QuickTestPipeline:
    """
    Quick testing pipeline that reuses pre-trained general models.
    快速测试流程，复用预训练的通用模型。
    """

    def __init__(self, test_file_name: str, verbose: bool = True):
        """
        Initialize quick test pipeline.
        初始化快速测试流程。

        Args:
            test_file_name: Name of test file / 测试文件名
            verbose: Print progress / 打印进度
        """
        self.test_file_name = test_file_name
        self.verbose = verbose
        self.output_suffix = '_quicktest'  # Suffix for output directories / 输出目录后缀

        # Create output directories with _quicktest suffix / 创建带_quicktest后缀的输出目录
        create_output_directories(test_file_name, suffix=self.output_suffix)

        # Results storage / 结果存储
        self.results = {}

    def run_quick_test(self):
        """
        Run quick test pipeline (loads models, then fine-tunes).
        运行快速测试流程（加载模型，然后微调）。
        """
        print("\n" + "=" * 80)
        print(" QUICK TEST PIPELINE - REUSING PRE-TRAINED MODELS")
        print(" 快速测试流程 - 复用预训练模型")
        print("=" * 80)

        start_time = time.time()

        # Step 1: Load pre-trained general models / 步骤1：加载预训练通用模型
        print("\n[STEP 1/5] Loading Pre-trained General Models...")
        general_model_sys, general_model_dia, scaler = self._load_pretrained_models()

        # Step 2: Load and process test data / 步骤2：加载并处理测试数据
        print("\n[STEP 2/5] Loading and Processing Test Data...")
        X_test, y_test_sys, y_test_dia = self._load_and_process_test_data(scaler)
        self.results['n_test'] = len(X_test)

        # Step 3: Split test data / 步骤3：分割测试数据
        print("\n[STEP 3/5] Splitting Test Data...")
        X_calib, y_calib_sys, y_calib_dia, X_eval, y_eval_sys, y_eval_dia = \
            self._split_test_data(X_test, y_test_sys, y_test_dia)

        # Step 4: Evaluate general models / 步骤4：评估通用模型
        print("\n[STEP 4/5] Evaluating General Models...")
        general_results = self._evaluate_general_models(
            general_model_sys, general_model_dia, X_eval, y_eval_sys, y_eval_dia
        )

        # Step 5: Fine-tune and evaluate / 步骤5：微调并评估
        print("\n[STEP 5/5] Fine-tuning Personalized Models...")
        personalized_results = self._fine_tune_and_evaluate(
            general_model_sys, general_model_dia,
            X_calib, y_calib_sys, y_calib_dia,
            X_eval, y_eval_sys, y_eval_dia
        )

        # Compare and report / 对比并报告
        self._compare_and_report_results(general_results, personalized_results)

        # Generate visualizations / 生成可视化
        self._generate_visualizations(
            general_results, personalized_results,
            X_eval, y_eval_sys, y_eval_dia
        )

        elapsed_time = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"Quick test completed in {elapsed_time:.2f} seconds")
        print(f"快速测试完成，耗时 {elapsed_time:.2f} 秒")
        print(f"{'=' * 80}\n")

    def _load_pretrained_models(self):
        """
        Load pre-trained general models and scaler.
        加载预训练的通用模型和标准化器。
        """
        model_dir = get_model_save_path('general')

        # Check if models exist / 检查模型是否存在
        sys_path = os.path.join(model_dir, 'general_model_systolic.pkl')
        if not os.path.exists(sys_path):
            print(f"\n❌ ERROR: Pre-trained models not found at: {model_dir}")
            print("Please run the full training pipeline first using:")
            print("  python main_transfer_learning.py")
            sys.exit(1)

        # Load models and scaler / 加载模型和标准化器
        general_model_sys, general_model_dia, scaler = ModelManager.load_general_models(
            model_dir, load_scaler=True
        )

        if scaler is None:
            print("\n⚠ WARNING: Scaler not found!")
            print("The scaler will be re-fitted on test data (not ideal).")
            print("For best results, re-run full training with updated code to save scaler.")

        return general_model_sys, general_model_dia, scaler

    def _load_and_process_test_data(self, scaler):
        """
        Load and process test data only.
        仅加载并处理测试数据。
        """
        # Load data / 加载数据
        _, _, pleth_test, art_test = load_train_test_data()

        if pleth_test is None or art_test is None:
            raise ValueError("Failed to load test data")

        if self.verbose:
            print(f"\n  Processing test data...")

        # Process signals / 处理信号
        processor = SignalProcessor()
        pleth_processed = processor.process_signal(pleth_test, 'PLETH')
        art_processed = processor.process_signal(art_test, 'ART')

        # Extract features / 提取特征
        extractor = CycleBasedFeatureExtractor()
        X_test, y_test_sys, y_test_dia = extractor.prepare_cycle_based_dataset(
            pleth_processed, art_processed
        )

        if len(X_test) == 0:
            raise ValueError("No valid test cycles extracted")

        # Standardize features / 标准化特征
        from sklearn.preprocessing import StandardScaler
        if scaler is None:
            # Re-fit scaler on test data (not ideal) / 在测试数据上重新拟合（不理想）
            print(f"\n  ⚠ Re-fitting scaler on test data...")
            scaler = StandardScaler()
            X_test = scaler.fit_transform(X_test)
        else:
            # Use saved scaler / 使用保存的标准化器
            X_test = scaler.transform(X_test)

        print(f"\n  Total test samples: {len(X_test)}")
        print(f"  Features: {X_test.shape[1]}")

        return X_test, y_test_sys, y_test_dia

    def _split_test_data(self, X_test, y_test_sys, y_test_dia):
        """
        Split test data into calibration and evaluation sets.
        将测试数据分割为校准集和评估集。
        """
        method = DATA_SPLIT_CONFIG['split_method']
        config = DATA_SPLIT_CONFIG[method]

        splitter = PatientDataSplitter(method=method, **config)
        X_calib, y_calib_sys, y_calib_dia, X_eval, y_eval_sys, y_eval_dia = \
            splitter.split(X_test, y_test_sys, y_test_dia)

        print(f"  Calibration samples: {len(X_calib)}")
        print(f"  Evaluation samples: {len(X_eval)}")

        self.results['n_calib'] = len(X_calib)
        self.results['n_eval'] = len(X_eval)

        return X_calib, y_calib_sys, y_calib_dia, X_eval, y_eval_sys, y_eval_dia

    def _evaluate_general_models(self, model_sys, model_dia, X_eval, y_eval_sys, y_eval_dia):
        """
        Evaluate general models.
        评估通用模型。
        """
        # Predict / 预测
        y_pred_sys = model_sys.predict(X_eval)
        y_pred_dia = model_dia.predict(X_eval)

        # Calculate metrics / 计算指标
        metrics_sys = calculate_metrics(y_eval_sys, y_pred_sys)
        metrics_dia = calculate_metrics(y_eval_dia, y_pred_dia)

        results = {
            'systolic': {
                'y_true': y_eval_sys,
                'y_pred': y_pred_sys,
                'metrics': metrics_sys
            },
            'diastolic': {
                'y_true': y_eval_dia,
                'y_pred': y_pred_dia,
                'metrics': metrics_dia
            }
        }

        if self.verbose:
            print(f"\n  Systolic:  MAE={metrics_sys['MAE']:.2f}, MSE={metrics_sys['MSE']:.2f}, R²={metrics_sys['R2']:.4f}")
            print(f"  Diastolic: MAE={metrics_dia['MAE']:.2f}, MSE={metrics_dia['MSE']:.2f}, R²={metrics_dia['R2']:.4f}")

        return results

    def _fine_tune_and_evaluate(self, general_model_sys, general_model_dia,
                                 X_calib, y_calib_sys, y_calib_dia,
                                 X_eval, y_eval_sys, y_eval_dia):
        """
        Fine-tune and evaluate personalized models.
        微调并评估个性化模型。
        """
        strategy = FINE_TUNING_CONFIG['strategy']

        # Prepare fine-tuning parameters / 准备微调参数
        fine_tune_params = FINE_TUNING_CONFIG['xgboost'].copy()
        fine_tune_params.pop('strategy', None)  # Avoid duplicate

        # Fine-tune / 微调
        fine_tuner = PersonalFineTuner(strategy=strategy, **fine_tune_params)
        fine_tuner.fine_tune(
            general_model_sys, general_model_dia,
            X_calib, y_calib_sys, y_calib_dia,
            verbose=self.verbose
        )

        # Predict / 预测
        y_pred_sys, y_pred_dia = fine_tuner.predict(X_eval)

        # Calculate metrics / 计算指标
        metrics_sys = calculate_metrics(y_eval_sys, y_pred_sys)
        metrics_dia = calculate_metrics(y_eval_dia, y_pred_dia)

        results = {
            'systolic': {
                'y_true': y_eval_sys,
                'y_pred': y_pred_sys,
                'metrics': metrics_sys
            },
            'diastolic': {
                'y_true': y_eval_dia,
                'y_pred': y_pred_dia,
                'metrics': metrics_dia
            }
        }

        if self.verbose:
            print(f"\n  Systolic:  MAE={metrics_sys['MAE']:.2f}, MSE={metrics_sys['MSE']:.2f}, R²={metrics_sys['R2']:.4f}")
            print(f"  Diastolic: MAE={metrics_dia['MAE']:.2f}, MSE={metrics_dia['MSE']:.2f}, R²={metrics_dia['R2']:.4f}")

        return results

    def _compare_and_report_results(self, general_results, personalized_results):
        """
        Compare and report results.
        对比并报告结果。
        """
        print("\n" + "=" * 80)
        print(" PERFORMANCE COMPARISON")
        print(" 性能对比")
        print("=" * 80)

        for bp_type in ['systolic', 'diastolic']:
            print(f"\n{bp_type.upper()} / {('收缩压' if bp_type == 'systolic' else '舒张压')}:")
            print(f"  General Model:      MAE={general_results[bp_type]['metrics']['MAE']:.2f}, "
                  f"MSE={general_results[bp_type]['metrics']['MSE']:.2f}, "
                  f"R²={general_results[bp_type]['metrics']['R2']:.4f}")
            print(f"  Personalized Model: MAE={personalized_results[bp_type]['metrics']['MAE']:.2f}, "
                  f"MSE={personalized_results[bp_type]['metrics']['MSE']:.2f}, "
                  f"R²={personalized_results[bp_type]['metrics']['R2']:.4f}")

            mae_improvement = (general_results[bp_type]['metrics']['MAE'] -
                              personalized_results[bp_type]['metrics']['MAE']) / \
                             general_results[bp_type]['metrics']['MAE'] * 100
            print(f"  Improvement: {mae_improvement:.1f}%")

        # Save report / 保存报告
        report_path = os.path.join(
            get_output_path(self.test_file_name, PATH_CONFIG['reports_dir'], suffix=self.output_suffix),
            'quick_test_report.txt'
        )
        generate_transfer_learning_report(
            general_results, personalized_results,
            self.results, report_path
        )

    def _generate_visualizations(self, general_results, personalized_results,
                                 X_eval, y_eval_sys, y_eval_dia):
        """
        Generate visualizations.
        生成可视化。
        """
        plots_dir = get_output_path(self.test_file_name, PATH_CONFIG['plots_dir'], suffix=self.output_suffix)

        # Time series comparison / 时间序列对比
        plot_transfer_learning_comparison(
            general_results, personalized_results,
            save_path=os.path.join(plots_dir, 'quick_test_comparison.png')
        )

        # Improvement summary / 改善总结
        plot_improvement_summary(
            general_results, personalized_results,
            save_path=os.path.join(plots_dir, 'quick_test_improvement.png')
        )

        print(f"\nVisualizations saved to: {plots_dir}")


def main():
    """Main entry point / 主入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Quick Transfer Learning Test - Reuse Pre-trained Models'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default=None,
        help='Test file path (optional, defaults to config.py setting)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed progress'
    )

    args = parser.parse_args()

    # Get test file path / 获取测试文件路径
    if args.test_file:
        # Use command line argument if provided / 如果提供了命令行参数则使用
        DATA_CONFIG['test_file_path'] = args.test_file

    # Get test file name from full path / 从完整路径获取文件名
    test_file_path = DATA_CONFIG['test_file_path']
    test_file_name = os.path.basename(test_file_path)

    print(f"Using test file: {test_file_name}")

    # Run quick test / 运行快速测试
    pipeline = QuickTestPipeline(
        test_file_name=test_file_name,
        verbose=args.verbose
    )
    pipeline.run_quick_test()


if __name__ == '__main__':
    main()
