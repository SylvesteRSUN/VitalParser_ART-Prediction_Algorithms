"""
评估和可视化模块 - 评估模型性能并生成可视化报告
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from config.config import EVALUATION_CONFIG

class ModelEvaluator:
    """模型评估器类"""
    
    def __init__(self, output_dir=None):
        """
        初始化评估器
        
        参数:
            output_dir: 输出目录，如果为None则使用config中的设置
        """
        self.output_dir = output_dir or EVALUATION_CONFIG['output_dir']
        self.figure_size = EVALUATION_CONFIG['figure_size']
        self.dpi = EVALUATION_CONFIG['dpi']
        self.save_figures = EVALUATION_CONFIG['save_figures']
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
    
    def plot_prediction_vs_actual(self, y_true, y_pred, model_name, target_name, split='test'):
        """
        绘制预测值 vs 真实值散点图
        
        参数:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            target_name: 目标名称
            split: 数据集类型（train/test）
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 散点图
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # 理想线（y=x）
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')
        
        # 标签和标题
        ax.set_xlabel('Actual Value (mmHg)', fontsize=12)
        ax.set_ylabel('Predicted Value (mmHg)', fontsize=12)
        ax.set_title(f'{model_name} - {target_name}\nPrediction vs Actual ({split})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_figures:
            filename = f'prediction_vs_actual_{target_name}_{model_name}_{split}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"  Saved figure: {filename}")
        
        return fig
    
    def plot_residuals(self, y_true, y_pred, model_name, target_name, split='test'):
        """
        绘制残差图
        
        参数:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            target_name: 目标名称
            split: 数据集类型
        """
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
        
        # 残差散点图
        ax1.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax1.axhline(y=0, color='r', linestyle='--', lw=2)
        ax1.set_xlabel('Predicted Value (mmHg)', fontsize=12)
        ax1.set_ylabel('Residuals (mmHg)', fontsize=12)
        ax1.set_title('Residual Distribution', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 残差直方图
        ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Residuals (mmHg)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Residual Histogram', fontsize=12)
        ax2.axvline(x=0, color='r', linestyle='--', lw=2)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f'{model_name} - {target_name} ({split})', fontsize=14)
        plt.tight_layout()
        
        if self.save_figures:
            filename = f'residuals_{target_name}_{model_name}_{split}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"  Saved figure: {filename}")
        
        return fig
    
    def plot_time_series_prediction(self, y_true, y_pred, model_name, target_name, split='test', max_points=250):
        """
        绘制时序预测对比图
        
        参数:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            target_name: 目标名称
            split: 数据集类型
            max_points: 最多显示的点数
        """
        # 如果数据太多，只显示一部分
        if len(y_true) > max_points:
            indices = np.linspace(0, len(y_true)-1, max_points, dtype=int)
            y_true_plot = y_true[indices]
            y_pred_plot = y_pred[indices]
        else:
            y_true_plot = y_true
            y_pred_plot = y_pred
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        x = np.arange(len(y_true_plot))
        ax.plot(x, y_true_plot, 'b-', label='Actual', alpha=0.7, linewidth=1.5)
        ax.plot(x, y_pred_plot, 'r-', label='Predicted', alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Blood Pressure (mmHg)', fontsize=12)
        ax.set_title(f'{model_name} - {target_name}\nTime Series Comparison ({split})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_figures:
            filename = f'timeseries_{target_name}_{model_name}_{split}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"  Saved figure: {filename}")
        
        return fig
    
    def evaluate_single_model(self, result, target_name):
        """
        评估单个模型并生成可视化
        
        参数:
            result: 模型结果字典
            target_name: 目标名称
        """
        model_name = result['model_name']
        print(f"\nGenerating visualization for {model_name}...")
        
        # 测试集可视化
        self.plot_prediction_vs_actual(
            result['y_test'], result['predictions_test'],
            model_name, target_name, 'test'
        )
        
        self.plot_residuals(
            result['y_test'], result['predictions_test'],
            model_name, target_name, 'test'
        )
        
        self.plot_time_series_prediction(
            result['y_test'], result['predictions_test'],
            model_name, target_name, 'test'
        )
        
        plt.close('all')  # 关闭所有图形以释放内存
    
    def create_summary_report(self, results_systolic, results_diastolic):
        """
        创建综合评估报告
        
        参数:
            results_systolic: 收缩压结果
            results_diastolic: 舒张压结果
        """
        print(f"\n{'='*60}")
        print("Generating Comprehensive Evaluation Report")
        print(f"{'='*60}")
        
        # 获取模型列表
        model_list = [k for k in results_systolic.keys() 
                     if k not in ['X_train', 'X_test', 'y_train', 'y_test']]
        
        # 创建性能对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 收缩压性能
        mae_sys = [results_systolic[m]['metrics_test']['mae'] for m in model_list]
        mse_sys = [results_systolic[m]['metrics_test']['mse'] for m in model_list]
        
        x = np.arange(len(model_list))
        width = 0.35
        
        ax1.bar(x - width/2, mae_sys, width, label='MAE', alpha=0.8)
        ax1.bar(x + width/2, mse_sys, width, label='MSE', alpha=0.8)
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Error', fontsize=12)
        ax1.set_title('Systolic BP Prediction Performance', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_list, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加阈值线
        ax1.axhline(y=EVALUATION_CONFIG['mae_threshold'], color='r', 
                   linestyle='--', label=f'MAE Threshold={EVALUATION_CONFIG["mae_threshold"]}')
        ax1.axhline(y=EVALUATION_CONFIG['mse_threshold_systolic'], color='orange', 
                   linestyle='--', label=f'MSE Threshold={EVALUATION_CONFIG["mse_threshold_systolic"]}')
        
        # 舒张压性能
        mae_dia = [results_diastolic[m]['metrics_test']['mae'] for m in model_list]
        mse_dia = [results_diastolic[m]['metrics_test']['mse'] for m in model_list]
        
        ax2.bar(x - width/2, mae_dia, width, label='MAE', alpha=0.8)
        ax2.bar(x + width/2, mse_dia, width, label='MSE', alpha=0.8)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('Error', fontsize=12)
        ax2.set_title('Diastolic BP Prediction Performance', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_list, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加阈值线
        ax2.axhline(y=EVALUATION_CONFIG['mae_threshold'], color='r', 
                   linestyle='--', label=f'MAE Threshold={EVALUATION_CONFIG["mae_threshold"]}')
        ax2.axhline(y=EVALUATION_CONFIG['mse_threshold_diastolic'], color='orange', 
                   linestyle='--', label=f'MSE Threshold={EVALUATION_CONFIG["mse_threshold_diastolic"]}')
        
        plt.tight_layout()
        
        if self.save_figures:
            filename = 'performance_comparison.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"  Saved summary report: {filename}")
        
        plt.close()
        
        # 生成文本报告
        self._save_text_report(results_systolic, results_diastolic, model_list)
    
    def _save_text_report(self, results_systolic, results_diastolic, model_list):
        """保存文本格式的评估报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'evaluation_report_{timestamp}.txt'
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("PLETH-BP Prediction Model Evaluation Report\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 收缩压结果
            f.write("Systolic Blood Pressure Prediction Results:\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Model':<25} {'MAE':<10} {'MSE':<10} {'RMSE':<10} {'R²':<10}\n")
            f.write("-"*60 + "\n")
            
            for model_name in model_list:
                metrics = results_systolic[model_name]['metrics_test']
                f.write(f"{model_name:<25} {metrics['mae']:<10.2f} {metrics['mse']:<10.2f} "
                       f"{metrics['rmse']:<10.2f} {metrics['r2']:<10.4f}\n")
            
            f.write("\n" + "="*60 + "\n\n")
            
            # 舒张压结果
            f.write("Diastolic Blood Pressure Prediction Results:\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Model':<25} {'MAE':<10} {'MSE':<10} {'RMSE':<10} {'R²':<10}\n")
            f.write("-"*60 + "\n")
            
            for model_name in model_list:
                metrics = results_diastolic[model_name]['metrics_test']
                f.write(f"{model_name:<25} {metrics['mae']:<10.2f} {metrics['mse']:<10.2f} "
                       f"{metrics['rmse']:<10.2f} {metrics['r2']:<10.4f}\n")
            
            f.write("\n" + "="*60 + "\n\n")
            
            # 性能指标达标情况
            f.write("Performance Target Achievement:\n")
            f.write("-"*60 + "\n")
            
            for model_name in model_list:
                sys_metrics = results_systolic[model_name]['metrics_test']
                dia_metrics = results_diastolic[model_name]['metrics_test']
                
                f.write(f"\n{model_name}:\n")
                f.write(f"  Systolic:\n")
                f.write(f"    MAE: {sys_metrics['mae']:.2f} {'✓' if sys_metrics['mae'] < 10 else '✗'} (Target: <10)\n")
                f.write(f"    MSE: {sys_metrics['mse']:.2f} {'✓' if sys_metrics['mse'] < 70 else '✗'} (Target: <70)\n")
                f.write(f"  Diastolic:\n")
                f.write(f"    MAE: {dia_metrics['mae']:.2f} {'✓' if dia_metrics['mae'] < 10 else '✗'} (Target: <10)\n")
                f.write(f"    MSE: {dia_metrics['mse']:.2f} {'✓' if dia_metrics['mse'] < 60 else '✗'} (Target: <60)\n")
        
        print(f"  Saved text report: {filename}")


    def plot_transfer_learning_comparison(self, y_true, y_pred_general, y_pred_personalized,
                                          target_name, max_points=500):
        """
        Plot comparison between general and personalized models.
        绘制通用模型和个性化模型的对比图。

        Args:
            y_true: True values / 真实值
            y_pred_general: General model predictions / 通用模型预测
            y_pred_personalized: Personalized model predictions / 个性化模型预测
            target_name: Target name (Systolic/Diastolic) / 目标名称
            max_points: Maximum points to display / 最多显示的点数
        """
        # Sample data if too large / 如果数据太大则采样
        if len(y_true) > max_points:
            indices = np.linspace(0, len(y_true)-1, max_points, dtype=int)
            y_true_plot = y_true[indices]
            y_pred_gen_plot = y_pred_general[indices]
            y_pred_pers_plot = y_pred_personalized[indices]
        else:
            y_true_plot = y_true
            y_pred_gen_plot = y_pred_general
            y_pred_pers_plot = y_pred_personalized

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Time series comparison / 时序对比
        ax1 = axes[0, 0]
        x = np.arange(len(y_true_plot))
        ax1.plot(x, y_true_plot, 'g-', label='Actual', alpha=0.7, linewidth=2)
        ax1.plot(x, y_pred_gen_plot, 'b--', label='General Model', alpha=0.6, linewidth=1.5)
        ax1.plot(x, y_pred_pers_plot, 'r-', label='Personalized Model', alpha=0.6, linewidth=1.5)
        ax1.set_xlabel('Sample Index', fontsize=11)
        ax1.set_ylabel('Blood Pressure (mmHg)', fontsize=11)
        ax1.set_title(f'{target_name} BP - Time Series Comparison', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. Scatter plots side by side / 散点图对比
        ax2 = axes[0, 1]
        min_val = min(y_true.min(), y_pred_general.min(), y_pred_personalized.min())
        max_val = max(y_true.max(), y_pred_general.max(), y_pred_personalized.max())

        ax2.scatter(y_true, y_pred_general, alpha=0.4, s=30, c='blue', label='General Model')
        ax2.scatter(y_true, y_pred_personalized, alpha=0.4, s=30, c='red', label='Personalized Model')
        ax2.plot([min_val, max_val], [min_val, max_val], 'g--', lw=2, label='Ideal Prediction')
        ax2.set_xlabel('Actual Value (mmHg)', fontsize=11)
        ax2.set_ylabel('Predicted Value (mmHg)', fontsize=11)
        ax2.set_title(f'{target_name} BP - Prediction Accuracy', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 3. Error distribution comparison / 误差分布对比
        ax3 = axes[1, 0]
        errors_gen = y_true - y_pred_general
        errors_pers = y_true - y_pred_personalized

        ax3.hist(errors_gen, bins=40, alpha=0.6, label='General Model', color='blue', edgecolor='black')
        ax3.hist(errors_pers, bins=40, alpha=0.6, label='Personalized Model', color='red', edgecolor='black')
        ax3.axvline(x=0, color='green', linestyle='--', lw=2, label='Zero Error')
        ax3.set_xlabel('Prediction Error (mmHg)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title(f'{target_name} BP - Error Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 4. Performance metrics comparison / 性能指标对比
        ax4 = axes[1, 1]

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae_gen = mean_absolute_error(y_true, y_pred_general)
        mae_pers = mean_absolute_error(y_true, y_pred_personalized)
        mse_gen = mean_squared_error(y_true, y_pred_general)
        mse_pers = mean_squared_error(y_true, y_pred_personalized)
        rmse_gen = np.sqrt(mse_gen)
        rmse_pers = np.sqrt(mse_pers)

        metrics = ['MAE', 'MSE', 'RMSE']
        gen_values = [mae_gen, mse_gen, rmse_gen]
        pers_values = [mae_pers, mse_pers, rmse_pers]

        x_pos = np.arange(len(metrics))
        width = 0.35

        bars1 = ax4.bar(x_pos - width/2, gen_values, width, label='General Model',
                       alpha=0.8, color='blue')
        bars2 = ax4.bar(x_pos + width/2, pers_values, width, label='Personalized Model',
                       alpha=0.8, color='red')

        # Add value labels on bars / 在柱状图上添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=9)

        ax4.set_ylabel('Error Value', fontsize=11)
        ax4.set_title(f'{target_name} BP - Performance Metrics Comparison',
                     fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(metrics)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if self.save_figures:
            filename = f'transfer_learning_comparison_{target_name}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"  Saved transfer learning comparison: {filename}")

        return fig

    def plot_improvement_summary(self, general_results, personalized_results):
        """
        Plot improvement summary for transfer learning.
        绘制迁移学习改善总结图。

        Args:
            general_results: General model results dict / 通用模型结果字典
            personalized_results: Personalized model results dict / 个性化模型结果字典
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        bp_types = ['Systolic', 'Diastolic']

        # Calculate improvements / 计算改善百分比
        mae_improvements = []
        mse_improvements = []

        for bp_type in ['systolic', 'diastolic']:
            mae_gen = general_results[bp_type]['MAE']
            mae_pers = personalized_results[bp_type]['MAE']
            mse_gen = general_results[bp_type]['MSE']
            mse_pers = personalized_results[bp_type]['MSE']

            mae_improvement = (mae_gen - mae_pers) / mae_gen * 100
            mse_improvement = (mse_gen - mse_pers) / mse_gen * 100

            mae_improvements.append(mae_improvement)
            mse_improvements.append(mse_improvement)

        # Plot MAE improvement / 绘制MAE改善
        x_pos = np.arange(len(bp_types))
        bars = ax1.bar(x_pos, mae_improvements, alpha=0.8, color=['#2E86AB', '#A23B72'])
        ax1.set_ylabel('MAE Improvement (%)', fontsize=12)
        ax1.set_title('MAE Improvement from Transfer Learning', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(bp_types)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # Add value labels / 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=11, fontweight='bold')

        # Plot MSE improvement / 绘制MSE改善
        bars = ax2.bar(x_pos, mse_improvements, alpha=0.8, color=['#2E86AB', '#A23B72'])
        ax2.set_ylabel('MSE Improvement (%)', fontsize=12)
        ax2.set_title('MSE Improvement from Transfer Learning', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(bp_types)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # Add value labels / 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=11, fontweight='bold')

        plt.tight_layout()

        if self.save_figures:
            filename = 'transfer_learning_improvement_summary.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"  Saved improvement summary: {filename}")

        return fig


if __name__ == '__main__':
    # Test code / 测试代码
    print("=" * 60)
    print("Evaluation Module Test")
    print("=" * 60)

    # Generate mock data / 生成模拟数据
    np.random.seed(42)
    y_true = np.random.randn(100) * 10 + 120
    y_pred = y_true + np.random.randn(100) * 5

    # Create evaluator / 创建评估器
    evaluator = ModelEvaluator()

    # Test visualization / 测试可视化
    mock_result = {
        'model_name': 'test_model',
        'y_test': y_true,
        'predictions_test': y_pred,
    }

    evaluator.plot_prediction_vs_actual(y_true, y_pred, 'test_model', 'Systolic', 'test')
    evaluator.plot_residuals(y_true, y_pred, 'test_model', 'Systolic', 'test')

    # Test transfer learning comparison / 测试迁移学习对比
    y_pred_general = y_true + np.random.randn(100) * 8
    y_pred_personalized = y_true + np.random.randn(100) * 3
    evaluator.plot_transfer_learning_comparison(y_true, y_pred_general, y_pred_personalized, 'Systolic')

    # Test improvement summary / 测试改善总结
    general_results = {
        'systolic': {'MAE': 10.5, 'MSE': 150},
        'diastolic': {'MAE': 8.2, 'MSE': 95}
    }
    personalized_results = {
        'systolic': {'MAE': 4.2, 'MSE': 45},
        'diastolic': {'MAE': 3.5, 'MSE': 30}
    }
    evaluator.plot_improvement_summary(general_results, personalized_results)

    print("\nAll tests completed!")