"""
Main Transfer Learning Pipeline
迁移学习主程序

Complete pipeline for transfer learning-based blood pressure prediction.
基于迁移学习的血压预测完整流程。
"""

import numpy as np
import os
import sys
from pathlib import Path
import argparse
import time

# Import existing modules / 导入现有模块
from config import DATA_CONFIG, SIGNAL_CONFIG, FEATURE_CONFIG, EVALUATION_CONFIG
from data_loader import load_train_test_data
from signal_processing import SignalProcessor
from feature_extraction import CycleBasedFeatureExtractor

# Import transfer learning modules / 导入迁移学习模块
from config_transfer import (
    GENERAL_MODEL_CONFIG, FINE_TUNING_CONFIG, DATA_SPLIT_CONFIG,
    EVALUATION_CONFIG as TL_EVAL_CONFIG, PATH_CONFIG, EXPERIMENT_CONFIG,
    create_output_directories, get_output_path, get_model_save_path
)
from data_splitter import PatientDataSplitter, validate_split_data
from transfer_learning import GeneralTrainer, PersonalFineTuner, ModelManager


class TransferLearningPipeline:
    """
    Complete transfer learning pipeline.
    完整的迁移学习流程。
    """

    def __init__(self, test_file_name: str, verbose: bool = True):
        """
        Initialize pipeline.
        初始化流程。

        Args:
            test_file_name: Name of test file / 测试文件名
            verbose: Print progress / 打印进度
        """
        self.test_file_name = test_file_name
        self.verbose = verbose

        # Create output directories / 创建输出目录
        create_output_directories(test_file_name)

        # Results storage / 结果存储
        self.results = {}

    def run_complete_pipeline(self):
        """
        Run the complete transfer learning pipeline.
        运行完整的迁移学习流程。
        """
        print("\n" + "=" * 80)
        print(" TRANSFER LEARNING PIPELINE FOR BLOOD PRESSURE PREDICTION")
        print(" 迁移学习血压预测流程")
        print("=" * 80)

        start_time = time.time()

        # Step 1: Load and process training data / 步骤1：加载并处理训练数据
        print("\n[STEP 1/7] Loading and Processing Training Data...")
        X_train, y_train_sys, y_train_dia, scaler = self._load_and_process_training_data()
        self.results['n_train'] = len(X_train)

        # Step 2: Train general models / 步骤2：训练通用模型
        print("\n[STEP 2/7] Training General Models...")
        general_model_sys, general_model_dia = self._train_general_models(
            X_train, y_train_sys, y_train_dia
        )

        # Step 3: Save general models / 步骤3：保存通用模型
        print("\n[STEP 3/7] Saving General Models...")
        self._save_general_models(general_model_sys, general_model_dia)

        # Step 4: Load and process test data / 步骤4：加载并处理测试数据
        print("\n[STEP 4/7] Loading and Processing Test Data...")
        X_test, y_test_sys, y_test_dia = self._load_and_process_test_data(scaler)
        self.results['n_test'] = len(X_test)

        # Step 5: Split test data into calibration and evaluation sets
        # 步骤5：将测试数据分割为校准集和评估集
        print("\n[STEP 5/7] Splitting Test Data...")
        X_calib, y_calib_sys, y_calib_dia, X_eval, y_eval_sys, y_eval_dia = \
            self._split_test_data(X_test, y_test_sys, y_test_dia)

        # Step 6: Evaluate general models / 步骤6：评估通用模型
        print("\n[STEP 6/7] Evaluating General Models...")
        general_results = self._evaluate_general_models(
            general_model_sys, general_model_dia, X_eval, y_eval_sys, y_eval_dia
        )

        # Step 7: Fine-tune and evaluate personalized models
        # 步骤7：微调并评估个性化模型
        print("\n[STEP 7/7] Fine-tuning Personalized Models...")
        personalized_results = self._fine_tune_and_evaluate(
            general_model_sys, general_model_dia,
            X_calib, y_calib_sys, y_calib_dia,
            X_eval, y_eval_sys, y_eval_dia
        )

        # Compare results / 对比结果
        self._compare_and_report_results(general_results, personalized_results)

        # Generate visualizations / 生成可视化
        self._generate_visualizations(
            general_results, personalized_results,
            X_eval, y_eval_sys, y_eval_dia
        )

        # Save predictions / 保存预测
        self._save_predictions(
            general_model_sys, general_model_dia,
            X_eval, y_eval_sys, y_eval_dia
        )

        elapsed_time = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"Pipeline completed in {elapsed_time:.2f} seconds")
        print(f"{'=' * 80}\n")

    def _load_and_process_training_data(self):
        """
        Load and process all training data.
        加载并处理所有训练数据。
        """
        # Load training files / 加载训练文件
        train_data_all, _ = load_train_test_data(DATA_CONFIG)

        # Process signals / 处理信号
        processor = SignalProcessor(SIGNAL_CONFIG)

        X_train_list = []
        y_train_sys_list = []
        y_train_dia_list = []

        for i, (pleth_raw, art_raw) in enumerate(train_data_all):
            if self.verbose:
                print(f"\n  Processing training file {i + 1}/{len(train_data_all)}...")

            # Apply signal processing / 应用信号处理
            pleth_processed = processor.process_signal(pleth_raw, 'PLETH')
            art_processed = processor.process_signal(art_raw, 'ART')

            # Extract features / 提取特征
            extractor = CycleBasedFeatureExtractor(FEATURE_CONFIG)
            X, y_sys, y_dia, _ = extractor.extract_features(
                pleth_processed, art_processed
            )

            if len(X) > 0:
                X_train_list.append(X)
                y_train_sys_list.append(y_sys)
                y_train_dia_list.append(y_dia)

                if self.verbose:
                    print(f"    Extracted {len(X)} cycles")

        # Combine all training data / 合并所有训练数据
        X_train = np.vstack(X_train_list)
        y_train_sys = np.concatenate(y_train_sys_list)
        y_train_dia = np.concatenate(y_train_dia_list)

        # Standardize features / 标准化特征
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        print(f"\n  Total training samples: {len(X_train)}")
        print(f"  Features: {X_train.shape[1]}")

        return X_train, y_train_sys, y_train_dia, scaler

    def _train_general_models(self, X_train, y_train_sys, y_train_dia):
        """
        Train general models.
        训练通用模型。
        """
        model_type = GENERAL_MODEL_CONFIG['model_type']
        model_params = GENERAL_MODEL_CONFIG[model_type]

        trainer = GeneralTrainer(model_type=model_type, **model_params)
        model_sys, model_dia = trainer.train(
            X_train, y_train_sys, y_train_dia, verbose=self.verbose
        )

        return model_sys, model_dia

    def _save_general_models(self, model_sys, model_dia):
        """Save general models / 保存通用模型"""
        save_dir = get_model_save_path('general')
        ModelManager.save_general_models(model_sys, model_dia, save_dir)

    def _load_and_process_test_data(self, scaler):
        """
        Load and process test data.
        加载并处理测试数据。
        """
        _, test_data = load_train_test_data(DATA_CONFIG)
        pleth_raw, art_raw = test_data

        # Process signals / 处理信号
        processor = SignalProcessor(SIGNAL_CONFIG)
        pleth_processed = processor.process_signal(pleth_raw, 'PLETH')
        art_processed = processor.process_signal(art_raw, 'ART')

        # Extract features / 提取特征
        extractor = CycleBasedFeatureExtractor(FEATURE_CONFIG)
        X_test, y_test_sys, y_test_dia, _ = extractor.extract_features(
            pleth_processed, art_processed
        )

        # Standardize using training scaler / 使用训练集的标准化器
        X_test = scaler.transform(X_test)

        print(f"  Test samples: {len(X_test)}")
        return X_test, y_test_sys, y_test_dia

    def _split_test_data(self, X_test, y_test_sys, y_test_dia):
        """
        Split test data into calibration and evaluation sets.
        将测试数据分割为校准集和评估集。
        """
        split_method = DATA_SPLIT_CONFIG['split_method']
        split_params = DATA_SPLIT_CONFIG[split_method]

        splitter = PatientDataSplitter(split_method=split_method, **split_params)
        X_calib, y_calib_sys, y_calib_dia, X_eval, y_eval_sys, y_eval_dia = \
            splitter.split(X_test, y_test_sys, y_test_dia)

        # Validate split / 验证分割
        validate_split_data(
            X_calib, y_calib_sys, y_calib_dia,
            X_eval, y_eval_sys, y_eval_dia
        )

        # Store split info / 存储分割信息
        self.results['n_calibration'] = len(X_calib)
        self.results['n_evaluation'] = len(X_eval)

        return X_calib, y_calib_sys, y_calib_dia, X_eval, y_eval_sys, y_eval_dia

    def _evaluate_general_models(self, model_sys, model_dia, X_eval, y_eval_sys, y_eval_dia):
        """
        Evaluate general models on evaluation set.
        在评估集上评估通用模型。
        """
        y_pred_sys = model_sys.predict(X_eval)
        y_pred_dia = model_dia.predict(X_eval)

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        results = {
            'systolic': {
                'MAE': mean_absolute_error(y_eval_sys, y_pred_sys),
                'MSE': mean_squared_error(y_eval_sys, y_pred_sys),
                'RMSE': np.sqrt(mean_squared_error(y_eval_sys, y_pred_sys)),
                'R2': r2_score(y_eval_sys, y_pred_sys),
                'predictions': y_pred_sys
            },
            'diastolic': {
                'MAE': mean_absolute_error(y_eval_dia, y_pred_dia),
                'MSE': mean_squared_error(y_eval_dia, y_pred_dia),
                'RMSE': np.sqrt(mean_squared_error(y_eval_dia, y_pred_dia)),
                'R2': r2_score(y_eval_dia, y_pred_dia),
                'predictions': y_pred_dia
            }
        }

        print("\n  General Model Performance on Evaluation Set:")
        print(f"    Systolic  - MAE: {results['systolic']['MAE']:.2f}, "
              f"MSE: {results['systolic']['MSE']:.2f}, R²: {results['systolic']['R2']:.3f}")
        print(f"    Diastolic - MAE: {results['diastolic']['MAE']:.2f}, "
              f"MSE: {results['diastolic']['MSE']:.2f}, R²: {results['diastolic']['R2']:.3f}")

        return results

    def _fine_tune_and_evaluate(self, general_model_sys, general_model_dia,
                                 X_calib, y_calib_sys, y_calib_dia,
                                 X_eval, y_eval_sys, y_eval_dia):
        """
        Fine-tune models and evaluate on evaluation set.
        微调模型并在评估集上评估。
        """
        model_type = GENERAL_MODEL_CONFIG['model_type']
        strategy = FINE_TUNING_CONFIG['strategy']
        fine_tune_params = FINE_TUNING_CONFIG.copy()

        # Create fine-tuner / 创建微调器
        fine_tuner = PersonalFineTuner(
            model_type=model_type,
            strategy=strategy,
            **fine_tune_params
        )

        # Fine-tune / 微调
        pers_model_sys, pers_model_dia = fine_tuner.fine_tune(
            general_model_sys, general_model_dia,
            X_calib, y_calib_sys, y_calib_dia,
            verbose=self.verbose
        )

        # Save personalized models / 保存个性化模型
        save_dir = get_model_save_path('personalized', self.test_file_name)
        ModelManager.save_personalized_models(pers_model_sys, pers_model_dia, save_dir)

        # Evaluate / 评估
        results = fine_tuner.evaluate(X_eval, y_eval_sys, y_eval_dia)

        # Store predictions / 存储预测
        y_pred_sys, y_pred_dia = fine_tuner.predict(X_eval)
        results['systolic']['predictions'] = y_pred_sys
        results['diastolic']['predictions'] = y_pred_dia

        print("\n  Personalized Model Performance on Evaluation Set:")
        print(f"    Systolic  - MAE: {results['systolic']['MAE']:.2f}, "
              f"MSE: {results['systolic']['MSE']:.2f}, R²: {results['systolic']['R2']:.3f}")
        print(f"    Diastolic - MAE: {results['diastolic']['MAE']:.2f}, "
              f"MSE: {results['diastolic']['MSE']:.2f}, R²: {results['diastolic']['R2']:.3f}")

        return results

    def _compare_and_report_results(self, general_results, personalized_results):
        """
        Compare and report results.
        对比并报告结果。
        """
        print("\n" + "=" * 80)
        print(" PERFORMANCE COMPARISON / 性能对比")
        print("=" * 80)

        for bp_type in ['systolic', 'diastolic']:
            print(f"\n{bp_type.upper()} BLOOD PRESSURE:")
            print("-" * 80)

            gen = general_results[bp_type]
            pers = personalized_results[bp_type]

            # Calculate improvements / 计算改善
            mae_improvement = (gen['MAE'] - pers['MAE']) / gen['MAE'] * 100
            mse_improvement = (gen['MSE'] - pers['MSE']) / gen['MSE'] * 100

            print(f"  Metric          General Model    Personalized    Improvement")
            print(f"  {'─' * 70}")
            print(f"  MAE             {gen['MAE']:>8.2f}         {pers['MAE']:>8.2f}        "
                  f"{mae_improvement:>6.1f}%")
            print(f"  MSE             {gen['MSE']:>8.2f}         {pers['MSE']:>8.2f}        "
                  f"{mse_improvement:>6.1f}%")
            print(f"  RMSE            {gen['RMSE']:>8.2f}         {pers['RMSE']:>8.2f}")
            print(f"  R²              {gen['R2']:>8.3f}         {pers['R2']:>8.3f}")

            # Check if meets targets / 检查是否达标
            targets = TL_EVAL_CONFIG['targets'][bp_type]
            print(f"\n  Target Achievement / 目标达成:")
            print(f"    MAE Target: {targets['MAE']:.0f}  - General: "
                  f"{'✓' if gen['MAE'] < targets['MAE'] else '✗'}, Personalized: "
                  f"{'✓' if pers['MAE'] < targets['MAE'] else '✗'}")
            print(f"    MSE Target: {targets['MSE']:.0f}  - General: "
                  f"{'✓' if gen['MSE'] < targets['MSE'] else '✗'}, Personalized: "
                  f"{'✓' if pers['MSE'] < targets['MSE'] else '✗'}")

        print("\n" + "=" * 80)

        # Save text report / 保存文本报告
        self._save_text_report(general_results, personalized_results)

    def _generate_visualizations(self, general_results, personalized_results,
                                 X_eval, y_eval_sys, y_eval_dia):
        """
        Generate transfer learning visualizations.
        生成迁移学习可视化图表。
        """
        print("\n" + "=" * 80)
        print(" GENERATING VISUALIZATIONS / 生成可视化")
        print("=" * 80)

        from evaluation import ModelEvaluator

        # Create evaluator with transfer learning output directory / 使用迁移学习输出目录创建评估器
        plots_dir = get_output_path(self.test_file_name, PATH_CONFIG['plots_dir'])
        evaluator = ModelEvaluator(output_dir=plots_dir)

        # Generate comparison plots for systolic BP / 生成收缩压对比图
        print("\nGenerating Systolic BP visualizations...")
        evaluator.plot_transfer_learning_comparison(
            y_eval_sys,
            general_results['systolic']['predictions'],
            personalized_results['systolic']['predictions'],
            'Systolic'
        )

        # Generate comparison plots for diastolic BP / 生成舒张压对比图
        print("\nGenerating Diastolic BP visualizations...")
        evaluator.plot_transfer_learning_comparison(
            y_eval_dia,
            general_results['diastolic']['predictions'],
            personalized_results['diastolic']['predictions'],
            'Diastolic'
        )

        # Generate improvement summary / 生成改善总结
        print("\nGenerating improvement summary...")
        evaluator.plot_improvement_summary(general_results, personalized_results)

        print("\n" + "=" * 80)
        print("Visualizations completed!")
        print("=" * 80)

    def _save_predictions(self, model_sys, model_dia, X_eval, y_eval_sys, y_eval_dia):
        """
        Save predictions to CSV.
        保存预测到CSV文件。
        """
        if not TL_EVAL_CONFIG['save_predictions']:
            return

        import pandas as pd

        y_pred_sys = model_sys.predict(X_eval)
        y_pred_dia = model_dia.predict(X_eval)

        # Create DataFrame / 创建数据框
        df = pd.DataFrame({
            'true_systolic': y_eval_sys,
            'pred_systolic_general': y_pred_sys,
            'true_diastolic': y_eval_dia,
            'pred_diastolic_general': y_pred_dia
        })

        # Save / 保存
        output_path = os.path.join(
            get_output_path(self.test_file_name, PATH_CONFIG['predictions_dir']),
            'predictions.csv'
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"\nPredictions saved to: {output_path}")

    def _save_text_report(self, general_results, personalized_results):
        """Save text report / 保存文本报告"""
        if not TL_EVAL_CONFIG['generate_report']:
            return

        report_path = os.path.join(
            get_output_path(self.test_file_name, PATH_CONFIG['reports_dir']),
            'transfer_learning_report.txt'
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(" TRANSFER LEARNING REPORT / 迁移学习报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Test File: {self.test_file_name}\n")
            f.write(f"Training Samples: {self.results.get('n_train', 'N/A')}\n")
            f.write(f"Calibration Samples: {self.results.get('n_calibration', 'N/A')}\n")
            f.write(f"Evaluation Samples: {self.results.get('n_evaluation', 'N/A')}\n\n")

            for bp_type in ['systolic', 'diastolic']:
                f.write(f"\n{bp_type.upper()} BLOOD PRESSURE:\n")
                f.write("-" * 80 + "\n")

                gen = general_results[bp_type]
                pers = personalized_results[bp_type]

                mae_improvement = (gen['MAE'] - pers['MAE']) / gen['MAE'] * 100
                mse_improvement = (gen['MSE'] - pers['MSE']) / gen['MSE'] * 100

                f.write(f"  General Model    - MAE: {gen['MAE']:.2f}, MSE: {gen['MSE']:.2f}, "
                        f"RMSE: {gen['RMSE']:.2f}, R²: {gen['R2']:.3f}\n")
                f.write(f"  Personalized     - MAE: {pers['MAE']:.2f}, MSE: {pers['MSE']:.2f}, "
                        f"RMSE: {pers['RMSE']:.2f}, R²: {pers['R2']:.3f}\n")
                f.write(f"  Improvement      - MAE: {mae_improvement:.1f}%, MSE: {mse_improvement:.1f}%\n")

        print(f"Report saved to: {report_path}")


def main():
    """Main function / 主函数"""
    parser = argparse.ArgumentParser(
        description='Transfer Learning Pipeline for Blood Pressure Prediction'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default=None,
        help='Path to test .vital file (default: from config.py)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['xgboost', 'lightgbm', 'gradient_boosting'],
        default=None,
        help='Model type to use (default: from config_transfer.py)'
    )
    parser.add_argument(
        '--calibration-samples',
        type=int,
        default=None,
        help='Number of calibration samples (default: from config_transfer.py)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed progress'
    )

    args = parser.parse_args()

    # Override configs if specified / 如果指定则覆盖配置
    if args.test_file:
        DATA_CONFIG['test_file_path'] = args.test_file

    if args.model_type:
        GENERAL_MODEL_CONFIG['model_type'] = args.model_type

    if args.calibration_samples:
        DATA_SPLIT_CONFIG['sample_based']['n_samples'] = args.calibration_samples

    # Get test file name / 获取测试文件名
    test_file_path = DATA_CONFIG['test_file_path']
    test_file_name = os.path.basename(test_file_path)

    # Run pipeline / 运行流程
    pipeline = TransferLearningPipeline(test_file_name, verbose=args.verbose)
    pipeline.run_complete_pipeline()


if __name__ == '__main__':
    main()
