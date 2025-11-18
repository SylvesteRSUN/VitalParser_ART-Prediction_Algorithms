"""
Test script for transfer learning modules
迁移学习模块测试脚本

Run this to verify all modules are working correctly.
运行此脚本以验证所有模块是否正常工作。
"""

import numpy as np
import sys
import os

print("=" * 80)
print(" TRANSFER LEARNING MODULES TEST / 迁移学习模块测试")
print("=" * 80)

# Test 1: Configuration module / 测试配置模块
print("\n[Test 1/5] Testing config_transfer.py...")
try:
    from config_transfer import (
        GENERAL_MODEL_CONFIG, FINE_TUNING_CONFIG, DATA_SPLIT_CONFIG,
        EVALUATION_CONFIG, PATH_CONFIG, EXPERIMENT_CONFIG,
        validate_config, create_output_directories
    )
    validate_config()
    print("✓ Configuration module loaded successfully")
except Exception as e:
    print(f"✗ Configuration module failed: {e}")
    sys.exit(1)

# Test 2: Data splitter / 测试数据分割器
print("\n[Test 2/5] Testing data_splitter.py...")
try:
    from data_splitter import PatientDataSplitter, MultiSizeSplitter, validate_split_data

    # Create dummy data / 创建虚拟数据
    np.random.seed(42)
    X_test = np.random.randn(500, 17)
    y_test_sys = np.random.uniform(90, 180, 500)
    y_test_dia = np.random.uniform(60, 110, 500)

    # Test sample-based split / 测试基于样本的分割
    splitter = PatientDataSplitter(split_method='sample_based', n_samples=200)
    X_calib, y_calib_sys, y_calib_dia, X_eval, y_eval_sys, y_eval_dia = splitter.split(
        X_test, y_test_sys, y_test_dia
    )

    # Validate / 验证
    validate_split_data(X_calib, y_calib_sys, y_calib_dia, X_eval, y_eval_sys, y_eval_dia)

    assert len(X_calib) == 200, f"Expected 200 calibration samples, got {len(X_calib)}"
    assert len(X_eval) == 300, f"Expected 300 evaluation samples, got {len(X_eval)}"

    print("✓ Data splitter working correctly")
except Exception as e:
    print(f"✗ Data splitter failed: {e}")
    sys.exit(1)

# Test 3: Transfer learning core / 测试迁移学习核心
print("\n[Test 3/5] Testing transfer_learning.py...")
try:
    from transfer_learning import GeneralTrainer, PersonalFineTuner, ModelManager

    # Create dummy training data / 创建虚拟训练数据
    X_train = np.random.randn(1000, 17)
    y_train_sys = np.random.uniform(90, 180, 1000)
    y_train_dia = np.random.uniform(60, 110, 1000)

    # Test GeneralTrainer / 测试通用训练器
    trainer = GeneralTrainer(
        model_type='gradient_boosting',  # Use sklearn (no external dependencies)
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    model_sys, model_dia = trainer.train(X_train, y_train_sys, y_train_dia, verbose=False)

    # Test predictions / 测试预测
    y_pred_sys = model_sys.predict(X_eval)
    y_pred_dia = model_dia.predict(X_eval)

    assert len(y_pred_sys) == len(X_eval), "Prediction length mismatch"

    # Test PersonalFineTuner / 测试个性化微调器
    fine_tuner = PersonalFineTuner(
        model_type='gradient_boosting',
        strategy='incremental',
        gradient_boosting={'n_estimators': 30, 'max_depth': 5, 'learning_rate': 0.01}
    )
    pers_sys, pers_dia = fine_tuner.fine_tune(
        model_sys, model_dia, X_calib, y_calib_sys, y_calib_dia, verbose=False
    )

    # Test personalized predictions / 测试个性化预测
    y_pers_sys, y_pers_dia = fine_tuner.predict(X_eval)

    assert len(y_pers_sys) == len(X_eval), "Personalized prediction length mismatch"

    # Test ModelManager / 测试模型管理器
    test_save_dir = 'test_temp_models'
    os.makedirs(test_save_dir, exist_ok=True)
    ModelManager.save_general_models(model_sys, model_dia, test_save_dir)
    loaded_sys, loaded_dia = ModelManager.load_general_models(test_save_dir)

    # Clean up / 清理
    import shutil
    shutil.rmtree(test_save_dir)

    print("✓ Transfer learning core working correctly")
except Exception as e:
    print(f"✗ Transfer learning core failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Evaluation visualizations / 测试评估可视化
print("\n[Test 4/5] Testing evaluation.py (transfer learning features)...")
try:
    from evaluation import ModelEvaluator

    # Create evaluator / 创建评估器
    import tempfile
    temp_dir = tempfile.mkdtemp()
    evaluator = ModelEvaluator(output_dir=temp_dir)

    # Test transfer learning comparison / 测试迁移学习对比
    evaluator.plot_transfer_learning_comparison(
        y_eval_sys, y_pred_sys, y_pers_sys, 'Systolic'
    )

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

    # Clean up / 清理
    import shutil
    shutil.rmtree(temp_dir)

    print("✓ Evaluation visualizations working correctly")
except Exception as e:
    print(f"✗ Evaluation visualizations failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Main pipeline (import check) / 测试主程序(导入检查)
print("\n[Test 5/5] Testing main_transfer_learning.py (import check)...")
try:
    from main_transfer_learning import TransferLearningPipeline
    print("✓ Main pipeline module loaded successfully")
except Exception as e:
    print(f"✗ Main pipeline failed: {e}")
    sys.exit(1)

# Summary / 总结
print("\n" + "=" * 80)
print(" ALL TESTS PASSED! / 所有测试通过!")
print("=" * 80)
print("\nYou can now run the transfer learning pipeline:")
print("您现在可以运行迁移学习流程:")
print("\n  python main_transfer_learning.py")
print("\nFor help:")
print("获取帮助:")
print("\n  python main_transfer_learning.py --help")
print("\nFor more information, see TRANSFER_LEARNING_README.md")
print("更多信息请参阅 TRANSFER_LEARNING_README.md")
print("=" * 80)
