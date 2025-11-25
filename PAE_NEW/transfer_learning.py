"""
Transfer Learning Core Module
迁移学习核心模块

This module implements transfer learning for blood pressure prediction.
该模块实现血压预测的迁移学习。
"""

import numpy as np
import pickle
import os
import warnings
from typing import Optional, Tuple, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Model imports / 模型导入
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

from sklearn.ensemble import GradientBoostingRegressor


class GeneralTrainer:
    """
    Train general models on all training patient data.
    在所有训练患者数据上训练通用模型。
    """

    def __init__(self, model_type='xgboost', **model_params):
        """
        Initialize general model trainer.
        初始化通用模型训练器。

        Args:
            model_type: 'xgboost', 'lightgbm', or 'gradient_boosting'
                       模型类型
            **model_params: Model hyperparameters / 模型超参数
        """
        self.model_type = model_type
        self.model_params = model_params

        # Models for systolic and diastolic BP / 收缩压和舒张压的模型
        self.model_sys = None
        self.model_dia = None

        # Training history / 训练历史
        self.training_history = {}

    def train(self, X_train: np.ndarray, y_train_sys: np.ndarray,
              y_train_dia: np.ndarray, verbose: bool = True) -> Tuple[Any, Any]:
        """
        Train general models for both systolic and diastolic BP.
        训练收缩压和舒张压的通用模型。

        Args:
            X_train: Training features / 训练特征
            y_train_sys: Systolic BP labels / 收缩压标签
            y_train_dia: Diastolic BP labels / 舒张压标签
            verbose: Print training progress / 打印训练进度

        Returns:
            Tuple of (model_sys, model_dia) / 收缩压和舒张压模型的元组
        """
        if verbose:
            print("\n" + "=" * 60)
            print("Training General Models / 训练通用模型")
            print("=" * 60)
            print(f"Model Type: {self.model_type}")
            print(f"Training Samples: {len(X_train)}")
            print(f"Features: {X_train.shape[1]}")

        # Train systolic BP model / 训练收缩压模型
        if verbose:
            print("\n[1/2] Training Systolic BP Model...")
        self.model_sys = self._train_single_model(X_train, y_train_sys, 'systolic')

        # Train diastolic BP model / 训练舒张压模型
        if verbose:
            print("\n[2/2] Training Diastolic BP Model...")
        self.model_dia = self._train_single_model(X_train, y_train_dia, 'diastolic')

        if verbose:
            print("\n" + "=" * 60)
            print("General Models Training Completed")
            print("=" * 60 + "\n")

        return self.model_sys, self.model_dia

    def _train_single_model(self, X: np.ndarray, y: np.ndarray, bp_type: str) -> Any:
        """
        Train a single model.
        训练单个模型。

        Args:
            X: Features / 特征
            y: Labels / 标签
            bp_type: 'systolic' or 'diastolic' / 血压类型

        Returns:
            Trained model / 训练好的模型
        """
        if self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed")
            model = xgb.XGBRegressor(**self.model_params)

        elif self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM is not installed")
            model = lgb.LGBMRegressor(**self.model_params)

        elif self.model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(**self.model_params)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Train model / 训练模型
        model.fit(X, y)

        # Store training info / 存储训练信息
        self.training_history[bp_type] = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'model_params': self.model_params
        }

        return model

    def evaluate(self, X_test: np.ndarray, y_test_sys: np.ndarray,
                 y_test_dia: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate general models on test data.
        在测试数据上评估通用模型。

        Args:
            X_test: Test features / 测试特征
            y_test_sys: Systolic BP / 收缩压
            y_test_dia: Diastolic BP / 舒张压

        Returns:
            Dictionary with performance metrics / 性能指标字典
        """
        # Predict / 预测
        y_pred_sys = self.model_sys.predict(X_test)
        y_pred_dia = self.model_dia.predict(X_test)

        # Calculate metrics / 计算指标
        results = {
            'systolic': self._calculate_metrics(y_test_sys, y_pred_sys),
            'diastolic': self._calculate_metrics(y_test_dia, y_pred_dia)
        }

        return results

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics / 计算回归指标"""
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }

    def get_models(self) -> Tuple[Any, Any]:
        """Get trained models / 获取训练好的模型"""
        return self.model_sys, self.model_dia


class PersonalFineTuner:
    """
    Fine-tune general model for individual patient.
    为个体患者微调通用模型。
    """

    def __init__(self, model_type='xgboost', strategy='incremental', **fine_tune_params):
        """
        Initialize fine-tuner.
        初始化微调器。

        Args:
            model_type: 'xgboost', 'lightgbm', or 'gradient_boosting'
            strategy: 'incremental' or 'correction_model' / 微调策略
            **fine_tune_params: Fine-tuning hyperparameters / 微调超参数
        """
        self.model_type = model_type
        self.strategy = strategy
        self.fine_tune_params = fine_tune_params

        # Fine-tuned models / 微调后的模型
        self.personalized_model_sys = None
        self.personalized_model_dia = None

        # Fine-tuning history / 微调历史
        self.fine_tuning_history = {}

    def fine_tune(self, general_model_sys: Any, general_model_dia: Any,
                  X_calib: np.ndarray, y_calib_sys: np.ndarray,
                  y_calib_dia: np.ndarray, verbose: bool = True
                  ) -> Tuple[Any, Any]:
        """
        Fine-tune general models using calibration data.
        使用校准数据微调通用模型。

        Args:
            general_model_sys: General systolic model / 通用收缩压模型
            general_model_dia: General diastolic model / 通用舒张压模型
            X_calib: Calibration features / 校准特征
            y_calib_sys: Calibration systolic BP / 校准收缩压
            y_calib_dia: Calibration diastolic BP / 校准舒张压
            verbose: Print progress / 打印进度

        Returns:
            Tuple of personalized models / 个性化模型元组
        """
        if verbose:
            print("\n" + "=" * 60)
            print("Fine-tuning Models / 微调模型")
            print("=" * 60)
            print(f"Strategy: {self.strategy}")
            print(f"Calibration Samples: {len(X_calib)}")

        # Fine-tune systolic model / 微调收缩压模型
        if verbose:
            print("\n[1/2] Fine-tuning Systolic BP Model...")
        self.personalized_model_sys = self._fine_tune_single_model(
            general_model_sys, X_calib, y_calib_sys, 'systolic', verbose
        )

        # Fine-tune diastolic model / 微调舒张压模型
        if verbose:
            print("\n[2/2] Fine-tuning Diastolic BP Model...")
        self.personalized_model_dia = self._fine_tune_single_model(
            general_model_dia, X_calib, y_calib_dia, 'diastolic', verbose
        )

        if verbose:
            print("\n" + "=" * 60)
            print("Fine-tuning Completed")
            print("=" * 60 + "\n")

        return self.personalized_model_sys, self.personalized_model_dia

    def _fine_tune_single_model(self, general_model: Any, X_calib: np.ndarray,
                                 y_calib: np.ndarray, bp_type: str,
                                 verbose: bool = True) -> Any:
        """
        Fine-tune a single model.
        微调单个模型。

        Args:
            general_model: General model to fine-tune / 要微调的通用模型
            X_calib: Calibration features / 校准特征
            y_calib: Calibration labels / 校准标签
            bp_type: 'systolic' or 'diastolic'
            verbose: Print progress / 打印进度

        Returns:
            Fine-tuned model / 微调后的模型
        """
        if self.strategy == 'incremental':
            return self._incremental_training(general_model, X_calib, y_calib, bp_type, verbose)
        elif self.strategy == 'correction_model':
            return self._correction_model(general_model, X_calib, y_calib, bp_type, verbose)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _incremental_training(self, general_model: Any, X_calib: np.ndarray,
                              y_calib: np.ndarray, bp_type: str,
                              verbose: bool = True) -> Any:
        """
        Incremental training: Continue training from general model.
        增量训练：从通用模型继续训练。

        Args:
            general_model: General model / 通用模型
            X_calib: Calibration features / 校准特征
            y_calib: Calibration labels / 校准标签
            bp_type: Blood pressure type / 血压类型
            verbose: Print progress / 打印进度

        Returns:
            Fine-tuned model / 微调后的模型
        """
        # Handle early stopping / 处理early stopping
        use_early_stopping = self.fine_tune_params.get('early_stopping', {}).get('enabled', False)

        if use_early_stopping:
            val_frac = self.fine_tune_params['early_stopping'].get('validation_fraction', 0.2)
            n_val = int(len(X_calib) * val_frac)

            if n_val < 10:
                if verbose:
                    print(f"  Warning: Validation set too small ({n_val}). Disabling early stopping.")
                use_early_stopping = False

        # Split calibration into train and validation / 将校准数据分为训练和验证
        if use_early_stopping:
            n_val = int(len(X_calib) * val_frac)
            X_train_calib = X_calib[:-n_val]
            y_train_calib = y_calib[:-n_val]
            X_val_calib = X_calib[-n_val:]
            y_val_calib = y_calib[-n_val:]
        else:
            X_train_calib = X_calib
            y_train_calib = y_calib

        # XGBoost incremental training / XGBoost增量训练
        if self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed")

            # Create new model with fine-tuning parameters / 使用微调参数创建新模型
            personalized_model = xgb.XGBRegressor(**self.fine_tune_params['xgboost'])

            # Continue training from general model / 从通用模型继续训练
            if use_early_stopping:
                personalized_model.fit(
                    X_train_calib, y_train_calib,
                    xgb_model=general_model.get_booster(),
                    eval_set=[(X_val_calib, y_val_calib)],
                    verbose=verbose
                )
            else:
                personalized_model.fit(
                    X_train_calib, y_train_calib,
                    xgb_model=general_model.get_booster(),
                    verbose=verbose
                )

        # LightGBM incremental training / LightGBM增量训练
        elif self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM is not installed")

            # Save general model temporarily / 临时保存通用模型
            temp_model_path = f'temp_general_model_{bp_type}.txt'
            general_model.booster_.save_model(temp_model_path)

            # Create new model / 创建新模型
            personalized_model = lgb.LGBMRegressor(**self.fine_tune_params['lightgbm'])

            # Continue training / 继续训练
            if use_early_stopping:
                personalized_model.fit(
                    X_train_calib, y_train_calib,
                    init_model=temp_model_path,
                    eval_set=[(X_val_calib, y_val_calib)],
                    callbacks=[lgb.early_stopping(
                        stopping_rounds=self.fine_tune_params['early_stopping'].get('rounds', 20)
                    )] if verbose else None
                )
            else:
                personalized_model.fit(
                    X_train_calib, y_train_calib,
                    init_model=temp_model_path
                )

            # Clean up temp file / 清理临时文件
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)

        # Sklearn Gradient Boosting (warm_start) / Sklearn梯度提升
        elif self.model_type == 'gradient_boosting':
            # Note: GradientBoostingRegressor doesn't support true incremental training
            # We'll retrain with warm_start / 注意：GradientBoostingRegressor不支持真正的增量训练
            warnings.warn(
                "GradientBoostingRegressor doesn't support true incremental training. "
                "Retraining from scratch with calibration data.",
                UserWarning
            )
            personalized_model = GradientBoostingRegressor(
                **self.fine_tune_params.get('gradient_boosting', {})
            )
            personalized_model.fit(X_train_calib, y_train_calib)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Store fine-tuning info / 存储微调信息
        self.fine_tuning_history[bp_type] = {
            'calibration_samples': len(X_calib),
            'strategy': 'incremental',
            'early_stopping': use_early_stopping
        }

        return personalized_model

    def _correction_model(self, general_model: Any, X_calib: np.ndarray,
                          y_calib: np.ndarray, bp_type: str,
                          verbose: bool = True) -> Dict[str, Any]:
        """
        Correction model: Train a small model to correct general model predictions.
        校正模型：训练一个小模型来校正通用模型的预测。

        Args:
            general_model: General model / 通用模型
            X_calib: Calibration features / 校准特征
            y_calib: Calibration labels / 校准标签
            bp_type: Blood pressure type / 血压类型
            verbose: Print progress / 打印进度

        Returns:
            Dictionary with general model and correction model
            包含通用模型和校正模型的字典
        """
        # Get general model predictions / 获取通用模型预测
        y_pred_general = general_model.predict(X_calib)

        # Calculate residuals / 计算残差
        residuals = y_calib - y_pred_general

        # Train correction model on residuals / 在残差上训练校正模型
        if self.model_type == 'xgboost':
            correction_model = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.01,
                random_state=42
            )
        elif self.model_type == 'lightgbm':
            correction_model = lgb.LGBMRegressor(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.01,
                random_state=42
            )
        else:
            correction_model = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.01,
                random_state=42
            )

        correction_model.fit(X_calib, residuals)

        if verbose:
            # Evaluate correction / 评估校正
            y_pred_corrected = y_pred_general + correction_model.predict(X_calib)
            mae_before = mean_absolute_error(y_calib, y_pred_general)
            mae_after = mean_absolute_error(y_calib, y_pred_corrected)
            print(f"  MAE before correction: {mae_before:.2f}")
            print(f"  MAE after correction: {mae_after:.2f}")
            print(f"  Improvement: {(mae_before - mae_after) / mae_before * 100:.1f}%")

        # Return both models / 返回两个模型
        personalized_model = {
            'general_model': general_model,
            'correction_model': correction_model,
            'strategy': 'correction'
        }

        # Store info / 存储信息
        self.fine_tuning_history[bp_type] = {
            'calibration_samples': len(X_calib),
            'strategy': 'correction_model'
        }

        return personalized_model

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using personalized models.
        使用个性化模型进行预测。

        Args:
            X: Features / 特征

        Returns:
            Tuple of (systolic predictions, diastolic predictions)
            收缩压和舒张压预测的元组
        """
        # Handle correction model strategy / 处理校正模型策略
        if self.strategy == 'correction_model':
            y_pred_sys = (self.personalized_model_sys['general_model'].predict(X) +
                          self.personalized_model_sys['correction_model'].predict(X))
            y_pred_dia = (self.personalized_model_dia['general_model'].predict(X) +
                          self.personalized_model_dia['correction_model'].predict(X))
        else:
            y_pred_sys = self.personalized_model_sys.predict(X)
            y_pred_dia = self.personalized_model_dia.predict(X)

        return y_pred_sys, y_pred_dia

    def evaluate(self, X_eval: np.ndarray, y_eval_sys: np.ndarray,
                 y_eval_dia: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate personalized models.
        评估个性化模型。

        Args:
            X_eval: Evaluation features / 评估特征
            y_eval_sys: Systolic BP / 收缩压
            y_eval_dia: Diastolic BP / 舒张压

        Returns:
            Performance metrics / 性能指标
        """
        y_pred_sys, y_pred_dia = self.predict(X_eval)

        results = {
            'systolic': self._calculate_metrics(y_eval_sys, y_pred_sys),
            'diastolic': self._calculate_metrics(y_eval_dia, y_pred_dia)
        }

        return results

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics / 计算回归指标"""
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }


class ModelManager:
    """
    Manage model saving and loading.
    管理模型保存和加载。
    """

    @staticmethod
    def save_general_models(model_sys: Any, model_dia: Any, save_dir: str, scaler: Any = None):
        """
        Save general models and optionally the scaler.
        保存通用模型和可选的特征标准化器。

        Args:
            model_sys: Systolic model / 收缩压模型
            model_dia: Diastolic model / 舒张压模型
            save_dir: Directory to save models / 保存目录
            scaler: Feature scaler (optional) / 特征标准化器（可选）
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save systolic model / 保存收缩压模型
        sys_path = os.path.join(save_dir, 'general_model_systolic.pkl')
        with open(sys_path, 'wb') as f:
            pickle.dump(model_sys, f)

        # Save diastolic model / 保存舒张压模型
        dia_path = os.path.join(save_dir, 'general_model_diastolic.pkl')
        with open(dia_path, 'wb') as f:
            pickle.dump(model_dia, f)

        # Save scaler if provided / 保存标准化器（如果提供）
        if scaler is not None:
            scaler_path = os.path.join(save_dir, 'feature_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"General models and scaler saved to: {save_dir}")
        else:
            print(f"General models saved to: {save_dir}")

    @staticmethod
    def load_general_models(load_dir: str, load_scaler: bool = True) -> Tuple[Any, Any, Any]:
        """
        Load general models and optionally the scaler.
        加载通用模型和可选的特征标准化器。

        Args:
            load_dir: Directory containing models / 包含模型的目录
            load_scaler: Whether to load scaler / 是否加载标准化器

        Returns:
            Tuple of (model_sys, model_dia, scaler) / 模型和标准化器元组
            If load_scaler=False or scaler doesn't exist, returns (model_sys, model_dia, None)
        """
        sys_path = os.path.join(load_dir, 'general_model_systolic.pkl')
        dia_path = os.path.join(load_dir, 'general_model_diastolic.pkl')
        scaler_path = os.path.join(load_dir, 'feature_scaler.pkl')

        with open(sys_path, 'rb') as f:
            model_sys = pickle.load(f)

        with open(dia_path, 'rb') as f:
            model_dia = pickle.load(f)

        scaler = None
        if load_scaler and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"General models and scaler loaded from: {load_dir}")
        else:
            print(f"General models loaded from: {load_dir}")
            if load_scaler:
                print(f"  ⚠ Warning: Scaler not found, will need to re-fit on data")

        return model_sys, model_dia, scaler

    @staticmethod
    def save_personalized_models(model_sys: Any, model_dia: Any, save_dir: str):
        """Save personalized models / 保存个性化模型"""
        os.makedirs(save_dir, exist_ok=True)

        sys_path = os.path.join(save_dir, 'personalized_model_systolic.pkl')
        dia_path = os.path.join(save_dir, 'personalized_model_diastolic.pkl')

        with open(sys_path, 'wb') as f:
            pickle.dump(model_sys, f)

        with open(dia_path, 'wb') as f:
            pickle.dump(model_dia, f)

        print(f"Personalized models saved to: {save_dir}")

    @staticmethod
    def load_personalized_models(load_dir: str) -> Tuple[Any, Any]:
        """Load personalized models / 加载个性化模型"""
        sys_path = os.path.join(load_dir, 'personalized_model_systolic.pkl')
        dia_path = os.path.join(load_dir, 'personalized_model_diastolic.pkl')

        with open(sys_path, 'rb') as f:
            model_sys = pickle.load(f)

        with open(dia_path, 'rb') as f:
            model_dia = pickle.load(f)

        print(f"Personalized models loaded from: {load_dir}")
        return model_sys, model_dia


# ============================================================================
# Testing / 测试
# ============================================================================

if __name__ == '__main__':
    print("Testing Transfer Learning Module...\n")

    # Create dummy data / 创建虚拟数据
    np.random.seed(42)
    X_train = np.random.randn(1000, 17)
    y_train_sys = np.random.uniform(90, 180, 1000)
    y_train_dia = np.random.uniform(60, 110, 1000)

    X_calib = np.random.randn(200, 17)
    y_calib_sys = np.random.uniform(90, 180, 200)
    y_calib_dia = np.random.uniform(60, 110, 200)

    X_eval = np.random.randn(300, 17)
    y_eval_sys = np.random.uniform(90, 180, 300)
    y_eval_dia = np.random.uniform(60, 110, 300)

    # Test GeneralTrainer / 测试通用训练器
    print("=" * 60)
    print("Test 1: GeneralTrainer")
    print("=" * 60)
    trainer = GeneralTrainer(model_type='gradient_boosting',
                             n_estimators=50, max_depth=5, random_state=42)
    model_sys, model_dia = trainer.train(X_train, y_train_sys, y_train_dia)
    results = trainer.evaluate(X_eval, y_eval_sys, y_eval_dia)
    print("\nGeneral Model Performance:")
    print(f"  Systolic MAE: {results['systolic']['MAE']:.2f}")
    print(f"  Diastolic MAE: {results['diastolic']['MAE']:.2f}")

    # Test PersonalFineTuner / 测试个性化微调器
    print("\n" + "=" * 60)
    print("Test 2: PersonalFineTuner")
    print("=" * 60)
    fine_tuner = PersonalFineTuner(
        model_type='gradient_boosting',
        strategy='incremental',
        gradient_boosting={'n_estimators': 30, 'max_depth': 5, 'learning_rate': 0.01}
    )
    pers_sys, pers_dia = fine_tuner.fine_tune(
        model_sys, model_dia, X_calib, y_calib_sys, y_calib_dia
    )
    pers_results = fine_tuner.evaluate(X_eval, y_eval_sys, y_eval_dia)
    print("\nPersonalized Model Performance:")
    print(f"  Systolic MAE: {pers_results['systolic']['MAE']:.2f}")
    print(f"  Diastolic MAE: {pers_results['diastolic']['MAE']:.2f}")

    # Test ModelManager / 测试模型管理器
    print("\n" + "=" * 60)
    print("Test 3: ModelManager")
    print("=" * 60)
    ModelManager.save_general_models(model_sys, model_dia, 'test_models/general')
    loaded_sys, loaded_dia = ModelManager.load_general_models('test_models/general')
    print("Models saved and loaded successfully!")

    print("\nAll tests completed!")
