"""
LSTM模型 - 基于时序的血压预测
使用过去N个心跳周期预测当前心跳的血压
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
layers = tf.keras.layers
Sequential = tf.keras.Sequential
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class LSTMBloodPressurePredictor:
    """LSTM血压预测器（带个体校准）"""
    
    def __init__(self, n_past=20, n_features=17):
        """
        初始化LSTM预测器
        
        参数:
            n_past: 使用过去多少个心跳周期
            n_features: 每个心跳周期的特征数量
        """
        self.n_past = n_past
        self.n_features = n_features
        self.scaler_X = StandardScaler()
        self.scaler_y_systolic = StandardScaler()  # 收缩压专用
        self.scaler_y_diastolic = StandardScaler()  # 舒张压专用
        self.model_systolic = None
        self.model_diastolic = None
        
        # 个体校准参数
        self.calibration_offset_systolic = 0.0
        self.calibration_offset_diastolic = 0.0
        self.is_calibrated = False
    
    def prepare_sequences(self, X, y_systolic, y_diastolic):
        """
        将特征矩阵转换为时序序列
        
        参数:
            X: 特征矩阵 (n_cycles, n_features)
            y_systolic: 收缩压数组 (n_cycles,)
            y_diastolic: 舒张压数组 (n_cycles,)
        
        返回:
            (X_seq, y_sys_seq, y_dia_seq): 时序序列
        """
        X_sequences = []
        y_systolic_sequences = []
        y_diastolic_sequences = []
        
        # 创建时序窗口
        for i in range(self.n_past, len(X)):
            # 过去n_past个心跳的特征
            X_sequences.append(X[i - self.n_past:i, :])
            # 当前心跳的血压
            y_systolic_sequences.append(y_systolic[i])
            y_diastolic_sequences.append(y_diastolic[i])
        
        X_seq = np.array(X_sequences)
        y_sys_seq = np.array(y_systolic_sequences)
        y_dia_seq = np.array(y_diastolic_sequences)
        
        print(f"\n✓ 序列准备完成:")
        print(f"  序列形状: {X_seq.shape}")
        print(f"  可用样本数: {len(X_seq)} (原始: {len(X)} - 序列长度: {self.n_past})")
        
        return X_seq, y_sys_seq, y_dia_seq
    
    def build_lstm_model(self, model_name="systolic"):
        """
        构建LSTM模型
        
        参数:
            model_name: 模型名称（用于打印）
        
        返回:
            model: 编译好的Keras模型
        """
        model = Sequential([
            # 第一层LSTM：128单元，返回序列
            layers.LSTM(128, activation='relu', 
                       input_shape=(self.n_past, self.n_features),
                       return_sequences=True),
            layers.Dropout(0.2),
            
            # 第二层LSTM：64单元，返回序列
            layers.LSTM(64, activation='relu', return_sequences=True),
            layers.Dropout(0.2),
            
            # 第三层LSTM：32单元，不返回序列
            layers.LSTM(32, activation='relu', return_sequences=False),
            layers.Dropout(0.2),
            
            # 全连接层
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            
            # 输出层
            layers.Dense(1)
        ])
        
        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"\n{'='*60}")
        print(f"{model_name} LSTM模型架构:")
        print(f"{'='*60}")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train_systolic, y_train_diastolic, 
              validation_split=0.15, epochs=100, batch_size=32, verbose=1):
        """
        训练LSTM模型
        
        参数:
            X_train: 训练特征 (n_samples, n_features)
            y_train_systolic: 训练收缩压
            y_train_diastolic: 训练舒张压
            validation_split: 验证集比例
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 详细程度
        
        返回:
            (history_sys, history_dia): 训练历史
        """
        print("\n" + "="*70)
        print("LSTM模型训练开始")
        print("="*70)
        
        # 1. 数据标准化
        print("\n步骤1: 数据标准化")
        X_scaled = self.scaler_X.fit_transform(X_train)
        y_sys_scaled = self.scaler_y_systolic.fit_transform(y_train_systolic.reshape(-1, 1)).flatten()
        y_dia_scaled = self.scaler_y_diastolic.fit_transform(y_train_diastolic.reshape(-1, 1)).flatten()
        
        # 2. 准备时序序列
        print("\n步骤2: 准备时序序列")
        X_seq, y_sys_seq, y_dia_seq = self.prepare_sequences(
            X_scaled, y_sys_scaled, y_dia_scaled
        )
        
        if len(X_seq) < 100:
            print(f"\n⚠ 警告: 序列数量太少 ({len(X_seq)})，建议至少100个")
        
        # 3. 构建模型
        print("\n步骤3: 构建LSTM模型")
        self.model_systolic = self.build_lstm_model("收缩压")
        self.model_diastolic = self.build_lstm_model("舒张压")
        
        # 4. 设置回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # 5. 训练收缩压模型
        print("\n" + "="*70)
        print("训练收缩压预测模型")
        print("="*70)
        history_sys = self.model_systolic.fit(
            X_seq, y_sys_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # 6. 训练舒张压模型
        print("\n" + "="*70)
        print("训练舒张压预测模型")
        print("="*70)
        history_dia = self.model_diastolic.fit(
            X_seq, y_dia_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("\n✓ LSTM模型训练完成")
        
        return history_sys, history_dia
    
    def calibrate(self, X_calib, y_calib_systolic, y_calib_diastolic):
        """
        使用校准数据计算个体偏移
        
        参数:
            X_calib: 校准特征（测试集的前N分钟）
            y_calib_systolic: 真实收缩压
            y_calib_diastolic: 真实舒张压
        """
        print("\n" + "="*70)
        print("个体校准")
        print("="*70)
        
        if len(X_calib) < self.n_past + 10:
            print(f"⚠ 警告: 校准数据不足 ({len(X_calib)} < {self.n_past + 10})")
            print("  将使用全部可用数据进行校准")
        
        # 标准化
        X_calib_scaled = self.scaler_X.transform(X_calib)
        
        # 准备序列
        X_calib_seq, y_calib_sys_seq, y_calib_dia_seq = self.prepare_sequences(
            X_calib_scaled, y_calib_systolic, y_calib_diastolic
        )
        
        if len(X_calib_seq) == 0:
            print("✗ 校准失败: 序列数量为0")
            return
        
        # 预测（标准化空间）
        y_sys_scaled = self.scaler_y_systolic.transform(y_calib_sys_seq.reshape(-1, 1))
        y_dia_scaled = self.scaler_y_diastolic.transform(y_calib_dia_seq.reshape(-1, 1))
        
        pred_sys_scaled = self.model_systolic.predict(X_calib_seq, verbose=0)
        pred_dia_scaled = self.model_diastolic.predict(X_calib_seq, verbose=0)
        
        # 计算偏移（在标准化空间）
        self.calibration_offset_systolic = np.mean(y_sys_scaled - pred_sys_scaled)
        self.calibration_offset_diastolic = np.mean(y_dia_scaled - pred_dia_scaled)
        
        # 转换偏移量到原始空间（只乘scale，不加mean）
        offset_sys_real = self.calibration_offset_systolic * self.scaler_y_systolic.scale_[0]
        offset_dia_real = self.calibration_offset_diastolic * self.scaler_y_diastolic.scale_[0]
        
        self.is_calibrated = True
        
        print(f"\n✓ 校准完成:")
        print(f"  使用样本数: {len(X_calib_seq)}")
        print(f"  收缩压偏移: {offset_sys_real:+.2f} mmHg")
        print(f"  舒张压偏移: {offset_dia_real:+.2f} mmHg")
    
    def predict(self, X_test, apply_calibration=True):
        """
        预测血压
        
        参数:
            X_test: 测试特征 (n_samples, n_features)
            apply_calibration: 是否应用个体校准
        
        返回:
            (pred_systolic, pred_diastolic): 预测的血压值
        """
        # 标准化
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # 准备序列
        X_test_seq = []
        for i in range(self.n_past, len(X_test_scaled)):
            X_test_seq.append(X_test_scaled[i - self.n_past:i, :])
        X_test_seq = np.array(X_test_seq)
        
        if len(X_test_seq) == 0:
            print("⚠ 警告: 测试序列数量为0")
            return np.array([]), np.array([])
        
        # 预测（标准化空间）
        pred_sys_scaled = self.model_systolic.predict(X_test_seq, verbose=0)
        pred_dia_scaled = self.model_diastolic.predict(X_test_seq, verbose=0)
        
        # 应用校准
        if apply_calibration and self.is_calibrated:
            pred_sys_scaled += self.calibration_offset_systolic
            pred_dia_scaled += self.calibration_offset_diastolic
        
        # 反标准化
        pred_systolic = self.scaler_y_systolic.inverse_transform(pred_sys_scaled)
        pred_diastolic = self.scaler_y_diastolic.inverse_transform(pred_dia_scaled)
        
        return pred_systolic.flatten(), pred_diastolic.flatten()
    
    def plot_training_history(self, history_sys, history_dia, save_path=None):
        """
        可视化训练历史
        
        参数:
            history_sys: 收缩压训练历史
            history_dia: 舒张压训练历史
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 收缩压 - Loss
        axes[0, 0].plot(history_sys.history['loss'], label='Training Loss')
        axes[0, 0].plot(history_sys.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Systolic BP - Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 收缩压 - MAE
        axes[0, 1].plot(history_sys.history['mae'], label='Training MAE')
        axes[0, 1].plot(history_sys.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Systolic BP - MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE (mmHg)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 舒张压 - Loss
        axes[1, 0].plot(history_dia.history['loss'], label='Training Loss')
        axes[1, 0].plot(history_dia.history['val_loss'], label='Validation Loss')
        axes[1, 0].set_title('Diastolic BP - Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss (MSE)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 舒张压 - MAE
        axes[1, 1].plot(history_dia.history['mae'], label='Training MAE')
        axes[1, 1].plot(history_dia.history['val_mae'], label='Validation MAE')
        axes[1, 1].set_title('Diastolic BP - MAE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE (mmHg)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"  训练历史已保存: {save_path}")
        
        return fig
    
    def save_model(self, save_dir):
        """保存模型"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        self.model_systolic.save(os.path.join(save_dir, 'lstm_systolic.h5'))
        self.model_diastolic.save(os.path.join(save_dir, 'lstm_diastolic.h5'))
        
        # 保存scaler和校准参数
        np.savez(
            os.path.join(save_dir, 'calibration_params.npz'),
            scaler_X_mean=self.scaler_X.mean_,
            scaler_X_scale=self.scaler_X.scale_,
            scaler_y_systolic_mean=self.scaler_y_systolic.mean_,
            scaler_y_systolic_scale=self.scaler_y_systolic.scale_,
            scaler_y_diastolic_mean=self.scaler_y_diastolic.mean_,
            scaler_y_diastolic_scale=self.scaler_y_diastolic.scale_,
            calibration_offset_systolic=self.calibration_offset_systolic,
            calibration_offset_diastolic=self.calibration_offset_diastolic,
            is_calibrated=self.is_calibrated
        )
        
        print(f"\n✓ 模型已保存到: {save_dir}")


if __name__ == '__main__':
    # 测试代码
    print("LSTM模型测试")
    
    # 生成模拟数据
    n_samples = 1000
    n_features = 17
    
    X = np.random.randn(n_samples, n_features)
    y_sys = 120 + X[:, 0] * 10 + np.random.randn(n_samples) * 3
    y_dia = 70 + X[:, 1] * 8 + np.random.randn(n_samples) * 2
    
    # 创建预测器
    predictor = LSTMBloodPressurePredictor(n_past=20, n_features=n_features)
    
    # 训练
    history_sys, history_dia = predictor.train(
        X[:800], y_sys[:800], y_dia[:800],
        epochs=10, batch_size=32
    )
    
    # 校准
    predictor.calibrate(X[800:850], y_sys[800:850], y_dia[800:850])
    
    # 预测
    pred_sys, pred_dia = predictor.predict(X[850:], apply_calibration=True)
    
    print(f"\n测试完成！预测了 {len(pred_sys)} 个样本")