"""
LSTM模型 - 修复版本
主要修改:
1. 不对Y值(血压)进行标准化,保持原始量纲
2. 修复校准机制,直接在原始空间计算偏移
3. 简化预测流程
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
    """LSTM血压预测器 - 修复版本"""
    
    def __init__(self, n_past=20, n_features=17):
        """
        初始化LSTM预测器
        
        参数:
            n_past: 使用过去多少个心跳周期
            n_features: 每个心跳周期的特征数量
        """
        self.n_past = n_past
        self.n_features = n_features
        self.scaler_X = StandardScaler()  # 只标准化X(特征)
        # 不再标准化Y(血压值)!
        self.model_systolic = None
        self.model_diastolic = None
        
        # 个体校准参数 - 直接在原始血压空间
        self.calibration_offset_systolic = 0.0
        self.calibration_offset_diastolic = 0.0
        self.is_calibrated = False
    
    def prepare_sequences(self, X, y_systolic, y_diastolic):
        """
        将特征矩阵转换为时序序列
        
        参数:
            X: 特征矩阵 (n_cycles, n_features) - 已标准化
            y_systolic: 收缩压数组 (n_cycles,) - 原始值,未标准化
            y_diastolic: 舒张压数组 (n_cycles,) - 原始值,未标准化
        
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
            # 当前心跳的血压 - 保持原始值
            y_systolic_sequences.append(y_systolic[i])
            y_diastolic_sequences.append(y_diastolic[i])
        
        X_seq = np.array(X_sequences)
        y_sys_seq = np.array(y_systolic_sequences)
        y_dia_seq = np.array(y_diastolic_sequences)
        
        print(f"\n✓ 序列准备完成:")
        print(f"  序列形状: {X_seq.shape}")
        print(f"  可用样本数: {len(X_seq)} (原始: {len(X)} - 序列长度: {self.n_past})")
        print(f"  血压范围: 收缩压[{y_sys_seq.min():.1f}, {y_sys_seq.max():.1f}], "
              f"舒张压[{y_dia_seq.min():.1f}, {y_dia_seq.max():.1f}] mmHg")
        
        return X_seq, y_sys_seq, y_dia_seq
    
    def build_lstm_model(self, model_name="systolic"):
        """构建LSTM模型"""
        model = Sequential([
            # 第一层LSTM:128单元,返回序列
            layers.LSTM(128, activation='relu', 
                       input_shape=(self.n_past, self.n_features),
                       return_sequences=True),
            layers.Dropout(0.2),
            
            # 第二层LSTM:64单元,返回序列
            layers.LSTM(64, activation='relu', return_sequences=True),
            layers.Dropout(0.2),
            
            # 第三层LSTM:32单元,不返回序列
            layers.LSTM(32, activation='relu', return_sequences=False),
            layers.Dropout(0.2),
            
            # 全连接层
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            
            # 输出层 - 直接输出血压值(mmHg)
            layers.Dense(1)  # 不用激活函数,线性输出
        ])
        
        # 编译模型 - MAE作为损失函数(因为是mmHg单位)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mae',  # Mean Absolute Error
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, X_train, y_train_systolic, y_train_diastolic, 
              validation_split=0.15, epochs=100, batch_size=32, verbose=1):
        """
        训练LSTM模型
        
        参数:
            X_train: 训练特征 (n_samples, n_features)
            y_train_systolic: 训练收缩压 - 原始值
            y_train_diastolic: 训练舒张压 - 原始值
            validation_split: 验证集比例
            epochs: 训练轮数
            batch_size: 批大小
            verbose: 详细程度
        
        返回:
            (history_sys, history_dia): 训练历史
        """
        print("\n" + "="*70)
        print("LSTM模型训练开始")
        print("="*70)
        
        # 1. 只标准化X(特征)
        print("\n步骤1: 特征标准化")
        X_scaled = self.scaler_X.fit_transform(X_train)
        print(f"  特征已标准化: mean≈0, std≈1")
        
        # 2. 准备时序序列 - Y保持原始值
        print("\n步骤2: 准备时序序列")
        X_seq, y_sys_seq, y_dia_seq = self.prepare_sequences(
            X_scaled, y_train_systolic, y_train_diastolic
        )
        
        if len(X_seq) < 100:
            raise ValueError(f"序列样本太少: {len(X_seq)}")
        
        # 3. 构建模型
        print("\n步骤3: 构建LSTM模型")
        self.model_systolic = self.build_lstm_model("systolic")
        self.model_diastolic = self.build_lstm_model("diastolic")
        
        print("\n" + "="*60)
        print("收缩压 LSTM模型架构:")
        print("="*60)
        self.model_systolic.summary()
        
        print("\n" + "="*60)
        print("舒张压 LSTM模型架构:")
        print("="*60)
        self.model_diastolic.summary()
        
        # 4. 设置回调
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
            X_seq, y_sys_seq,  # Y是原始血压值
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
            X_seq, y_dia_seq,  # Y是原始血压值
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
        使用校准数据计算个体偏移 - 在原始血压空间
        
        参数:
            X_calib: 校准特征(测试集的前N分钟)
            y_calib_systolic: 真实收缩压 - 原始值
            y_calib_diastolic: 真实舒张压 - 原始值
        """
        print("\n" + "="*70)
        print("个体校准")
        print("="*70)
        
        if len(X_calib) < self.n_past + 10:
            print(f"⚠ 警告: 校准数据不足 ({len(X_calib)} < {self.n_past + 10})")
            print("  将使用全部可用数据进行校准")
        
        # 标准化特征
        X_calib_scaled = self.scaler_X.transform(X_calib)
        
        # 准备序列
        X_calib_seq, y_calib_sys_seq, y_calib_dia_seq = self.prepare_sequences(
            X_calib_scaled, y_calib_systolic, y_calib_diastolic
        )
        
        if len(X_calib_seq) == 0:
            print("✗ 校准失败: 序列数量为0")
            return
        
        # 预测 - 直接得到原始血压值
        pred_sys = self.model_systolic.predict(X_calib_seq, verbose=0).flatten()
        pred_dia = self.model_diastolic.predict(X_calib_seq, verbose=0).flatten()
        
        # 计算偏移 - 直接在原始血压空间
        self.calibration_offset_systolic = np.mean(y_calib_sys_seq - pred_sys)
        self.calibration_offset_diastolic = np.mean(y_calib_dia_seq - pred_dia)
        
        self.is_calibrated = True
        
        print(f"\n✓ 校准完成:")
        print(f"  使用样本数: {len(X_calib_seq)}")
        print(f"  收缩压偏移: {self.calibration_offset_systolic:+.2f} mmHg")
        print(f"  舒张压偏移: {self.calibration_offset_diastolic:+.2f} mmHg")
        print(f"  (正值表示模型预测偏低,需要向上调整)")
    
    def predict(self, X_test, apply_calibration=True):
        """
        预测血压
        
        参数:
            X_test: 测试特征 (n_samples, n_features)
            apply_calibration: 是否应用个体校准
        
        返回:
            (pred_systolic, pred_diastolic): 预测的血压值(mmHg)
        """
        # 标准化特征
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # 准备序列
        X_test_seq = []
        for i in range(self.n_past, len(X_test_scaled)):
            X_test_seq.append(X_test_scaled[i - self.n_past:i, :])
        X_test_seq = np.array(X_test_seq)
        
        if len(X_test_seq) == 0:
            print("⚠ 警告: 测试序列数量为0")
            return np.array([]), np.array([])
        
        # 预测 - 直接得到原始血压值(mmHg)
        pred_systolic = self.model_systolic.predict(X_test_seq, verbose=0).flatten()
        pred_diastolic = self.model_diastolic.predict(X_test_seq, verbose=0).flatten()
        
        # 应用校准 - 直接加偏移量(都在mmHg单位)
        if apply_calibration and self.is_calibrated:
            pred_systolic += self.calibration_offset_systolic
            pred_diastolic += self.calibration_offset_diastolic
        
        return pred_systolic, pred_diastolic
    
    def plot_training_history(self, history_sys, history_dia, save_path=None):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 收缩压 - Loss
        axes[0, 0].plot(history_sys.history['loss'], label='Training Loss')
        axes[0, 0].plot(history_sys.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Systolic BP - Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MAE)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 收缩压 - MAE
        axes[0, 1].plot(history_sys.history['mae'], label='Training MAE')
        axes[0, 1].plot(history_sys.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Systolic BP - MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE (mmHg)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 舒张压 - Loss
        axes[1, 0].plot(history_dia.history['loss'], label='Training Loss')
        axes[1, 0].plot(history_dia.history['val_loss'], label='Validation Loss')
        axes[1, 0].set_title('Diastolic BP - Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss (MAE)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 舒张压 - MAE
        axes[1, 1].plot(history_dia.history['mae'], label='Training MAE')
        axes[1, 1].plot(history_dia.history['val_mae'], label='Validation MAE')
        axes[1, 1].set_title('Diastolic BP - MAE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE (mmHg)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  训练历史已保存: {save_path}")
        
        plt.close()
        
        return fig


if __name__ == '__main__':
    # 测试代码
    print("=" * 70)
    print("LSTM模型修复版本测试")
    print("=" * 70)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 17
    
    # 模拟特征
    X = np.random.randn(n_samples, n_features)
    
    # 模拟血压 - 原始值(mmHg)
    y_sys = 120 + X[:, 0] * 10 + np.random.randn(n_samples) * 3
    y_dia = 70 + X[:, 1] * 8 + np.random.randn(n_samples) * 2
    
    print(f"\n模拟数据:")
    print(f"  样本数: {n_samples}")
    print(f"  特征数: {n_features}")
    print(f"  收缩压范围: [{y_sys.min():.1f}, {y_sys.max():.1f}] mmHg")
    print(f"  舒张压范围: [{y_dia.min():.1f}, {y_dia.max():.1f}] mmHg")
    
    # 创建预测器
    predictor = LSTMBloodPressurePredictor(n_past=20, n_features=n_features)
    
    # 训练
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)
    history_sys, history_dia = predictor.train(
        X[:800], y_sys[:800], y_dia[:800],
        epochs=10, batch_size=32, verbose=0
    )
    
    # 校准
    print("\n" + "=" * 70)
    print("使用50个样本进行校准...")
    print("=" * 70)
    predictor.calibrate(X[800:850], y_sys[800:850], y_dia[800:850])
    
    # 预测
    print("\n" + "=" * 70)
    print("预测剩余数据...")
    print("=" * 70)
    pred_sys, pred_dia = predictor.predict(X[850:], apply_calibration=True)
    
    # 评估
    y_true_sys = y_sys[850+predictor.n_past:]
    y_true_dia = y_dia[850+predictor.n_past:]
    
    mae_sys = np.mean(np.abs(y_true_sys - pred_sys))
    mae_dia = np.mean(np.abs(y_true_dia - pred_dia))
    
    print(f"\n✓ 测试完成!")
    print(f"  预测样本数: {len(pred_sys)}")
    print(f"  收缩压MAE: {mae_sys:.2f} mmHg")
    print(f"  舒张压MAE: {mae_dia:.2f} mmHg")