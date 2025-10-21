"""
模型训练模块 - 训练和评估ML模型
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn import metrics
from config import MODEL_CONFIG, DATA_CONFIG

class ModelTrainer:
    """模型训练器类"""
    
    def __init__(self):
        """初始化模型训练器"""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = float('inf')
    
    def create_model(self, model_name):
        """
        创建指定的模型
        
        参数:
            model_name: 模型名称
        
        返回:
            model: 创建的模型实例
        """
        if model_name == 'linear_regression':
            return LinearRegression()
        
        elif model_name == 'random_forest':
            params = MODEL_CONFIG.get('random_forest', {})
            return RandomForestRegressor(**params)
        
        elif model_name == 'gradient_boosting':
            params = MODEL_CONFIG.get('gradient_boosting', {})
            return GradientBoostingRegressor(**params)
        
        elif model_name == 'hist_gradient_boosting':
            params = MODEL_CONFIG.get('hist_gradient_boosting', {})
            return HistGradientBoostingRegressor(**params)
        
        elif model_name == 'stacking_ensemble':
            # 创建基学习器
            estimators = [
                ('rf', RandomForestRegressor(**MODEL_CONFIG.get('random_forest', {}))),
                ('gb', GradientBoostingRegressor(**MODEL_CONFIG.get('gradient_boosting', {}))),
                ('hgb', HistGradientBoostingRegressor(**MODEL_CONFIG.get('hist_gradient_boosting', {}))),
            ]
            
            # 元学习器
            final_estimator = GradientBoostingRegressor(n_estimators=30, subsample=0.5, random_state=42)
            
            return StackingRegressor(estimators=estimators, final_estimator=final_estimator, n_jobs=-1)
        
        else:
            raise ValueError(f"未知的模型类型: {model_name}")
    
    def train_single_model(self, model_name, X_train, y_train, X_test, y_test, target_name="Target"):
        """
        训练单个模型并评估
        
        参数:
            model_name: 模型名称
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            target_name: 目标名称（用于日志）
        
        返回:
            result: 包含模型和评估指标的字典
        """
        print(f"\n  训练 {model_name} 模型...")
        
        # 创建模型
        model = self.create_model(model_name)
        
        # 训练
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估
        result = {
            'model': model,
            'model_name': model_name,
            'predictions_train': y_pred_train,
            'predictions_test': y_pred_test,
            'metrics_train': {
                'mae': metrics.mean_absolute_error(y_train, y_pred_train),
                'mse': metrics.mean_squared_error(y_train, y_pred_train),
                'rmse': np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)),
                'r2': metrics.r2_score(y_train, y_pred_train),
            },
            'metrics_test': {
                'mae': metrics.mean_absolute_error(y_test, y_pred_test),
                'mse': metrics.mean_squared_error(y_test, y_pred_test),
                'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)),
                'r2': metrics.r2_score(y_test, y_pred_test),
            }
        }
        
        # 打印结果
        print(f"    训练集 - MAE: {result['metrics_train']['mae']:.2f}, MSE: {result['metrics_train']['mse']:.2f}, R²: {result['metrics_train']['r2']:.4f}")
        print(f"    测试集 - MAE: {result['metrics_test']['mae']:.2f}, MSE: {result['metrics_test']['mse']:.2f}, R²: {result['metrics_test']['r2']:.4f}")
        
        return result
    
    def train_all_models(self, X, y, target_name="Target"):
        """
        训练所有配置的模型
        
        参数:
            X: 特征矩阵
            y: 目标向量
            target_name: 目标名称（用于日志）
        
        返回:
            results: 所有模型的结果字典
        """
        print(f"\n{'='*60}")
        print(f"训练模型 - {target_name}")
        print(f"{'='*60}")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=1-DATA_CONFIG['train_test_split'],
            random_state=DATA_CONFIG['random_seed']
        )
        
        print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        
        # 训练每个模型
        results = {}
        model_list = MODEL_CONFIG['models']
        
        for model_name in model_list:
            try:
                result = self.train_single_model(
                    model_name, X_train, y_train, X_test, y_test, target_name
                )
                results[model_name] = result
                
                # 更新最佳模型（基于测试集MSE）
                test_mse = result['metrics_test']['mse']
                if test_mse < self.best_score:
                    self.best_score = test_mse
                    self.best_model = model_name
                    
            except Exception as e:
                print(f"    ✗ 训练失败: {e}")
                continue
        
        # 保存数据用于后续评估
        results['X_train'] = X_train
        results['X_test'] = X_test
        results['y_train'] = y_train
        results['y_test'] = y_test
        
        return results
    
    def train_for_targets(self, X, y_systolic, y_diastolic):
        """
        为两个目标（收缩压和舒张压）训练模型
        
        参数:
            X: 特征矩阵
            y_systolic: 收缩压目标
            y_diastolic: 舒张压目标
        
        返回:
            (results_systolic, results_diastolic): 两个目标的结果字典
        """
        # 训练收缩压模型
        results_systolic = self.train_all_models(X, y_systolic, "收缩压 (Systolic)")
        
        # 训练舒张压模型
        results_diastolic = self.train_all_models(X, y_diastolic, "舒张压 (Diastolic)")
        
        return results_systolic, results_diastolic
    
    def print_model_comparison(self, results, target_name="Target"):
        """
        打印所有模型的性能对比
        
        参数:
            results: 模型结果字典
            target_name: 目标名称
        """
        print(f"\n{'='*60}")
        print(f"模型性能对比 - {target_name}")
        print(f"{'='*60}")
        print(f"\n{'模型':<25} {'测试MAE':<12} {'测试MSE':<12} {'测试R²':<10}")
        print("-" * 60)
        
        model_list = [k for k in results.keys() if k not in ['X_train', 'X_test', 'y_train', 'y_test']]
        
        for model_name in model_list:
            result = results[model_name]
            mae = result['metrics_test']['mae']
            mse = result['metrics_test']['mse']
            r2 = result['metrics_test']['r2']
            
            marker = "⭐" if model_name == self.best_model else "  "
            print(f"{marker} {model_name:<23} {mae:<12.2f} {mse:<12.2f} {r2:<10.4f}")
        
        print(f"\n最佳模型: {self.best_model} (MSE: {self.best_score:.2f})")

    def train_models_with_split(self, X_train, y_train, X_test, y_test, target_name="Target"):
        """
        使用已分割好的训练/测试集训练模型
    
        参数:
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            target_name: 目标名称
    
        返回:
            results: 所有模型的结果字典
        """
        print(f"\n{'='*60}")
        print(f"训练模型 - {target_name}")
        print(f"{'='*60}")
        print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    
        # 训练每个模型
        results = {}
        model_list = MODEL_CONFIG['models']
    
        for model_name in model_list:
            try:
                result = self.train_single_model(
                    model_name, X_train, y_train, X_test, y_test, target_name
                )
                results[model_name] = result
            
                # 更新最佳模型（基于测试集MSE）
                test_mse = result['metrics_test']['mse']
                if test_mse < self.best_score:
                    self.best_score = test_mse
                    self.best_model = model_name
                
            except Exception as e:
                print(f"    ✗ 训练失败: {e}")
                continue
    
        # 保存数据用于后续评估
        results['X_train'] = X_train
        results['X_test'] = X_test
        results['y_train'] = y_train
        results['y_test'] = y_test
    
        return results


if __name__ == '__main__':
    # 测试代码
    print("=" * 60)
    print("模型训练模块测试")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(1000) * 0.5
    
    # 创建训练器
    trainer = ModelTrainer()
    
    # 训练模型
    results = trainer.train_all_models(X, y, "模拟目标")
    
    # 打印对比
    trainer.print_model_comparison(results, "模拟目标")