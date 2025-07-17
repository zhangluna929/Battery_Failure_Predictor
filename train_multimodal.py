import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from multimodal_model import BatteryMultiModalModel
from model_interpretability import BatteryModelInterpreter, BatteryModelEvaluator
from feature_engineering import add_physics_informed_features # Import the new function
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib # Import joblib

def load_and_preprocess_data(time_series_path, eis_path, capacity_path, num_classes=4):
    """
    加载和预处理多模态数据
    """
    # 加载数据
    time_series_data = pd.read_csv(time_series_path)
    eis_data = pd.read_csv(eis_path)
    capacity_data = pd.read_csv(capacity_path)
    
    # Add physics-informed features
    time_series_data = add_physics_informed_features(time_series_data)
    
    # Define original and new features
    time_series_feature_names = [
        'voltage', 'current', 'temperature', 'soc',
        'voltage_roc', 'temperature_roc', 'coulombic_efficiency_approx', 'power'
    ]

    # Preprocess time series data
    time_series_features = time_series_data[time_series_feature_names].values
    time_series_scaler = StandardScaler()
    time_series_scaled = time_series_scaler.fit_transform(time_series_features)
    
    # 预处理EIS数据
    eis_scaler = StandardScaler()
    eis_scaled = eis_scaler.fit_transform(eis_data)
    
    # 预处理容量数据
    capacity_scaler = StandardScaler()
    capacity_scaled = capacity_scaler.fit_transform(capacity_data)

    # Save the scalers for later use in production/monitoring
    joblib.dump(time_series_scaler, 'ts_scaler.joblib')
    joblib.dump(eis_scaler, 'eis_scaler.joblib')
    joblib.dump(capacity_scaler, 'capacity_scaler.joblib')
    
    # 获取标签
    fault_labels = time_series_data['failure'].values
    soc_values = time_series_data['soc'].values
    
    # 转换为分类标签的one-hot编码
    fault_labels_onehot = tf.keras.utils.to_categorical(fault_labels, num_classes=num_classes)
    
    return {
        'time_series': time_series_scaled,
        'eis': eis_scaled,
        'capacity': capacity_scaled,
        'fault_labels': fault_labels_onehot,
        'soc_values': soc_values,
        'feature_names': {
            'time_series': time_series_feature_names,
            'eis': [f'eis_feature_{i}' for i in range(eis_data.shape[1])],
            'capacity': [f'capacity_feature_{i}' for i in range(capacity_data.shape[1])]
        }
    }

def split_data(data_dict, test_size=0.2, val_size=0.2):
    """
    分割数据集为训练集、验证集和测试集
    """
    # 首先分割出测试集
    train_val_indices, test_indices = train_test_split(
        np.arange(len(data_dict['time_series'])),
        test_size=test_size,
        random_state=42
    )
    
    # 从剩余数据中分割出验证集
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size,
        random_state=42
    )
    
    # 创建数据集字典
    datasets = {
        'train': {
            'time_series': data_dict['time_series'][train_indices],
            'eis': data_dict['eis'][train_indices],
            'capacity': data_dict['capacity'][train_indices],
            'fault_labels': data_dict['fault_labels'][train_indices],
            'soc_values': data_dict['soc_values'][train_indices]
        },
        'val': {
            'time_series': data_dict['time_series'][val_indices],
            'eis': data_dict['eis'][val_indices],
            'capacity': data_dict['capacity'][val_indices],
            'fault_labels': data_dict['fault_labels'][val_indices],
            'soc_values': data_dict['soc_values'][val_indices]
        },
        'test': {
            'time_series': data_dict['time_series'][test_indices],
            'eis': data_dict['eis'][test_indices],
            'capacity': data_dict['capacity'][test_indices],
            'fault_labels': data_dict['fault_labels'][test_indices],
            'soc_values': data_dict['soc_values'][test_indices]
        }
    }
    
    return datasets

def train_model(datasets, model_params):
    """
    训练多模态模型
    """
    # 创建模型
    model = BatteryMultiModalModel(
        time_series_shape=model_params['time_series_shape'],
        eis_shape=model_params['eis_shape'],
        capacity_shape=model_params['capacity_shape'],
        num_fault_classes=model_params['num_fault_classes']
    ).build_model()
    
    # 设置回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # 训练模型
    history = model.fit(
        [
            datasets['train']['time_series'],
            datasets['train']['eis'],
            datasets['train']['capacity']
        ],
        [
            datasets['train']['fault_labels'],
            datasets['train']['soc_values']
        ],
        validation_data=(
            [
                datasets['val']['time_series'],
                datasets['val']['eis'],
                datasets['val']['capacity']
            ],
            [
                datasets['val']['fault_labels'],
                datasets['val']['soc_values']
            ]
        ),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )
    
    return model, history

def evaluate_and_interpret(model, datasets, data_dict):
    """
    评估模型性能并解释预测结果
    """
    # 创建评估器和解释器
    evaluator = BatteryModelEvaluator(model)
    interpreter = BatteryModelInterpreter(
        model,
        data_dict['feature_names']['time_series'] +
        data_dict['feature_names']['eis'] +
        data_dict['feature_names']['capacity']
    )
    
    # 评估分类性能
    print("\n=== 故障预测性能评估 ===")
    evaluator.evaluate_classification_metrics(
        [
            datasets['test']['time_series'],
            datasets['test']['eis'],
            datasets['test']['capacity']
        ],
        np.argmax(datasets['test']['fault_labels'], axis=1)
    )
    
    # 评估SOC估算性能
    print("\n=== SOC估算性能评估 ===")
    evaluator.evaluate_soc_estimation(
        [
            datasets['test']['time_series'],
            datasets['test']['eis'],
            datasets['test']['capacity']
        ],
        datasets['test']['soc_values']
    )
    
    # 绘制ROC曲线
    evaluator.plot_roc_curve(
        [
            datasets['test']['time_series'],
            datasets['test']['eis'],
            datasets['test']['capacity']
        ],
        np.argmax(datasets['test']['fault_labels'], axis=1)
    )
    
    # 使用SHAP解释模型预测
    print("\n=== 模型解释性分析 ===")
    interpreter.explain_with_shap(
        [
            datasets['test']['time_series'][:100],
            datasets['test']['eis'][:100],
            datasets['test']['capacity'][:100]
        ]
    )

def main():
    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 加载数据
    data_dict = load_and_preprocess_data(
        'time_series_data.csv',
        'eis_data.csv',
        'capacity_data.csv'
    )
    
    # 分割数据集
    datasets = split_data(data_dict)
    
    # Set model parameters, updating time_series_shape
    model_params = {
        'time_series_shape': (datasets['train']['time_series'].shape[1],), # Updated shape
        'eis_shape': (datasets['train']['eis'].shape[1],),
        'capacity_shape': (datasets['train']['capacity'].shape[1],),
        'num_fault_classes': 4
    }
    
    # 训练模型
    model, history = train_model(datasets, model_params)
    
    # 评估和解释模型
    evaluate_and_interpret(model, datasets, data_dict)
    
    # 保存模型
    model.save('battery_multimodal_model.h5')

if __name__ == "__main__":
    main() 