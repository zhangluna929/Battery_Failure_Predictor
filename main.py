import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_and_preprocess_data(file_path):
    """
    加载和预处理电池数据集。
    :param file_path: 数据文件路径
    :return: 返回训练集和测试集的输入特征和标签
    """
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"加载数据时出错: {e}")
        raise

    # 特征和标签分离
    features = data[['voltage', 'current', 'temperature', 'soc']].values
    labels = data['failure'].values

    # 数据标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    # 调整为RNN的输入格式
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    return X_train, X_test, y_train, y_test


def build_rnn_model(input_shape):
    """
    构建一个基于LSTM的RNN模型进行电池故障预测。
    :param input_shape: 输入数据的形状
    :return: 编译好的RNN模型
    """
    model = Sequential()

    # 第一层 LSTM
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # 第二层 LSTM
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))

    # 输出层（二分类问题）
    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    训练并评估RNN模型。
    :param X_train: 训练集特征
    :param X_test: 测试集特征
    :param y_train: 训练集标签
    :param y_test: 测试集标签
    :return: 训练好的模型
    """
    # 构建RNN模型
    model = build_rnn_model(X_train.shape[1:])

    # 模型训练
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # 模型评估
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"测试集准确率: {accuracy * 100:.2f}%")

    # 绘制训练过程中的损失和准确率
    plt.figure(figsize=(12, 6))

    # 损失图
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss over epochs')
    plt.legend()

    # 准确率图
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy over epochs')
    plt.legend()

    plt.show()

    return model


def predict_battery_failure(model, data):
    """
    使用训练好的模型预测电池是否发生故障。
    :param model: 训练好的模型
    :param data: 新的数据样本（特征）
    :return: 预测结果
    """
    prediction = model.predict(data)
    return prediction


# 主函数
if __name__ == "__main__":
    # 加载并预处理数据
    file_path = 'battery_data.csv'  # 假设数据文件路径
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    # 训练和评估模型
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    # 使用模型进行预测（例如：预测某些电池样本的故障）
    sample_data = np.array([[3.7, 1.0, 30, 0.8]])  # 假设的样本数据
    sample_data_scaled = StandardScaler().fit(X_train.reshape(-1, 4)).transform(sample_data)
    sample_data_scaled = sample_data_scaled.reshape(sample_data_scaled.shape[0], 1, sample_data_scaled.shape[1])
    
    prediction = predict_battery_failure(model, sample_data_scaled)

    print(f"预测的电池故障概率: {prediction[0][0]:.2f}")
    if prediction[0][0] > 0.5:
        print("预测：电池故障")
    else:
        print("预测：电池正常")
