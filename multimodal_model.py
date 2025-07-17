import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, 
    Flatten, Concatenate, BatchNormalization, Reshape
)

class BatteryMultiModalModel:
    def __init__(self, 
                 time_series_shape,
                 eis_shape,
                 capacity_shape,
                 num_fault_classes=4):  # Updated default
        """
        初始化多模态电池预测模型
        :param time_series_shape: 时间序列数据的形状 (seq_len, features)
        :param eis_shape: EIS数据的形状
        :param capacity_shape: 容量数据的形状
        :param num_fault_classes: 故障类别数量
        """
        self.time_series_shape = time_series_shape
        self.eis_shape = eis_shape
        self.capacity_shape = capacity_shape
        self.num_fault_classes = num_fault_classes
        
    def build_time_series_branch(self, inputs):
        """构建时间序列处理分支（LSTM）"""
        # Reshape input to (batch_size, timesteps, features)
        x = Reshape((1, self.time_series_shape[0]))(inputs)
        
        x = LSTM(128, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = LSTM(64, return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        return x
    
    def build_eis_branch(self, inputs):
        """构建EIS数据处理分支（CNN）"""
        x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        
        x = Conv1D(32, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        return x
    
    def build_capacity_branch(self, inputs):
        """构建容量数据处理分支（Dense）"""
        x = Dense(32, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(16, activation='relu')(x)
        x = BatchNormalization()(x)
        
        return x
    
    def build_model(self):
        """构建完整的多模态模型"""
        # 输入层
        time_series_input = Input(shape=self.time_series_shape, name='time_series_input')
        eis_input = Input(shape=self.eis_shape, name='eis_input')
        capacity_input = Input(shape=self.capacity_shape, name='capacity_input')
        
        # 处理各个模态的数据
        time_series_features = self.build_time_series_branch(time_series_input)
        eis_features = self.build_eis_branch(eis_input)
        capacity_features = self.build_capacity_branch(capacity_input)
        
        # 特征融合
        combined_features = Concatenate()([
            time_series_features,
            eis_features,
            capacity_features
        ])
        
        # 共享层
        shared = Dense(64, activation='relu')(combined_features)
        shared = BatchNormalization()(shared)
        shared = Dropout(0.3)(shared)
        
        # Multi-task output
        fault_prediction = Dense(self.num_fault_classes, activation='softmax', 
                               name='fault_prediction')(shared)
        soc_estimation = Dense(1, activation='linear', # Changed to linear for regression
                             name='soc_estimation')(shared)
        
        # Create model
        model = Model(
            inputs=[time_series_input, eis_input, capacity_input],
            outputs=[fault_prediction, soc_estimation]
        )
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss={
                'fault_prediction': 'categorical_crossentropy',
                'soc_estimation': 'mse'
            },
            loss_weights={
                'fault_prediction': 1.0,
                'soc_estimation': 0.5
            },
            metrics={
                'fault_prediction': ['accuracy'],
                'soc_estimation': ['mae']
            }
        )
        
        return model 