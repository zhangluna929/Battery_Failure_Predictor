import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class BatteryDataAugmentation:
    def __init__(self, data):
        """
        初始化数据增强类
        :param data: 原始数据DataFrame，包含 voltage, current, temperature, soc, failure 列
        """
        self.data = data
        self.features = ['voltage', 'current', 'temperature', 'soc']
        
    def add_gaussian_noise(self, noise_factor=0.02):
        """
        添加高斯噪声
        :param noise_factor: 噪声强度因子
        :return: 增强后的数据
        """
        noisy_data = self.data.copy()
        for feature in self.features:
            noise = np.random.normal(0, noise_factor * self.data[feature].std(), size=len(self.data))
            noisy_data[feature] = self.data[feature] + noise
        return noisy_data

    def scale_features(self, scale_range=(0.9, 1.1)):
        """
        特征缩放
        :param scale_range: 缩放范围
        :return: 增强后的数据
        """
        scaled_data = self.data.copy()
        for feature in self.features:
            scale_factor = np.random.uniform(scale_range[0], scale_range[1], size=len(self.data))
            scaled_data[feature] = self.data[feature] * scale_factor
        return scaled_data

    def time_shift(self, shift_range=(-5, 5)):
        """
        时间序列平移
        :param shift_range: 平移范围
        :return: 增强后的数据
        """
        shifted_data = self.data.copy()
        shift_steps = np.random.randint(shift_range[0], shift_range[1])
        for feature in self.features:
            shifted_data[feature] = np.roll(self.data[feature], shift_steps)
        return shifted_data

    def generate_synthetic_samples(self, num_synthetic_samples=1000):
        """
        生成合成样本
        :param num_synthetic_samples: 要生成的合成样本数量
        :return: 增强后的数据
        """
        synthetic_samples = []
        
        # 使用不同的增强方法生成样本
        for _ in range(num_synthetic_samples // 3):
            # 高斯噪声
            noisy_data = self.add_gaussian_noise()
            synthetic_samples.append(noisy_data)
            
            # 特征缩放
            scaled_data = self.scale_features()
            synthetic_samples.append(scaled_data)
            
            # 时间平移
            shifted_data = self.time_shift()
            synthetic_samples.append(shifted_data)
        
        # 合并所有增强的数据
        augmented_data = pd.concat([self.data] + synthetic_samples, axis=0)
        augmented_data = augmented_data.reset_index(drop=True)
        
        return augmented_data

    def validate_data(self, data):
        """
        验证增强后的数据是否合理
        :param data: 待验证的数据
        :return: 验证后的数据
        """
        # 确保电压在合理范围内 (例如 2.5V - 4.2V)
        data.loc[data['voltage'] > 4.2, 'voltage'] = 4.2
        data.loc[data['voltage'] < 2.5, 'voltage'] = 2.5
        
        # 确保温度在合理范围内 (例如 -20°C - 60°C)
        data.loc[data['temperature'] > 60, 'temperature'] = 60
        data.loc[data['temperature'] < -20, 'temperature'] = -20
        
        # 确保SOC在0-1之间
        data.loc[data['soc'] > 1, 'soc'] = 1
        data.loc[data['soc'] < 0, 'soc'] = 0
        
        return data 