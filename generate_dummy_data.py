import pandas as pd
import numpy as np
from tqdm import tqdm

def generate_time_series_data(num_samples=1000, seq_length=10):
    """
    生成模拟的时间序列数据，包含多种故障类型
    """
    data = []
    fault_types = {
        0: 'Normal',
        1: 'Capacity Fade',
        2: 'Internal Resistance Increase',
        3: 'Overheating'
    }
    
    for _ in tqdm(range(num_samples), desc="Generating Time Series Data"):
        fault_type = np.random.choice(list(fault_types.keys()), p=[0.7, 0.1, 0.1, 0.1])
        
        # 根据故障类型生成不同的数据模式
        if fault_type == 0: # Normal
            voltage = np.random.normal(3.7, 0.1, seq_length)
            current = np.random.normal(1.5, 0.3, seq_length)
            temperature = np.random.normal(25, 3, seq_length)
            soc = np.linspace(0.8, 0.7, seq_length)
        elif fault_type == 1: # Capacity Fade
            voltage = np.random.normal(3.5, 0.2, seq_length)
            current = np.random.normal(1.5, 0.3, seq_length)
            temperature = np.random.normal(28, 5, seq_length)
            soc = np.linspace(0.6, 0.4, seq_length)
        elif fault_type == 2: # Internal Resistance Increase
            voltage = np.random.normal(3.6, 0.15, seq_length)
            current = np.random.normal(1.2, 0.4, seq_length) # Lower current due to resistance
            temperature = np.random.normal(35, 6, seq_length) # Higher temp
            soc = np.linspace(0.7, 0.5, seq_length)
        elif fault_type == 3: # Overheating
            voltage = np.random.normal(3.7, 0.1, seq_length)
            current = np.random.normal(1.8, 0.4, seq_length) # Higher current
            temperature = np.random.normal(50, 8, seq_length) # High temp
            soc = np.linspace(0.8, 0.6, seq_length)
            
        for i in range(seq_length):
            data.append([voltage[i], current[i], temperature[i], soc[i], fault_type])
            
    return pd.DataFrame(data, columns=['voltage', 'current', 'temperature', 'soc', 'failure'])

def generate_eis_data(num_samples=1000, num_features=50):
    """
    生成模拟的EIS数据
    """
    data = np.random.rand(num_samples, num_features)
    return pd.DataFrame(data, columns=[f'eis_{i}' for i in range(num_features)])

def generate_capacity_data(num_samples=1000, num_features=5):
    """
    生成模拟的容量数据
    """
    data = np.random.rand(num_samples, num_features)
    return pd.DataFrame(data, columns=[f'capacity_{i}' for i in range(num_features)])

def main():
    time_series_df = generate_time_series_data()
    eis_df = generate_eis_data(num_samples=len(time_series_df))
    capacity_df = generate_capacity_data(num_samples=len(time_series_df))
    
    time_series_df.to_csv('time_series_data.csv', index=False)
    eis_df.to_csv('eis_data.csv', index=False)
    capacity_df.to_csv('capacity_data.csv', index=False)
    
    print("Dummy data generated successfully!")

if __name__ == "__main__":
    main() 