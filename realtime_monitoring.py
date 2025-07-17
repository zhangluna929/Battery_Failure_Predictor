import time
import numpy as np
import pandas as pd
import tensorflow as tf
from feature_engineering import add_physics_informed_features
from sklearn.preprocessing import StandardScaler
import joblib

# --- Configuration ---
MODEL_PATH = 'battery_multimodal_model.h5'
TS_SCALER_PATH = 'ts_scaler.joblib'
EIS_SCALER_PATH = 'eis_scaler.joblib'
CAPACITY_SCALER_PATH = 'capacity_scaler.joblib'
DATA_STREAM_INTERVAL = 2  # seconds
FAULT_THRESHOLD = 0.75  # Probability threshold to trigger an alert
FAULT_CLASS_NAMES = {
    0: 'Normal',
    1: 'Capacity Fade',
    2: 'Internal Resistance Increase',
    3: 'Overheating'
}

def data_stream_generator():
    """
    Simulates a real-time data stream from battery sensors.
    Yields a new data point (as a DataFrame) every few seconds.
    """
    while True:
        # Simulate a single row of sensor data
        data = {
            'voltage': [np.random.normal(3.7, 0.2)],
            'current': [np.random.normal(1.5, 0.5)],
            'temperature': [np.random.normal(30, 10)], # Wider range for potential faults
            'soc': [np.random.uniform(0.3, 0.9)]
        }
        # Simulate some dummy EIS and capacity data for the model input
        eis_data = np.random.rand(1, 50)
        capacity_data = np.random.rand(1, 5)

        yield pd.DataFrame(data), eis_data, capacity_data
        time.sleep(DATA_STREAM_INTERVAL)

def run_monitoring_system():
    """
    Main function to run the real-time monitoring and alerting system.
    """
    print("--- Starting Real-time Battery Monitoring System ---")
    
    # 1. Load the trained model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Multimodal model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load the scalers
    try:
        time_series_scaler = joblib.load(TS_SCALER_PATH)
        eis_scaler = joblib.load(EIS_SCALER_PATH)
        capacity_scaler = joblib.load(CAPACITY_SCALER_PATH)
        print("Scalers loaded successfully.")
    except Exception as e:
        print(f"Error loading scalers: {e}")
        print("Please ensure you have run the training script (`train_multimodal.py`) first to generate the scaler files.")
        return

    # 2. Initialize the data stream
    data_generator = data_stream_generator()
    
    # We need a small history to calculate rate-of-change features
    data_history = pd.DataFrame()
    
    print("System is now live. Monitoring for potential faults...")
    print("-" * 50)

    # 3. Prediction Loop
    for new_data_df, new_eis, new_capacity in data_generator:
        
        # Maintain a rolling history for feature calculation
        data_history = pd.concat([data_history, new_data_df], ignore_index=True)
        if len(data_history) > 10: # Keep history size manageable
            data_history = data_history.iloc[-10:]
        
        # Add physics-informed features
        processed_df = add_physics_informed_features(data_history.copy())
        
        # Get the latest data point with its new features
        latest_point = processed_df.iloc[[-1]]

        # Prepare data for the model
        time_series_feature_names = [
            'voltage', 'current', 'temperature', 'soc',
            'voltage_roc', 'temperature_roc', 'coulombic_efficiency_approx', 'power'
        ]
        time_series_features = latest_point[time_series_feature_names]
        
        # Scale the features
        ts_scaled = time_series_scaler.transform(time_series_features)
        eis_scaled = eis_scaler.transform(new_eis)
        capacity_scaled = capacity_scaler.transform(new_capacity)

        # Make a prediction
        fault_probs, soc_estimate = model.predict([ts_scaled, eis_scaled, capacity_scaled])
        
        fault_prediction = np.argmax(fault_probs[0])
        highest_prob = np.max(fault_probs[0])
        
        # 4. Alerting Logic
        print(f"Timestamp: {time.ctime()} | Predicted SOC: {soc_estimate[0][0]:.2f} | "
              f"Status: {FAULT_CLASS_NAMES[fault_prediction]} (Prob: {highest_prob:.2f})")

        if fault_prediction != 0 and highest_prob > FAULT_THRESHOLD:
            print("\n" + "="*20 + " ALERT! " + "="*20)
            print(f"High probability of fault detected: {FAULT_CLASS_NAMES[fault_prediction]}")
            print(f"Confidence: {highest_prob:.2f}")
            print("="*48 + "\n")


if __name__ == "__main__":
    run_monitoring_system() 