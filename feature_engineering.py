import pandas as pd
import numpy as np

def add_physics_informed_features(df):
    """
    Adds physics-informed features to the time series data.

    Args:
        df (pd.DataFrame): The input time series data with columns 
                           ['voltage', 'current', 'temperature', 'soc'].

    Returns:
        pd.DataFrame: The dataframe with added features.
    """
    df_sorted = df.sort_index()

    # Calculate time difference (assuming uniform time steps)
    dt = 1 # Assuming 1 time unit between samples

    # 1. Voltage Change Rate
    df_sorted['voltage_roc'] = df_sorted['voltage'].diff() / dt
    
    # 2. Temperature Change Rate
    df_sorted['temperature_roc'] = df_sorted['temperature'].diff() / dt

    # 3. Approximate Coulombic Efficiency
    # Change in SOC vs. Current. Add a small epsilon to avoid division by zero.
    df_sorted['coulombic_efficiency_approx'] = (df_sorted['soc'].diff() / dt) / (df_sorted['current'] + 1e-6)

    # 4. Power (Voltage * Current)
    df_sorted['power'] = df_sorted['voltage'] * df_sorted['current']

    # Fill NaN values that result from .diff() using backfill then forward fill
    df_sorted.fillna(method='bfill', inplace=True)
    df_sorted.fillna(method='ffill', inplace=True)

    return df_sorted 