import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from UpdateCleanDataset import load_all_csvs_combined
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,Dense,Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def prepare_data_cylindrical(df, sequence_length=60):
    """
    FIXED VERSION: Properly handles NaN and data cleaning
    """
    print("="*60)
    print("DATA PREPARATION - FIXED VERSION")
    print("="*60)
    
    # Convert to DataFrame if needed
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    
    print(f"Original data shape: {df.shape}")
    
    # 1. Convert ALL columns to numeric first
    print("\n1. Converting columns to numeric...")
    numeric_data = []
    column_stats = []
    
    for i in range(min(4, df.shape[1])):
        col_data = df.iloc[:, i]
        numeric = pd.to_numeric(col_data, errors='coerce')
        numeric_data.append(numeric.values)
        
        stats = {
            'col': i,
            'total': len(numeric),
            'non_nan': numeric.notna().sum(),
            'nan': numeric.isna().sum(),
            'min': numeric.min() if numeric.notna().any() else np.nan,
            'max': numeric.max() if numeric.notna().any() else np.nan
        }
        column_stats.append(stats)
        
        print(f"   Column {i}: {stats['non_nan']}/{stats['total']} valid, "
              f"NaN={stats['nan']}, range={stats['min']:.2f} to {stats['max']:.2f}")
    
    # Stack as columns
    data_array = np.column_stack(numeric_data)
    print(f"\n2. Combined array shape: {data_array.shape}")
    print(f"   Total NaN in array: {np.isnan(data_array).sum()}")
    
    # 3. Remove rows with ANY NaN
    print("\n3. Removing rows with NaN values...")
    valid_mask = ~np.isnan(data_array).any(axis=1)
    
    print(f"   Valid rows: {valid_mask.sum()}")
    print(f"   Invalid rows (with NaN): {len(valid_mask) - valid_mask.sum()}")
    
    if valid_mask.sum() == 0:
        print("❌ ERROR: No valid rows after NaN removal!")
        return None, None, None
    
    # Apply mask
    clean_data = data_array[valid_mask]
    print(f"   Clean data shape: {clean_data.shape}")
    
    # 4. Extract clean columns
    year = clean_data[:, 0]
    month = clean_data[:, 1]
    day = clean_data[:, 2]
    temperature = clean_data[:, 3]
    
    print(f"\n4. Clean data ranges:")
    print(f"   Year: {year.min()} to {year.max()}")
    print(f"   Month: {month.min()} to {month.max()}")
    print(f"   Day: {day.min()} to {day.max()}")
    print(f"   Temperature: {temperature.min():.2f} to {temperature.max():.2f}")
    
    # 5. Create cylindrical encoding
    print("\n5. Creating cylindrical encoding...")
    
    # Normalize with safe division
    year_range = year.max() - year.min()
    if year_range == 0:
        year_norm = np.zeros_like(year)
    else:
        year_norm = (year - year.min()) / year_range
    
    month_norm = (month - 1) / 11
    day_norm = (day - 1) / 30
    
    # Clip to avoid any edge issues
    month_norm = np.clip(month_norm, 0, 1)
    day_norm = np.clip(day_norm, 0, 1)
    
    # Convert to angles
    #year_angle = 2 * np.pi * year_norm
    month_angle = 2 * np.pi * month_norm
    day_angle = 2 * np.pi * day_norm
    
    # Create features
    features = np.column_stack([
        temperature,                    # Temperature
        year_norm,
        #np.sin(year_angle), np.cos(year_angle),
        np.sin(month_angle), np.cos(month_angle),
        np.sin(day_angle), np.cos(day_angle)
    ])
    
    print(f"   Features shape: {features.shape}")
    print(f"   NaN in features: {np.isnan(features).sum()}")
    print(f"   Inf in features: {np.isinf(features).sum()}")
    
    # 6. Scale features
    print("\n6. Scaling features...")
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    print(f"   Scaled features shape: {scaled_features.shape}")
    print(f"   Scaled range: {scaled_features.min():.4f} to {scaled_features.max():.4f}")
    
    # 7. Check if enough data
    if len(scaled_features) <= sequence_length:
        print(f"❌ ERROR: Not enough data. Need > {sequence_length}, have {len(scaled_features)}")
        return None, None, None
    
    # 8. Create sequences
    print(f"\n7. Creating sequences (length={sequence_length})...")
    x, y = [], []
    
    for i in range(sequence_length, len(scaled_features)):
        x.append(scaled_features[i-sequence_length:i, :])
        y.append(scaled_features[i, 0])  # Temperature is first feature
    
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"\n✅ DATA PREPARATION COMPLETE!")
    print(f"   x shape: {x.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Features per timestep: {x.shape[2]}")
    print(f"   Total sequences: {x.shape[0]}")
    
    return x, y, scaler

# Use the fixed version
x, y, scaler = prepare_data_cylindrical(load_all_csvs_combined(), sequence_length=60)