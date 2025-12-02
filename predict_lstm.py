import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_model_and_scaler():
    """Load trained model and scaler if they exist"""
    if not os.path.exists("trained_lstm_model.h5"):
        print("❌ Error: trained_lstm_model.h5 not found!")
        print("   Please run train_lstm.py first to train the model.")
        return None, None
    
    if not os.path.exists("temperature_scaler.pkl"):
        print("❌ Error: temperature_scaler.pkl not found!")
        return None, None
    
    try:
        print("Loading trained LSTM model...")
        model = tf.keras.models.load_model("trained_lstm_model.h5", custom_objects={ "mse": MeanSquaredError()})
        print("✅ Model loaded successfully")
        
        print("Loading scaler...")
        scaler = joblib.load("temperature_scaler.pkl")
        print("✅ Scaler loaded successfully")
        
        return model, scaler
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return None, None

def plot_predictions_vs_actual(model, X_val, y_val, scaler, filename="lstm_predictions_vs_actual.png"):
    """
    Create prediction chart: Actual vs Predicted values for LSTM
    """
    print("\n[1] Making predictions...")
    
    y_pred_scaled = model.predict(X_val, verbose=0)
    n_features = X_val.shape[2]
    
    # Inverse transform
    y_pred_dummy = np.zeros((len(y_pred_scaled), n_features))
    y_val_dummy = np.zeros((len(y_val), n_features))
    
    y_pred_dummy[:, 0] = y_pred_scaled.flatten()
    y_val_dummy[:, 0] = y_val
    y_pred_dummy[:, 1:] = 0.5
    y_val_dummy[:, 1:] = 0.5
    
    y_pred_original = scaler.inverse_transform(y_pred_dummy)[:, 0]
    y_val_original = scaler.inverse_transform(y_val_dummy)[:, 0]
    
    # Calculate metrics
    mae = mean_absolute_error(y_val_original, y_pred_original)
    mse = mean_squared_error(y_val_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val_original, y_pred_original)
    
    error_abs = np.abs(y_val_original - y_pred_original)
    accuracy_1c = np.mean(error_abs <= 1.0) * 100
    accuracy_2c = np.mean(error_abs <= 2.0) * 100
    
    print(f"📊 LSTM Prediction Results:")
    print(f"  MAE:  {mae:.2f}°C")
    print(f"  RMSE: {rmse:.2f}°C")
    print(f"  R²:   {r2:.4f}")
    print(f"  Accuracy (±1°C): {accuracy_1c:.1f}%")
    print(f"  Accuracy (±2°C): {accuracy_2c:.1f}%")
    
    # Create chart
    print("\n[2] Generating prediction chart...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Time series
    n_plot = min(1000, len(y_val_original))
    time_points = range(n_plot)
    
    axes[0, 0].plot(time_points, y_val_original[:n_plot], 'b-', 
                    label='Actual', linewidth=2, alpha=0.8)
    axes[0, 0].plot(time_points, y_pred_original[:n_plot], 'g--', 
                    label='LSTM Predicted', linewidth=2, alpha=0.8)
    
    axes[0, 0].fill_between(time_points,
                           y_val_original[:n_plot] - mae,
                           y_val_original[:n_plot] + mae,
                           alpha=0.2, color='gray', label=f'±{mae:.1f}°C')
    
    axes[0, 0].set_title(f'LSTM: Predictions vs Actual (First {n_plot} Samples)', fontsize=14)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    axes[0, 1].scatter(y_val_original, y_pred_original, alpha=0.5, s=20, color='green')
    
    min_val = min(y_val_original.min(), y_pred_original.min())
    max_val = max(y_val_original.max(), y_pred_original.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 
                    'r--', label='Perfect Prediction', linewidth=2)
    
    axes[0, 1].set_title('LSTM: Scatter Plot', fontsize=14)
    axes[0, 1].set_xlabel('Actual Temperature (°C)')
    axes[0, 1].set_ylabel('Predicted Temperature (°C)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    errors = y_val_original - y_pred_original
    axes[1, 0].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1, 0].axvline(x=errors.mean(), color='darkgreen', linestyle='--', 
                       linewidth=2, label=f'Mean: {errors.mean():.2f}°C')
    
    axes[1, 0].set_title('LSTM: Error Distribution', fontsize=14)
    axes[1, 0].set_xlabel('Error (Actual - Predicted) °C')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Error over time
    axes[1, 1].plot(range(len(errors)), errors, 'gray', alpha=0.7, linewidth=1)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].axhline(y=errors.mean(), color='darkgreen', linestyle='--', 
                       linewidth=2, label=f'Mean Error: {errors.mean():.2f}°C')
    
    window_size = min(50, len(errors) // 10)
    if window_size > 1:
        rolling_mean = pd.Series(errors).rolling(window=window_size).mean()
        axes[1, 1].plot(range(len(rolling_mean)), rolling_mean, 
                       'blue', linewidth=2, label=f'{window_size}-point Moving Avg')
    
    axes[1, 1].set_title('LSTM: Error Over Time', fontsize=14)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Error (°C)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Statistics text box
    stats_text = f"""LSTM Statistics:
MAE: {mae:.2f}°C
RMSE: {rmse:.2f}°C
R²: {r2:.4f}
Accuracy (±1°C): {accuracy_1c:.1f}%
Accuracy (±2°C): {accuracy_2c:.1f}%
N = {len(y_val_original)} samples"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('LSTM Model: Predictions vs Actual - Hong Kong Grass Temperature', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ LSTM prediction chart saved as: {filename}")
    plt.show()
    plt.close(fig)
    
    return y_val_original, y_pred_original, mae, rmse, r2

def forecast_future(model, scaler, last_sequence, days=30):
    """
    Forecast future temperatures
    """
    print(f"\n[3] Generating {days}-day forecast...")
    
    n_features = last_sequence.shape[1]
    sequence_length = last_sequence.shape[0]
    
    forecasts_scaled = []
    current_sequence = last_sequence.copy().reshape(1, sequence_length, n_features)
    
    for day in range(days):
        pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
        forecasts_scaled.append(pred_scaled)
        
        new_seq = current_sequence[0, 1:, :].copy()
        new_row = current_sequence[0, -1, :].copy()
        new_row[0] = pred_scaled
        new_seq = np.vstack([new_seq, new_row])
        current_sequence = new_seq.reshape(1, sequence_length, n_features)
    
    # Convert to original scale
    forecasts_dummy = np.zeros((len(forecasts_scaled), n_features))
    forecasts_dummy[:, 0] = forecasts_scaled
    forecasts_dummy[:, 1:] = 0.5
    
    forecasts_original = scaler.inverse_transform(forecasts_dummy)[:, 0]
    
    return forecasts_original

def plot_forecast(forecasts, filename="lstm_forecast.png"):
    """Plot forecasted temperatures"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    days = range(1, len(forecasts) + 1)
    ax.plot(days, forecasts, 'go-', linewidth=2, markersize=8, 
            label='LSTM Forecasted Temperature')
    
    std_error = np.std(forecasts) * 0.5
    ax.fill_between(days, 
                   forecasts - std_error,
                   forecasts + std_error,
                   alpha=0.2, color='green', label='Uncertainty Band')
    
    ax.set_title(f'LSTM: {len(forecasts)}-Day Grass Temperature Forecast', fontsize=16)
    ax.set_xlabel('Day')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xticks(range(1, len(forecasts) + 1, max(1, len(forecasts)//10)))
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    for i, temp in enumerate(forecasts):
        if i % max(1, len(forecasts)//10) == 0:
            ax.text(i+1, temp + 0.2, f'{temp:.1f}°C', 
                   ha='center', fontsize=9, fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    print(f"\n🌡️ LSTM {len(forecasts)}-DAY FORECAST:")
    print("-"*40)
    for i, temp in enumerate(forecasts, 1):
        print(f"Day {i:2d}: {temp:5.1f}°C")
    
    print(f"\n📊 LSTM Forecast Summary:")
    print(f"  Average: {forecasts.mean():.1f}°C")
    print(f"  Maximum: {forecasts.max():.1f}°C")
    print(f"  Minimum: {forecasts.min():.1f}°C")
    print(f"  Range:   {forecasts.max() - forecasts.min():.1f}°C")
    
    return forecasts

def save_forecast_csv(forecasts, filename="lstm_forecast.csv"):
    """Save forecast to CSV"""
    forecast_df = pd.DataFrame({
        'day': range(1, len(forecasts) + 1),
        'lstm_forecasted_temperature': forecasts
    })
    forecast_df.to_csv(filename, index=False)
    print(f"✅ LSTM forecast saved: {filename}")

# Main prediction execution
if 1:
    print("="*60)
    print("LSTM PREDICTION PIPELINE")
    print("="*60)
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        exit()
    
    print("\nModel Summary:")
    model.summary()
    
    # Load or prepare validation data
    # Note: You need to load your validation data here
    # This is just a placeholder - you'll need to adapt this part
    print("\n⚠️  Note: You need to load validation data for predictions")
    print("   This requires your prepare_data_cylindrical function")
    print("   or loading pre-saved validation data")
    
    # Example placeholder - you need to implement this
    try:
        from prepare_data import prepare_data_cylindrical
        from UpdateCleanDataset import load_all_csvs_combined
        
        print("\nLoading data for predictions...")
        data = load_all_csvs_combined()
        if data.empty:
            print("❌ No data loaded!")
        else:
            print(f"✅ Data loaded: {data.shape}")
            
            # Prepare data (get validation portion)
            x, y, _ = prepare_data_cylindrical(data, sequence_length=60)
            
            if x is not None and len(x) > 0:
                # Use last 20% for validation
                split_idx = int(0.8 * len(x))
                X_val, y_val = x[split_idx:], y[split_idx:]
                
                print(f"\nValidation data shape: {X_val.shape}")
                
                # Make predictions
                y_true, y_pred, mae, rmse, r2 = plot_predictions_vs_actual(
                    model, X_val, y_val, scaler
                )
                
                # Save predictions
                predictions_df = pd.DataFrame({
                    'actual_temperature': y_true,
                    'lstm_predicted_temperature': y_pred,
                    'lstm_error': y_true - y_pred
                })
                predictions_df.to_csv('lstm_predictions.csv', index=False)
                print("✅ Predictions saved to lstm_predictions.csv")
                
                # Generate forecast if we have recent data
                if len(X_val) > 0:
                    last_sequence = X_val[-1]
                    forecast = forecast_future(model, scaler, last_sequence, days=30)
                    plot_forecast(forecast)
                    save_forecast_csv(forecast)
                
    except ImportError:
        print("\n⚠️  Cannot import data preparation functions")
        print("   Make sure UpdateCleanDataset.py and prepare_data.py are in the same directory")
    
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        print("   You may need to manually load validation data")
    
    print("\n" + "="*60)
    print("✅ PREDICTION COMPLETE!")
    print("="*60)