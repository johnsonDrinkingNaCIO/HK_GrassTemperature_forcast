import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from UpdateCleanDataset import load_all_csvs_combined
from prepare_data import prepare_data_cylindrical
from rnn import create_rnn_model_for_features

def train_rnn_model(X_train, y_train, X_val, y_val):
    """
    Train the RNN model - FIXED VERSION
    """
    # CRITICAL FIX 1: Get sequence_length and n_features from data
    sequence_length = X_train.shape[1]
    n_features = X_train.shape[2]
    
    print(f"Creating RNN model for input shape: ({sequence_length}, {n_features})")
    
    # CRITICAL FIX 2: Pass parameters to model creation
    model = create_rnn_model_for_features(
        sequence_length=sequence_length,
        n_features=n_features
    )

    # CRITICAL FIX 3: Better callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,  # Increased from 15
            min_delta=0.001,  # Minimum improvement required
            restore_best_weights=True,
            verbose=1  # Show when early stopping triggers
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Less aggressive (was 0.2)
            patience=10,
            min_lr=0.00001,  # Lower minimum
            verbose=1
        )
    ]

    print(f"\nStarting training for up to 200 epochs...")
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=200,  # Increased from 100
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    return model, history

def save_training_plots(history, filename="rnn_training_history.png"):
    """Save training plots properly"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title('Model Loss (MSE)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['mae'], label='Train MAE')
    ax2.plot(history.history['val_mae'], label='Val MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE (°C)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # CRITICAL FIX 4: Save BEFORE show()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved as: {filename}")
    
    # Then show
    plt.show()
    
    # Close figure
    plt.close(fig)

def plot_predictions_vs_actual(model, X_val, y_val, scaler, filename="predictions_vs_actual.png"):
    """
    Create prediction chart: Actual vs Predicted values
    """
    print("\n[6] Making predictions on validation set...")
    
    # Make predictions
    y_pred_scaled = model.predict(X_val, verbose=0)
    
    # Get number of features
    n_features = X_val.shape[2]
    
    # Inverse transform predictions and actual values
    # Create dummy arrays with same number of features as scaler expects
    y_pred_dummy = np.zeros((len(y_pred_scaled), n_features))
    y_val_dummy = np.zeros((len(y_val), n_features))
    
    # Fill first column with predictions/actual (temperature is first feature)
    y_pred_dummy[:, 0] = y_pred_scaled.flatten()
    y_val_dummy[:, 0] = y_val
    
    # Fill remaining columns with zeros (or any value)
    y_pred_dummy[:, 1:] = 0.5  # Mid-range for scaled features
    y_val_dummy[:, 1:] = 0.5
    
    # Inverse transform
    y_pred_original = scaler.inverse_transform(y_pred_dummy)[:, 0]
    y_val_original = scaler.inverse_transform(y_val_dummy)[:, 0]
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_val_original, y_pred_original)
    mse = mean_squared_error(y_val_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val_original, y_pred_original)
    
    # Calculate accuracy within error bands
    error_abs = np.abs(y_val_original - y_pred_original)
    accuracy_1c = np.mean(error_abs <= 1.0) * 100
    accuracy_2c = np.mean(error_abs <= 2.0) * 100
    
    print(f"📊 Validation Results:")
    print(f"  MAE:  {mae:.2f}°C")
    print(f"  RMSE: {rmse:.2f}°C")
    print(f"  R²:   {r2:.4f}")
    print(f"  Accuracy (±1°C): {accuracy_1c:.1f}%")
    print(f"  Accuracy (±2°C): {accuracy_2c:.1f}%")
    
    # Create prediction chart
    print("\n[7] Generating prediction chart...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Predictions vs Actual (time series)
    n_plot = min(1000, len(y_val_original))
    time_points = range(n_plot)
    
    axes[0, 0].plot(time_points, y_val_original[:n_plot], 'b-', 
                    label='Actual', linewidth=2, alpha=0.8)
    axes[0, 0].plot(time_points, y_pred_original[:n_plot], 'r--', 
                    label='Predicted', linewidth=2, alpha=0.8)
    
    # Add error band
    axes[0, 0].fill_between(time_points,
                           y_val_original[:n_plot] - mae,
                           y_val_original[:n_plot] + mae,
                           alpha=0.2, color='gray', label=f'±{mae:.1f}°C')
    
    axes[0, 0].set_title(f'Predictions vs Actual (First {n_plot} Samples)', fontsize=14)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot with perfect prediction line
    axes[0, 1].scatter(y_val_original, y_pred_original, alpha=0.5, s=20)
    
    # Add perfect prediction line (y=x)
    min_val = min(y_val_original.min(), y_pred_original.min())
    max_val = max(y_val_original.max(), y_pred_original.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 
                    'r--', label='Perfect Prediction', linewidth=2)
    
    axes[0, 1].set_title('Scatter Plot: Actual vs Predicted', fontsize=14)
    axes[0, 1].set_xlabel('Actual Temperature (°C)')
    axes[0, 1].set_ylabel('Predicted Temperature (°C)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    errors = y_val_original - y_pred_original
    axes[1, 0].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1, 0].axvline(x=errors.mean(), color='green', linestyle='--', 
                       linewidth=2, label=f'Mean: {errors.mean():.2f}°C')
    
    axes[1, 0].set_title('Prediction Error Distribution', fontsize=14)
    axes[1, 0].set_xlabel('Error (Actual - Predicted) °C')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Error over time
    axes[1, 1].plot(range(len(errors)), errors, 'gray', alpha=0.7, linewidth=1)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].axhline(y=errors.mean(), color='green', linestyle='--', 
                       linewidth=2, label=f'Mean Error: {errors.mean():.2f}°C')
    
    # Add rolling mean of error
    window_size = min(50, len(errors) // 10)
    if window_size > 1:
        rolling_mean = pd.Series(errors).rolling(window=window_size).mean()
        axes[1, 1].plot(range(len(rolling_mean)), rolling_mean, 
                       'blue', linewidth=2, label=f'{window_size}-point Moving Avg')
    
    axes[1, 1].set_title('Prediction Error Over Time', fontsize=14)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Error (°C)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f"""Statistics:
MAE: {mae:.2f}°C
RMSE: {rmse:.2f}°C
R²: {r2:.4f}
Accuracy (±1°C): {accuracy_1c:.1f}%
Accuracy (±2°C): {accuracy_2c:.1f}%
N = {len(y_val_original)} samples"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('RNN Model: Predictions vs Actual - Hong Kong Grass Temperature', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Prediction chart saved as: {filename}")
    
    plt.show()
    plt.close(fig)
    
    return y_val_original, y_pred_original, mae, rmse, r2

def forecast_next_month(model, scaler, last_sequence, sequence_length=60):
    """
    Forecast temperatures for the next 30 days (1 month)
    """
    print("\n[8] Generating 30-day forecast...")
    
    # Get number of features
    n_features = last_sequence.shape[1]
    
    # Forecast day by day (autoregressive)
    forecasts_scaled = []
    current_sequence = last_sequence.copy().reshape(1, sequence_length, n_features)
    
    for day in range(30):
        # Predict next day
        pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
        forecasts_scaled.append(pred_scaled)
        
        # Create new sequence
        new_seq = current_sequence[0, 1:, :].copy()  # Remove oldest
        
        # Create new row with predicted temperature
        new_row = current_sequence[0, -1, :].copy()  # Copy last row
        new_row[0] = pred_scaled  # Update temperature
        
        # Update time features (simplified - in reality should increment dates)
        # For demo, we'll just keep them as is
        
        # Add new row
        new_seq = np.vstack([new_seq, new_row])
        current_sequence = new_seq.reshape(1, sequence_length, n_features)
        
        if (day + 1) % 10 == 0:
            print(f"  Forecasted day {day + 1}")
    
    # Convert to original scale
    forecasts_dummy = np.zeros((len(forecasts_scaled), n_features))
    forecasts_dummy[:, 0] = forecasts_scaled
    forecasts_dummy[:, 1:] = 0.5  # Placeholder for time features
    
    forecasts_original = scaler.inverse_transform(forecasts_dummy)[:, 0]
    
    # Plot forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    
    days = range(1, 31)
    ax.plot(days, forecasts_original, 'ro-', linewidth=2, markersize=8, 
            label='Forecasted Temperature')
    
    # Add confidence interval (simplified)
    std_error = np.std(forecasts_original) * 0.5  # Approximate
    ax.fill_between(days, 
                   forecasts_original - std_error,
                   forecasts_original + std_error,
                   alpha=0.2, color='red', label='Uncertainty Band')
    
    ax.set_title('30-Day Grass Temperature Forecast', fontsize=16)
    ax.set_xlabel('Day')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xticks(range(1, 31, 5))
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add value labels for key days
    for i, temp in enumerate(forecasts_original):
        if i % 5 == 0:  # Label every 5th day
            ax.text(i+1, temp + 0.2, f'{temp:.1f}°C', 
                   ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('30_day_forecast.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    print(f"\n🌡️ 30-DAY FORECAST:")
    print("-"*40)
    for i, temp in enumerate(forecasts_original, 1):
        print(f"Day {i:2d}: {temp:5.1f}°C")
    
    print(f"\n📊 Forecast Summary:")
    print(f"  Average: {forecasts_original.mean():.1f}°C")
    print(f"  Maximum: {forecasts_original.max():.1f}°C")
    print(f"  Minimum: {forecasts_original.min():.1f}°C")
    print(f"  Range:   {forecasts_original.max() - forecasts_original.min():.1f}°C")
    
    return forecasts_original

# Main execution
print("="*60)
print("RNN Training Pipeline - Hong Kong Grass Temperature")
print("="*60)

# Load and prepare data
print("\n[1] Loading data...")
data = load_all_csvs_combined()
if data.empty:
    print("❌ No data loaded! Check your CSV files.")
else:
    print(f"✅ Data loaded: {data.shape}")

print("\n[2] Preparing data with cylindrical encoding...")
x, y, scaler = prepare_data_cylindrical(data, sequence_length=60)

if x is None or len(x) == 0:
    print("❌ Data preparation failed!")
else:
    print(f"✅ Data prepared: X shape={x.shape}, y shape={y.shape}")
    print(f"   Features per time step: {x.shape[2]}")

    # Split data
    print("\n[3] Splitting data (80% train, 20% validation)...")
    split_idx = int(0.8 * len(x))
    X_train, X_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"✅ Training samples: {len(X_train)}")
    print(f"✅ Validation samples: {len(X_val)}")

    # Train the model
    print("\n[4] Training RNN model...")
    print("-"*40)
    
    rnn_model, rnn_history = train_rnn_model(X_train, y_train, X_val, y_val)
    
    # Results
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    
    print(f"\nTraining Summary:")
    print(f"  Total epochs: {len(rnn_history.history['loss'])}")
    print(f"  Final training loss: {rnn_history.history['loss'][-1]:.6f}")
    print(f"  Final validation loss: {rnn_history.history['val_loss'][-1]:.6f}")
    print(f"  Final training MAE: {rnn_history.history['mae'][-1]:.4f}°C")
    print(f"  Final validation MAE: {rnn_history.history['val_mae'][-1]:.4f}°C")
    
    # Save plots
    print("\n[5] Generating and saving training plots...")
    save_training_plots(rnn_history, "rnn_training_history.png")
    
    # Generate prediction chart
    y_val_original, y_pred_original, mae, rmse, r2 = plot_predictions_vs_actual(
        rnn_model, X_val, y_val, scaler, "predictions_vs_actual.png"
    )
    
    # Generate 30-day forecast
    if X_val.shape[0] > 0:
        last_sequence = X_val[-1]  # Use last validation sequence
        forecast = forecast_next_month(rnn_model, scaler, last_sequence, sequence_length=60)
    
    # Save model
    print("\n[9] Saving model...")
    rnn_model.save("trained_rnn_model.h5")
    print("✅ Model saved: trained_rnn_model.h5")
    
    # Save scaler
    import joblib
    joblib.dump(scaler, "temperature_scaler.pkl")
    print("✅ Scaler saved: temperature_scaler.pkl")
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'actual_temperature': y_val_original,
        'predicted_temperature': y_pred_original,
        'error': y_val_original - y_pred_original
    })
    predictions_df.to_csv('validation_predictions.csv', index=False)
    print("✅ Predictions saved: validation_predictions.csv")
    
    if 'forecast' in locals():
        forecast_df = pd.DataFrame({
            'day': range(1, 31),
            'forecasted_temperature': forecast
        })
        forecast_df.to_csv('30_day_forecast.csv', index=False)
        print("✅ Forecast saved: 30_day_forecast.csv")
    
    print("\n" + "="*60)
    print("✅ ALL DONE!")
    print("="*60)