from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, LSTM, GRU

def create_rnn_model_for_features(sequence_length=60, n_features=7, units=64):
    """
    Create RNN model for multiple features.
    
    Args:
        sequence_length: Number of time steps (e.g., 60 days)
        n_features: Number of features per time step (e.g., 7)
        units: Number of RNN units
    """
    model = Sequential([
        # First RNN layer - accepts n_features
        SimpleRNN(units=units,
                 return_sequences=True,
                 input_shape=(sequence_length, n_features)),  # CHANGED!
        Dropout(0.2),

        # Second RNN layer
        SimpleRNN(units=units // 2,  # Reduced units
                 return_sequences=False),
        Dropout(0.2),

        # Fully connected layers
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)  # Output: predicted temperature
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Create model for your 7 features
rnn_model = create_rnn_model_for_features(
    sequence_length=30, 
    n_features=6,  # Temperature + 6 cylindrical time features -1
    units=64
)
print(rnn_model.summary())