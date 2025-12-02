from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import tensorflow as tf



def create_lstm_model(sequence_length=30, n_features=6, lstm_units=[100, 50], dropout_rate=0.2):
    """
    Create BIDIRECTIONAL stacked LSTM model
    """
    
    model = Sequential()
    
    # First LSTM layer - BIDIRECTIONAL
    model.add(Bidirectional(
        LSTM(
            units=lstm_units[0],
            return_sequences=len(lstm_units) > 1
        ),
        input_shape=(sequence_length, n_features)
    ))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer - BIDIRECTIONAL
    if len(lstm_units) > 1:
        model.add(Bidirectional(
            LSTM(
                units=lstm_units[1],
                return_sequences=False
            )
        ))
        model.add(Dropout(dropout_rate))
    
    # Dense output layers (same)
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Usage (same as RNN):
lstm_model = create_lstm_model(sequence_length=30, n_features=6)
lstm_model.summary()