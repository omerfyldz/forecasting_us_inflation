"""
LSTM (Long Short-Term Memory) functions for inflation forecasting.
Using TensorFlow/Keras for LSTM implementation.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors, plot_forecast

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Warning: TensorFlow/Keras not available. Install with: pip install tensorflow")


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def run_lstm(Y, indice, lag, lstm_units=50, dropout_rate=0.2):
    """
    Run LSTM model for forecasting.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix (last column may be dummy variable)
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    lstm_units : int, default=50
        Number of LSTM units
    dropout_rate : float, default=0.2
        Dropout rate for regularization
    
    Returns:
    --------
    dict : Dictionary with 'model' and 'pred'
    """
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required for LSTM models")
    
    # Convert to 0-indexed
    indice = indice - 1
    
    Y = np.array(Y)
    
    # Remove dummy variable if present (last column)
    if Y.shape[1] > 2:
        dum = Y[:, -1]
        Y = Y[:, :-1]
    
    # Compute PCA scores (returns tuple: scores, Y_filled)
    scores, _ = compute_pca_scores(Y, n_components=4, scale=False)
    
    # Combine original data with PCA scores
    Y2 = np.column_stack([Y, scores])
    
    # Create embedded matrix
    aux = embed(Y2, 4)
    y = aux[:, indice]
    n_cols_Y2 = Y2.shape[1]
    X = aux[:, n_cols_Y2 * lag:]
    
    # Prepare out-of-sample data
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        aux_trimmed = aux[:, :aux.shape[1] - n_cols_Y2 * (lag - 1)]
        X_out = aux_trimmed[-1, :X.shape[1]]
    
    # Adjust for lag
    y = y[:len(y) - lag + 1]
    X = X[:X.shape[0] - lag + 1, :]
    
    # Standardize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_scaled = (X - X_mean) / X_std
    X_out_scaled = (X_out - X_mean) / X_std
    
    # Reshape for LSTM (samples, timesteps, features)
    # We'll treat each feature as a timestep
    lookback = min(4, X.shape[1])  # Use 4 timesteps
    n_features = X.shape[1] // lookback
    
    # If dimensions don't divide evenly, pad with zeros
    if X.shape[1] % lookback != 0:
        n_features = X.shape[1] // lookback + 1
        pad_width = n_features * lookback - X.shape[1]
        X_scaled = np.pad(X_scaled, ((0, 0), (0, pad_width)), mode='constant')
        X_out_scaled = np.pad(X_out_scaled, (0, pad_width), mode='constant')
    
    X_lstm = X_scaled.reshape(X_scaled.shape[0], lookback, n_features)
    X_out_lstm = X_out_scaled.reshape(1, lookback, n_features)
    
    # Build LSTM model
    model = Sequential([
        LSTM(lstm_units, activation='tanh', return_sequences=True, input_shape=(lookback, n_features)),
        Dropout(dropout_rate),
        LSTM(lstm_units // 2, activation='tanh'),
        Dropout(dropout_rate),
        Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse'
    )
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    # Train the model
    model.fit(
        X_lstm, y,
        epochs=100,
        batch_size=16,
        verbose=0,
        callbacks=[early_stop]
    )
    
    # Make prediction
    pred = model.predict(X_out_lstm, verbose=0)[0, 0]
    
    return {
        'model': model,
        'pred': pred
    }


def lstm_rolling_window(Y, nprev, indice, lag, lstm_units=50, dropout_rate=0.2):
    """
    Run LSTM model with rolling window approach.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    nprev : int
        Number of observations to use for training
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    lstm_units : int, default=50
        Number of LSTM units
    dropout_rate : float, default=0.2
        Dropout rate for regularization
    
    Returns:
    --------
    dict : Dictionary with 'pred' array and 'errors'
    """
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required for LSTM models")
    
    Y = np.array(Y)
    nobs = Y.shape[0]
    
    # Predictions array
    predictions = []
    actuals = []
    
    # Rolling window forecasting
    for i in range(nprev, nobs - lag + 1):
        Y_train = Y[:i, :]
        
        result = run_lstm(Y_train, indice, lag, lstm_units, dropout_rate)
        pred = result['pred']
        predictions.append(pred)
        
        # Get actual value
        actual = Y[i + lag - 1, indice - 1]
        actuals.append(actual)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate errors
    errors = calculate_errors(actuals, predictions)
    
    return {
        'pred': predictions,
        'actuals': actuals,
        'errors': errors
    }
