"""
Neural Network functions for inflation forecasting.
Using TensorFlow/Keras as Python equivalent to H2O deep learning.
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
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Warning: TensorFlow/Keras not available. Install with: pip install tensorflow")


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def run_nn(Y, indice, lag):
    """
    Run Neural Network model for forecasting.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix (last column may be dummy variable)
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    
    Returns:
    --------
    dict : Dictionary with 'model' and 'pred'
    """
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required for neural network models")
    
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
    
    # Build neural network (similar to R's h2o.deeplearning)
    # Architecture: hidden = c(32, 16, 8)
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X.shape[1],)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse'
    )
    
    # Train model
    np.random.seed(1)
    model.fit(
        X_scaled, y,
        epochs=100,
        batch_size=32,
        verbose=0,
        validation_split=0.1
    )
    
    # Make prediction
    pred = model.predict(X_out_scaled.reshape(1, -1), verbose=0)[0, 0]
    
    return {'model': model, 'pred': pred}


def nn_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window Neural Network forecasting.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    nprev : int
        Number of out-of-sample predictions
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    
    Returns:
    --------
    dict : Dictionary with 'pred' and 'errors'
    """
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required for neural network models")
    
    Y = np.array(Y)
    def process_single_iteration(i, Y, nprev, indice, lag):
        """Process a single iteration - designed for parallel execution."""
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        result = run_nn(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred']
    
    print(f"Running {nprev} NN iterations in parallel (N_JOBS={N_JOBS})...")
    
    # Parallel execution
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_single_iteration)(i, Y, nprev, indice, lag)
        for i in range(nprev, 0, -1)
    )
    
    # Collect results
    save_pred = np.full((nprev, 1), np.nan)
    for idx, pred in results:
        save_pred[idx, 0] = pred
    
    # Calculate errors
    real = Y[:, indice - 1]  # Convert to 0-indexed
    actual = real[-nprev:]
    errors = calculate_errors(actual, save_pred.flatten())
    
    return {'pred': save_pred, 'errors': errors}


if __name__ == "__main__":
    if KERAS_AVAILABLE:
        # Test with sample data
        np.random.seed(42)
        Y = np.random.randn(200, 5)
        
        result = nn_rolling_window(Y, nprev=10, indice=1, lag=1)
        print(f"NN RMSE: {result['errors']['rmse']:.4f}")
        print(f"NN MAE: {result['errors']['mae']:.4f}")
    else:
        print("Skipping test: TensorFlow/Keras not available")
