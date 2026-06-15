"""
Neural Network Functions for Inflation Forecasting - Second Sample
With dummy variable handling
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def run_nn(Y, indice, lag, hidden_layers=[50, 50], epochs=100, dropout=0.2):
    """
    Run Neural Network for inflation forecasting.
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    has_dummy = Y.shape[1] > 2
    if has_dummy:
        Y_main = Y[:, :-1]
    else:
        Y_main = Y
    
    pca_scores, _ = compute_pca_scores(Y_main)
    pca_scores = pca_scores[:, :4]
    Y2 = np.column_stack([Y_main, pca_scores])
    
    aux = embed(Y2, 4)
    y = aux[:, indice - 1]
    X = aux[:, (Y2.shape[1] * lag):]
    
    X_out = aux[-1, :X.shape[1]].reshape(1, -1)
    
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    # Standardize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std
    X_out_scaled = (X_out - X_mean) / X_std
    
    if HAS_TENSORFLOW:
        model = Sequential()
        model.add(Dense(hidden_layers[0], activation='relu', input_shape=(X.shape[1],)))
        model.add(Dropout(dropout))
        
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout))
        
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mse')
        
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        
        model.fit(X_scaled, y, epochs=epochs, batch_size=32, verbose=0, callbacks=[early_stop])
        
        pred = model.predict(X_out_scaled, verbose=0)[0, 0]
    else:
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=tuple(hidden_layers), max_iter=epochs, random_state=42)
        model.fit(X_scaled, y)
        pred = model.predict(X_out_scaled)[0]
    
    return {
        'pred': pred,
        'model': model
    }


def nn_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window forecasting with Neural Network (PARALLELIZED).
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    save_pred = np.full((nprev, 1), np.nan)
    
    def _single_iteration(i):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        result = run_nn(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred']
    
    print(f"Running {nprev} NN iterations in parallel (N_JOBS={N_JOBS})...")
    
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_single_iteration)(i) for i in range(nprev, 0, -1)
    )
    
    for idx, pred in results:
        save_pred[idx, 0] = pred
    
    print(f"Completed {nprev} iterations.")
    
    real = Y[:, indice - 1]
    errors = calculate_errors(real[-nprev:], save_pred.flatten())
    
    return {
        'pred': save_pred,
        'errors': errors
    }
