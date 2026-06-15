"""
Random Walk (RW) model functions for inflation forecasting.
The random walk model uses the most recent observation as the forecast.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rw_rolling_window(Y, nprev, indice, lag):
    """
    Random Walk rolling window forecasting.
    
    The random walk model predicts that the value at time t+h will be equal to 
    the value at time t (no change expected).
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix (T x N)
    nprev : int
        Number of out-of-sample forecasts
    indice : int
        Column index of target variable (1-indexed, 1=CPI, 2=PCE)
    lag : int
        Forecast horizon (1-12 months ahead)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'pred': array of predictions
        - 'errors': dict with 'rmse' and 'mae'
    """
    Y = np.array(Y)
    save_pred = np.full((nprev, 1), np.nan)
    
    # Rolling window forecasting
    for i in range(nprev, 0, -1):
        # Window selection - same pattern as AR model
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        
        # Get the target series from window (0-indexed)
        y_series = Y_window[:, indice - 1]
        
        # Random walk forecast: y_{t+h} = y_t
        # Use the last available value in the window
        pred = y_series[-1]
        
        idx = nprev - i
        save_pred[idx, 0] = pred
        
        print(f"iteration {idx + 1}")
    
    # Calculate errors - use actual values from original data
    real = Y[:, indice - 1]  # Convert to 0-indexed
    actual = real[-nprev:]
    
    from utils import calculate_errors
    errors = calculate_errors(actual, save_pred.flatten())
    
    return {
        'pred': save_pred,
        'errors': errors
    }
