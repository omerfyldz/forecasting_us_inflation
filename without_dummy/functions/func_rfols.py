"""
Random Forest OLS (RFOLS) functions for inflation forecasting.
Combines Random Forest variable selection with OLS estimation.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors, plot_forecast


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def run_rfols(Y, indice, lag):
    """
    Run Random Forest OLS model for forecasting.
    Uses RF for variable selection, then fits OLS on selected variables.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    
    Returns:
    --------
    float : Prediction value
    """
    # Convert to 0-indexed
    indice = indice - 1
    
    Y = np.array(Y)
    
    # Remove dummy if present (last column)
    if Y.shape[1] > 2:
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
    
    n, p = X.shape
    n_trees = 500
    predictions = []
    
    for k in range(n_trees):
        # Bootstrap sample
        bootstrap_idx = np.random.choice(n, size=n, replace=True)
        X_boot = X[bootstrap_idx, :]
        y_boot = y[bootstrap_idx]
        
        # Fit single tree to get variable importance
        rf = RandomForestRegressor(
            n_estimators=1, 
            max_depth=None,
            max_features='sqrt',
            bootstrap=False,
            random_state=k
        )
        rf.fit(X_boot, y_boot)
        
        # Get selected variables (non-zero importance)
        importance = rf.feature_importances_
        selected = np.where(importance > 0)[0]
        
        if len(selected) == 0:
            selected = np.array([0])  # Use at least one variable
        
        # Fit OLS on selected variables
        X_selected = X_boot[:, selected]
        ols = LinearRegression()
        ols.fit(X_selected, y_boot)
        
        # Predict
        X_out_selected = X_out[selected]
        pred = ols.predict(X_out_selected.reshape(1, -1))[0]
        predictions.append(pred)
    
    # Average predictions
    final_pred = np.mean(predictions)
    
    return {'pred': final_pred, 'model': 'rfols'}


def _rfols_single_iteration(i, Y, nprev, indice, lag):
    """Single iteration for parallel RFOLS rolling window."""
    Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
    result = run_rfols(Y_window, indice, lag)
    idx = nprev - i
    return idx, result['pred']


def rfols_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window Random Forest OLS forecasting (PARALLELIZED).
    
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
    Y = np.array(Y)
    save_pred = np.full((nprev, 1), np.nan)
    
    # PARALLEL execution of rolling window
    print(f"    Running {nprev} RFOLS iterations in parallel...")
    results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(_rfols_single_iteration)(i, Y, nprev, indice, lag)
        for i in range(nprev, 0, -1)
    )
    
    # Sort by index and extract predictions
    results.sort(key=lambda x: x[0])
    for idx, pred in results:
        save_pred[idx, 0] = pred
    
    # Calculate errors
    real = Y[:, indice - 1]  # Convert to 0-indexed
    actual = real[-nprev:]
    errors = calculate_errors(actual, save_pred.flatten())
    
    return {'pred': save_pred, 'errors': errors}


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(1)
    Y = np.random.randn(200, 5)
    
    result = rfols_rolling_window(Y, nprev=10, indice=1, lag=1)
    print(f"RFOLS RMSE: {result['errors']['rmse']:.4f}")
    print(f"RFOLS MAE: {result['errors']['mae']:.4f}")
