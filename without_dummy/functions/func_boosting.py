"""
Boosting functions for inflation forecasting.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors, plot_forecast


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def boosting(y, X, v=0.2, max_iter=1000, tol=1e-6):
    """
    L2 Boosting for regression.
    
    Parameters:
    -----------
    y : ndarray
        Target variable
    X : ndarray
        Feature matrix
    v : float
        Learning rate / shrinkage parameter
    max_iter : int
        Maximum number of boosting iterations
    tol : float
        Tolerance for convergence
    
    Returns:
    --------
    dict : Dictionary with 'coef' and fitted model info
    """
    n, p = X.shape
    
    # Initialize
    coef = np.zeros(p + 1)  # Including intercept
    residuals = y - np.mean(y)
    coef[0] = np.mean(y)
    
    # Standardize X for boosting
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_scaled = (X - X_mean) / X_std
    
    # Boosting iterations
    for iteration in range(max_iter):
        # Find best variable
        correlations = np.abs(np.dot(X_scaled.T, residuals))
        best_j = np.argmax(correlations)
        
        # Update coefficient
        x_j = X_scaled[:, best_j]
        gamma = np.dot(x_j, residuals) / np.dot(x_j, x_j)
        
        # Shrink update
        coef[best_j + 1] += v * gamma / X_std[best_j]
        
        # Update residuals
        residuals = residuals - v * gamma * x_j
        
        # Check convergence
        if np.sum(residuals ** 2) < tol:
            break
    
    # Adjust intercept for centered data
    coef[0] = np.mean(y) - np.dot(X_mean, coef[1:])
    
    return {'coef': coef}


def run_boost(Y, indice, lag):
    """
    Run Boosting model for forecasting.
    
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
    dict : Dictionary with 'model', 'pred', and 'coef'
    """
    # Convert to 0-indexed
    indice = indice - 1
    
    Y = np.array(Y)
    
    # Compute PCA scores (using 8 components as in R code, returns tuple)
    scores, _ = compute_pca_scores(Y, n_components=8, scale=False)
    
    # Combine target with PCA scores
    Y2 = np.column_stack([Y[:, indice].reshape(-1, 1), scores])
    
    # Create embedded matrix
    aux = embed(Y2, 4)
    y = aux[:, 0]
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
    
    # Fit boosting model
    model = boosting(y, X, v=0.2)
    coef = model['coef']
    
    # Make prediction
    pred = coef[0] + np.dot(X_out, coef[1:])
    
    return {'model': model, 'pred': pred, 'coef': coef}


def boosting_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window Boosting forecasting (PARALLELIZED).
    
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
    dict : Dictionary with 'pred', 'coef', and 'errors'
    """
    Y = np.array(Y)
    n_coef = 36  # Based on R code
    
    save_coef = np.full((nprev, n_coef), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    def _single_iteration(i):
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        result = run_boost(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred'], result['coef']
    
    print(f"Running {nprev} Boosting iterations in parallel (N_JOBS={N_JOBS})...")
    
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_single_iteration)(i) for i in range(nprev, 0, -1)
    )
    
    for idx, pred, coef in results:
        save_coef[idx, :len(coef)] = coef
        save_pred[idx, 0] = pred
    
    print(f"Completed {nprev} iterations.")
    
    # Calculate errors
    real = Y[:, indice - 1]  # Convert to 0-indexed
    actual = real[-nprev:]
    errors = calculate_errors(actual, save_pred.flatten())
    
    return {'pred': save_pred, 'coef': save_coef, 'errors': errors}


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    Y = np.random.randn(200, 5)
    
    result = boosting_rolling_window(Y, nprev=10, indice=1, lag=1)
    print(f"Boosting RMSE: {result['errors']['rmse']:.4f}")
    print(f"Boosting MAE: {result['errors']['mae']:.4f}")
