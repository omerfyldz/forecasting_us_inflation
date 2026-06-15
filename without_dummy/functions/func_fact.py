"""
Factor Model functions for inflation forecasting.
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

def run_fact(Y, indice, lag):
    """
    Run Factor Model for forecasting.
    
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
    
    # Compute PCA scores (returns tuple: scores, Y_filled)
    scores, _ = compute_pca_scores(Y, n_components=4, scale=False)
    
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
    
    n = len(y)
    
    # Select number of factors using BIC
    best_bic = np.inf
    best_coef = None
    best_model = None
    
    for n_factors in range(5, min(21, X.shape[1] + 1), 5):
        X_subset = X[:, :n_factors]
        model = LinearRegression()
        model.fit(X_subset, y)
        
        # Calculate BIC
        y_pred = model.predict(X_subset)
        residuals = y - y_pred
        mse = np.mean(residuals ** 2)
        n_params = n_factors + 1
        bic = n * np.log(mse) + n_params * np.log(n)
        
        if bic < best_bic:
            best_bic = bic
            f_coef = np.concatenate([[model.intercept_], model.coef_])
            best_model = model
    
    # Pad coefficients to match full size
    coef = np.zeros(X.shape[1] + 1)
    coef[:len(f_coef)] = f_coef
    
    # Make prediction
    pred = coef[0] + np.dot(X_out, coef[1:])
    
    return {'model': best_model, 'pred': pred, 'coef': coef}


def fact_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window Factor Model forecasting.
    
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
    n_coef = 21  # Based on R code
    
    save_coef = np.full((nprev, n_coef), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        # Window selection
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        
        # Run Factor model
        result = run_fact(Y_window, indice, lag)
        
        idx = nprev - i
        coef = result['coef']
        save_coef[idx, :min(len(coef), n_coef)] = coef[:n_coef]
        save_pred[idx, 0] = result['pred']
        
        print(f"iteration {idx + 1}")
    
    # Calculate errors
    real = Y[:, indice - 1]  # Convert to 0-indexed
    actual = real[-nprev:]
    errors = calculate_errors(actual, save_pred.flatten())
    
    return {'pred': save_pred, 'coef': save_coef, 'errors': errors}


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    Y = np.random.randn(200, 5)
    
    result = fact_rolling_window(Y, nprev=10, indice=1, lag=1)
    print(f"Factor RMSE: {result['errors']['rmse']:.4f}")
    print(f"Factor MAE: {result['errors']['mae']:.4f}")
