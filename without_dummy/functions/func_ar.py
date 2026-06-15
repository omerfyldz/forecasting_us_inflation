"""
Autoregressive (AR) model functions for inflation forecasting.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joblib import Parallel, delayed
from utils import embed, calculate_errors, plot_forecast


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def run_ar(Y, indice, lag, model_type="fixed"):
    """
    Run AR model for forecasting.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    indice : int
        Column index of target variable (1-indexed as in R)
    lag : int
        Forecast horizon
    model_type : str
        "fixed" for fixed AR(4) or "bic" for BIC-selected order
    
    Returns:
    --------
    dict : Dictionary with 'model', 'pred', and 'coef'
    """
    # Convert to 0-indexed
    indice = indice - 1
    
    Y = np.array(Y)
    Y2 = Y[:, indice].reshape(-1, 1)
    
    # Create embedded matrix
    aux = embed(Y2, 4)
    y = aux[:, 0]
    X = aux[:, lag:]
    
    # Prepare out-of-sample data
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        aux_trimmed = aux[:, :aux.shape[1] - (lag - 1)]
        X_out = aux_trimmed[-1, :X.shape[1]]
    
    # Adjust for lag
    y = y[:len(y) - lag + 1]
    X = X[:X.shape[0] - lag + 1, :]
    
    if model_type == "fixed":
        model = LinearRegression()
        model.fit(X, y)
        coef = np.concatenate([[model.intercept_], model.coef_])
    
    elif model_type == "bic":
        best_bic = np.inf
        best_coef = None
        best_model = None
        
        n = len(y)
        for i in range(1, X.shape[1] + 1):
            X_subset = X[:, :i]
            model = LinearRegression()
            model.fit(X_subset, y)
            
            # Calculate BIC
            y_pred = model.predict(X_subset)
            residuals = y - y_pred
            mse = np.mean(residuals ** 2)
            n_params = i + 1  # coefficients + intercept
            bic = n * np.log(mse) + n_params * np.log(n)
            
            if bic < best_bic:
                best_bic = bic
                ar_coef = np.concatenate([[model.intercept_], model.coef_])
                best_model = model
        
        # Pad coefficients to match full size
        coef = np.zeros(X.shape[1] + 1)
        coef[:len(ar_coef)] = ar_coef
        model = best_model
    
    # Make prediction
    pred = coef[0] + np.dot(X_out, coef[1:])
    
    return {'model': model, 'pred': pred, 'coef': coef}


def ar_rolling_window(Y, nprev, indice=1, lag=1, model_type="fixed"):
    """
    Rolling window AR forecasting.
    
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
    model_type : str
        "fixed" or "bic"
    
    Returns:
    --------
    dict : Dictionary with 'pred', 'coef', and 'errors'
    """
    Y = np.array(Y)
    save_coef = np.full((nprev, 5), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        # Window selection
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        
        # Run AR model
        result = run_ar(Y_window, indice, lag, model_type)
        
        idx = nprev - i
        save_coef[idx, :] = result['coef'][:5]
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
    
    result = ar_rolling_window(Y, nprev=10, indice=1, lag=1, model_type="fixed")
    print(f"RMSE: {result['errors']['rmse']:.4f}")
    print(f"MAE: {result['errors']['mae']:.4f}")
