"""
Adaptive LASSO with Random Forest for inflation forecasting.
Combines adaptive LASSO with RF for improved variable selection.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors, plot_forecast
from .func_lasso import ICGlmnet


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def run_adalasso_rf(Y, indice, lag, alpha=1.0, model_type="lasso"):
    """
    Run Adaptive LASSO with Random Forest model for forecasting.
    First fits LASSO, selects variables, then uses RF for prediction.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    alpha : float
        Mixing parameter
    model_type : str
        "lasso", "adalasso", or "fal"
    
    Returns:
    --------
    dict : Dictionary with 'pred'
    """
    # Convert to 0-indexed
    indice = indice - 1
    
    Y = np.array(Y)
    
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
    
    # Fit initial model (LASSO or adaptive LASSO)
    model = ICGlmnet(alpha=alpha)
    model.fit(X, y)
    coef = model.coef
    
    if model_type == "adalasso":
        penalty = (np.abs(coef[1:]) + 1/np.sqrt(len(y))) ** (-1)
        model = ICGlmnet(alpha=alpha)
        model.fit(X, y, penalty_factor=penalty)
        coef = model.coef
    
    elif model_type == "fal":
        taus = list(np.arange(0.1, 1.1, 0.1)) + [1.25, 1.5, 2, 3, 4, 5, 7, 10]
        alphas = np.arange(0, 1.1, 0.1)
        best_bic = np.inf
        
        for a in alphas:
            m0 = ICGlmnet(alpha=a)
            m0.fit(X, y)
            coef_init = m0.coef
            
            for tau in taus:
                penalty = (np.abs(coef_init[1:]) + 1/np.sqrt(len(y))) ** (-tau)
                m = ICGlmnet(alpha=1.0)
                m.fit(X, y, penalty_factor=penalty)
                
                if m.bic < best_bic:
                    model = m
                    best_bic = m.bic
        
        coef = model.coef
    
    # Select variables with non-zero coefficients
    selected = np.where(coef[1:] != 0)[0]
    
    if len(selected) < 2:
        selected = np.array([0, 1])  # Use at least 2 variables
    
    # Fit Random Forest on selected variables
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X[:, selected], y)
    
    # Make prediction
    pred = rf.predict(X_out[selected].reshape(1, -1))[0]
    
    return {'pred': pred}


def adalasso_rf_rolling_window(Y, nprev, indice=1, lag=1, alpha=1.0, model_type="lasso"):
    """
    Rolling window Adaptive LASSO RF forecasting.
    
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
    alpha : float
        Mixing parameter
    model_type : str
        "lasso", "adalasso", or "fal"
    
    Returns:
    --------
    dict : Dictionary with 'pred' and 'errors'
    """
    Y = np.array(Y)
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        # Window selection
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        
        # Run AdaLASSO-RF model
        result = run_adalasso_rf(Y_window, indice, lag, alpha, model_type)
        
        idx = nprev - i
        save_pred[idx, 0] = result['pred']
        
        print(f"iteration {idx + 1}")
    
    # Calculate errors
    real = Y[:, indice - 1]  # Convert to 0-indexed
    actual = real[-nprev:]
    errors = calculate_errors(actual, save_pred.flatten())
    
    return {'pred': save_pred, 'errors': errors}


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    Y = np.random.randn(200, 5)
    
    result = adalasso_rf_rolling_window(Y, nprev=10, indice=1, lag=1, model_type="adalasso")
    print(f"AdaLASSO-RF RMSE: {result['errors']['rmse']:.4f}")
    print(f"AdaLASSO-RF MAE: {result['errors']['mae']:.4f}")
