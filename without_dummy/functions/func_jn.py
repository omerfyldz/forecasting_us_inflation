"""
Jackknife Model Averaging functions for inflation forecasting.
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

def jackknife(X_list, y, lag=4, fixed_controls=None):
    """
    Jackknife Model Averaging.
    
    Parameters:
    -----------
    X_list : list of ndarrays
        List of feature matrices for different lag groups
    y : ndarray
        Target variable
    lag : int
        Number of lag groups
    fixed_controls : int or list
        Index of fixed control variable
    
    Returns:
    --------
    dict : Dictionary with 'coef', 'weights', and predictions
    """
    n = len(y)
    n_groups = len(X_list)
    
    # Combine all X matrices
    X_full = np.hstack(X_list)
    p = X_full.shape[1]
    
    # Initialize weights uniformly
    weights = np.ones(n_groups) / n_groups
    
    # Leave-one-out predictions for each model
    loo_predictions = np.zeros((n, n_groups))
    
    for g, X_g in enumerate(X_list):
        for i in range(n):
            # Leave one out
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            
            X_train = X_g[mask, :]
            y_train = y[mask]
            X_test = X_g[i:i+1, :]
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            loo_predictions[i, g] = model.predict(X_test)[0]
    
    # Optimize weights using quadratic programming (simplified)
    # Minimize sum of squared errors of weighted predictions
    from scipy.optimize import minimize
    
    def objective(w):
        pred = np.dot(loo_predictions, w)
        return np.sum((y - pred) ** 2)
    
    # Constraints: weights sum to 1, weights >= 0
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_groups)]
    
    result = minimize(objective, weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    weights = result.x
    
    # Fit final models and compute coefficients
    coef_list = []
    for g, X_g in enumerate(X_list):
        model = LinearRegression()
        model.fit(X_g, y)
        coef_list.append(np.concatenate([[model.intercept_], model.coef_]))
    
    return {'coef': coef_list, 'weights': weights}


def run_jn(Y, indice, lag):
    """
    Run Jackknife Model Averaging for forecasting.
    
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
    dict : Dictionary with 'model', 'pred', 'weights', and 'coef'
    """
    # Convert to 0-indexed
    indice_0 = indice - 1
    
    Y = np.array(Y)
    
    # Compute PCA scores (returns tuple: scores, Y_filled)
    scores, _ = compute_pca_scores(Y, n_components=4, scale=False)
    
    # Combine original data with PCA scores
    Y2 = np.column_stack([Y, scores])
    
    # Create embedded matrix
    aux = embed(Y2, 4)
    y = aux[:, indice_0]
    n_cols_Y2 = Y2.shape[1]
    X = aux[:, n_cols_Y2 * lag:]
    
    # Adjust for lag
    y = y[:len(y) - lag + 1]
    X = X[:X.shape[0] - lag + 1, :]
    
    # Split X into lag groups
    n_Y = Y.shape[1] + 4  # Original variables + PCA
    i1 = list(range(0, X.shape[1], n_Y))
    i2 = list(range(n_Y - 1, X.shape[1], n_Y))
    
    X_list = []
    X_out_list = []
    
    for i in range(min(4, len(i1))):
        start = i1[i]
        end = min(i2[i] + 1, X.shape[1])
        X_list.append(X[:, start:end])
        
        # Prepare out-of-sample data
        if lag == 1:
            X_out_list.append(aux[-1, start:end])
        else:
            aux_trimmed = aux[:, :aux.shape[1] - n_cols_Y2 * (lag - 1)]
            X_out_list.append(aux_trimmed[-1, start:end])
    
    # Run jackknife
    result = jackknife(X_list, y, lag=4, fixed_controls=indice_0)
    weights = result['weights']
    coef_list = result['coef']
    
    # Make prediction using weighted average
    pred = 0
    for g, (X_out, coef) in enumerate(zip(X_out_list, coef_list)):
        pred_g = coef[0] + np.dot(X_out, coef[1:])
        pred += weights[g] * pred_g
    
    # Flatten weights for storage
    all_weights = np.concatenate([weights, np.zeros(Y.shape[1] - 1)])
    
    return {'model': result, 'pred': pred, 'weights': all_weights, 'coef': coef_list}


def jackknife_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window Jackknife Model Averaging forecasting.
    
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
    dict : Dictionary with 'pred', 'weights', and 'errors'
    """
    Y = np.array(Y)
    n_weights = Y.shape[1] - 1 + 4  # Based on R code
    
    save_weights = np.full((nprev, n_weights), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        # Window selection
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        
        # Run Jackknife model
        result = run_jn(Y_window, indice, lag)
        
        idx = nprev - i
        save_pred[idx, 0] = result['pred']
        weights = result['weights']
        save_weights[idx, :len(weights)] = weights
        
        print(f"iteration {idx + 1}")
    
    # Calculate errors
    real = Y[:, indice - 1]  # Convert to 0-indexed
    actual = real[-nprev:]
    errors = calculate_errors(actual, save_pred.flatten())
    
    return {'pred': save_pred, 'weights': save_weights, 'errors': errors}


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    Y = np.random.randn(200, 5)
    
    result = jackknife_rolling_window(Y, nprev=10, indice=1, lag=1)
    print(f"Jackknife RMSE: {result['errors']['rmse']:.4f}")
    print(f"Jackknife MAE: {result['errors']['mae']:.4f}")
