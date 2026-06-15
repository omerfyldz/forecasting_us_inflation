"""
Jackknife Model Averaging Functions for Inflation Forecasting - Second Sample
With dummy variable handling
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def jackknife(X, y):
    """
    Jackknife Model Averaging.
    """
    n, p = X.shape
    p = min(p, 10)
    X = X[:, :p]
    
    models = []
    loo_residuals = np.zeros((n, p))
    
    for k in range(1, p + 1):
        X_k = X[:, :k]
        
        for i in range(n):
            X_loo = np.delete(X_k, i, axis=0)
            y_loo = np.delete(y, i)
            
            model = LinearRegression()
            model.fit(X_loo, y_loo)
            
            pred_i = model.intercept_ + X_k[i] @ model.coef_
            loo_residuals[i, k-1] = y[i] - pred_i
        
        full_model = LinearRegression()
        full_model.fit(X_k, y)
        models.append(full_model)
    
    # Compute optimal weights
    def objective(w):
        combined_residuals = loo_residuals @ w
        return np.sum(combined_residuals ** 2)
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]
    bounds = [(0, 1) for _ in range(p)]
    
    w0 = np.ones(p) / p
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = result.x
    
    return {
        'models': models,
        'weights': weights
    }


def run_jn(Y, indice, lag):
    """
    Run Jackknife Model Averaging for inflation forecasting.
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
    
    X_out = aux[-1, :X.shape[1]]
    
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    result = jackknife(X, y)
    
    # Weighted prediction
    pred = 0
    for k, (model, weight) in enumerate(zip(result['models'], result['weights'])):
        X_out_k = X_out[:k+1]
        pred += weight * (model.intercept_ + X_out_k @ model.coef_)
    
    return {
        'pred': pred,
        'model': result
    }


def jackknife_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window forecasting with Jackknife Model Averaging (PARALLELIZED).
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    save_pred = np.full((nprev, 1), np.nan)
    
    def _single_iteration(i):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        result = run_jn(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred']
    
    print(f"Running {nprev} Jackknife iterations in parallel (N_JOBS={N_JOBS})...")
    
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
