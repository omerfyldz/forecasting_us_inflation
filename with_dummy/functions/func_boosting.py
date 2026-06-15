"""
Boosting Functions for Inflation Forecasting - Second Sample
With dummy variable handling
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def boosting(X, y, max_iter=100, nu=0.1):
    """L2 Boosting algorithm."""
    n, p = X.shape
    
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_centered = y - y_mean
    
    coef = np.zeros(p)
    residuals = y_centered.copy()
    
    for _ in range(max_iter):
        best_j = 0
        best_reduction = 0
        best_coef = 0
        
        for j in range(p):
            x_j = X_scaled[:, j]
            gamma = np.sum(x_j * residuals) / (np.sum(x_j ** 2) + 1e-8)
            
            new_residuals = residuals - nu * gamma * x_j
            reduction = np.sum(residuals ** 2) - np.sum(new_residuals ** 2)
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_j = j
                best_coef = gamma
        
        if best_reduction < 1e-10:
            break
        
        coef[best_j] += nu * best_coef
        residuals = residuals - nu * best_coef * X_scaled[:, best_j]
    
    coef_original = coef / X_std
    intercept = y_mean - X_mean @ coef_original
    
    return {
        'coef': coef_original,
        'intercept': intercept
    }


def run_boost(Y, indice, lag):
    """
    Run Boosting for inflation forecasting.
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
    
    model = boosting(X, y)
    pred = model['intercept'] + X_out @ model['coef']
    
    return {
        'pred': pred,
        'model': model
    }


def boosting_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window forecasting with Boosting (PARALLELIZED).
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    save_pred = np.full((nprev, 1), np.nan)
    
    def _single_iteration(i):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        result = run_boost(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred']
    
    print(f"Running {nprev} Boosting iterations in parallel (N_JOBS={N_JOBS})...")
    
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
