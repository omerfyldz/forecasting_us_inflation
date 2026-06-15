"""
Bagging Functions for Inflation Forecasting - Second Sample
With dummy variable handling
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def pre_testing_group_joint(X, y, alpha=0.05):
    """Pre-testing for variable selection."""
    n, p = X.shape
    
    selected = []
    for j in range(p):
        X_j = X[:, j].reshape(-1, 1)
        model = LinearRegression()
        model.fit(X_j, y)
        
        y_pred = model.predict(X_j)
        residuals = y - y_pred
        mse = np.sum(residuals ** 2) / (n - 2)
        se = np.sqrt(mse / np.sum((X_j - X_j.mean()) ** 2))
        
        t_stat = model.coef_[0] / se if se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        if p_value < alpha:
            selected.append(j)
    
    return selected if selected else [0]


def run_bagg(Y, indice, lag, n_iter=100, alpha=0.05):
    """
    Run Bagging for inflation forecasting.
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
    
    # Pre-testing
    selected = pre_testing_group_joint(X, y, alpha)
    X = X[:, selected]
    X_out = X_out[selected]
    
    n = len(y)
    predictions = np.zeros(n_iter)
    
    for i in range(n_iter):
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = X[idx]
        y_boot = y[idx]
        
        model = LinearRegression()
        model.fit(X_boot, y_boot)
        predictions[i] = model.intercept_ + X_out @ model.coef_
    
    pred = np.mean(predictions)
    
    return {
        'pred': pred,
        'model': None
    }


def bagg_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window forecasting with Bagging (PARALLELIZED).
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    save_pred = np.full((nprev, 1), np.nan)
    
    def _single_iteration(i):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        result = run_bagg(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred']
    
    print(f"Running {nprev} Bagging iterations in parallel (N_JOBS={N_JOBS})...")
    
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
