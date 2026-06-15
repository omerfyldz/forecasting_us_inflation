"""
Targeted Factor Model Functions for Inflation Forecasting - Second Sample
With dummy variable handling
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def baggit_pretest(X, y, n_iter=100, alpha=0.05):
    """
    Bagging with pre-testing for targeted factors.
    """
    n, p = X.shape
    
    # Pre-testing
    selected = []
    for j in range(p):
        X_j = X[:, j].reshape(-1, 1)
        model = LinearRegression()
        model.fit(X_j, y)
        
        y_pred = model.predict(X_j)
        residuals = y - y_pred
        mse = np.sum(residuals ** 2) / (n - 2)
        se = np.sqrt(mse / (np.sum((X_j - X_j.mean()) ** 2) + 1e-8))
        
        t_stat = model.coef_[0] / se if se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        if p_value < alpha:
            selected.append(j)
    
    if not selected:
        selected = list(range(min(4, p)))
    
    X_selected = X[:, selected]
    
    # Extract factors from selected variables
    n_factors = min(4, X_selected.shape[1])
    if n_factors > 0:
        pca = PCA(n_components=n_factors)
        factors = pca.fit_transform(X_selected)
    else:
        factors = X_selected
    
    return {
        'factors': factors,
        'selected': selected,
        'pca': pca if n_factors > 0 else None
    }


def run_tfact(Y, indice, lag, n_factors=4):
    """
    Run Targeted Factor Model for inflation forecasting.
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    has_dummy = Y.shape[1] > 2
    if has_dummy:
        Y_main = Y[:, :-1]
    else:
        Y_main = Y
    
    y_target = Y_main[:, indice - 1]
    X_other = np.delete(Y_main, indice - 1, axis=1)
    
    # Get targeted factors
    result = baggit_pretest(X_other, y_target)
    factors = result['factors']
    
    # Combine target with factors
    Y2 = np.column_stack([y_target, factors])
    
    aux = embed(Y2, 4)
    y = aux[:, 0]
    X = aux[:, (Y2.shape[1] * lag):]
    
    X_out = aux[-1, :X.shape[1]]
    
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    model = LinearRegression()
    model.fit(X, y)
    
    pred = model.intercept_ + X_out @ model.coef_
    
    return {
        'pred': pred,
        'model': model
    }


def tfact_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window forecasting with Targeted Factor Model (PARALLELIZED).
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    save_pred = np.full((nprev, 1), np.nan)
    
    def _single_iteration(i):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        result = run_tfact(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred']
    
    print(f"Running {nprev} Targeted Factor iterations in parallel (N_JOBS={N_JOBS})...")
    
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
