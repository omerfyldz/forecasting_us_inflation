"""
CSR (Complete Subset Regression) Functions for Inflation Forecasting - Second Sample
With dummy variable handling
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from itertools import combinations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def csr(X, y, k_max=5):
    """
    Complete Subset Regression.
    """
    n, p = X.shape
    k_max = min(k_max, p)
    
    predictions = []
    weights = []
    
    for k in range(1, k_max + 1):
        for indices in combinations(range(p), k):
            X_subset = X[:, list(indices)]
            model = LinearRegression()
            model.fit(X_subset, y)
            
            y_pred = model.predict(X_subset)
            rss = np.sum((y - y_pred) ** 2)
            bic = n * np.log(rss / n) + k * np.log(n)
            
            weight = np.exp(-0.5 * bic)
            predictions.append((model, list(indices), weight))
            weights.append(weight)
    
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1 / len(weights)] * len(weights)
    
    return {
        'models': predictions,
        'weights': weights
    }


def run_csr(Y, indice, lag, k_max=5):
    """
    Run CSR for inflation forecasting.
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
    
    result = csr(X, y, k_max)
    
    # Weighted prediction
    pred = 0
    for (model, indices, _), weight in zip(result['models'], result['weights']):
        X_out_subset = X_out[indices]
        pred += weight * (model.intercept_ + X_out_subset @ model.coef_)
    
    return {
        'pred': pred,
        'model': result
    }


def csr_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window forecasting with CSR (PARALLELIZED).
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    save_pred = np.full((nprev, 1), np.nan)
    
    def _single_iteration(i):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        result = run_csr(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred']
    
    print(f"Running {nprev} CSR iterations in parallel (N_JOBS={N_JOBS})...")
    
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
