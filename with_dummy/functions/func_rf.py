"""
Random Forest Functions for Inflation Forecasting - Second Sample
With dummy variable handling
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def run_rf(Y, indice, lag):
    """
    Run Random Forest for inflation forecasting with dummy variable.
    Parameters match without_dummy: n_estimators=500
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    # Check for dummy variable
    has_dummy = Y.shape[1] > 2
    if has_dummy:
        Y_main = Y[:, :-1]
    else:
        Y_main = Y
    
    # Compute PCA scores (returns tuple: scores, Y_filled)
    pca_scores, _ = compute_pca_scores(Y_main)
    pca_scores = pca_scores[:, :4]
    Y2 = np.column_stack([Y_main, pca_scores])
    
    # Create embedded matrix
    aux = embed(Y2, 4)
    y = aux[:, indice - 1]
    X = aux[:, (Y2.shape[1] * lag):]
    
    X_out = aux[-1, :X.shape[1]].reshape(1, -1)
    
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    # FIX: Match without_dummy parameters
    model = RandomForestRegressor(
        n_estimators=500,       # Fixed: consistent with without_dummy
        random_state=42,
        n_jobs=-1               # Added: for internal parallelization
    )
    model.fit(X, y)
    
    pred = model.predict(X_out)[0]
    
    return {
        'pred': pred,
        'model': model
    }


def rf_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window forecasting with Random Forest (PARALLELIZED).
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    save_pred = np.full((nprev, 1), np.nan)
    
    # Helper function for parallel processing
    def _single_iteration(i):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        result = run_rf(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred']
    
    # Parallelize rolling window
    print(f"Running {nprev} RF iterations in parallel (N_JOBS={N_JOBS})...")
    
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_single_iteration)(i) for i in range(nprev, 0, -1)
    )
    
    # Aggregate results
    for idx, pred in results:
        save_pred[idx, 0] = pred
    
    print(f"Completed {nprev} iterations.")
    
    real = Y[:, indice - 1]
    errors = calculate_errors(real[-nprev:], save_pred.flatten())
    
    return {
        'pred': save_pred,
        'errors': errors
    }
