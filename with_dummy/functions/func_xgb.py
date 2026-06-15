"""
XGBoost Functions for Inflation Forecasting - Second Sample
With dummy variable handling
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def run_xgb(Y, indice, lag):
    """
    Run XGBoost for inflation forecasting.
    Parameters match R code: n_estimators=1000, eta=0.05, max_depth=4
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
    
    X_out = aux[-1, :X.shape[1]].reshape(1, -1)
    
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    # FIX: Use parameters matching R code and without_dummy
    model = xgb.XGBRegressor(
        n_estimators=1000,          # Changed from 100 to 1000
        learning_rate=0.05,         # Changed from 0.1 to 0.05 (eta)
        max_depth=4,                # Changed from 6 to 4
        colsample_bylevel=2/3,      # Added: match R code
        subsample=1.0,              # Added: match R code
        min_child_weight=X.shape[0] / 200,  # Added: match R code
        random_state=42,
        verbosity=0,
        n_jobs=1                    # Added: match R code (nthread=1)
    )
    model.fit(X, y)
    
    pred = model.predict(X_out)[0]
    
    return {
        'pred': pred,
        'model': model
    }


def xgb_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window forecasting with XGBoost (PARALLELIZED).
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    save_pred = np.full((nprev, 1), np.nan)
    
    # Helper function for parallel processing
    def _single_iteration(i):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        result = run_xgb(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred']
    
    # Parallelize rolling window
    print(f"Running {nprev} XGBoost iterations in parallel (N_JOBS={N_JOBS})...")
    
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
