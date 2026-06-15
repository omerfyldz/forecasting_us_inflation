"""
Polynomial LASSO Functions for Inflation Forecasting - Second Sample
With dummy variable handling
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def run_polilasso(Y, indice, lag, degree=2):
    """
    Run Polynomial LASSO for inflation forecasting.
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
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_out_poly = poly.transform(X_out.reshape(1, -1))[0]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    X_out_scaled = (X_out_poly - scaler.mean_) / scaler.scale_
    
    try:
        model = LassoCV(cv=5, max_iter=10000)
        model.fit(X_scaled, y)
        pred = model.intercept_ + X_out_scaled @ model.coef_
    except Exception:
        model = LinearRegression()
        model.fit(X_scaled, y)
        pred = model.intercept_ + X_out_scaled @ model.coef_
    
    return {
        'pred': pred,
        'model': model
    }


def polilasso_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window forecasting with Polynomial LASSO (PARALLELIZED).
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    save_pred = np.full((nprev, 1), np.nan)
    
    def _single_iteration(i):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        result = run_polilasso(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred']
    
    print(f"Running {nprev} Polynomial LASSO iterations in parallel (N_JOBS={N_JOBS})...")
    
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
