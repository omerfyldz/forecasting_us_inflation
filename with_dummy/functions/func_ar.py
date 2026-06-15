"""
AR Model Functions for Inflation Forecasting - Second Sample
Adapted from R code with dummy variable handling

This module contains functions for AR (Autoregressive) models with support
for a dummy variable in the last column of the data.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, calculate_errors


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def calculate_aic(y, y_pred, k):
    """Calculate AIC."""
    n = len(y)
    rss = np.sum((y - y_pred) ** 2)
    return n * np.log(rss / n) + 2 * k


def calculate_bic(y, y_pred, k):
    """Calculate BIC."""
    n = len(y)
    rss = np.sum((y - y_pred) ** 2)
    return n * np.log(rss / n) + k * np.log(n)


def run_ar(Y, indice, lag, model_type='fixed', max_lag=12):
    """
    Run AR model for inflation forecasting with dummy variable.
    
    Parameters
    ----------
    Y : array-like
        Full dataset with dummy variable in last column
    indice : int
        Index of target variable (1 for CPI, 2 for PCE)
    lag : int
        Forecast horizon
    model_type : str
        Type of lag selection ('fixed', 'bic', 'aic')
    max_lag : int
        Maximum lag for model selection (only used when model_type != 'fixed')
    
    Returns
    -------
    dict with 'pred' (prediction) and 'model' (fitted model)
    """
    Y = np.array(Y)
    
    # Extract dummy variable if present (last column)
    if Y.shape[1] > 2:  # More than just 2 target variables
        dummy = Y[:, -1]
        Y_main = Y[:, :-1]
    else:
        dummy = None
        Y_main = Y
    
    # Get target series
    y_full = Y_main[:, indice - 1]
    
    if model_type == 'fixed':
        # Use fixed number of lags
        n_lags = lag
        
        # Create embedded matrix
        aux = embed(y_full.reshape(-1, 1), n_lags + lag)
        y = aux[:, 0]
        X = aux[:, lag:(lag + n_lags)]
        
        # Create out-of-sample X
        X_out = aux[-1, :n_lags]
        
        # Adjust for forecast horizon
        y = y[:(len(y) - lag + 1)]
        X = X[:(X.shape[0] - lag + 1), :]
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict
        pred = model.intercept_ + X_out @ model.coef_
        
    else:
        # Select lags using information criterion
        best_ic = np.inf
        best_model = None
        best_pred = None
        best_lag = 1
        
        for n_lags in range(1, max_lag + 1):
            try:
                aux = embed(y_full.reshape(-1, 1), n_lags + lag)
                y = aux[:, 0]
                X = aux[:, lag:(lag + n_lags)]
                
                X_out = aux[-1, :n_lags]
                
                y = y[:(len(y) - lag + 1)]
                X = X[:(X.shape[0] - lag + 1), :]
                
                model = LinearRegression()
                model.fit(X, y)
                
                y_pred = model.predict(X)
                
                if model_type == 'bic':
                    ic = calculate_bic(y, y_pred, n_lags + 1)
                else:  # aic
                    ic = calculate_aic(y, y_pred, n_lags + 1)
                
                if ic < best_ic:
                    best_ic = ic
                    best_model = model
                    best_pred = model.intercept_ + X_out @ model.coef_
                    best_lag = n_lags
                    
            except Exception:
                continue
        
        model = best_model
        pred = best_pred
    
    return {
        'pred': pred,
        'model': model
    }


def ar_rolling_window(Y, nprev, indice=1, lag=1, model_type='fixed'):
    """
    Rolling window forecasting with AR model.
    
    Parameters
    ----------
    Y : array-like
        Full dataset with dummy variable in last column
    nprev : int
        Number of out-of-sample forecasts
    indice : int
        Index of target variable
    lag : int
        Forecast horizon
    model_type : str
        Type of lag selection ('fixed', 'bic', 'aic')
    
    Returns
    -------
    dict with 'pred' and 'errors'
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    save_pred = np.full((nprev, 1), np.nan)
    
    def _single_iteration(i):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        result = run_ar(Y_window, indice, lag, model_type)
        idx = nprev - i
        return idx, result['pred']
    
    print(f"Running {nprev} AR iterations in parallel (N_JOBS={N_JOBS})...")
    
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_single_iteration)(i) for i in range(nprev, 0, -1)
    )
    
    for idx, pred in results:
        save_pred[idx, 0] = pred
    
    print(f"Completed {nprev} iterations.")
    
    # Calculate errors
    real = Y[:, indice - 1]
    errors = calculate_errors(real[-nprev:], save_pred.flatten())
    
    return {
        'pred': save_pred,
        'errors': errors
    }
