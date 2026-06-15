"""
LBVAR Functions for Inflation Forecasting - Second Sample
With dummy variable handling
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import embed, calculate_errors


class LBVAR:
    """
    Large Bayesian VAR with Minnesota prior.
    """
    
    def __init__(self, lag=1, lambda_=0.1, tau=10, epsilon=0.001):
        self.lag = lag
        self.lambda_ = lambda_
        self.tau = tau
        self.epsilon = epsilon
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, Y):
        """Fit LBVAR model."""
        Y = np.array(Y)
        n, m = Y.shape
        
        # Create VAR structure
        aux = embed(Y, self.lag + 1)
        Y_dep = aux[:, :m]
        X = aux[:, m:]
        
        n_obs = X.shape[0]
        
        # Minnesota prior shrinkage
        shrinkage = self.lambda_ / (np.arange(1, self.lag + 1) ** 2)
        
        # Fit equation by equation with Ridge
        self.coef_ = np.zeros((m, X.shape[1]))
        self.intercept_ = np.zeros(m)
        
        for j in range(m):
            alpha = self.lambda_ * self.tau
            model = Ridge(alpha=alpha)
            model.fit(X, Y_dep[:, j])
            
            self.coef_[j] = model.coef_
            self.intercept_[j] = model.intercept_
        
        return self
    
    def predict(self, X):
        """Predict using fitted model."""
        return self.intercept_ + X @ self.coef_.T


def lbvar_rw(Y, nprev, indice=1, lag=1, lambda_=0.1):
    """
    Rolling window forecasting with LBVAR (PARALLELIZED).
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    # Check for dummy
    has_dummy = Y.shape[1] > 2
    if has_dummy:
        Y_main = Y[:, :-1]
    else:
        Y_main = Y
    
    save_pred = np.full((nprev, 1), np.nan)
    
    def _single_iteration(i):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y_main[start_idx:end_idx, :]
        
        model = LBVAR(lag=lag, lambda_=lambda_)
        model.fit(Y_window)
        
        aux = embed(Y_window, lag + 1)
        X_out = aux[-1, Y_window.shape[1]:]
        
        pred = model.predict(X_out.reshape(1, -1))
        idx = nprev - i
        return idx, pred[0, indice - 1]
    
    print(f"Running {nprev} LBVAR iterations in parallel (N_JOBS={N_JOBS})...")
    
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_single_iteration)(i) for i in range(nprev, 0, -1)
    )
    
    for idx, pred in results:
        save_pred[idx, 0] = pred
    
    print(f"Completed {nprev} iterations.")
    
    real = Y_main[:, indice - 1]
    errors = calculate_errors(real[-nprev:], save_pred.flatten())
    
    return {
        'pred': save_pred,
        'errors': errors
    }
