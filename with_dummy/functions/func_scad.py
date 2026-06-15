"""
SCAD Functions for Inflation Forecasting - Second Sample
With dummy variable handling
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors


class SCADRegressor:
    """
    SCAD (Smoothly Clipped Absolute Deviations) Regressor.
    """
    
    def __init__(self, lambda_=0.1, a=3.7, max_iter=1000, tol=1e-6):
        self.lambda_ = lambda_
        self.a = a
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        
    def _scad_penalty_derivative(self, beta):
        """Compute SCAD penalty derivative."""
        abs_beta = np.abs(beta)
        
        if abs_beta <= self.lambda_:
            return self.lambda_
        elif abs_beta <= self.a * self.lambda_:
            return (self.a * self.lambda_ - abs_beta) / (self.a - 1)
        else:
            return 0
    
    def fit(self, X, y):
        """Fit SCAD regression using coordinate descent."""
        n, p = X.shape
        
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        y_mean = y.mean()
        y_centered = y - y_mean
        
        beta = np.zeros(p)
        
        for iteration in range(self.max_iter):
            beta_old = beta.copy()
            
            for j in range(p):
                X_j = X_scaled[:, j]
                residual = y_centered - X_scaled @ beta + beta[j] * X_j
                
                rho = np.sum(X_j * residual) / n
                
                penalty = self._scad_penalty_derivative(beta_old[j])
                
                if np.abs(rho) <= penalty:
                    beta[j] = 0
                else:
                    beta[j] = np.sign(rho) * (np.abs(rho) - penalty)
            
            if np.sum((beta - beta_old) ** 2) < self.tol:
                break
        
        self.coef_ = beta / self.scaler_.scale_
        self.intercept_ = y_mean - self.scaler_.mean_ @ self.coef_
        
        return self
    
    def predict(self, X):
        """Predict using fitted model."""
        return self.intercept_ + X @ self.coef_


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def run_scad(Y, indice, lag, lambda_=0.1):
    """
    Run SCAD for inflation forecasting.
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
    
    model = SCADRegressor(lambda_=lambda_)
    model.fit(X, y)
    
    pred = model.predict(X_out.reshape(1, -1))[0]
    
    return {
        'pred': pred,
        'model': model
    }


def scad_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window forecasting with SCAD (PARALLELIZED).
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    save_pred = np.full((nprev, 1), np.nan)
    
    def _single_iteration(i):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        result = run_scad(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred']
    
    print(f"Running {nprev} SCAD iterations in parallel (N_JOBS={N_JOBS})...")
    
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
