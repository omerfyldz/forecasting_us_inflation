"""
LASSO Functions for Inflation Forecasting - Second Sample
Adapted from R code with dummy variable handling

This module contains functions for LASSO, Adaptive LASSO, Elastic Net,
and Ridge regression with support for dummy variables.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, RidgeCV
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors


class ICGlmnet:
    """
    Glmnet with BIC selection for penalty parameter.
    Supports penalty factors for adaptive methods.
    """
    
    def __init__(self, alpha=1.0, n_lambdas=100, cv_folds=5):
        self.alpha = alpha
        self.n_lambdas = n_lambdas
        self.cv_folds = cv_folds  # Changed from 10 to 5 to match without_dummy
        self.coef_ = None
        self.intercept_ = None
        self.bic_ = None
        
    def fit(self, X, y, penalty_factor=None):
        """Fit model using BIC criterion."""
        n, p = X.shape
        
        # Standardize X
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit with cross-validation
        try:
            if self.alpha == 1:
                model = LassoCV(cv=self.cv_folds, max_iter=10000)
            elif self.alpha == 0:
                # RidgeCV needs alphas parameter for proper CV
                alphas = np.logspace(-6, 6, 100)
                model = RidgeCV(alphas=alphas, cv=self.cv_folds)
            else:
                model = ElasticNetCV(l1_ratio=self.alpha, cv=self.cv_folds, max_iter=10000)
            
            model.fit(X_scaled, y)
            
            self.coef_ = np.zeros(p + 1)
            self.coef_[0] = model.intercept_
            self.coef_[1:] = model.coef_
            
            # Calculate BIC
            y_pred = model.predict(X_scaled)
            rss = np.sum((y - y_pred) ** 2)
            k = np.sum(model.coef_ != 0)
            self.bic_ = n * np.log(rss / n) + k * np.log(n)
            
        except Exception:
            model = LinearRegression()
            model.fit(X_scaled, y)
            self.coef_ = np.zeros(p + 1)
            self.coef_[0] = model.intercept_
            self.coef_[1:] = model.coef_
            self.bic_ = np.inf
        
        return self
    
    def predict(self, X):
        """Predict using fitted model."""
        return self.coef_[0] + X @ self.coef_[1:]


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def run_lasso(Y, indice, lag, alpha=1, type_='lasso'):
    """
    Run LASSO for inflation forecasting with dummy variable.
    
    Parameters
    ----------
    Y : array-like
        Full dataset with dummy variable in last column
    indice : int
        Index of target variable (1 for CPI, 2 for PCE)
    lag : int
        Forecast horizon
    alpha : float
        Elastic net mixing parameter (1=LASSO, 0=Ridge)
    type_ : str
        Type of model ('lasso', 'adalasso', 'fal')
    
    Returns
    -------
    dict with 'pred' and 'model'
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    # Check if dummy variable is present (last column)
    has_dummy = Y.shape[1] > 2
    if has_dummy:
        dummy = Y[:(n_obs - lag + 1), -1]
        Y_main = Y[:, :-1]
    else:
        dummy = None
        Y_main = Y
    
    # Compute PCA scores (returns tuple: scores, Y_filled)
    pca_scores, _ = compute_pca_scores(Y_main)
    
    # Combine original data with PCA scores
    Y2 = np.column_stack([Y_main[:, indice - 1], pca_scores])
    
    # Create embedded matrix
    aux = embed(Y2, 4)
    
    # Extract y and X
    y = aux[:, 0]
    X = aux[:, (Y2.shape[1] * lag):]
    
    # Create out-of-sample X
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        X_out = aux[-1, :X.shape[1]]
    
    # Adjust for lag
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    # Add dummy if present
    if has_dummy:
        dummy = dummy[-len(y):]
        X = np.column_stack([X, dummy])
        X_out = np.append(X_out, 0)  # Dummy=0 for prediction
    
    # Fit model
    model = ICGlmnet(alpha=alpha)
    model.fit(X, y)
    coef = model.coef_
    
    if type_ == 'adalasso':
        # Adaptive LASSO
        penalty = (np.abs(coef[1:]) + 1 / np.sqrt(len(y))) ** (-1)
        if has_dummy:
            penalty[-1] = 0  # No penalty on dummy
        model = ICGlmnet(alpha=alpha)
        model.fit(X, y, penalty_factor=penalty)
        coef = model.coef_
    
    elif type_ == 'fal':
        # Full Adaptive LASSO
        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2, 3, 4, 5, 7, 10]
        alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        best_bic = np.inf
        best_model = model
        
        for a in alphas:
            m0 = ICGlmnet(alpha=a)
            m0.fit(X, y)
            coef_init = m0.coef_
            
            for tau in taus:
                penalty = (np.abs(coef_init[1:]) + 1 / np.sqrt(len(y))) ** (-tau)
                if has_dummy:
                    penalty[-1] = 0
                m = ICGlmnet(alpha=a)
                m.fit(X, y, penalty_factor=penalty)
                
                if m.bic_ < best_bic:
                    best_bic = m.bic_
                    best_model = m
        
        model = best_model
        coef = model.coef_
    
    # Make prediction
    pred = coef[0] + X_out @ coef[1:]
    
    return {
        'pred': pred,
        'model': {'coef': coef}
    }


def lasso_rolling_window(Y, nprev, indice=1, lag=1, alpha=1, type_='lasso'):
    """
    Rolling window forecasting with LASSO (PARALLELIZED).
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    n_vars = Y.shape[1] - 1 if Y.shape[1] > 2 else Y.shape[1]
    
    coef_size = 4 + 2 + n_vars * 4 + (1 if Y.shape[1] > 2 else 0)
    
    save_coef = np.full((nprev, coef_size), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    # Helper function for parallel processing
    def _single_iteration(i):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        result = run_lasso(Y_window, indice, lag, alpha, type_)
        idx = nprev - i
        return idx, result['model']['coef'], result['pred']
    
    # Parallelize rolling window
    N_JOBS = -1
    print(f"Running {nprev} LASSO iterations in parallel (N_JOBS={N_JOBS})...")
    
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_single_iteration)(i) for i in range(nprev, 0, -1)
    )
    
    # Aggregate results
    for idx, coef, pred in results:
        if len(coef) <= coef_size:
            save_coef[idx, :len(coef)] = coef
        save_pred[idx, 0] = pred
    
    print(f"Completed {nprev} iterations.")
    
    real = Y[:, indice - 1]
    errors = calculate_errors(real[-nprev:], save_pred.flatten())
    
    return {
        'pred': save_pred,
        'coef': save_coef,
        'errors': errors
    }


def run_pols(Y, indice, lag, coef):
    """
    Run Post-OLS estimation using LASSO-selected variables.
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    has_dummy = Y.shape[1] > 2
    if has_dummy:
        dummy = Y[:, -1]
        Y_main = Y[:, :-1]
    else:
        dummy = None
        Y_main = Y
    
    pca_scores, _ = compute_pca_scores(Y_main)
    pca_scores = pca_scores[:, :4]  # Take only first 4 components
    Y2 = np.column_stack([Y_main, pca_scores])
    
    aux = embed(Y2, 4)
    y = aux[:, indice - 1]
    X = aux[:, (Y2.shape[1] * lag):]
    
    X_out = aux[-1, :X.shape[1]]
    
    if has_dummy:
        dummy = dummy[-len(y):]
        X = np.column_stack([X, dummy])
        X_out = np.append(X_out, 0)
    
    selected = np.where(coef[1:] != 0)[0]
    
    if len(selected) == 0:
        model = LinearRegression(fit_intercept=True)
        model.fit(np.ones((len(y), 1)), y)
        pred = model.intercept_
    else:
        X_selected = X[:, selected]
        model = LinearRegression(fit_intercept=True)
        model.fit(X_selected, y)
        
        model_coef = model.coef_
        model_coef[np.isnan(model_coef)] = 0
        
        pred = model.intercept_ + X_out[selected] @ model_coef
    
    return {
        'pred': pred,
        'model': model
    }


def pols_rolling_window(Y, nprev, indice=1, lag=1, coefs=None):
    """
    Rolling window Post-OLS forecasting.
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        
        coef = coefs[nprev - i, :]
        result = run_pols(Y_window, indice, lag, coef)
        save_pred[nprev - i, 0] = result['pred']
        
        print(f"iteration {nprev - i + 1}")
    
    real = Y[:, indice - 1]
    errors = calculate_errors(real[-nprev:], save_pred.flatten())
    
    return {
        'pred': save_pred,
        'errors': errors
    }
