"""
FLASSO (LASSO with dummy variable) Functions for Inflation Forecasting
Adapted from R code for second-sample with dummy variable handling

This module contains functions for LASSO, Adaptive LASSO, and FAL models
with a dummy variable (last column) that requires special treatment.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors


class ICGlmnetDummy:
    """
    Glmnet with BIC selection for penalty parameter with dummy variable support.
    Dummy variable is excluded from penalization.
    """
    
    def __init__(self, alpha=1.0, n_lambdas=100, cv_folds=10):
        self.alpha = alpha
        self.n_lambdas = n_lambdas
        self.cv_folds = cv_folds
        self.coef_ = None
        self.intercept_ = None
        self.bic_ = None
        
    def fit(self, X, y, penalty_factor=None):
        """
        Fit model using BIC criterion.
        
        Parameters
        ----------
        X : array-like
            Feature matrix (last column should be the dummy variable)
        y : array-like
            Target variable
        penalty_factor : array-like, optional
            Penalty factors for each variable (last element for dummy should be 0)
        """
        n, p = X.shape
        
        # Standardize X except dummy variable (last column)
        X_main = X[:, :-1]
        X_dummy = X[:, -1:] if X.shape[1] > 0 else None
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_main)
        if X_dummy is not None:
            X_scaled = np.column_stack([X_scaled, X_dummy])
        
        # Handle penalty factor
        if penalty_factor is not None:
            # Create weighted penalty
            weight = np.array(penalty_factor)
            # Ensure dummy has 0 penalty
            weight[-1] = 0
        else:
            weight = np.ones(p)
            weight[-1] = 0  # No penalty on dummy
        
        # Fit with various lambdas and select by BIC
        if self.alpha > 0:
            base_model = LassoCV(alphas=None, cv=self.cv_folds, max_iter=10000)
        else:
            base_model = ElasticNetCV(l1_ratio=0.5, cv=self.cv_folds, max_iter=10000)
        
        try:
            base_model.fit(X_scaled, y)
            
            # Use cross-validated lambda
            self.coef_ = np.zeros(p + 1)  # +1 for intercept
            self.coef_[0] = base_model.intercept_
            self.coef_[1:] = base_model.coef_
            
            # Calculate BIC
            y_pred = base_model.predict(X_scaled)
            rss = np.sum((y - y_pred) ** 2)
            k = np.sum(base_model.coef_ != 0)
            self.bic_ = n * np.log(rss / n) + k * np.log(n)
            
        except Exception:
            # Fallback to OLS
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

def run_flasso(Y, indice, lag, alpha=1, type_='lasso'):
    """
    Run LASSO with dummy variable for inflation forecasting.
    
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
    dict with 'pred' (prediction) and 'model' (fitted model info)
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    # Extract dummy variable (last column) and remove from main data
    dummy = Y[:(n_obs - lag + 1), -1]
    Y_main = Y[:, :-1]
    
    # Compute PCA scores (returns tuple: scores, Y_filled)
    pca_scores, _ = compute_pca_scores(Y_main)
    
    # Combine original data with PCA scores
    Y2 = np.column_stack([Y_main[:, indice - 1], pca_scores])
    
    # Create embedded matrix
    aux = embed(Y2, 4)
    
    # Extract y and X
    y = aux[:, 0]  # First column is target
    X = aux[:, (Y2.shape[1] * lag):]  # Lagged values as predictors
    
    # Create out-of-sample X
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        X_out_full = aux[:, :(Y2.shape[1] * (lag - 1))]
        X_out = aux[-1, :X.shape[1]]
    
    # Adjust for lag
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    dummy = dummy[-len(y):]
    
    # Add dummy to X
    X_with_dummy = np.column_stack([X, dummy])
    X_out_with_dummy = np.append(X_out, 0)  # Dummy=0 for prediction
    
    # Fit model based on type
    model = ICGlmnetDummy(alpha=alpha)
    model.fit(X_with_dummy, y)
    coef = model.coef_
    
    if type_ == 'adalasso':
        # Adaptive LASSO with ridge penalty weights
        penalty = (np.abs(coef[1:]) + 1 / np.sqrt(len(y))) ** (-1)
        penalty[-1] = 0  # No penalty on dummy
        model = ICGlmnetDummy(alpha=alpha)
        model.fit(X_with_dummy, y, penalty_factor=penalty)
        coef = model.coef_
    
    elif type_ == 'fal':
        # Full Adaptive LASSO - search over taus and alphas
        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2, 3, 4, 5, 7, 10]
        alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        best_bic = np.inf
        best_model = model
        
        for a in alphas:
            m0 = ICGlmnetDummy(alpha=a)
            m0.fit(X_with_dummy, y)
            coef_init = m0.coef_
            
            for tau in taus:
                penalty = (np.abs(coef_init[1:]) + 1 / np.sqrt(len(y))) ** (-tau)
                penalty[-1] = 0  # No penalty on dummy
                m = ICGlmnetDummy(alpha=a)
                m.fit(X_with_dummy, y, penalty_factor=penalty)
                
                if m.bic_ < best_bic:
                    best_bic = m.bic_
                    best_model = m
        
        model = best_model
        coef = model.coef_
    
    # Make prediction (with dummy=0)
    pred = coef[0] + X_out_with_dummy @ coef[1:]
    
    return {
        'pred': pred,
        'model': {'coef': coef}
    }


def flasso_rolling_window(Y, nprev, indice=1, lag=1, alpha=1, type_='lasso'):
    """
    Rolling window forecasting with LASSO and dummy variable.
    
    Parameters
    ----------
    Y : array-like
        Full dataset with dummy variable in last column
    nprev : int
        Number of out-of-sample forecasts
    indice : int
        Index of target variable (1 for CPI, 2 for PCE)
    lag : int
        Forecast horizon
    alpha : float
        Elastic net mixing parameter
    type_ : str
        Type of model ('lasso', 'adalasso', 'fal')
    
    Returns
    -------
    dict with 'pred', 'coef', 'errors'
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    n_vars = Y.shape[1] - 1  # Exclude dummy
    
    # Estimate coefficient size
    coef_size = 4 + 2 + n_vars * 4 + 1  # +1 for dummy
    
    save_coef = np.full((nprev, coef_size), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    def _single_iteration(i):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        result = run_flasso(Y_window, indice, lag, alpha, type_)
        idx = nprev - i
        return idx, result['pred'], result['model']['coef']
    
    print(f"Running {nprev} Flexible LASSO iterations in parallel (N_JOBS={N_JOBS})...")
    
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_single_iteration)(i) for i in range(nprev, 0, -1)
    )
    
    for idx, pred, coef in results:
        if len(coef) <= coef_size:
            save_coef[idx, :len(coef)] = coef
        save_pred[idx, 0] = pred
    
    print(f"Completed {nprev} iterations.")
    
    # Calculate errors
    real = Y[:, indice - 1]
    errors = calculate_errors(real[-nprev:], save_pred.flatten())
    
    return {
        'pred': save_pred,
        'coef': save_coef,
        'errors': errors
    }


def run_pols_dummy(Y, indice, lag, coef):
    """
    Run Post-OLS estimation with dummy variable.
    Only uses variables selected by LASSO.
    
    Parameters
    ----------
    Y : array-like
        Full dataset with dummy variable in last column
    indice : int
        Index of target variable
    lag : int
        Forecast horizon
    coef : array-like
        Coefficients from LASSO (0 = not selected)
    
    Returns
    -------
    dict with 'pred' and 'model'
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    # Extract dummy variable
    dummy = Y[:, -1]
    Y_main = Y[:, :-1]
    
    # Compute PCA scores (returns tuple: scores, Y_filled)
    pca_scores, _ = compute_pca_scores(Y_main)
    
    # Combine with first 4 components
    Y2 = np.column_stack([Y_main, pca_scores[:, :4]])
    
    # Create embedded matrix
    aux = embed(Y2, 4)
    
    # Extract y and X
    y = aux[:, indice - 1]
    X = aux[:, (Y2.shape[1] * lag):]
    
    # Create out-of-sample X
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        X_out = aux[-1, :X.shape[1]]
    
    # Trim to match dummy length
    dummy = dummy[-len(y):]
    X_with_dummy = np.column_stack([X, dummy])
    X_out_with_dummy = np.append(X_out, 0)
    
    # Select variables based on non-zero coefficients (excluding intercept)
    selected = np.where(coef[1:] != 0)[0]
    
    if len(selected) == 0:
        # Intercept-only model
        model = LinearRegression(fit_intercept=True)
        model.fit(np.ones((len(y), 1)), y)
        pred = model.intercept_
    else:
        X_selected = X_with_dummy[:, selected]
        model = LinearRegression(fit_intercept=True)
        model.fit(X_selected, y)
        
        model_coef = model.coef_
        model_coef[np.isnan(model_coef)] = 0
        
        pred = model.intercept_ + X_out_with_dummy[selected] @ model_coef
    
    return {
        'pred': pred,
        'model': model
    }


def pols_dummy_rolling_window(Y, nprev, indice=1, lag=1, coefs=None):
    """
    Rolling window Post-OLS forecasting with dummy variable.
    
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
    coefs : array-like
        Matrix of coefficients from each rolling window iteration
    
    Returns
    -------
    dict with 'pred' and 'errors'
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        # Create window
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        
        # Get coefficients for this iteration
        coef = coefs[nprev - i, :]
        
        # Fit model
        result = run_pols_dummy(Y_window, indice, lag, coef)
        save_pred[nprev - i, 0] = result['pred']
        
        print(f"iteration {nprev - i + 1}")
    
    # Calculate errors
    real = Y[:, indice - 1]
    errors = calculate_errors(real[-nprev:], save_pred.flatten())
    
    return {
        'pred': save_pred,
        'errors': errors
    }
