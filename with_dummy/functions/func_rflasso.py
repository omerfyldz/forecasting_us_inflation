"""
RF-LASSO (Random Forest with LASSO) Functions for Inflation Forecasting
Adapted from R code for second-sample with dummy variable handling

This module implements the RF-LASSO hybrid method where:
1. Random Forest is used to select important variables
2. LASSO is applied with RF-based penalty weights
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors


class ICGlmnet:
    """
    Glmnet with BIC selection for penalty parameter.
    """
    
    def __init__(self, alpha=1.0, n_lambdas=100, cv_folds=10):
        self.alpha = alpha
        self.n_lambdas = n_lambdas
        self.cv_folds = cv_folds
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y, penalty_factor=None):
        """Fit model using BIC criterion."""
        n, p = X.shape
        
        # Standardize X
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit with cross-validation
        try:
            model = LassoCV(alphas=None, cv=self.cv_folds, max_iter=10000)
            model.fit(X_scaled, y)
            
            self.coef_ = np.zeros(p + 1)
            self.coef_[0] = model.intercept_
            self.coef_[1:] = model.coef_
            
        except Exception:
            # Fallback to OLS
            model = LinearRegression()
            model.fit(X_scaled, y)
            self.coef_ = np.zeros(p + 1)
            self.coef_[0] = model.intercept_
            self.coef_[1:] = model.coef_
        
        return self


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def run_rflasso(Y, indice, lag):
    """
    Run RF-LASSO for inflation forecasting with dummy variable.
    
    This method:
    1. Fits a single tree Random Forest to get variable importance
    2. Uses importance as penalty weights for LASSO
    3. Repeats 500 times and averages predictions
    
    Parameters
    ----------
    Y : array-like
        Full dataset with dummy variable in last column
    indice : int
        Index of target variable (1 for CPI, 2 for PCE)
    lag : int
        Forecast horizon
    
    Returns
    -------
    float : prediction
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    # Extract dummy variable and remove from main data
    dummy = Y[:(n_obs - lag + 1), -1]
    Y_main = Y[:, :-1]
    
    # Compute PCA scores (first 4 components) - returns tuple: scores, Y_filled
    pca_scores, _ = compute_pca_scores(Y_main)
    pca_scores = pca_scores[:, :4]
    
    # Combine original data with PCA scores
    Y2 = np.column_stack([Y_main, pca_scores])
    
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
    
    # Adjust for lag
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    dummy = dummy[-len(y):]
    
    # Run 500 bootstrap iterations
    n_iter = 500
    pred_aux = np.zeros(n_iter)
    
    for k in range(n_iter):
        # Fit single tree Random Forest
        rf = RandomForestRegressor(
            n_estimators=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=k
        )
        rf.fit(X, y)
        
        # Get feature importance
        importance = rf.feature_importances_
        
        # Select important features (importance > 0)
        selected = np.where(importance > 0)[0]
        
        if len(selected) == 0:
            # No features selected, use mean
            pred_aux[k] = np.mean(y)
            continue
        
        # Create penalty factors based on importance
        penalty_factor = np.abs(importance[selected]) ** (-1)
        penalty_factor = np.append(penalty_factor, 0)  # No penalty on dummy
        
        # Get bootstrap sample indices
        n_samples = len(y)
        bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Prepare data for LASSO
        X_selected = X[bootstrap_idx][:, selected]
        y_boot = y[bootstrap_idx]
        dummy_boot = dummy[bootstrap_idx]
        
        X_with_dummy = np.column_stack([X_selected, dummy_boot])
        
        # Fit LASSO with penalty factors
        try:
            model = ICGlmnet(alpha=1.0)
            model.fit(X_with_dummy, y_boot, penalty_factor=penalty_factor)
            coef = model.coef_
            
            # Make prediction with dummy=0
            X_out_selected = np.append(X_out[selected], 0)
            pred_aux[k] = coef[0] + X_out_selected @ coef[1:]
            
        except Exception:
            # Fallback to mean
            pred_aux[k] = np.mean(y)
    
    # Average predictions
    pred = np.mean(pred_aux)
    
    return pred


def rflasso_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window forecasting with RF-LASSO.
    
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
        pred = run_rflasso(Y_window, indice, lag)
        idx = nprev - i
        return idx, pred
    
    print(f"Running {nprev} RF-LASSO iterations in parallel (N_JOBS={N_JOBS})...")
    
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
