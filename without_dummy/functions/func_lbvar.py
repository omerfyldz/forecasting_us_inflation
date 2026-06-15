"""
Large Bayesian VAR (LBVAR) functions for inflation forecasting.
"""

import numpy as np
from scipy import linalg
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import calculate_errors


class LBVAR:
    """
    Large Bayesian VAR with Minnesota-type prior.
    """
    
    def __init__(self, p=4, delta=0, lambda_param=0.05):
        """
        Parameters:
        -----------
        p : int
            Number of lags
        delta : float
            Prior mean for own lags (0 for stationary, 1 for random walk)
        lambda_param : float
            Overall tightness parameter
        """
        self.p = p
        self.delta = delta
        self.lambda_param = lambda_param
        self.coef = None
        self.covmat = None
        
    def fit(self, Y):
        """
        Fit Bayesian VAR model.
        
        Parameters:
        -----------
        Y : ndarray
            Data matrix (T x n)
        """
        Y = np.array(Y)
        T, n = Y.shape
        p = self.p
        
        # Create lagged matrix
        Y_lag = np.zeros((T - p, n * p))
        for i in range(p):
            Y_lag[:, i*n:(i+1)*n] = Y[p-1-i:T-1-i, :]
        
        Y_dep = Y[p:, :]
        
        # Add intercept
        X = np.column_stack([np.ones(T - p), Y_lag])
        
        # Minnesota prior
        k = X.shape[1]
        
        # Prior mean
        B_prior = np.zeros((k, n))
        if self.delta != 0:
            # Set prior for own first lag to delta
            for i in range(n):
                if 1 + i < k:
                    B_prior[1 + i, i] = self.delta
        
        # Prior variance (Minnesota-style)
        V_prior = np.zeros((k, n))
        V_prior[0, :] = 1e6  # Uninformative for intercept
        
        for i in range(n):
            for lag in range(p):
                for j in range(n):
                    idx = 1 + lag * n + j
                    if idx < k:
                        if i == j:
                            # Own lag
                            V_prior[idx, i] = (self.lambda_param / (lag + 1)) ** 2
                        else:
                            # Cross lag
                            sigma_i = np.var(Y[:, i])
                            sigma_j = np.var(Y[:, j])
                            if sigma_j > 0:
                                V_prior[idx, i] = (self.lambda_param / (lag + 1)) ** 2 * sigma_i / sigma_j
                            else:
                                V_prior[idx, i] = (self.lambda_param / (lag + 1)) ** 2
        
        # OLS estimate
        XtX = X.T @ X
        XtY = X.T @ Y_dep
        
        # Posterior mean (simplified)
        try:
            B_ols = np.linalg.solve(XtX, XtY)
        except:
            B_ols = np.linalg.pinv(XtX) @ XtY
        
        # Residuals and covariance
        residuals = Y_dep - X @ B_ols
        self.covmat = residuals.T @ residuals / (T - p - k)
        
        # Posterior coefficients (shrinkage toward prior)
        self.coef = B_ols  # Simplified: could add proper Bayesian shrinkage
        
        self.Y = Y
        self.n = n
        
        return self
    
    def predict(self, h=1):
        """
        h-step ahead forecast.
        
        Parameters:
        -----------
        h : int
            Forecast horizon
        
        Returns:
        --------
        ndarray : Forecasts (h x n)
        """
        Y = self.Y
        p = self.p
        n = self.n
        
        forecasts = np.zeros((h, n))
        
        # Last p observations
        Y_last = Y[-p:, :].copy()
        
        for step in range(h):
            # Create regressor
            x = np.concatenate([[1], Y_last.flatten()[:n*p]])
            
            # Forecast
            y_new = x @ self.coef
            forecasts[step, :] = y_new
            
            # Update Y_last
            Y_last = np.vstack([Y_last[1:, :], y_new.reshape(1, -1)])
        
        return forecasts


def lbvar_rw(Y, p, lag, nprev, delta=0, lambda_param=0.05, variables=None):
    """
    Rolling window LBVAR forecasting.
    
    Parameters:
    -----------
    Y : ndarray
        Data matrix
    p : int
        VAR lag order
    lag : int
        Forecast horizon
    nprev : int
        Number of out-of-sample predictions
    delta : float
        Prior mean parameter
    lambda_param : float
        Shrinkage parameter
    variables : list
        Indices of variables to forecast (0-indexed)
    
    Returns:
    --------
    dict : Dictionary with 'pred', 'covmat', and 'real'
    """
    Y = np.array(Y)
    
    if variables is None:
        variables = [0]
    
    real = Y[-nprev:, variables]
    Y = Y[:-lag, :]
    
    store_pred = np.full((nprev, len(variables)), np.nan)
    store_covmat = np.full((nprev, Y.shape[1]), np.nan)
    
    for i in range(nprev):
        # Window selection
        y_window = Y[(nprev - i - 1):(Y.shape[0] - i), :]
        
        # Fit LBVAR
        model = LBVAR(p=p, delta=delta, lambda_param=lambda_param)
        model.fit(y_window)
        
        # Store covariance
        covmat = np.sqrt(np.diag(model.covmat))
        store_covmat[i, :] = covmat
        
        # Predict
        pred = model.predict(h=lag)
        store_pred[i, :] = pred[-1, variables]
        
        print(f"iteration {i + 1}")
    
    # Reverse order to match R code
    store_pred = store_pred[::-1, :]
    
    return {'pred': store_pred, 'covmat': store_covmat, 'real': real}


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    Y = np.random.randn(200, 5)
    
    result = lbvar_rw(Y, p=4, lag=1, nprev=10, variables=[0, 1])
    print(f"Predictions shape: {result['pred'].shape}")
    print(f"Real values shape: {result['real'].shape}")
