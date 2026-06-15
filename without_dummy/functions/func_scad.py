"""
SCAD (Smoothly Clipped Absolute Deviation) penalty functions for inflation forecasting.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors, plot_forecast


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def scad_penalty(beta, lambda_val, a=3.7):
    """
    SCAD penalty function.
    
    Parameters:
    -----------
    beta : float or ndarray
        Coefficient value(s)
    lambda_val : float
        Regularization parameter
    a : float
        SCAD parameter (typically 3.7)
    
    Returns:
    --------
    penalty : float or ndarray
        SCAD penalty value
    """
    abs_beta = np.abs(beta)
    
    penalty = np.where(
        abs_beta <= lambda_val,
        lambda_val * abs_beta,
        np.where(
            abs_beta <= a * lambda_val,
            (2 * a * lambda_val * abs_beta - beta**2 - lambda_val**2) / (2 * (a - 1)),
            lambda_val**2 * (a + 1) / 2
        )
    )
    
    return penalty


def scad_derivative(beta, lambda_val, a=3.7):
    """
    Derivative of SCAD penalty for coordinate descent.
    """
    abs_beta = np.abs(beta)
    sign_beta = np.sign(beta)
    
    deriv = np.where(
        abs_beta <= lambda_val,
        lambda_val * sign_beta,
        np.where(
            abs_beta <= a * lambda_val,
            (a * lambda_val * sign_beta - beta) / (a - 1),
            0
        )
    )
    
    return deriv


class SCADRegressor:
    """
    SCAD penalized regression using coordinate descent.
    """
    
    def __init__(self, alpha=1.0, n_lambdas=100, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.n_lambdas = n_lambdas
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.best_lambda = None
        
    def fit(self, X, y, criterion='bic'):
        """
        Fit SCAD model with IC-based lambda selection.
        """
        n, p = X.shape
        
        # Standardize
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1
        X_scaled = (X - X_mean) / X_std
        y_mean = y.mean()
        y_centered = y - y_mean
        
        # Lambda sequence
        lambda_max = np.max(np.abs(np.dot(X_scaled.T, y_centered))) / n
        lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_max * 0.001), self.n_lambdas)
        
        best_ic = np.inf
        best_coef = None
        best_lambda = None
        
        for lam in lambdas:
            # Coordinate descent
            coef = np.zeros(p)
            
            for _ in range(self.max_iter):
                coef_old = coef.copy()
                
                for j in range(p):
                    # Partial residual
                    r_j = y_centered - np.dot(X_scaled, coef) + coef[j] * X_scaled[:, j]
                    
                    # Update
                    z_j = np.dot(X_scaled[:, j], r_j) / n
                    
                    # Soft thresholding with SCAD
                    if np.abs(z_j) <= 2 * lam:
                        coef[j] = np.sign(z_j) * max(np.abs(z_j) - lam, 0)
                    elif np.abs(z_j) <= 3.7 * lam:
                        coef[j] = ((3.7 - 1) * z_j - np.sign(z_j) * 3.7 * lam) / (3.7 - 2)
                    else:
                        coef[j] = z_j
                
                # Check convergence
                if np.max(np.abs(coef - coef_old)) < self.tol:
                    break
            
            # Calculate IC
            y_pred = y_mean + np.dot(X_scaled, coef)
            residuals = y - y_pred
            mse = np.mean(residuals ** 2)
            n_nonzero = np.sum(np.abs(coef) > 1e-10) + 1
            
            if criterion == 'bic':
                ic = n * np.log(mse) + n_nonzero * np.log(n)
            elif criterion == 'aic':
                ic = n * np.log(mse) + 2 * n_nonzero
            else:
                ic = n * np.log(mse) + n_nonzero * np.log(n)
            
            if ic < best_ic:
                best_ic = ic
                best_coef = coef.copy()
                best_lambda = lam
        
        # Unscale coefficients
        self.coef_ = best_coef / X_std
        self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        self.best_lambda = best_lambda
        self.bic = best_ic
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.intercept_ + np.dot(X, self.coef_)
    
    @property
    def coef(self):
        """Return coefficients including intercept."""
        return np.concatenate([[self.intercept_], self.coef_])


def run_scad(Y, indice, lag, alpha=1.0):
    """
    Run SCAD model for forecasting.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    alpha : float
        Not used, kept for compatibility
    
    Returns:
    --------
    dict : Dictionary with 'model' and 'pred'
    """
    # Convert to 0-indexed
    indice = indice - 1
    
    Y = np.array(Y)
    
    # Compute PCA scores (returns tuple: scores, Y_filled)
    scores, _ = compute_pca_scores(Y, n_components=4, scale=False)
    
    # Combine original data with PCA scores
    Y2 = np.column_stack([Y, scores])
    
    # Create embedded matrix
    aux = embed(Y2, 4)
    y = aux[:, indice]
    n_cols_Y2 = Y2.shape[1]
    X = aux[:, n_cols_Y2 * lag:]
    
    # Prepare out-of-sample data
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        aux_trimmed = aux[:, :aux.shape[1] - n_cols_Y2 * (lag - 1)]
        X_out = aux_trimmed[-1, :X.shape[1]]
    
    # Adjust for lag
    y = y[:len(y) - lag + 1]
    X = X[:X.shape[0] - lag + 1, :]
    
    # Fit SCAD model
    model = SCADRegressor()
    model.fit(X, y)
    
    # Make prediction
    pred = model.predict(X_out)
    
    return {'model': model, 'pred': pred[0]}


def scad_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window SCAD forecasting.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    nprev : int
        Number of out-of-sample predictions
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    
    Returns:
    --------
    dict : Dictionary with 'pred', 'coef', and 'errors'
    """
    Y = np.array(Y)
    n_coef = 21 + (Y.shape[1] - 1) * 4
    
    save_coef = np.full((nprev, n_coef), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        # Window selection
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        
        # Run SCAD model
        result = run_scad(Y_window, indice, lag)
        
        idx = nprev - i
        coef = result['model'].coef
        save_coef[idx, :len(coef)] = coef
        save_pred[idx, 0] = result['pred']
        
        print(f"iteration {idx + 1}")
    
    # Calculate errors
    real = Y[:, indice - 1]  # Convert to 0-indexed
    actual = real[-nprev:]
    errors = calculate_errors(actual, save_pred.flatten())
    
    return {'pred': save_pred, 'coef': save_coef, 'errors': errors}


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    Y = np.random.randn(200, 5)
    
    result = scad_rolling_window(Y, nprev=10, indice=1, lag=1)
    print(f"SCAD RMSE: {result['errors']['rmse']:.4f}")
    print(f"SCAD MAE: {result['errors']['mae']:.4f}")
