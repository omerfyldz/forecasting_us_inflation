"""
LASSO and related penalized regression functions for inflation forecasting.
Includes: LASSO, Adaptive LASSO, Elastic Net, Ridge, and post-OLS variants.
"""

import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors, plot_forecast


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1


class ICGlmnet:
    """
    Information Criterion based GLMnet model selection.
    Similar to R's ic.glmnet from HDeconometrics package.
    """
    
    def __init__(self, alpha=1.0, criterion='bic'):
        """
        Parameters:
        -----------
        alpha : float
            Mixing parameter (1=LASSO, 0=Ridge, 0<alpha<1=Elastic Net)
        criterion : str
            Information criterion ('bic', 'aic', 'aicc', 'hqc')
        """
        self.alpha = alpha
        self.criterion = criterion
        self.coef_ = None
        self.intercept_ = None
        self.bic = None
        self.best_lambda = None
        
    def fit(self, X, y, penalty_factor=None):
        """
        Fit the model using cross-validation and select best model by IC.
        """
        n, p = X.shape
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle near-zero variance features
        scale = self.scaler.scale_.copy()
        scale[scale < 1e-10] = 1.0  # Prevent division by zero
        
        # Use cross-validation to get lambda path
        if self.alpha == 1.0:
            model = LassoCV(cv=5, max_iter=10000, random_state=42)
        elif self.alpha == 0.0:
            # RidgeCV needs alphas parameter, not just cv
            alphas = np.logspace(-6, 6, 100)
            model = RidgeCV(alphas=alphas, cv=5)
        else:
            model = ElasticNetCV(l1_ratio=self.alpha, cv=5, max_iter=10000, random_state=42)
        
        # Apply penalty factor if provided
        if penalty_factor is not None:
            # Adjust features by penalty factor
            penalty_factor = np.array(penalty_factor)
            penalty_factor[penalty_factor == 0] = 1e-10
            X_adjusted = X_scaled / penalty_factor.reshape(1, -1)
            model.fit(X_adjusted, y)
            
            # Adjust coefficients back
            if hasattr(model, 'coef_'):
                coef_adjusted = model.coef_ / penalty_factor
            else:
                coef_adjusted = model.coef_
        else:
            model.fit(X_scaled, y)
            coef_adjusted = model.coef_ / scale
        
        # Store coefficients (unscaled)
        self.coef_ = coef_adjusted
        self.intercept_ = y.mean() - np.dot(self.scaler.mean_, self.coef_)
        
        # Calculate information criteria
        y_pred = self.predict(X)
        residuals = y - y_pred
        mse = np.mean(residuals ** 2)
        sse = np.sum(residuals ** 2)
        
        n_nonzero = np.sum(np.abs(self.coef_) > 1e-10) + 1  # +1 for intercept
        
        self.bic = n * np.log(mse) + n_nonzero * np.log(n)
        self.aic = n * np.log(mse) + 2 * n_nonzero
        
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


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def run_lasso(Y, indice, lag, alpha=1.0, model_type="lasso"):
    """
    Run LASSO/Elastic Net model for forecasting.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    alpha : float
        Mixing parameter (1=LASSO, 0=Ridge)
    model_type : str
        "lasso", "adalasso" (adaptive), or "fal" (flexible adaptive)
    
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
    
    # Fit initial model
    model = ICGlmnet(alpha=alpha)
    model.fit(X, y)
    coef = model.coef
    
    if model_type == "adalasso":
        # Adaptive LASSO: use initial estimates for penalty weights
        penalty = (np.abs(coef[1:]) + 1/np.sqrt(len(y))) ** (-1)
        model = ICGlmnet(alpha=alpha)
        model.fit(X, y, penalty_factor=penalty)
    
    elif model_type == "fal":
        # Flexible Adaptive LASSO
        taus = list(np.arange(0.1, 1.1, 0.1)) + [1.25, 1.5, 2, 3, 4, 5, 7, 10]
        alphas = np.arange(0, 1.1, 0.1)
        best_bic = np.inf
        
        for a in alphas:
            m0 = ICGlmnet(alpha=a)
            m0.fit(X, y)
            coef_init = m0.coef
            
            for tau in taus:
                penalty = (np.abs(coef_init[1:]) + 1/np.sqrt(len(y))) ** (-tau)
                m = ICGlmnet(alpha=1.0)  # Use LASSO with adaptive weights
                m.fit(X, y, penalty_factor=penalty)
                
                if m.bic < best_bic:
                    model = m
                    best_bic = m.bic
    
    # Make prediction
    pred = model.predict(X_out)
    pred = float(pred.flatten()[0]) if hasattr(pred, 'flatten') else float(pred)
    
    return {'model': model, 'pred': pred}


def run_pols(Y, indice, lag, coef):
    """
    Post-OLS: Run OLS on variables selected by penalized regression.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    coef : ndarray
        Coefficients from penalized regression
    
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
    
    # Get selected variables (non-zero coefficients, excluding intercept)
    selected = np.where(coef[1:] != 0)[0]
    
    if len(selected) == 0:
        # Intercept-only model
        model = LinearRegression()
        y_adjusted = y[:len(y) - lag + 1]
        X_dummy = np.ones((len(y_adjusted), 1))
        model.fit(X_dummy, y_adjusted)
        pred = model.intercept_
    else:
        # OLS on selected variables
        X_adjusted = X[:X.shape[0] - lag + 1, :]
        y_adjusted = y[:len(y) - lag + 1]
        
        X_selected = X_adjusted[:, selected]
        model = LinearRegression()
        model.fit(X_selected, y_adjusted)
        
        # Predict
        X_out_selected = X_out[selected]
        pred = model.intercept_ + np.dot(X_out_selected, model.coef_)
    
    return {'model': model, 'pred': pred}


def _lasso_single_iteration(i, Y, nprev, indice, lag, alpha, model_type):
    """Single iteration for parallel processing."""
    Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
    result = run_lasso(Y_window, indice, lag, alpha, model_type)
    idx = nprev - i
    return idx, result['model'].coef, result['pred']


def lasso_rolling_window(Y, nprev, indice=1, lag=1, alpha=1.0, model_type="lasso"):
    """
    Rolling window LASSO forecasting (PARALLELIZED).
    
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
    alpha : float
        Mixing parameter
    model_type : str
        "lasso", "adalasso", or "fal"
    
    Returns:
    --------
    dict : Dictionary with 'pred', 'coef', and 'errors'
    """
    Y = np.array(Y)
    n_coef = 21 + (Y.shape[1] - 1) * 4  # Approximate coefficient size
    
    save_coef = np.full((nprev, n_coef), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    # PARALLEL execution of rolling window
    print(f"Running {nprev} iterations in parallel...")
    results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(_lasso_single_iteration)(i, Y, nprev, indice, lag, alpha, model_type)
        for i in range(nprev, 0, -1)
    )
    
    # Collect results
    for idx, coef, pred in results:
        save_coef[idx, :len(coef)] = coef
        save_pred[idx, 0] = pred
    
    # Calculate errors
    real = Y[:, indice - 1]  # Convert to 0-indexed
    actual = real[-nprev:]
    errors = calculate_errors(actual, save_pred.flatten())
    
    return {'pred': save_pred, 'coef': save_coef, 'errors': errors}


def _pols_single_iteration(i, Y, nprev, indice, lag, coef_matrix):
    """Single iteration for parallel Post-OLS."""
    idx = nprev - i
    Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
    coef = coef_matrix[idx, :]
    coef = coef[~np.isnan(coef)]
    result = run_pols(Y_window, indice, lag, coef)
    return idx, result['pred']


def pols_rolling_window(Y, nprev, indice=1, lag=1, coef_matrix=None):
    """
    Rolling window Post-OLS forecasting (PARALLELIZED).
    
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
    coef_matrix : ndarray
        Matrix of coefficients from penalized regression (nprev x n_coef)
    
    Returns:
    --------
    dict : Dictionary with 'pred' and 'errors'
    """
    Y = np.array(Y)
    save_pred = np.full((nprev, 1), np.nan)
    
    # PARALLEL execution
    print(f"Running {nprev} Post-OLS iterations in parallel...")
    results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(_pols_single_iteration)(i, Y, nprev, indice, lag, coef_matrix)
        for i in range(nprev, 0, -1)
    )
    
    for idx, pred in results:
        save_pred[idx, 0] = pred
    
    # Calculate errors
    real = Y[:, indice - 1]  # Convert to 0-indexed
    actual = real[-nprev:]
    errors = calculate_errors(actual, save_pred.flatten())
    
    return {'pred': save_pred, 'errors': errors}


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    Y = np.random.randn(200, 5)
    
    print("Testing LASSO...")
    result = lasso_rolling_window(Y, nprev=10, indice=1, lag=1, alpha=1.0, model_type="lasso")
    print(f"LASSO RMSE: {result['errors']['rmse']:.4f}")
    
    print("\nTesting Adaptive LASSO...")
    result = lasso_rolling_window(Y, nprev=10, indice=1, lag=1, alpha=1.0, model_type="adalasso")
    print(f"AdaLASSO RMSE: {result['errors']['rmse']:.4f}")

