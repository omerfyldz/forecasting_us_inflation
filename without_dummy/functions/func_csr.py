"""
Complete Subset Regression (CSR) functions for inflation forecasting.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import combinations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors, plot_forecast


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def csr(X, y, fixed_controls=None, k_max=None):
    """
    Complete Subset Regression.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target variable
    fixed_controls : list or ndarray
        Indices of fixed control variables (0-indexed)
    k_max : int
        Maximum subset size
    
    Returns:
    --------
    dict : Model with averaged coefficients and predictions
    """
    n, p = X.shape
    
    if k_max is None:
        k_max = min(p, n // 10)
    
    if fixed_controls is None:
        fixed_controls = []
    else:
        fixed_controls = list(fixed_controls)
    
    # Variables available for selection (excluding fixed controls)
    available = [i for i in range(p) if i not in fixed_controls]
    
    # Store predictions from all models
    all_predictions = []
    all_weights = []
    
    # Fixed controls part
    X_fixed = X[:, fixed_controls] if fixed_controls else np.zeros((n, 0))
    
    # Iterate through subset sizes
    for k in range(1, min(k_max, len(available)) + 1):
        for subset in combinations(available, k):
            # Combine fixed controls with subset
            selected = list(fixed_controls) + list(subset)
            X_subset = X[:, selected]
            
            # Fit model
            model = LinearRegression()
            model.fit(X_subset, y)
            
            # Calculate BIC for weighting
            y_pred = model.predict(X_subset)
            residuals = y - y_pred
            mse = np.mean(residuals ** 2)
            n_params = len(selected) + 1
            bic = n * np.log(mse) + n_params * np.log(n)
            
            # Weight based on BIC
            weight = np.exp(-0.5 * bic)
            
            all_predictions.append(y_pred)
            all_weights.append(weight)
    
    # Normalize weights
    weights = np.array(all_weights)
    weights = weights / np.sum(weights)
    
    # Averaged predictions
    predictions = np.zeros(n)
    for pred, w in zip(all_predictions, weights):
        predictions += w * pred
    
    # Store for out-of-sample prediction
    return {
        'weights': weights,
        'fitted': predictions,
        'models': None  # Would need to store models for prediction
    }


def run_csr(Y, indice, lag):
    """
    Run CSR model for forecasting.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    
    Returns:
    --------
    dict : Dictionary with 'model' and 'pred'
    """
    # Convert to 0-indexed
    indice_0 = indice - 1
    
    Y = np.array(Y)
    
    # Compute PCA scores (returns tuple: scores, Y_filled)
    scores, _ = compute_pca_scores(Y, n_components=4, scale=False)
    
    # Combine original data with PCA scores
    Y2 = np.column_stack([Y, scores])
    
    # Create embedded matrix
    aux = embed(Y2, 4)
    y = aux[:, indice_0]
    n_cols_Y2 = Y2.shape[1]
    X = aux[:, n_cols_Y2 * lag:]
    
    # Fixed controls: target variable at each lag
    f_seq = list(range(indice_0, X.shape[1], n_cols_Y2))
    
    # Prepare out-of-sample data
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        aux_trimmed = aux[:, :aux.shape[1] - n_cols_Y2 * (lag - 1)]
        X_out = aux_trimmed[-1, :X.shape[1]]
    
    # Adjust for lag
    y = y[:len(y) - lag + 1]
    X = X[:X.shape[0] - lag + 1, :]
    
    # Simplified CSR: use best subset based on BIC
    n = len(y)
    best_bic = np.inf
    best_model = None
    best_selected = None
    
    # Always include fixed controls, try adding other variables
    available = [i for i in range(X.shape[1]) if i not in f_seq]
    
    # Try different subsets
    for k in range(min(5, len(available) + 1)):
        if k == 0:
            subsets = [[]]
        else:
            subsets = list(combinations(available, k))
        
        for subset in subsets:
            selected = list(f_seq) + list(subset)
            selected = [s for s in selected if s < X.shape[1]]
            
            if len(selected) == 0:
                continue
                
            X_subset = X[:, selected]
            model = LinearRegression()
            model.fit(X_subset, y)
            
            y_pred = model.predict(X_subset)
            residuals = y - y_pred
            mse = np.mean(residuals ** 2)
            n_params = len(selected) + 1
            bic = n * np.log(mse) + n_params * np.log(n)
            
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_selected = selected
    
    # Make prediction
    X_out_selected = X_out[best_selected]
    pred = best_model.predict(X_out_selected.reshape(1, -1))[0]
    
    return {'model': best_model, 'pred': pred}


def csr_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window CSR forecasting.
    
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
    dict : Dictionary with 'pred' and 'errors'
    """
    Y = np.array(Y)
    def process_single_iteration(i, Y, nprev, indice, lag):
        """Process a single iteration - designed for parallel execution."""
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        result = run_csr(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred']
    
    print(f"Running {nprev} CSR iterations in parallel (N_JOBS={N_JOBS})...")
    
    # Parallel execution
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_single_iteration)(i, Y, nprev, indice, lag)
        for i in range(nprev, 0, -1)
    )
    
    # Collect results
    save_pred = np.full((nprev, 1), np.nan)
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
    
    result = csr_rolling_window(Y, nprev=10, indice=1, lag=1)
    print(f"CSR RMSE: {result['errors']['rmse']:.4f}")
    print(f"CSR MAE: {result['errors']['mae']:.4f}")
