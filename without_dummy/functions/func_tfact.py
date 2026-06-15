"""
Targeted Factor Model functions for inflation forecasting.
Uses pre-testing to select variables before factor extraction.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors, plot_forecast
from .func_fact import run_fact


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def baggit_pretest(X, y, fixed_controls=None, significance=0.1):
    """
    Pre-testing for variable selection using individual tests.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target variable
    fixed_controls : list
        Indices of fixed control variables
    significance : float
        Significance level for testing
    
    Returns:
    --------
    selected : ndarray
        Boolean array indicating selected variables
    """
    from scipy import stats
    
    n, p = X.shape
    selected = np.zeros(p)
    
    if fixed_controls is None:
        fixed_controls = []
    
    # Mark fixed controls as selected
    for fc in fixed_controls:
        if fc < p:
            selected[fc] = 1
    
    # Test each variable individually
    for j in range(p):
        if j in fixed_controls:
            continue
        
        # Include fixed controls and this variable
        vars_to_use = list(fixed_controls) + [j]
        X_subset = X[:, vars_to_use]
        
        model = LinearRegression()
        model.fit(X_subset, y)
        
        # t-test for the last coefficient (variable j)
        y_pred = model.predict(X_subset)
        residuals = y - y_pred
        mse = np.sum(residuals ** 2) / (n - len(vars_to_use) - 1)
        
        # Standard error of coefficient
        XtX_inv = np.linalg.pinv(np.dot(X_subset.T, X_subset))
        se = np.sqrt(mse * XtX_inv[-1, -1])
        
        if se > 0:
            t_stat = model.coef_[-1] / se
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - len(vars_to_use) - 1))
            
            if p_value < significance:
                selected[j] = 1
    
    return selected


def run_tfact(Y, indice, lag):
    """
    Run Targeted Factor Model for forecasting.
    
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
    dict : Dictionary with 'coef' and 'pred'
    """
    # Convert to 0-indexed
    indice_0 = indice - 1
    
    Y = np.array(Y)
    
    # Prepare data for pre-testing
    y_full = Y[:, indice_0]
    X_full = np.delete(Y, indice_0, axis=1)
    
    # Adjust for lag
    y = y_full[:len(y_full) - lag + 1]
    X = X_full[:X_full.shape[0] - lag + 1, :]
    
    # Create lagged target for controls
    y_embed = embed(y.reshape(-1, 1), 5)
    y_target = y_embed[:, 0]
    y_lags = y_embed[:, 1:]
    
    # Combine lags with other variables
    X_combined = np.column_stack([y_lags, X[4:, :]])
    
    # Pre-testing: select predictive variables
    pretest = baggit_pretest(X_combined, y_target, fixed_controls=list(range(4)))
    
    # Get selected indices for original Y (excluding target)
    pretest_vars = pretest[4:]  # Exclude lag controls
    
    # Reconstruct selection for full Y
    selected_cols = [indice_0]  # Always include target
    other_cols = [i for i in range(Y.shape[1]) if i != indice_0]
    for i, col in enumerate(other_cols):
        if i < len(pretest_vars) and pretest_vars[i] == 1:
            selected_cols.append(col)
    
    # If no variables selected, use all
    if len(selected_cols) < 2:
        selected_cols = list(range(Y.shape[1]))
    
    # Extract selected variables
    Y2 = Y[:, selected_cols]
    
    # Find new index of target variable
    new_indice = selected_cols.index(indice_0) + 1  # Convert back to 1-indexed
    
    # Run factor model on selected variables
    result = run_fact(Y2, new_indice, lag)
    
    return {'coef': result['coef'], 'pred': result['pred']}


def tfact_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window Targeted Factor Model forecasting.
    
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
    n_coef = 21  # Based on R code
    
    save_coef = np.full((nprev, n_coef), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        # Window selection
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        
        # Run Targeted Factor model
        result = run_tfact(Y_window, indice, lag)
        
        idx = nprev - i
        coef = result['coef']
        save_coef[idx, :min(len(coef), n_coef)] = coef[:n_coef]
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
    
    result = tfact_rolling_window(Y, nprev=10, indice=1, lag=1)
    print(f"TFactor RMSE: {result['errors']['rmse']:.4f}")
    print(f"TFactor MAE: {result['errors']['mae']:.4f}")
