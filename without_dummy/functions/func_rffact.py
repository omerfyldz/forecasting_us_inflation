"""
Random Forest Factors (RFFACT) functions for inflation forecasting.
Uses importance weights from previous Random Forest runs to construct factors.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import embed, calculate_errors


def run_rffact(Y, indice, lag, importance_weights, n_factors=4):
    """
    Run Random Forest Factor model for forecasting.
    Uses RF importance weights to construct weighted factors.

    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    importance_weights : ndarray
        Variable importance weights from previous RF run (nprev x n_features)
    n_factors : int
        Number of factors to extract

    Returns:
    --------
    dict : Dictionary with 'pred' and 'model'
    """
    # Convert to 0-indexed
    indice = indice - 1

    Y = np.array(Y)
    n_obs, n_vars = Y.shape

    # Get the latest importance weights (use mean if multiple)
    if importance_weights.ndim == 2:
        weights = np.mean(importance_weights, axis=0)
    else:
        weights = importance_weights

    # Normalize weights
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(weights)) / len(weights)

    # Create weighted data matrix
    # Expand weights to match Y columns (importance is typically for lagged variables)
    if len(weights) > n_vars:
        # Weights include lagged variables - use only first n_vars
        var_weights = weights[:n_vars]
    elif len(weights) < n_vars:
        # Pad with equal weights
        var_weights = np.concatenate([weights, np.ones(n_vars - len(weights)) / (n_vars - len(weights))])
    else:
        var_weights = weights

    # Apply weights to standardized data
    Y_std = (Y - np.mean(Y, axis=0)) / (np.std(Y, axis=0) + 1e-10)
    Y_weighted = Y_std * np.sqrt(var_weights)

    # Extract factors using PCA on weighted data
    n_factors_use = min(n_factors, n_vars - 1, n_obs - 1)
    pca = PCA(n_components=n_factors_use)
    factors = pca.fit_transform(Y_weighted)

    # Combine original target with factors
    Y2 = np.column_stack([Y[:, indice], factors])

    # Create embedded matrix for forecasting
    aux = embed(Y2, 4)
    y = aux[:, 0]  # Target is first column
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

    # Fit OLS model
    model = LinearRegression()
    model.fit(X, y)

    # Predict
    pred = model.predict(X_out.reshape(1, -1))[0]

    return {'pred': pred, 'model': model, 'factors': factors}


def rffact_rolling_window(Y, nprev, indice, lag, importance_matrix, n_factors=4):
    """
    Rolling window forecasting using RF-weighted factors.

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
    importance_matrix : ndarray
        Variable importance matrix from RF (nprev x n_features)
    n_factors : int
        Number of factors to extract

    Returns:
    --------
    dict : Dictionary with 'pred' and 'errors'
    """
    Y = np.array(Y)
    importance_matrix = np.array(importance_matrix)

    save_pred = np.full((nprev, 1), np.nan)

    for i in range(nprev, 0, -1):
        # Window selection
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]

        # Get corresponding importance weights
        # Use the importance from the same iteration of the RF run
        iter_idx = nprev - i
        if importance_matrix.ndim == 2 and iter_idx < importance_matrix.shape[0]:
            importance_weights = importance_matrix[iter_idx, :]
        else:
            importance_weights = importance_matrix

        # Run RF-factor model
        result = run_rffact(Y_window, indice, lag, importance_weights, n_factors)

        save_pred[iter_idx, 0] = result['pred']

        print(f"iteration {iter_idx + 1}")

    # Calculate errors
    real = Y[:, indice - 1]  # Convert to 0-indexed
    actual = real[-nprev:]
    errors = calculate_errors(actual, save_pred.flatten())

    return {'pred': save_pred, 'errors': errors}


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    Y = np.random.randn(200, 10)
    
    # Simulate importance weights from RF
    importance = np.random.rand(50, 10)
    
    result = rffact_rolling_window(Y, nprev=10, indice=1, lag=1, 
                                    importance_matrix=importance)
    print(f"RFFACT RMSE: {result['errors']['rmse']:.4f}")
