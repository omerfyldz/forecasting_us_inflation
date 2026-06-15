"""
Random Forest functions for inflation forecasting.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors, plot_forecast


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def run_rf(Y, indice, lag):
    """
    Run Random Forest model for forecasting.
    
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
    indice = indice - 1
    
    Y = np.array(Y)
    
    # Compute PCA scores and get filled Y
    scores, Y_filled = compute_pca_scores(Y, n_components=4, scale=False)
    
    # Combine filled data with PCA scores
    Y2 = np.column_stack([Y_filled, scores])
    
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
    
    # Fit Random Forest
    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Make prediction
    pred = model.predict(X_out.reshape(1, -1))[0]
    
    return {'model': model, 'pred': pred}


def rf_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window Random Forest forecasting.
    
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
    dict : Dictionary with 'pred', 'errors', and 'save_importance'
    """
    Y = np.array(Y)
    def process_single_iteration(i, Y, nprev, indice, lag):
        """Process a single iteration - designed for parallel execution."""
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        result = run_rf(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred'], result['model'].feature_importances_
    
    print(f"Running {nprev} RF iterations in parallel (N_JOBS={N_JOBS})...")
    
    # Parallel execution
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_single_iteration)(i, Y, nprev, indice, lag)
        for i in range(nprev, 0, -1)
    )
    
    # Collect results
    save_pred = np.full((nprev, 1), np.nan)
    save_importance = [None] * nprev
    for idx, pred, importance in results:
        save_pred[idx, 0] = pred
        save_importance[idx] = importance
    
    # Calculate errors
    real = Y[:, indice - 1]  # Convert to 0-indexed
    actual = real[-nprev:]
    errors = calculate_errors(actual, save_pred.flatten())
    
    return {'pred': save_pred, 'errors': errors, 'save_importance': save_importance}


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(123)
    Y = np.random.randn(200, 5)
    
    result = rf_rolling_window(Y, nprev=10, indice=1, lag=1)
    print(f"RF RMSE: {result['errors']['rmse']:.4f}")
    print(f"RF MAE: {result['errors']['mae']:.4f}")
