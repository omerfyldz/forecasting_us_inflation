"""
Bagging (Bootstrap Aggregation) functions for inflation forecasting.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors, plot_forecast


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def pre_testing_group_joint(X, y, significance=0.05):
    """
    Group-joint pre-testing for variable selection.
    Tests groups of variables jointly.

    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target variable
    significance : float
        Significance level for testing

    Returns:
    --------
    selected : ndarray
        Boolean array of selected features
    """
    from scipy import stats

    n, p = X.shape
    selected = np.ones(p, dtype=bool)

    # Fit full model
    model_full = LinearRegression()
    model_full.fit(X, y)
    y_pred_full = model_full.predict(X)
    sse_full = np.sum((y - y_pred_full) ** 2)

    # Test each variable
    for j in range(p):
        X_reduced = np.delete(X, j, axis=1)
        model_reduced = LinearRegression()
        model_reduced.fit(X_reduced, y)
        y_pred_reduced = model_reduced.predict(X_reduced)
        sse_reduced = np.sum((y - y_pred_reduced) ** 2)

        # F-test
        df1 = 1
        df2 = n - p
        if df2 > 0 and sse_full > 0:
            f_stat = ((sse_reduced - sse_full) / df1) / (sse_full / df2)
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)
            if p_value > significance:
                selected[j] = False

    # Ensure at least one variable is selected
    if not np.any(selected):
        # Select top 5 variables with highest absolute correlation
        correlations = np.abs([np.corrcoef(X[:, j], y)[0, 1] for j in range(p)])
        top_indices = np.argsort(correlations)[-min(5, p):]
        selected[top_indices] = True

    return selected


def run_bagg(Y, indice, lag, R=100, l=5):
    """
    Run Bagging model for forecasting.

    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    R : int
        Number of bootstrap replications
    l : int
        Number of variables per model

    Returns:
    --------
    dict : Dictionary with 'model', 'pred', and 'nselect'
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

    n, p = X.shape
    predictions = []
    coef_matrix = np.zeros((R, p))

    # Pre-testing for variable selection
    selected_vars = pre_testing_group_joint(X, y)
    X_selected = X[:, selected_vars]
    X_out_selected = X_out[selected_vars]
    p_selected = X_selected.shape[1]

    for r in range(R):
        # Bootstrap sample
        indices = np.random.choice(n, size=n, replace=True)
        X_boot = X_selected[indices, :]
        y_boot = y[indices]

        # Random subset of variables
        n_vars = min(l, p_selected)
        var_indices = np.random.choice(p_selected, size=n_vars, replace=False)

        # Fit model
        model = LinearRegression()
        model.fit(X_boot[:, var_indices], y_boot)

        # Predict
        pred = model.predict(X_out_selected[var_indices].reshape(1, -1))[0]
        predictions.append(pred)

        # Store coefficient selection
        full_indices = np.where(selected_vars)[0][var_indices]
        coef_matrix[r, full_indices] = 1

    # Aggregate predictions
    final_pred = np.mean(predictions)

    # Count selections
    nselect = np.sum(coef_matrix, axis=0)

    return {'model': None, 'pred': final_pred, 'nselect': nselect}


def bagg_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window Bagging forecasting (PARALLELIZED).
    
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
    dict : Dictionary with 'pred', 'errors', and 'nselect'
    """
    Y = np.array(Y)
    n_features = 21 + (Y.shape[1] - 1) * 4

    save_nselect = np.full((nprev, n_features), np.nan)
    save_pred = np.full((nprev, 1), np.nan)

    def _single_iteration(i):
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        result = run_bagg(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred'], result['nselect']
    
    print(f"Running {nprev} Bagging iterations in parallel (N_JOBS={N_JOBS})...")
    
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_single_iteration)(i) for i in range(nprev, 0, -1)
    )
    
    for idx, pred, nselect in results:
        save_pred[idx, 0] = pred
        save_nselect[idx, :len(nselect)] = nselect
    
    print(f"Completed {nprev} iterations.")

    # Calculate errors
    real = Y[:, indice - 1]  # Convert to 0-indexed
    actual = real[-nprev:]
    errors = calculate_errors(actual, save_pred.flatten())

    return {'pred': save_pred, 'errors': errors, 'nselect': save_nselect}


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    Y = np.random.randn(200, 5)

    result = bagg_rolling_window(Y, nprev=10, indice=1, lag=1)
    print(f"Bagging RMSE: {result['errors']['rmse']:.4f}")
