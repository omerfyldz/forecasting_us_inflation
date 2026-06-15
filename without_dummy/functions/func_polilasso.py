"""
Polynomial LASSO for inflation forecasting.
Creates polynomial interactions of features before applying LASSO.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors, plot_forecast
from .func_lasso import ICGlmnet


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def run_polilasso(Y, indice, lag, alpha=1.0, model_type="lasso"):
    """
    Run Polynomial LASSO model for forecasting.
    Creates polynomial features (interactions) before applying LASSO.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    alpha : float
        Mixing parameter
    model_type : str
        "lasso" or "adalasso"
    
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
    
    # Create polynomial features (interactions within each lag group)
    n_obs = X.shape[0]
    z = X.shape[1] // 4  # Features per lag
    
    poly_features = []
    poly_features_out = []
    
    for u in range(4):  # 4 lag groups
        start = u * z
        end = min((u + 1) * z, X.shape[1])
        
        comb0 = X[:, start:end]
        comb0_out = X_out[start:end]
        
        # Create interactions within this lag group
        n_vars = comb0.shape[1]
        for i in range(n_vars):
            for j in range(n_vars):
                poly_features.append(comb0[:, i] * comb0[:, j])
                poly_features_out.append(comb0_out[i] * comb0_out[j])
    
    X_poly = np.column_stack(poly_features)
    X_out_poly = np.array(poly_features_out)
    
    # Fit model
    model = ICGlmnet(alpha=alpha)
    model.fit(X_poly, y)
    coef = model.coef
    
    if model_type == "adalasso":
        penalty = (np.abs(coef[1:]) + 1/np.sqrt(len(y))) ** (-1)
        model = ICGlmnet(alpha=alpha)
        model.fit(X_poly, y, penalty_factor=penalty)
    
    # Make prediction
    pred = model.predict(X_out_poly)
    pred = float(pred.flatten()[0]) if hasattr(pred, 'flatten') else float(pred)
    
    return {'model': model, 'pred': pred}


def polilasso_rolling_window(Y, nprev, indice=1, lag=1, alpha=1.0, model_type="lasso"):
    """
    Rolling window Polynomial LASSO forecasting.
    
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
        "lasso" or "adalasso"
    
    Returns:
    --------
    dict : Dictionary with 'pred', 'coef', and 'errors'
    """
    Y = np.array(Y)
    
    # Estimate coefficient size
    z = (Y.shape[1] + 4) // 4
    n_coef = 1 + (z ** 2) * 4
    
    save_coef = np.full((nprev, n_coef), np.nan)
    save_pred = np.full((nprev, 1), np.nan)
    
    def _single_iteration(i):
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        result = run_polilasso(Y_window, indice, lag, alpha, model_type)
        idx = nprev - i
        return idx, result['pred'], result['model'].coef
    
    print(f"Running {nprev} Polynomial LASSO iterations in parallel (N_JOBS={N_JOBS})...")
    
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_single_iteration)(i) for i in range(nprev, 0, -1)
    )
    
    for idx, pred, coef in results:
        save_coef[idx, :min(len(coef), n_coef)] = coef[:n_coef]
        save_pred[idx, 0] = pred
    
    print(f"Completed {nprev} iterations.")
    
    # Calculate errors
    real = Y[:, indice - 1]  # Convert to 0-indexed
    actual = real[-nprev:]
    errors = calculate_errors(actual, save_pred.flatten())
    
    return {'pred': save_pred, 'coef': save_coef, 'errors': errors}


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    Y = np.random.randn(200, 5)
    
    result = polilasso_rolling_window(Y, nprev=10, indice=1, lag=1, model_type="adalasso")
    print(f"PolyLASSO RMSE: {result['errors']['rmse']:.4f}")
    print(f"PolyLASSO MAE: {result['errors']['mae']:.4f}")

