"""
Utility functions for time series forecasting.
Common helper functions used across all forecasting methods.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def embed(data, dimension):
    """
    Create lagged matrix similar to R's embed function.
    
    Parameters:
    -----------
    data : array-like
        Input time series data (1D or 2D array)
    dimension : int
        Number of lags + 1 (embedding dimension)
    
    Returns:
    --------
    embedded : ndarray
        Matrix with lagged values
    """
    data = np.array(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n_rows = data.shape[0] - dimension + 1
    n_cols = data.shape[1]
    
    result = np.zeros((n_rows, n_cols * dimension))
    
    for i in range(dimension):
        result[:, i*n_cols:(i+1)*n_cols] = data[dimension-1-i:n_rows+dimension-1-i, :]
    
    return result


def compute_pca_scores(Y, n_components=4, scale=False):
    """
    Compute PCA scores similar to R's princomp.
    
    **DATA LEAKAGE WARNING:**
    This function computes PCA on the ENTIRE dataset Y.
    For forecasting applications, this may include future information
    that should not be available at prediction time.
    
    Consider computing PCA only on training data and applying
    the transformation to test data separately.
    
    Parameters:
    -----------
    Y : array-like
        Input data matrix
    n_components : int
        Number of principal components to retain
    scale : bool
        Whether to scale data (only center by default like R's princomp)
    
    Returns:
    --------
    scores : ndarray
        PCA scores
    Y_filled : ndarray
        Data with NaN values filled (returned as second value)
    """
    Y = np.array(Y)
    
    # Handle missing values using forward fill then backward fill
    # This is more aligned with time series data assumptions
    Y_filled = Y.copy()
    
    # Forward fill (use previous value for NaN)
    for col in range(Y.shape[1]):
        mask = np.isnan(Y_filled[:, col])
        if mask.any():
            # Forward fill
            idx = np.where(~mask)[0]
            if len(idx) > 0:
                Y_filled[:, col] = np.interp(
                    np.arange(len(Y_filled[:, col])),
                    idx,
                    Y_filled[idx, col],
                    left=np.nan, right=np.nan
                )
            # If still NaN at beginning, use column mean
            if np.isnan(Y_filled[0, col]):
                col_mean = np.nanmean(Y_filled[:, col])
                Y_filled[np.isnan(Y_filled[:, col]), col] = col_mean if not np.isnan(col_mean) else 0
    
    # Center the data (like R's scale with scale=FALSE)
    Y_centered = Y_filled - np.mean(Y_filled, axis=0)
    
    pca = PCA(n_components=min(n_components, Y.shape[1]))
    scores = pca.fit_transform(Y_centered)
    
    return scores, Y_filled


def calculate_rmse(actual, predicted):
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_mae(actual, predicted):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(actual - predicted))


def calculate_errors(actual, predicted):
    """
    Calculate RMSE and MAE.
    
    Returns:
    --------
    dict : Dictionary with 'rmse' and 'mae' keys
    """
    return {
        'rmse': calculate_rmse(actual, predicted),
        'mae': calculate_mae(actual, predicted)
    }


def prepare_forecast_data(Y, indice, lag, n_pca_components=4):
    """
    Prepare data for forecasting models.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    indice : int
        Column index of target variable (0-indexed)
    lag : int
        Forecast horizon
    n_pca_components : int
        Number of PCA components to add
    
    Returns:
    --------
    tuple : (y, X, X_out) - target, features, out-of-sample features
    """
    Y = np.array(Y)
    
    # Compute PCA scores (returns tuple: scores, Y_filled)
    scores, _ = compute_pca_scores(Y, n_components=n_pca_components, scale=False)
    
    # Combine original data with PCA scores
    Y2 = np.column_stack([Y, scores])
    
    # Create embedded matrix
    aux = embed(Y2, 4)
    
    # Extract target variable
    y = aux[:, indice]
    
    # Extract features (exclude first lag columns)
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
    
    return y, X, X_out


def plot_forecast(real, predictions, nprev, title="Forecast vs Actual"):
    """
    Plot actual vs predicted values.
    
    Parameters:
    -----------
    real : array-like
        Actual values
    predictions : array-like
        Predicted values
    nprev : int
        Number of predictions
    title : str
        Plot title
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(real, 'b-', label='Actual')
    
    # Create prediction series with NAs at the beginning
    pred_series = np.full(len(real), np.nan)
    pred_series[-nprev:] = predictions.flatten()
    plt.plot(pred_series, 'r-', label='Predicted')
    
    plt.title(title)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()


def load_rdata(filepath):
    """
    Load R data file (.rda or .RData).
    
    Parameters:
    -----------
    filepath : str
        Path to the R data file
    
    Returns:
    --------
    dict : Dictionary with data objects
    """
    import pyreadr
    result = pyreadr.read_r(filepath)
    return result


def save_forecasts(data, filepath, sep=";"):
    """
    Save forecasts to CSV file.
    
    Parameters:
    -----------
    data : array-like
        Forecast data to save
    filepath : str
        Output file path
    sep : str
        Separator (default semicolon like R code)
    """
    df = pd.DataFrame(data)
    df.to_csv(filepath, sep=sep, header=False, index=False)


def load_csv(filepath, sep=",", header=None):
    """
    Load prepared data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    sep : str
        Separator (default comma)
    header : int or None
        Row number to use as column names (None if no header)
    
    Returns:
    --------
    ndarray : Data as numpy array
    """
    df = pd.read_csv(filepath, sep=sep, header=header)
    return df.values


def add_outlier_dummy(Y, target_col=0):
    """
    Add dummy variable for outliers (second sample methodology).
    Creates a dummy variable that equals 1 at the minimum value of the target column.
    
    This is used in the second sample to handle the COVID-19 outlier in inflation data,
    as described in the paper methodology.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    target_col : int
        Column index to find the minimum value (0-indexed, default is column 1/CPI)
    
    Returns:
    --------
    Y_with_dummy : ndarray
        Data matrix with appended dummy column
    
    Example:
    --------
    >>> Y = load_csv('rawdata_1990_2022.csv')
    >>> Y = add_outlier_dummy(Y, target_col=0)  # CPI is column 0 (1 in R indexing)
    """
    Y = np.array(Y)
    n_rows = Y.shape[0]
    
    # Create dummy variable (all zeros)
    dum = np.zeros(n_rows)
    
    # Set 1 at the index of minimum value in target column
    min_idx = np.argmin(Y[:, target_col])
    dum[min_idx] = 1
    
    # Append dummy column to data
    Y_with_dummy = np.column_stack([Y, dum])
    
    return Y_with_dummy
