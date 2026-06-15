
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def calculate_zscore(series, window=12):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def check_stationarity(series):
    # ADF Test
    try:
        result = adfuller(series.dropna())
        return {'stationary': result[1] < 0.05, 'p_value': result[1]}
    except:
        return {'stationary': False, 'p_value': 1.0}

def calculate_rolling_statistics(series, windows):
    stats = pd.DataFrame(index=series.index)
    for w in windows:
        stats[f'mean_{w}'] = series.rolling(w).mean()
        stats[f'std_{w}'] = series.rolling(w).std()
        stats[f'max_{w}'] = series.rolling(w).max()
        stats[f'min_{w}'] = series.rolling(w).min()
    return stats

def calculate_momentum_features(series, horizons):
    momentum = pd.DataFrame(index=series.index)
    for h in horizons:
        momentum[f'roc_{h}'] = series.pct_change(h)
        momentum[f'acc_{h}'] = series.diff(h).diff(h)
    return momentum

def calculate_volatility_features(series, windows):
    vol = pd.DataFrame(index=series.index)
    for w in windows:
        # Standard deviation of returns
        vol[f'vol_{w}'] = series.pct_change().rolling(w).std()
    return vol

def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def create_lagged_features(df, lags):
    lagged = pd.DataFrame(index=df.index)
    for col in df.columns:
        for lag in lags:
            lagged[f'{col}_lag{lag}'] = df[col].shift(lag)
    return lagged

def exponential_smoothing(series, alpha=0.3):
    return series.ewm(alpha=alpha).mean()
