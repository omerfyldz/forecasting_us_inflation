"""
Random Walk (RW) forecasting for second sample
"""

import os
import sys
import numpy as np

# Set up absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Period configuration - with_dummy
# Valid periods: 2001_2015, 1990_2022
PERIOD_CONFIG = {
    '2001_2015': {'nprev': 84},
    '1990_2022': {'nprev': 132},
}
CURRENT_PERIOD = '1990_2022'  # Change this to run different period

DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), f'rawdata_{CURRENT_PERIOD}.csv')
FORECAST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'forecasts')

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import load_csv, save_forecasts
from with_dummy.functions.func_rw import rw_rolling_window

# Parameters
nprev = PERIOD_CONFIG[CURRENT_PERIOD]['nprev']  # Out-of-sample forecasts for second sample (consistent with other models)

# Load data
Y = load_csv(DATA_PATH)

# Create forecast directory if it doesn't exist
os.makedirs(FORECAST_DIR, exist_ok=True)

# CPI forecasts (indice=1)
print("Running Random Walk for CPI...")
cpi_forecasts = {}
for lag in range(1, 13):
    print(f"  Lag {lag}...")
    result = rw_rolling_window(Y, nprev, indice=1, lag=lag)
    cpi_forecasts[f'lag_{lag}'] = result['pred'].flatten()
    print(f"    RMSE: {result['errors']['rmse']:.4f}, MAE: {result['errors']['mae']:.4f}")

# Save CPI forecasts
cpi_file = os.path.join(FORECAST_DIR, 'rw-cpi.csv')
save_forecasts(cpi_forecasts, cpi_file)
print(f"CPI forecasts saved to: {cpi_file}")

# PCE forecasts (indice=2)
print("\nRunning Random Walk for PCE...")
pce_forecasts = {}
for lag in range(1, 13):
    print(f"  Lag {lag}...")
    result = rw_rolling_window(Y, nprev, indice=2, lag=lag)
    pce_forecasts[f'lag_{lag}'] = result['pred'].flatten()
    print(f"    RMSE: {result['errors']['rmse']:.4f}, MAE: {result['errors']['mae']:.4f}")

# Save PCE forecasts
pce_file = os.path.join(FORECAST_DIR, 'rw-pce.csv')
save_forecasts(pce_forecasts, pce_file)
print(f"PCE forecasts saved to: {pce_file}")

print("\nRandom Walk forecasting complete!")
