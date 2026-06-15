"""
Adaptive LASSO + Random Forest forecasting run script for second sample
Runs adalasso-rf with dummy variable for all lags (PCE only based on R code)
"""
import sys
import os
import numpy as np
import pandas as pd

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Path constants for absolute paths
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

from with_dummy.functions.func_adalassorf import adalasso_rf_rolling_window
from utils import load_csv, save_forecasts, add_outlier_dummy

def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    # Add dummy variable for outliers (COVID period) as in the R code
    Y = add_outlier_dummy(Y, target_col=0)
    
    nprev = PERIOD_CONFIG[CURRENT_PERIOD]['nprev']  # Out-of-sample 2001-2025
    
    print("Running AdaLASSO+RF forecasts (second sample with dummy)...")
    
    # Storage for results - Note: R code only runs PCE for this model
    pce_results = []
    
    # Run for each lag (only PCE based on original R code)
    for lag in range(1, 13):
        print(f"  Lag {lag}/12...")
        
        # PCE (indice=2) only
        result_pce = adalasso_rf_rolling_window(Y, nprev, indice=2, lag=lag)
        pce_results.append(result_pce['pred'])
    
    # Combine results
    pce = np.column_stack(pce_results)
    
    # Save forecasts
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    save_forecasts(pce, os.path.join(FORECAST_DIR, 'adalassorf-pce.csv'))
    
    print(f"AdaLASSO+RF forecasts saved to {FORECAST_DIR}")

if __name__ == "__main__":
    main()
