"""
RF-LASSO Rolling Window Forecasts - Second Sample
"""
import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
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

from utils import load_csv, save_forecasts, add_outlier_dummy
from with_dummy.functions.func_rflasso import rflasso_rolling_window


def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    # Add dummy variable for outliers (COVID period) as in the R code
    Y = add_outlier_dummy(Y, target_col=0)
    
    nprev = PERIOD_CONFIG[CURRENT_PERIOD]['nprev']  # Out-of-sample 2001-2025
    np.random.seed(123)
    
    results = {}
    rmse_cpi = {}
    rmse_pce = {}
    
    # RF-LASSO models
    print("Running RF-LASSO models...")
    for lag in range(1, 13):
        print(f"  RFLASSO lag={lag}")
        results[f'rflasso{lag}c'] = rflasso_rolling_window(Y, nprev, indice=1, lag=lag)
        results[f'rflasso{lag}p'] = rflasso_rolling_window(Y, nprev, indice=2, lag=lag)
        rmse_cpi[lag] = results[f'rflasso{lag}c']['errors']['rmse']
        rmse_pce[lag] = results[f'rflasso{lag}p']['errors']['rmse']
    
    # Combine results
    cpi_rflasso = np.column_stack([results[f'rflasso{lag}c']['pred'] for lag in range(1, 13)])
    pce_rflasso = np.column_stack([results[f'rflasso{lag}p']['pred'] for lag in range(1, 13)])
    
    # Create output directory
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_rflasso, os.path.join(FORECAST_DIR, 'rflasso-cpi.csv'))
    save_forecasts(pce_rflasso, os.path.join(FORECAST_DIR, 'rflasso-pce.csv'))
    
    print(f"Done! Forecasts saved to {FORECAST_DIR}")
    # Print RMSE by horizon
    print("\nRMSE BY HORIZON:")
    print(f"{'Horizon':<8} {'CPI':<12} {'PCE':<12}")
    for h in range(1, 13):
        print(f"h={h:<6} {rmse_cpi.get(h, 0):<12.6f} {rmse_pce.get(h, 0):<12.6f}")
    print(f"Average: {np.mean(list(rmse_cpi.values())):.6f}  {np.mean(list(rmse_pce.values())):.6f}")

    
    return results


if __name__ == '__main__':
    main()
