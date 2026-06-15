"""
Ridge Regression Rolling Window Forecasts
"""
import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Path constants for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Period configuration - without_dummy
# Valid periods: 1990_2000, 2016_2022, 2020_2022
PERIOD_CONFIG = {
    '1990_2000': {'nprev': 60},
    '2016_2022': {'nprev': 48},
    '2020_2022': {'nprev': 24},
}
CURRENT_PERIOD = '2016_2022'  # Change this to run different period

DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), f'rawdata_{CURRENT_PERIOD}.csv')
FORECAST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'forecasts')

from utils import load_csv, save_forecasts
from without_dummy.functions.func_lasso import lasso_rolling_window


def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    nprev = PERIOD_CONFIG[CURRENT_PERIOD]['nprev']
    alpha = 0  # alpha=0 for Ridge (pure L2)
    
    results = {}
    rmse_cpi = {}
    rmse_pce = {}
    
    # Ridge models
    print("Running Ridge models...")
    for lag in range(1, 13):
        print(f"  Ridge lag={lag}")
        results[f'ridge{lag}c'] = lasso_rolling_window(Y, nprev, indice=1, lag=lag, alpha=alpha, model_type='lasso')
        results[f'ridge{lag}p'] = lasso_rolling_window(Y, nprev, indice=2, lag=lag, alpha=alpha, model_type='lasso')
        rmse_cpi[lag] = results[f'ridge{lag}c']['errors']['rmse']
        rmse_pce[lag] = results[f'ridge{lag}p']['errors']['rmse']
    
    # Combine results for CPI (indice=1)
    cpi_ridge = np.column_stack([results[f'ridge{lag}c']['pred'] for lag in range(1, 13)])
    
    # Combine results for PCE (indice=2)
    pce_ridge = np.column_stack([results[f'ridge{lag}p']['pred'] for lag in range(1, 13)])
    
    # Create output directory
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_ridge, os.path.join(FORECAST_DIR, 'ridge-cpi.csv'))
    save_forecasts(pce_ridge, os.path.join(FORECAST_DIR, 'ridge-pce.csv'))
    
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
