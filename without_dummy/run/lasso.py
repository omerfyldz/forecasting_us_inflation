"""
LASSO Rolling Window Forecasts
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
from without_dummy.functions.func_lasso import lasso_rolling_window, pols_rolling_window


def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    nprev = PERIOD_CONFIG[CURRENT_PERIOD]['nprev']
    alpha = 1  # alpha=1 for LASSO
    
    results = {}
    rmse_cpi = {}
    rmse_pce = {}
    pols_results = {}
    rmse_cpi = {}
    rmse_pce = {}
    
    # LASSO models
    print("Running LASSO models...")
    for lag in range(1, 13):
        print(f"  LASSO lag={lag}")
        results[f'lasso{lag}c'] = lasso_rolling_window(Y, nprev, indice=1, lag=lag, alpha=alpha, model_type='lasso')
        results[f'lasso{lag}p'] = lasso_rolling_window(Y, nprev, indice=2, lag=lag, alpha=alpha, model_type='lasso')
        rmse_cpi[lag] = results[f'lasso{lag}c']['errors']['rmse']
        rmse_pce[lag] = results[f'lasso{lag}p']['errors']['rmse']
    
    # POLS (Post-OLS) models using LASSO selected variables
    print("Running POLS models...")
    for lag in range(1, 13):
        print(f"  POLS lag={lag}")
        pols_results[f'pols_lasso{lag}c'] = pols_rolling_window(Y, nprev, indice=1, lag=lag, 
                                                                  coef_matrix=results[f'lasso{lag}c']['coef'])
        pols_results[f'pols_lasso{lag}p'] = pols_rolling_window(Y, nprev, indice=2, lag=lag,
                                                                  coef_matrix=results[f'lasso{lag}p']['coef'])
    
    # Combine results for CPI (indice=1)
    cpi_lasso = np.column_stack([results[f'lasso{lag}c']['pred'] for lag in range(1, 13)])
    cpi_pols = np.column_stack([pols_results[f'pols_lasso{lag}c']['pred'] for lag in range(1, 13)])
    
    # Combine results for PCE (indice=2)
    pce_lasso = np.column_stack([results[f'lasso{lag}p']['pred'] for lag in range(1, 13)])
    pce_pols = np.column_stack([pols_results[f'pols_lasso{lag}p']['pred'] for lag in range(1, 13)])
    
    # Create output directory
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_lasso, os.path.join(FORECAST_DIR, 'lasso-cpi.csv'))
    save_forecasts(pce_lasso, os.path.join(FORECAST_DIR, 'lasso-pce.csv'))
    save_forecasts(cpi_pols, os.path.join(FORECAST_DIR, 'pols-lasso-cpi.csv'))
    save_forecasts(pce_pols, os.path.join(FORECAST_DIR, 'pols-lasso-pce.csv'))
    
    print(f"Done! Forecasts saved to {FORECAST_DIR}")
    # Print RMSE by horizon
    print("\nRMSE BY HORIZON:")
    print(f"{'Horizon':<8} {'CPI':<12} {'PCE':<12}")
    for h in range(1, 13):
        print(f"h={h:<6} {rmse_cpi.get(h, 0):<12.6f} {rmse_pce.get(h, 0):<12.6f}")
    print(f"Average: {np.mean(list(rmse_cpi.values())):.6f}  {np.mean(list(rmse_pce.values())):.6f}")

    
    return results, pols_results


if __name__ == '__main__':
    main()
