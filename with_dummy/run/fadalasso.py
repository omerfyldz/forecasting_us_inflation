"""
FLASSO (with dummy variable) Rolling Window Forecasts - Second Sample
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
from with_dummy.functions.func_flasso import flasso_rolling_window, pols_dummy_rolling_window


def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    nprev = PERIOD_CONFIG[CURRENT_PERIOD]['nprev']  # Out-of-sample 2001-2025
    alpha = 1
    
    results = {}
    rmse_cpi = {}
    rmse_pce = {}
    pols_results = {}
    rmse_cpi = {}
    rmse_pce = {}
    
    # LASSO with dummy models
    print("Running LASSO with dummy models...")
    for lag in range(1, 13):
        print(f"  FLASSO lag={lag}")
        results[f'flasso{lag}c'] = flasso_rolling_window(Y, nprev, indice=1, lag=lag, alpha=alpha, type_='lasso')
        results[f'flasso{lag}p'] = flasso_rolling_window(Y, nprev, indice=2, lag=lag, alpha=alpha, type_='lasso')
        rmse_cpi[lag] = results[f'flasso{lag}c']['errors']['rmse']
        rmse_pce[lag] = results[f'flasso{lag}p']['errors']['rmse']
    
    # POLS models
    print("Running POLS models...")
    for lag in range(1, 13):
        print(f"  POLS lag={lag}")
        pols_results[f'pols{lag}c'] = pols_dummy_rolling_window(Y, nprev, indice=1, lag=lag, 
                                                                 coefs=results[f'flasso{lag}c']['coef'])
        pols_results[f'pols{lag}p'] = pols_dummy_rolling_window(Y, nprev, indice=2, lag=lag,
                                                                 coefs=results[f'flasso{lag}p']['coef'])
    
    # Combine results
    cpi_flasso = np.column_stack([results[f'flasso{lag}c']['pred'] for lag in range(1, 13)])
    cpi_pols = np.column_stack([pols_results[f'pols{lag}c']['pred'] for lag in range(1, 13)])
    
    pce_flasso = np.column_stack([results[f'flasso{lag}p']['pred'] for lag in range(1, 13)])
    pce_pols = np.column_stack([pols_results[f'pols{lag}p']['pred'] for lag in range(1, 13)])
    
    # Create output directory
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_flasso, os.path.join(FORECAST_DIR, 'flasso-cpi.csv'))
    save_forecasts(pce_flasso, os.path.join(FORECAST_DIR, 'flasso-pce.csv'))
    save_forecasts(cpi_pols, os.path.join(FORECAST_DIR, 'pols-flasso-cpi.csv'))
    save_forecasts(pce_pols, os.path.join(FORECAST_DIR, 'pols-flasso-pce.csv'))
    
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
