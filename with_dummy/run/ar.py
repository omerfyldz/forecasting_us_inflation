"""
AR Model Rolling Window Forecasts - Second Sample
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
from with_dummy.functions.func_ar import ar_rolling_window


def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    # Add dummy variable for outliers (COVID period) as in the R code
    # dum[which.min(Y[,1])]=1  # Creates dummy at minimum CPI value
    Y = add_outlier_dummy(Y, target_col=0)
    
    nprev = PERIOD_CONFIG[CURRENT_PERIOD]['nprev']  # Out-of-sample 2001-2025
    
    results = {}
    rmse_cpi = {}
    rmse_pce = {}
    
    # Fixed AR models
    print("Running fixed AR models...")
    for lag in range(1, 13):
        print(f"  AR lag={lag}")
        results[f'ar{lag}c'] = ar_rolling_window(Y, nprev, indice=1, lag=lag, model_type='fixed')
        results[f'ar{lag}p'] = ar_rolling_window(Y, nprev, indice=2, lag=lag, model_type='fixed')
        rmse_cpi[lag] = results[f'ar{lag}c']['errors']['rmse']
        rmse_pce[lag] = results[f'ar{lag}p']['errors']['rmse']
    
    # BIC AR models
    print("Running BIC AR models...")
    for lag in range(1, 13):
        print(f"  BIC AR lag={lag}")
        results[f'bar{lag}c'] = ar_rolling_window(Y, nprev, indice=1, lag=lag, model_type='bic')
        results[f'bar{lag}p'] = ar_rolling_window(Y, nprev, indice=2, lag=lag, model_type='bic')
    
    # Combine results
    cpi_fixed = np.column_stack([results[f'ar{lag}c']['pred'] for lag in range(1, 13)])
    cpi_bic = np.column_stack([results[f'bar{lag}c']['pred'] for lag in range(1, 13)])
    
    pce_fixed = np.column_stack([results[f'ar{lag}p']['pred'] for lag in range(1, 13)])
    pce_bic = np.column_stack([results[f'bar{lag}p']['pred'] for lag in range(1, 13)])
    
    # Create output directory
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_fixed, os.path.join(FORECAST_DIR, 'ar-fixed-cpi.csv'))
    save_forecasts(pce_fixed, os.path.join(FORECAST_DIR, 'ar-fixed-pce.csv'))
    save_forecasts(cpi_bic, os.path.join(FORECAST_DIR, 'ar-bic-cpi.csv'))
    save_forecasts(pce_bic, os.path.join(FORECAST_DIR, 'ar-bic-pce.csv'))
    
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
