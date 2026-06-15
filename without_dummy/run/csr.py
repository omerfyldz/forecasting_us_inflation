"""
CSR (Complete Subset Regression) Rolling Window Forecasts
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
from without_dummy.functions.func_csr import csr_rolling_window


def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    nprev = PERIOD_CONFIG[CURRENT_PERIOD]['nprev']
    np.random.seed(123)
    
    results = {}
    rmse_cpi = {}
    rmse_pce = {}
    
    # CSR models
    print("Running CSR models...")
    for lag in range(1, 13):
        print(f"  CSR lag={lag}")
        results[f'csr{lag}c'] = csr_rolling_window(Y, nprev, indice=1, lag=lag)
        results[f'csr{lag}p'] = csr_rolling_window(Y, nprev, indice=2, lag=lag)
        rmse_cpi[lag] = results[f'csr{lag}c']['errors']['rmse']
        rmse_pce[lag] = results[f'csr{lag}p']['errors']['rmse']
    
    # Combine results for CPI (indice=1)
    cpi_csr = np.column_stack([results[f'csr{lag}c']['pred'] for lag in range(1, 13)])
    
    # Combine results for PCE (indice=2)
    pce_csr = np.column_stack([results[f'csr{lag}p']['pred'] for lag in range(1, 13)])
    
    # Create output directory
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_csr, os.path.join(FORECAST_DIR, 'csr-cpi.csv'))
    save_forecasts(pce_csr, os.path.join(FORECAST_DIR, 'csr-pce.csv'))
    
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
