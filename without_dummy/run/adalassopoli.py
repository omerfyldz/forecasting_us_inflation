"""
AdaLASSO with Polynomial Features Run Script
Uses polynomial LASSO functions for inflation forecasting
"""

import sys
import os
import numpy as np
import pandas as pd

# Add parent directories to path
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

from without_dummy.functions.func_polilasso import polilasso_rolling_window
from utils import load_csv, save_forecasts

def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    nprev = PERIOD_CONFIG[CURRENT_PERIOD]['nprev']
    alpha = 1.0  # LASSO
    
    # Storage for results
    cpi_results = []
    pce_results = []
    
    print("Running AdaLASSO Polynomial forecasts...")
    
    # Run for each lag (1-12)
    for lag in range(1, 13):
        print(f"  Processing lag {lag}/12...")
        
        # CPI (indice=1)
        result_cpi = polilasso_rolling_window(Y, nprev, indice=1, lag=lag, 
                                               alpha=alpha, model_type='adalasso')
        cpi_results.append(result_cpi['pred'])
        
        # PCE (indice=2)
        result_pce = polilasso_rolling_window(Y, nprev, indice=2, lag=lag,
                                               alpha=alpha, model_type='adalasso')
        pce_results.append(result_pce['pred'])
    
    # Combine results
    cpi_matrix = np.column_stack(cpi_results)
    pce_matrix = np.column_stack(pce_results)
    
    # Save forecasts
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    save_forecasts(cpi_matrix, os.path.join(FORECAST_DIR, 'adalassopoli-cpi.csv'))
    save_forecasts(pce_matrix, os.path.join(FORECAST_DIR, 'adalassopoli-pce.csv'))
    
    print("AdaLASSO Polynomial forecasts completed!")
    print(f"Results saved to {FORECAST_DIR}")

if __name__ == "__main__":
    main()
