"""
Combination Model Rolling Window Forecasts - Second Sample
This script combines multiple model forecasts.
"""
import os
import sys
import numpy as np
import pandas as pd

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

from utils import load_csv, save_forecasts


def load_forecasts(path):
    """Load forecasts from CSV file."""
    return pd.read_csv(path, header=None, sep=';').values


def main():
    """Combine forecasts from multiple models."""
    
    # Load all available forecasts
    models = ['ar-fixed', 'lasso', 'rf', 'xgb', 'nn', 'bagging', 'boosting', 'factors']
    
    cpi_forecasts = []
    pce_forecasts = []
    
    for model in models:
        try:
            cpi_path = os.path.join(FORECAST_DIR, f'{model}-cpi.csv')
            pce_path = os.path.join(FORECAST_DIR, f'{model}-pce.csv')
            
            if os.path.exists(cpi_path):
                cpi_forecasts.append(load_forecasts(cpi_path))
            if os.path.exists(pce_path):
                pce_forecasts.append(load_forecasts(pce_path))
        except Exception as e:
            print(f"Could not load {model}: {e}")
    
    if cpi_forecasts:
        # Simple average combination
        cpi_combined = np.mean(cpi_forecasts, axis=0)
        save_forecasts(cpi_combined, os.path.join(FORECAST_DIR, 'combined-cpi.csv'))
        print(f"Combined {len(cpi_forecasts)} CPI forecasts")
    
    if pce_forecasts:
        pce_combined = np.mean(pce_forecasts, axis=0)
        save_forecasts(pce_combined, os.path.join(FORECAST_DIR, 'combined-pce.csv'))
        print(f"Combined {len(pce_forecasts)} PCE forecasts")
    
    print(f"Done! Forecasts saved to {FORECAST_DIR}")


if __name__ == '__main__':
    main()
