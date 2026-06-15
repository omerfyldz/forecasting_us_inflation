"""
Random Forest Rolling Window Forecasts - Second Sample
Estimates RMSE for all horizons h1-h12
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

from utils import load_csv, save_forecasts, add_outlier_dummy
from with_dummy.functions.func_rf import rf_rolling_window


def main():
    Y = load_csv(DATA_PATH)
    
    # Add dummy variable for outliers (COVID period) as in the R code
    Y = add_outlier_dummy(Y, target_col=0)
    
    nprev = PERIOD_CONFIG[CURRENT_PERIOD]['nprev']  # Out-of-sample 2001-2025
    np.random.seed(123)
    
    results = {}
    rmse_cpi = {}
    rmse_pce = {}
    
    # Random Forest models for horizons h1-h12
    print("=" * 60)
    print("Random Forest - RMSE by Horizon (h1-h12)")
    print("=" * 60)
    
    for lag in range(1, 13):
        print(f"  Running h={lag}...")
        results[f'rf{lag}c'] = rf_rolling_window(Y, nprev, indice=1, lag=lag)
        results[f'rf{lag}p'] = rf_rolling_window(Y, nprev, indice=2, lag=lag)
        rmse_cpi[lag] = results[f'rf{lag}c']['errors']['rmse']
        rmse_pce[lag] = results[f'rf{lag}p']['errors']['rmse']
    
    # Print RMSE summary by horizon
    print("\n" + "=" * 60)
    print("RMSE BY HORIZON")
    print("=" * 60)
    print(f"{'Horizon':<10} {'CPI RMSE':<15} {'PCE RMSE':<15}")
    print("-" * 40)
    for h in range(1, 13):
        print(f"h={h:<7} {rmse_cpi[h]:<15.6f} {rmse_pce[h]:<15.6f}")
    
    # Print average RMSE
    avg_cpi = np.mean(list(rmse_cpi.values()))
    avg_pce = np.mean(list(rmse_pce.values()))
    print("-" * 40)
    print(f"{'Average':<10} {avg_cpi:<15.6f} {avg_pce:<15.6f}")
    print("=" * 60)
    
    # Combine forecasts
    cpi_rf = np.column_stack([results[f'rf{lag}c']['pred'] for lag in range(1, 13)])
    pce_rf = np.column_stack([results[f'rf{lag}p']['pred'] for lag in range(1, 13)])
    
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_rf, os.path.join(FORECAST_DIR, 'rf-cpi.csv'))
    save_forecasts(pce_rf, os.path.join(FORECAST_DIR, 'rf-pce.csv'))
    
    # Save RMSE summary
    rmse_df = pd.DataFrame({
        'Horizon': list(range(1, 13)),
        'CPI_RMSE': [rmse_cpi[h] for h in range(1, 13)],
        'PCE_RMSE': [rmse_pce[h] for h in range(1, 13)]
    })
    rmse_df.to_csv(os.path.join(FORECAST_DIR, 'rf-rmse.csv'), index=False)
    
    print(f"\nForecasts saved to {FORECAST_DIR}")
    print(f"RMSE summary saved to {os.path.join(FORECAST_DIR, 'rf-rmse.csv')}")
    
    return results, rmse_cpi, rmse_pce


if __name__ == '__main__':
    main()
