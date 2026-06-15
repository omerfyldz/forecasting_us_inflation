"""
AR Model Rolling Window Forecasts
Estimates RMSE for all horizons h1-h12
"""
import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import load_csv, save_forecasts
from without_dummy.functions.func_ar import ar_rolling_window

# Get the directory where this script is located
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
FORECAST_DIR = os.path.join(SCRIPT_DIR, '..', 'forecasts')


def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    nprev = PERIOD_CONFIG[CURRENT_PERIOD]['nprev']
    
    results = {}
    rmse_fixed_cpi = {}
    rmse_fixed_pce = {}
    rmse_bic_cpi = {}
    rmse_bic_pce = {}
    
    # Fixed AR models
    print("=" * 60)
    print("AR (Fixed) - RMSE by Horizon (h1-h12)")
    print("=" * 60)
    for lag in range(1, 13):
        print(f"  Running h={lag}...")
        results[f'ar{lag}c'] = ar_rolling_window(Y, nprev, indice=1, lag=lag, model_type='fixed')
        results[f'ar{lag}p'] = ar_rolling_window(Y, nprev, indice=2, lag=lag, model_type='fixed')
        rmse_fixed_cpi[lag] = results[f'ar{lag}c']['errors']['rmse']
        rmse_fixed_pce[lag] = results[f'ar{lag}p']['errors']['rmse']
    
    # Print Fixed AR RMSE summary
    print("\n" + "=" * 60)
    print("RMSE BY HORIZON - AR (Fixed)")
    print("=" * 60)
    print(f"{'Horizon':<10} {'CPI RMSE':<15} {'PCE RMSE':<15}")
    print("-" * 40)
    for h in range(1, 13):
        print(f"h={h:<7} {rmse_fixed_cpi[h]:<15.6f} {rmse_fixed_pce[h]:<15.6f}")
    avg_cpi = np.mean(list(rmse_fixed_cpi.values()))
    avg_pce = np.mean(list(rmse_fixed_pce.values()))
    print("-" * 40)
    print(f"{'Average':<10} {avg_cpi:<15.6f} {avg_pce:<15.6f}")
    
    # BIC AR models
    print("\n" + "=" * 60)
    print("AR (BIC) - RMSE by Horizon (h1-h12)")
    print("=" * 60)
    for lag in range(1, 13):
        print(f"  Running h={lag}...")
        results[f'bar{lag}c'] = ar_rolling_window(Y, nprev, indice=1, lag=lag, model_type='bic')
        results[f'bar{lag}p'] = ar_rolling_window(Y, nprev, indice=2, lag=lag, model_type='bic')
        rmse_bic_cpi[lag] = results[f'bar{lag}c']['errors']['rmse']
        rmse_bic_pce[lag] = results[f'bar{lag}p']['errors']['rmse']
    
    # Print BIC AR RMSE summary
    print("\n" + "=" * 60)
    print("RMSE BY HORIZON - AR (BIC)")
    print("=" * 60)
    print(f"{'Horizon':<10} {'CPI RMSE':<15} {'PCE RMSE':<15}")
    print("-" * 40)
    for h in range(1, 13):
        print(f"h={h:<7} {rmse_bic_cpi[h]:<15.6f} {rmse_bic_pce[h]:<15.6f}")
    avg_cpi = np.mean(list(rmse_bic_cpi.values()))
    avg_pce = np.mean(list(rmse_bic_pce.values()))
    print("-" * 40)
    print(f"{'Average':<10} {avg_cpi:<15.6f} {avg_pce:<15.6f}")
    print("=" * 60)
    
    # Combine results for CPI (indice=1)
    cpi_fixed = np.column_stack([results[f'ar{lag}c']['pred'] for lag in range(1, 13)])
    cpi_bic = np.column_stack([results[f'bar{lag}c']['pred'] for lag in range(1, 13)])
    
    # Combine results for PCE (indice=2)
    pce_fixed = np.column_stack([results[f'ar{lag}p']['pred'] for lag in range(1, 13)])
    pce_bic = np.column_stack([results[f'bar{lag}p']['pred'] for lag in range(1, 13)])
    
    # Create output directory
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_fixed, os.path.join(FORECAST_DIR, 'ar-fixed-cpi.csv'))
    save_forecasts(pce_fixed, os.path.join(FORECAST_DIR, 'ar-fixed-pce.csv'))
    save_forecasts(cpi_bic, os.path.join(FORECAST_DIR, 'ar-bic-cpi.csv'))
    save_forecasts(pce_bic, os.path.join(FORECAST_DIR, 'ar-bic-pce.csv'))
    
    # Save RMSE summaries
    rmse_fixed_df = pd.DataFrame({
        'Horizon': list(range(1, 13)),
        'CPI_RMSE': [rmse_fixed_cpi[h] for h in range(1, 13)],
        'PCE_RMSE': [rmse_fixed_pce[h] for h in range(1, 13)]
    })
    rmse_fixed_df.to_csv(os.path.join(FORECAST_DIR, 'ar-fixed-rmse.csv'), index=False)
    
    rmse_bic_df = pd.DataFrame({
        'Horizon': list(range(1, 13)),
        'CPI_RMSE': [rmse_bic_cpi[h] for h in range(1, 13)],
        'PCE_RMSE': [rmse_bic_pce[h] for h in range(1, 13)]
    })
    rmse_bic_df.to_csv(os.path.join(FORECAST_DIR, 'ar-bic-rmse.csv'), index=False)
    
    print(f"\nForecasts saved to {FORECAST_DIR}")
    
    return results


if __name__ == '__main__':
    main()
