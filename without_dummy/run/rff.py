"""
Random Forest Factors (RFFACT) run script.
Uses importance weights from previous Random Forest runs to construct factor-based forecasts.

Note: This script requires pre-computed RF importance matrices from rf.py runs.
Run rf.py first and save the importance weights, then use them here.
"""

import numpy as np
import pandas as pd
import os
import sys

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

from without_dummy.functions.func_rffact import rffact_rolling_window
from utils import load_csv

# Try to load data
if os.path.exists(DATA_PATH):
    Y = load_csv(DATA_PATH)
else:
    print("Warning: rawdata.csv not found, using simulated data")
    np.random.seed(42)
    Y = np.random.randn(300, 10)

nprev = PERIOD_CONFIG[CURRENT_PERIOD]['nprev']


def load_rf_importance(filepath):
    """Load RF importance matrix from file."""
    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        print(f"Warning: {filepath} not found, using random importance weights")
        return None


def run_rffact_experiments(Y, nprev, importance_dir='rf_importance'):
    """
    Run all RFFACT experiments for different horizons.
    
    Parameters:
    -----------
    Y : ndarray
        Input data
    nprev : int
        Number of out-of-sample forecasts
    importance_dir : str
        Directory containing RF importance matrices
    """
    results_cpi = {}
    results_pce = {}
    
    # Horizons 1-12
    for h in range(1, 13):
        print(f"\n=== Horizon {h} ===")
        
        # Load importance weights (or simulate if not available)
        imp_cpi_path = os.path.join(importance_dir, f'rf{h}c_importance.npy')
        imp_pce_path = os.path.join(importance_dir, f'rf{h}p_importance.npy')
        
        imp_cpi = load_rf_importance(imp_cpi_path)
        imp_pce = load_rf_importance(imp_pce_path)
        
        # Use simulated importance if files don't exist
        n_features = Y.shape[1] * 4 + 4  # Approximate number of features
        if imp_cpi is None:
            imp_cpi = np.random.rand(nprev, n_features)
        if imp_pce is None:
            imp_pce = np.random.rand(nprev, n_features)
        
        # CPI (indice=1)
        print(f"Running RFFACT for CPI, horizon {h}...")
        rff_cpi = rffact_rolling_window(Y, nprev, indice=1, lag=h, 
                                         importance_matrix=imp_cpi)
        results_cpi[f'rff{h}c'] = rff_cpi
        
        # PCE (indice=2)
        print(f"Running RFFACT for PCE, horizon {h}...")
        rff_pce = rffact_rolling_window(Y, nprev, indice=2, lag=h, 
                                         importance_matrix=imp_pce)
        results_pce[f'rff{h}p'] = rff_pce
    
    return results_cpi, results_pce


def save_results(results_cpi, results_pce, output_dir=None):
    """Save forecast results to CSV files."""
    if output_dir is None:
        output_dir = FORECAST_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine CPI predictions
    cpi_preds = np.column_stack([results_cpi[f'rff{h}c']['pred'] for h in range(1, 13)])
    pce_preds = np.column_stack([results_pce[f'rff{h}p']['pred'] for h in range(1, 13)])
    
    # Save
    np.savetxt(os.path.join(output_dir, 'rffact-cpi.csv'), cpi_preds, delimiter=';')
    np.savetxt(os.path.join(output_dir, 'rffact-pce.csv'), pce_preds, delimiter=';')
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    print("=" * 60)
    print("RANDOM FOREST FACTORS (RFFACT) FORECASTING")
    print("=" * 60)
    print(f"Data shape: {Y.shape}")
    print(f"Number of out-of-sample forecasts: {nprev}")
    print("\nNote: This script uses RF importance weights from previous runs.")
    print("If importance files are not found, random weights are used.")
    print("=" * 60)
    
    # Check if we have enough data
    if Y.shape[0] < nprev + 20:
        print(f"\nWarning: Not enough data. Need at least {nprev + 20} observations.")
        print("Running with reduced nprev...")
        nprev = min(10, Y.shape[0] - 20)
    
    # Run experiments
    results_cpi, results_pce = run_rffact_experiments(Y, nprev)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - RMSE by Horizon")
    print("=" * 60)
    print(f"{'Horizon':<10} {'CPI RMSE':<15} {'PCE RMSE':<15}")
    print("-" * 40)
    for h in range(1, 13):
        cpi_rmse = results_cpi[f'rff{h}c']['errors']['rmse']
        pce_rmse = results_pce[f'rff{h}p']['errors']['rmse']
        print(f"{h:<10} {cpi_rmse:<15.4f} {pce_rmse:<15.4f}")
    
    # Save results
    save_results(results_cpi, results_pce)
