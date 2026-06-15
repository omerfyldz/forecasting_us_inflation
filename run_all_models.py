"""
Master Script: Run All Models Across All Sample Periods

This script provides a unified interface to run all forecasting models
across all 5 sample periods. It handles:
- Data loading for each period
- Model execution
- Result aggregation and saving

Usage:
    python run_all_models.py                    # Run all models, all periods
    python run_all_models.py --period 1990_2000 # Run specific period
    python run_all_models.py --model rf_fe      # Run specific model
    python run_all_models.py --sample first     # Run only without_dummy

Sample Periods:
    1990_2000: Low volatility (Great Moderation)
    2001_2015: Financial crisis and recovery
    2016_2022: COVID-19 and inflation surge
    2020_2022: Pandemic subset
    1990_2022: Full extended sample
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_csv, save_forecasts, embed, calculate_errors

# =============================================================================
# CONFIGURATION
# =============================================================================

# Sample periods
SAMPLE_PERIODS = ['1990_2000', '2001_2015', '2016_2022', '2020_2022', '1990_2022']

# Sample directories
SAMPLE_DIRS = ['without_dummy', 'with_dummy']

# nprev values for each sample period (forecast evaluation months)
NPREV_CONFIG = {
    '1990_2000': 60,   # 5 years
    '2001_2015': 84,   # 7 years
    '2016_2022': 48,   # 4 years
    '2020_2022': 24,   # 2 years
    '1990_2022': 132,  # 11 years (same as original without_dummy)
    'original': {
        'without_dummy': 132,
        'with_dummy': 298
    }
}

# Available models
AVAILABLE_MODELS = {
    # Basic models (no feature engineering)
    'rw': 'Random Walk (benchmark)',
    'ar': 'Autoregressive models',
    'lasso': 'LASSO regression',
    'adalasso': 'Adaptive LASSO',
    'elasticnet': 'Elastic Net',
    'ridge': 'Ridge regression',
    'rf': 'Random Forest',
    'xgb': 'XGBoost',
    'lstm': 'LSTM',
    'nn': 'Neural Networks',
    'boosting': 'Gradient Boosting',
    'bagging': 'Bagging',
}


# =============================================================================
# DATA LOADING
# =============================================================================

def get_data_path(sample_dir, period=None):
    """Get path to data file for given sample and period."""
    base_dir = PROJECT_ROOT / sample_dir
    
    if period is None or period == 'original':
        filename = 'rawdata_1990_2022.csv'
    else:
        filename = f'rawdata_{period}.csv'
    
    return base_dir / filename


def load_data(sample_dir, period=None):
    """Load data for given sample and period."""
    path = get_data_path(sample_dir, period)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    data = pd.read_csv(path, header=None).values
    return data


def get_nprev(sample_dir, period=None):
    """Get nprev value for given sample and period."""
    if period is None or period == 'original':
        return NPREV_CONFIG['original'][sample_dir]
    return NPREV_CONFIG.get(period, 132)


# =============================================================================
# MODEL RUNNERS
# =============================================================================

def run_model(model_name, sample_dir, period=None, horizons=range(1, 13)):
    """
    Run a specific model for a sample/period combination.
    
    Parameters
    ----------
    model_name : str
        Name of the model to run
    sample_dir : str
        'without_dummy' or 'with_dummy'
    period : str, optional
        Sample period (e.g., '1990_2000')
    horizons : list
        Forecast horizons to run
        
    Returns
    -------
    dict
        Results with forecasts and metrics
    """
    print(f"\n  Running {model_name} on {sample_dir}/{period or 'original'}...")
    
    # Load data
    try:
        Y = load_data(sample_dir, period)
    except FileNotFoundError as e:
        print(f"    ERROR: {e}")
        return None
    
    nprev = get_nprev(sample_dir, period)
    
    # Import and run the appropriate model function
    results = {}
    
    try:
        if model_name == 'rw':
            from without_dummy.functions.func_rw import rw_rolling_window
            for lag in horizons:
                results[f'rw{lag}c'] = rw_rolling_window(Y, nprev, indice=1, lag=lag)
                results[f'rw{lag}p'] = rw_rolling_window(Y, nprev, indice=2, lag=lag)
                
        elif model_name == 'ar':
            if sample_dir == 'without_dummy':
                from without_dummy.functions.func_ar import ar_rolling_window
            else:
                from with_dummy.functions.func_ar import ar_rolling_window
            for lag in horizons:
                results[f'ar{lag}c'] = ar_rolling_window(Y, nprev, indice=1, lag=lag)
                results[f'ar{lag}p'] = ar_rolling_window(Y, nprev, indice=2, lag=lag)
                
        elif model_name == 'rf':
            if sample_dir == 'without_dummy':
                from without_dummy.functions.func_rf import rf_rolling_window
            else:
                from with_dummy.functions.func_rf import rf_rolling_window
            for lag in horizons:
                results[f'rf{lag}c'] = rf_rolling_window(Y, nprev, indice=1, lag=lag)
                results[f'rf{lag}p'] = rf_rolling_window(Y, nprev, indice=2, lag=lag)
        
        else:
            print(f"    Model {model_name} not yet implemented in master runner")
            return None

            
    except ImportError as e:
        print(f"    Import error: {e}")
        return None
    except Exception as e:
        print(f"    Error running model: {e}")
        return None
    
    return results


def save_results(results, model_name, sample_dir, period, output_dir):
    """Save model results to CSV files."""
    if results is None:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine period suffix
    period_suffix = f"_{period}" if period else ""
    
    # Aggregate CPI forecasts
    cpi_keys = sorted([k for k in results.keys() if k.endswith('c')])
    if cpi_keys:
        cpi_forecasts = np.column_stack([results[k]['pred'] for k in cpi_keys])
        filename = f"{model_name}-cpi{period_suffix}.csv"
        np.savetxt(output_dir / filename, cpi_forecasts, delimiter=',')
        print(f"    Saved: {filename}")
    
    # Aggregate PCE forecasts
    pce_keys = sorted([k for k in results.keys() if k.endswith('p')])
    if pce_keys:
        pce_forecasts = np.column_stack([results[k]['pred'] for k in pce_keys])
        filename = f"{model_name}-pce{period_suffix}.csv"
        np.savetxt(output_dir / filename, pce_forecasts, delimiter=',')
        print(f"    Saved: {filename}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all(models=None, periods=None, samples=None, save_results_flag=True):
    """
    Run all specified models across all specified periods and samples.
    
    Parameters
    ----------
    models : list, optional
        List of model names to run (default: all)
    periods : list, optional
        List of periods to run (default: all)
    samples : list, optional
        List of sample directories (default: both)
    save_results_flag : bool
        Whether to save results to CSV
        
    Returns
    -------
    dict
        All results organized by model/sample/period
    """
    if models is None:
        models = list(AVAILABLE_MODELS.keys())
    if periods is None:
        periods = SAMPLE_PERIODS + ['original']
    if samples is None:
        samples = SAMPLE_DIRS
    
    print("="*60)
    print("MULTI-SAMPLE MODEL EXECUTION")
    print("="*60)
    print(f"\nModels: {models}")
    print(f"Periods: {periods}")
    print(f"Samples: {samples}")
    print(f"Start time: {datetime.now()}")
    
    all_results = {}
    
    for sample_dir in samples:
        print(f"\n{'#'*60}")
        print(f"# Sample: {sample_dir}")
        print(f"{'#'*60}")
        
        all_results[sample_dir] = {}
        
        for period in periods:
            print(f"\n  Period: {period}")
            print("-"*40)
            
            all_results[sample_dir][period] = {}
            
            for model_name in models:
                results = run_model(model_name, sample_dir, period)
                
                if results:
                    all_results[sample_dir][period][model_name] = results
                    
                    if save_results_flag:
                        output_dir = PROJECT_ROOT / sample_dir / 'forecasts'
                        save_results(results, model_name, sample_dir, period, output_dir)
    
    print(f"\n{'='*60}")
    print(f"EXECUTION COMPLETE")
    print(f"End time: {datetime.now()}")
    print(f"{'='*60}")
    
    return all_results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Run all models across sample periods')
    parser.add_argument('--model', '-m', type=str, help='Specific model to run')
    parser.add_argument('--period', '-p', type=str, help='Specific period to run')
    parser.add_argument('--sample', '-s', type=str, choices=['first', 'second', 'both'], 
                       default='both', help='Sample to run')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nAvailable models:")
        for name, desc in AVAILABLE_MODELS.items():
            print(f"  {name:12s} - {desc}")
        return
    
    # Parse arguments
    models = [args.model] if args.model else None
    periods = [args.period] if args.period else None
    
    if args.sample == 'first':
        samples = ['without_dummy']
    elif args.sample == 'second':
        samples = ['with_dummy']
    else:
        samples = SAMPLE_DIRS
    
    # Run models
    results = run_all(
        models=models,
        periods=periods,
        samples=samples,
        save_results_flag=not args.no_save
    )
    
    return results


if __name__ == "__main__":
    main()
