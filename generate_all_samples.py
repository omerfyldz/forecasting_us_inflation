"""
Generate All Sample Period Datasets

This script generates datasets for all 5 sample periods used in the analysis:
1. 1990-2000: Low volatility period (Great Moderation)
2. 2001-2015: Financial crisis and recovery  
3. 2016-2022: COVID-19 and inflation surge
4. 2020-2022: Pandemic and inflation surge (subset)
5. 1990-2022: Full extended sample

For each period, generates:
- rawdata_{period}.csv: FRED-MD transformed data

Usage:
    python generate_all_samples.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from fred_md_loader import FREDMDLoader

# =============================================================================
# SAMPLE PERIOD CONFIGURATIONS
# =============================================================================

SAMPLE_PERIODS = {
    '1990_2000': {
        'start_date': '1990-01-01',
        'end_date': '2000-12-31',
        'description': 'Low volatility period (Great Moderation)',
        'nprev': 60  # 5 years evaluation
    },
    '2001_2015': {
        'start_date': '2001-01-01',
        'end_date': '2015-12-31',
        'description': 'Financial crisis and recovery',
        'nprev': 84  # 7 years evaluation
    },
    '2016_2022': {
        'start_date': '2016-01-01',
        'end_date': '2022-12-31',
        'description': 'COVID-19 and inflation surge',
        'nprev': 48  # 4 years evaluation
    },
    '2020_2022': {
        'start_date': '2020-01-01',
        'end_date': '2022-12-31',
        'description': 'Pandemic and inflation surge',
        'nprev': 24  # 2 years evaluation
    },
    '1990_2022': {
        'start_date': '1990-01-01',
        'end_date': '2022-12-31',
        'description': 'Full extended sample',
        'nprev': 132  # 11 years evaluation
    }
}


def load_fred_md_data(data_path=None):
    """Load and transform FRED-MD data."""
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "2025-11-MD.csv"
    
    loader = FREDMDLoader(str(data_path))
    loader.load_raw_data()
    df = loader.transform_data()
    
    return df


def generate_period_data(df_full, period_name, config, output_dir):
    """Generate data for a specific sample period."""
    print(f"\n  {period_name}: {config['description']}")
    
    start_date = pd.to_datetime(config['start_date'])
    end_date = pd.to_datetime(config['end_date'])
    
    df_period = df_full[(df_full.index >= start_date) & (df_full.index <= end_date)]
    
    if len(df_period) == 0:
        print(f"    WARNING: No data for period")
        return None
    
    # Save raw data
    raw_filename = f"rawdata_{period_name}.csv"
    raw_path = output_dir / raw_filename
    df_period.to_csv(raw_path, header=False, index=False)
    print(f"    Saved: {raw_filename} ({len(df_period)} rows, {df_period.shape[1]} cols)")
    
    return {
        'period': period_name,
        'observations': len(df_period),
        'features': df_period.shape[1]
    }


def generate_all_samples(data_path=None):
    """Generate all sample period datasets."""
    print("=" * 60)
    print("SAMPLE DATA GENERATION")
    print("=" * 60)
    
    df_full = load_fred_md_data(data_path)
    print(f"Full data: {len(df_full)} obs, {df_full.shape[1]} features")
    print(f"Date range: {df_full.index.min()} to {df_full.index.max()}")
    
    results = []
    
    for sample_dir in ['without_dummy', 'with_dummy']:
        output_dir = PROJECT_ROOT / sample_dir
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{sample_dir}/:")
        
        for period_name, config in SAMPLE_PERIODS.items():
            result = generate_period_data(df_full, period_name, config, output_dir)
            if result:
                result['sample_dir'] = sample_dir
                results.append(result)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['sample_dir']}/{r['period']}: {r['observations']} obs")
    
    return results


if __name__ == "__main__":
    generate_all_samples()
