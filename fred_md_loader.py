"""
FRED-MD Data Loader with Stationarity Transformations

This module loads raw FRED-MD data and applies the official stationarity 
transformations as specified in the transformation codes row.

FRED-MD Transformation Codes:
    1 = No transformation (level)
    2 = First difference: Δx_t = x_t - x_{t-1}
    3 = Second difference: Δ²x_t = Δx_t - Δx_{t-1}
    4 = Log: ln(x_t)
    5 = Log first difference (growth rate): Δln(x_t) = ln(x_t) - ln(x_{t-1})
    6 = Log second difference: Δ²ln(x_t)
    7 = Delta (x_t/x_{t-1} - 1): percentage change formula

Reference: McCracken & Ng (2016) "FRED-MD: A Monthly Database for Macroeconomic Research"
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


class FREDMDLoader:
    """
    Loads and transforms FRED-MD data according to stationarity codes.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the FRED-MD loader.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to raw FRED-MD CSV file
        """
        self.data_path = data_path
        self.raw_data = None
        self.transformed_data = None
        self.transformation_codes = None
        self.variable_names = None
        self.date_index = None
        
    def load_raw_data(self, data_path=None):
        """
        Load raw FRED-MD data from CSV.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to FRED-MD CSV file
            
        Returns:
        --------
        pd.DataFrame : Raw data with dates as index
        """
        if data_path:
            self.data_path = data_path
            
        if self.data_path is None:
            raise ValueError("No data path provided")
        
        print(f"Loading raw FRED-MD data from: {self.data_path}")
        
        # Read header row (variable names)
        df_header = pd.read_csv(self.data_path, nrows=0)
        self.variable_names = df_header.columns.tolist()
        
        # Read transformation codes (row 2)
        df_codes = pd.read_csv(self.data_path, skiprows=[0], nrows=1, header=None)
        self.transformation_codes = df_codes.iloc[0, 1:].astype(float).values  # Skip date column
        
        # Read data (skip header and transformation row)
        df = pd.read_csv(self.data_path, skiprows=[1])
        
        # Parse dates
        df['sasdate'] = pd.to_datetime(df['sasdate'], format='%m/%d/%Y')
        df = df.set_index('sasdate')
        self.date_index = df.index
        
        # Store raw data
        self.raw_data = df
        
        print(f"Loaded {len(df)} observations")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Variables: {len(df.columns)}")
        
        return df
    
    def apply_transformation(self, series, code):
        """
        Apply FRED-MD transformation based on code.
        
        Parameters:
        -----------
        series : pd.Series
            Time series to transform
        code : int
            Transformation code (1-7)
            
        Returns:
        --------
        pd.Series : Transformed series
        """
        code = int(code)
        
        if code == 1:
            # No transformation
            return series
        elif code == 2:
            # First difference
            return series.diff()
        elif code == 3:
            # Second difference
            return series.diff().diff()
        elif code == 4:
            # Log
            return np.log(series.replace(0, np.nan))
        elif code == 5:
            # Log first difference (growth rate)
            return np.log(series.replace(0, np.nan)).diff()
        elif code == 6:
            # Log second difference
            return np.log(series.replace(0, np.nan)).diff().diff()
        elif code == 7:
            # Delta (percentage change)
            return series.pct_change()
        else:
            print(f"Warning: Unknown transformation code {code}, using no transformation")
            return series
    
    def transform_data(self, drop_nan_rows='beginning', fill_method='ffill'):
        """
        Apply stationarity transformations to all variables.
        
        Parameters:
        -----------
        drop_nan_rows : str or bool
            'beginning' (default): Only drop initial NaN rows from differencing
            True: Drop all rows with any NaN
            False: Keep all rows
        fill_method : str
            Method to fill remaining NaN: 'ffill', 'bfill', 'mean', or None
            
        Returns:
        --------
        pd.DataFrame : Transformed data
        """
        if self.raw_data is None:
            raise ValueError("No raw data loaded. Call load_raw_data() first.")
        
        print("Applying FRED-MD stationarity transformations...")
        
        transformed = pd.DataFrame(index=self.raw_data.index)
        
        for i, col in enumerate(self.raw_data.columns):
            if i < len(self.transformation_codes):
                code = self.transformation_codes[i]
                if np.isnan(code):
                    code = 1  # Default to no transformation
            else:
                code = 1  # Default to no transformation
                
            transformed[col] = self.apply_transformation(self.raw_data[col], code)
        
        initial_rows = len(transformed)
        
        if drop_nan_rows == 'beginning':
            # Only drop initial rows that are ALL NaN or have NaN in most columns
            # This handles the differencing issue without dropping too many rows
            # Find first row where most columns have valid data (>80% non-NaN)
            threshold = 0.8 * len(transformed.columns)
            first_valid_idx = 0
            for i in range(len(transformed)):
                valid_count = transformed.iloc[i].notna().sum()
                if valid_count >= threshold:
                    first_valid_idx = i
                    break
            
            # Also ensure we skip the first 2 rows for second differencing
            first_valid_idx = max(first_valid_idx, 2)
            transformed = transformed.iloc[first_valid_idx:]
            dropped = first_valid_idx
            print(f"Dropped first {dropped} rows (differencing)")
            
            # Fill remaining NaN values
            if fill_method == 'ffill':
                transformed = transformed.ffill().bfill()
            elif fill_method == 'bfill':
                transformed = transformed.bfill().ffill()
            elif fill_method == 'mean':
                transformed = transformed.fillna(transformed.mean())
                
        elif drop_nan_rows == True:
            # Drop all rows with any NaN
            transformed = transformed.dropna()
            dropped = initial_rows - len(transformed)
            print(f"Dropped {dropped} rows with NaN values")
            print(f"Dropped {dropped} rows with NaN values")
        
        self.transformed_data = transformed
        
        print(f"Transformation complete: {len(transformed)} observations, {len(transformed.columns)} variables")
        
        return transformed
    
    def filter_by_date(self, start_date=None, end_date=None):
        """
        Filter data by date range.
        
        Parameters:
        -----------
        start_date : str or datetime
            Start date (inclusive)
        end_date : str or datetime
            End date (inclusive)
            
        Returns:
        --------
        pd.DataFrame : Filtered data
        """
        if self.transformed_data is None:
            raise ValueError("No transformed data. Call transform_data() first.")
        
        df = self.transformed_data.copy()
        
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        print(f"Filtered to {len(df)} observations")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def get_sample_period(self, sample_type='first'):
        """
        Get data for a specific sample period matching the R project.
        
        Parameters:
        -----------
        sample_type : str
            'first' for without_dummy (periods without 2008 crisis)
            'second' for with_dummy (periods with 2008 crisis)
            
        Returns:
        --------
        pd.DataFrame : Data for the specified sample
        """
        if self.transformed_data is None:
            raise ValueError("No transformed data. Call transform_data() first.")
        
        if sample_type == 'first':
            # First sample: roughly 2000-2020 (503 observations with nprev=132)
            # This means data from around 1959 to some point
            # nprev=132 means 11 years of test data (132 months)
            # Training starts at some point, evaluation for last 132 months
            print("Getting first sample period (similar to R rawdata2000)")
            return self.transformed_data.copy()
        
        elif sample_type == 'second':
            # Second sample: longer period with nprev=298
            print("Getting second sample period (extended)")
            return self.transformed_data.copy()
        
        else:
            raise ValueError(f"Unknown sample type: {sample_type}")
    
    def to_numpy(self, include_dates=False):
        """
        Convert transformed data to numpy array.
        
        Parameters:
        -----------
        include_dates : bool
            Whether to return dates as well
            
        Returns:
        --------
        np.ndarray or tuple : Data array (and dates if include_dates=True)
        """
        if self.transformed_data is None:
            raise ValueError("No transformed data. Call transform_data() first.")
        
        data = self.transformed_data.values
        
        if include_dates:
            return data, self.transformed_data.index
        return data
    
    def save_transformed(self, output_path):
        """
        Save transformed data to CSV.
        
        Parameters:
        -----------
        output_path : str
            Output file path
        """
        if self.transformed_data is None:
            raise ValueError("No transformed data. Call transform_data() first.")
        
        self.transformed_data.to_csv(output_path, index=True)
        print(f"Saved transformed data to: {output_path}")


def load_fred_md(data_path, start_date=None, end_date=None, return_numpy=True):
    """
    Convenience function to load and transform FRED-MD data.
    
    Parameters:
    -----------
    data_path : str
        Path to FRED-MD CSV file
    start_date : str, optional
        Start date for filtering
    end_date : str, optional
        End date for filtering
    return_numpy : bool
        Whether to return numpy array (True) or DataFrame (False)
        
    Returns:
    --------
    np.ndarray or pd.DataFrame : Transformed data
    """
    loader = FREDMDLoader(data_path)
    loader.load_raw_data()
    loader.transform_data()
    
    if start_date or end_date:
        df = loader.filter_by_date(start_date, end_date)
    else:
        df = loader.transformed_data
    
    if return_numpy:
        return df.values
    return df


def create_aligned_dataset(fred_md_path, target_rows=503, from_end=True):
    """
    Create a dataset aligned with the original project's data structure.
    
    The original rawdata.csv has 503 rows. This function loads raw FRED-MD,
    transforms it, and returns the same number of rows for compatibility.
    
    Parameters:
    -----------
    fred_md_path : str
        Path to raw FRED-MD CSV
    target_rows : int
        Number of rows to return (default: 503 to match without_dummy)
    from_end : bool
        If True, take last N rows; if False, take first N rows
        
    Returns:
    --------
    tuple : (data_array, dates_index)
    """
    loader = FREDMDLoader(fred_md_path)
    loader.load_raw_data()
    df = loader.transform_data()
    
    if len(df) < target_rows:
        print(f"Warning: Only {len(df)} rows available, requested {target_rows}")
        return df.values, df.index
    
    if from_end:
        df = df.tail(target_rows)
    else:
        df = df.head(target_rows)
    
    print(f"Aligned dataset: {len(df)} rows")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df.values, df.index


# =============================================================================
# Test the loader
# =============================================================================

if __name__ == "__main__":
    import os
    
    # Find data file
    script_dir = Path(__file__).parent
    data_path = script_dir / "data" / "2025-11-MD.csv"
    
    if data_path.exists():
        print("=" * 60)
        print("Testing FRED-MD Loader")
        print("=" * 60)
        
        # Load and transform
        loader = FREDMDLoader(str(data_path))
        loader.load_raw_data()
        df = loader.transform_data()
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nTransformation codes (first 10):")
        print(loader.transformation_codes[:10])
        
        print("\n" + "=" * 60)
        print("Creating aligned dataset (503 rows):")
        data, dates = create_aligned_dataset(str(data_path), target_rows=503)
        print(f"Shape: {data.shape}")
        print(f"Dates: {dates[0]} to {dates[-1]}")
    else:
        print(f"Data file not found: {data_path}")
        print("Please place 2025-11-MD.csv in the data/ folder")
