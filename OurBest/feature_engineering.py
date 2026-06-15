# Advanced Feature Engineering for Inflation Forecasting
# Creates only stationary features from raw macroeconomic data

import pandas as pd
import numpy as np
from config import RAW_FEATURES, ROLLING_WINDOWS, MOMENTUM_HORIZONS
from utils import (calculate_zscore, check_stationarity, calculate_rolling_statistics,
                   calculate_momentum_features, calculate_volatility_features,
                   safe_divide, create_lagged_features, exponential_smoothing)
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StationaryFeatureEngineer:
    """
    Creates stationary features for inflation forecasting.
    NO RAW FEATURES ARE INCLUDED - only transformations.
    """

    def __init__(self, raw_features=None):
        """
        Initialize feature engineer

        Parameters:
        -----------
        raw_features : list
            List of raw feature names to transform
        """
        self.raw_features = raw_features or RAW_FEATURES
        self.transformed_features = []
        self.feature_info = {}  # Store metadata about each feature
        self.stationarity_results = {}

        logger.info(f"Initialized with {len(self.raw_features)} raw features to transform")

    def load_and_prepare_data(self, data_path="../2025-11-MD.csv"):
        """
        Load FRED-MD data and prepare for feature engineering

        Parameters:
        -----------
        data_path : str
            Path to FRED-MD CSV file

        Returns:
        --------
        pd.DataFrame : Prepared dataframe
        """
        logger.info(f"Loading data from {data_path}")

        # Load data
        df = pd.read_csv(data_path, header=0, skiprows=[1])

        # Convert date
        df['sasdate'] = pd.to_datetime(df['sasdate'], format='%m/%d/%Y')
        df = df.set_index('sasdate')

        # SAFE METHOD: Calculate inflation_rate BEFORE renaming columns
        # Target = Headline CPI (CPIAUCSL) Log Difference
        # This avoids column index shifting issues after sasdate is removed
        df['inflation_rate'] = np.log(df['CPIAUCSL']).diff()
        
        # DEBUG: Print magnitude to confirm Log-Diff vs Pct-Change
        mean_abs_inf = df['inflation_rate'].abs().mean()
        logger.info(f"DEBUG: Mean Abs Inflation Rate (All) = {mean_abs_inf:.6f}")
        logger.info(f"DEBUG: Target Variable = CPIAUCSL (Headline CPI) Log Difference")
        
        # DEBUG: Calculate RW RMSE for 2016-2022 period specifically
        y_debug = df['inflation_rate']
        y_test_debug = y_debug.loc['2016-01-01':'2022-09-01']
        actuals_debug = y_test_debug.values
        preds_debug = y_debug.shift(1).loc['2016-01-01':'2022-09-01'].values
        rw_rmse_debug = np.sqrt(np.nanmean((actuals_debug - preds_debug)**2))
        logger.info(f"DEBUG: RW RMSE (2016-2022) = {rw_rmse_debug:.6f} (Expected: ~0.0027)")
        
        if mean_abs_inf > 0.1:
            logger.warning("ALARM: Inflation rate seems too high (>0.1). Is this Pct Change? It should be ~0.003 for Log Diff.")

        # NOW rename columns to V1, V2, etc. (after target is calculated)
        # Note: inflation_rate column will also be renamed, so we save and restore it
        inflation_rate_backup = df['inflation_rate'].copy()
        df = df.drop('inflation_rate', axis=1)
        df.columns = [f'V{i+1}' for i in range(len(df.columns))]
        df['inflation_rate'] = inflation_rate_backup

        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

        return df

    def apply_basic_transforms(self, df):
        """
        Apply basic stationary transformations to all raw features

        Parameters:
        -----------
        df : pd.DataFrame
            Raw macroeconomic data

        Returns:
        --------
        pd.DataFrame : Basic transformed features
        """
        logger.info("Applying basic transformations...")
        transformed_df = pd.DataFrame(index=df.index)

        for feature in self.raw_features:
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found in data, skipping")
                continue

            series = df[feature]

            # Skip if all NaN
            if series.isna().all():
                logger.warning(f"Feature {feature} is all NaN, skipping")
                continue

            # Percentage change (stationary for most economic series)
            pct_change = series.pct_change()
            transformed_df[f'{feature}_pct_change'] = pct_change

            # First difference
            diff = series.diff()
            transformed_df[f'{feature}_diff'] = diff

            # Log difference (for positive series)
            if (series > 0).all():
                log_diff = np.log(series).diff()
                transformed_df[f'{feature}_log_diff'] = log_diff

            # Year-over-year change (seasonally adjusted)
            yoy_change = (series - series.shift(12)) / series.shift(12).abs()
            transformed_df[f'{feature}_yoy_change'] = yoy_change

            # Quarter-over-quarter change
            qoq_change = (series - series.shift(3)) / series.shift(3).abs()
            transformed_df[f'{feature}_qoq_change'] = qoq_change

            # Store feature metadata
            self.feature_info[f'{feature}_pct_change'] = {
                'source': feature,
                'transform': 'pct_change',
                'description': f'{feature} percentage change'
            }
            self.feature_info[f'{feature}_diff'] = {
                'source': feature,
                'transform': 'diff',
                'description': f'{feature} first difference'
            }

        logger.info(f"Basic transforms completed: {len(transformed_df.columns)} features created")
        return transformed_df

    def apply_rolling_statistics(self, df):
        """
        Apply rolling statistics to raw features

        Parameters:
        -----------
        df : pd.DataFrame
            Raw macroeconomic data

        Returns:
        --------
        pd.DataFrame : Rolling statistics features
        """
        logger.info("Applying rolling statistics...")
        rolling_df = pd.DataFrame(index=df.index)

        for feature in self.raw_features:
            if feature not in df.columns:
                continue

            series = df[feature]

            # Calculate rolling statistics for different windows
            rolling_stats = calculate_rolling_statistics(series, windows=ROLLING_WINDOWS)

            # Add to main dataframe with proper naming
            for col in rolling_stats.columns:
                new_name = f'{feature}_{col}'
                rolling_df[new_name] = rolling_stats[col]

                self.feature_info[new_name] = {
                    'source': feature,
                    'transform': 'rolling_stat',
                    'window': int(col.split('_')[-1]),
                    'stat_type': '_'.join(col.split('_')[:-1]),
                    'description': f'{feature} {col}'
                }

        logger.info(f"Rolling statistics completed: {len(rolling_df.columns)} features created")
        return rolling_df

    def apply_momentum_features(self, df):
        """
        Apply momentum and acceleration features

        Parameters:
        -----------
        df : pd.DataFrame
            Raw macroeconomic data

        Returns:
        --------
        pd.DataFrame : Momentum features
        """
        logger.info("Applying momentum features...")
        momentum_df = pd.DataFrame(index=df.index)

        for feature in self.raw_features:
            if feature not in df.columns:
                continue

            series = df[feature]

            # Calculate momentum features
            momentum_features = calculate_momentum_features(series, horizons=MOMENTUM_HORIZONS)

            # Add to main dataframe
            for col in momentum_features.columns:
                new_name = f'{feature}_{col}'
                momentum_df[new_name] = momentum_features[col]

                self.feature_info[new_name] = {
                    'source': feature,
                    'transform': 'momentum',
                    'horizon': int(col.split('_')[-1]),
                    'momentum_type': col.split('_')[0],
                    'description': f'{feature} {col}'
                }

        logger.info(f"Momentum features completed: {len(momentum_df.columns)} features created")
        return momentum_df

    def apply_volatility_features(self, df):
        """
        Apply volatility-based features

        Parameters:
        -----------
        df : pd.DataFrame
            Raw macroeconomic data

        Returns:
        --------
        pd.DataFrame : Volatility features
        """
        logger.info("Applying volatility features...")
        volatility_df = pd.DataFrame(index=df.index)

        for feature in self.raw_features:
            if feature not in df.columns:
                continue

            series = df[feature]

            # Calculate volatility features
            vol_features = calculate_volatility_features(series, windows=ROLLING_WINDOWS)

            # Add to main dataframe
            for col in vol_features.columns:
                new_name = f'{feature}_{col}'
                volatility_df[new_name] = vol_features[col]

                self.feature_info[new_name] = {
                    'source': feature,
                    'transform': 'volatility',
                    'window': int(col.split('_')[-1]) if col.split('_')[-1].isdigit() else None,
                    'vol_type': '_'.join(col.split('_')[:-1]),
                    'description': f'{feature} {col}'
                }

        logger.info(f"Volatility features completed: {len(volatility_df.columns)} features created")
        return volatility_df

    def apply_zscore_features(self, df):
        """
        Apply z-score and outlier detection features

        Parameters:
        -----------
        df : pd.DataFrame
            Raw macroeconomic data

        Returns:
        --------
        pd.DataFrame : Z-score features
        """
        logger.info("Applying z-score features...")
        zscore_df = pd.DataFrame(index=df.index)

        for feature in self.raw_features:
            if feature not in df.columns:
                continue

            series = df[feature]

            # Z-score with default window (12 months)
            zscore = calculate_zscore(series, window=12)
            zscore_df[f'{feature}_zscore'] = zscore

            # Outlier flags
            outlier_flag = abs(zscore) > 2
            zscore_df[f'{feature}_outlier_flag'] = outlier_flag.astype(int)

            # Outlier magnitude
            outlier_magnitude = abs(zscore) * outlier_flag.astype(int)
            zscore_df[f'{feature}_outlier_magnitude'] = outlier_magnitude

            # Z-score change (acceleration of deviations)
            zscore_change = zscore.diff()
            zscore_df[f'{feature}_zscore_change'] = zscore_change

            # Store metadata
            for suffix in ['zscore', 'outlier_flag', 'outlier_magnitude', 'zscore_change']:
                col_name = f'{feature}_{suffix}'
                self.feature_info[col_name] = {
                    'source': feature,
                    'transform': 'zscore',
                    'zscore_type': suffix,
                    'description': f'{feature} {suffix}'
                }

        logger.info(f"Z-score features completed: {len(zscore_df.columns)} features created")
        return zscore_df

    def apply_cross_sectional_features(self, df):
        """
        Apply cross-sectional features (ratios, spreads, correlations)

        Parameters:
        -----------
        df : pd.DataFrame
            Raw macroeconomic data

        Returns:
        --------
        pd.DataFrame : Cross-sectional features
        """
        logger.info("Applying cross-sectional features...")
        cross_df = pd.DataFrame(index=df.index)

        # Pre-calculate key series
        # NOTE: After sasdate is set as index, all columns shift by 1
        # FEDFUNDS=V77, GS5=V82, GS10=V83, AAA=V84, BAA=V85, UNRATE=V24, PAYEMS=V32, CPIAUCSL=V105, OIL=V103
        if 'V77' in df.columns and 'V82' in df.columns:  # FEDFUNDS and GS5
            # Yield curve slope (GS5 - FEDFUNDS)
            yield_curve = df['V82'] - df['V77']  # GS5 - FEDFUNDS
            cross_df['yield_curve_slope'] = yield_curve

            # Term premium (GS10 - GS5) if GS10 exists
            if 'V83' in df.columns:  # GS10
                term_premium = df['V83'] - df['V82']  # GS10 - GS5
                cross_df['term_premium'] = term_premium

        if 'V84' in df.columns and 'V85' in df.columns:  # AAA and BAA spreads
            # Credit spread
            credit_spread = df['V85'] - df['V84']  # BAA - AAA
            cross_df['credit_spread'] = credit_spread

        if 'V24' in df.columns and 'V32' in df.columns:  # UNRATE and PAYEMS
            # Employment intensity (inverse unemployment adjusted)
            employment_intensity = df['V32'].pct_change() / (1 + df['V24']/100)
            cross_df['employment_intensity'] = employment_intensity

        # Rolling correlations (if sufficient data)
        if len(df) > 60:  # At least 5 years of data
            # Correlation between UNRATE and CPIAUCSL
            if 'V24' in df.columns and 'V105' in df.columns:
                corr_unrate_cpi = df['V24'].rolling(60).corr(df['V105'])
                cross_df['corr_unrate_cpi'] = corr_unrate_cpi

            # Correlation between OILPRICEx and CPIAUCSL
            if 'V103' in df.columns and 'V105' in df.columns:
                corr_oil_cpi = df['V103'].rolling(60).corr(df['V105'])
                cross_df['corr_oil_cpi'] = corr_oil_cpi

        # Store metadata
        for col in cross_df.columns:
            self.feature_info[col] = {
                'source': 'cross_sectional',
                'transform': 'cross_sectional',
                'description': f'Cross-sectional feature: {col}'
            }

        logger.info(f"Cross-sectional features completed: {len(cross_df.columns)} features created")
        return cross_df

    def get_all_stationary_features(self, df):
        """
        Apply all feature engineering transformations

        Parameters:
        -----------
        df : pd.DataFrame
            Raw macroeconomic data

        Returns:
        --------
        pd.DataFrame : All stationary features
        """
        logger.info("Starting complete feature engineering pipeline...")

        # Apply all transformations
        basic_features = self.apply_basic_transforms(df)
        rolling_features = self.apply_rolling_statistics(df)
        momentum_features = self.apply_momentum_features(df)
        volatility_features = self.apply_volatility_features(df)
        zscore_features = self.apply_zscore_features(df)
        cross_sectional_features = self.apply_cross_sectional_features(df)

        # Combine all features
        all_features = pd.concat([
            basic_features,
            rolling_features,
            momentum_features,
            volatility_features,
            zscore_features,
            cross_sectional_features
        ], axis=1)

        # Remove any remaining NaN columns
        all_features = all_features.dropna(axis=1, how='all')

        # Store final feature list
        self.transformed_features = all_features.columns.tolist()

        logger.info(f"Feature engineering completed: {len(self.transformed_features)} stationary features created")
        logger.info(f"Feature matrix shape: {all_features.shape}")

        return all_features

    def check_feature_stationarity(self, features_df, sample_size=0.1):
        """
        Check stationarity of transformed features

        Parameters:
        -----------
        features_df : pd.DataFrame
            Transformed features
        sample_size : float
            Fraction of features to check (for performance)

        Returns:
        --------
        dict : Stationarity results
        """
        logger.info(f"Checking stationarity for {int(len(features_df.columns) * sample_size)} features...")

        results = {}
        features_to_check = np.random.choice(
            features_df.columns,
            size=int(len(features_df.columns) * sample_size),
            replace=False
        )

        for feature in features_to_check:
            series = features_df[feature].dropna()
            if len(series) > 20:  # Minimum observations for test
                results[feature] = check_stationarity(series)

        stationary_count = sum(1 for r in results.values() if r['stationary'])
        total_checked = len(results)

        logger.info(f"Stationarity check: {stationary_count}/{total_checked} features are stationary")

        self.stationarity_results = results
        return results

    def get_feature_summary(self):
        """
        Get summary of created features

        Returns:
        --------
        dict : Feature summary
        """
        transform_counts = {}
        source_counts = {}

        for feature, info in self.feature_info.items():
            # Count by transform type
            transform_type = info.get('transform', 'unknown')
            transform_counts[transform_type] = transform_counts.get(transform_type, 0) + 1

            # Count by source
            source = info.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1

        return {
            'total_features': len(self.transformed_features),
            'by_transform': transform_counts,
            'by_source': source_counts,
            'stationarity_sample': len(self.stationarity_results),
            'stationary_ratio': sum(1 for r in self.stationarity_results.values() if r['stationary']) / max(1, len(self.stationarity_results))
        }
