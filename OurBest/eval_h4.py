"""
H4 ENHANCED - Horizon-Specific Feature Engineering
With longer rolling windows, trend features, and AR terms.
ABSOLUTELY NO LEAKAGE - Every feature is carefully verified.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from feature_engineering import StationaryFeatureEngineer
import warnings
warnings.filterwarnings('ignore')

HORIZON = 4
RF_PARAMS = {'n_estimators': 500, 'max_depth': 15, 'min_samples_leaf': 3, 'random_state': 42, 'n_jobs': -1}
XGB_PARAMS = {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'n_jobs': -1, 'verbosity': 0}
RF_WEIGHT = 0.5
N_FEATURES = 1000

print("="*70)
print(f"H4 ENHANCED - HORIZON-SPECIFIC FEATURES")
print("="*70)

# Load base features
fe = StationaryFeatureEngineer()
df_raw = fe.load_and_prepare_data("2025-12-MD.csv")
df_features = fe.get_all_stationary_features(df_raw)

monthly_inf = df_raw['inflation_rate']

print("\n=== ADDING HORIZON-SPECIFIC FEATURES ===")

# =============================================================================
# 1. LONGER ROLLING WINDOWS (24, 36 months)
# =============================================================================
print("1. Adding longer rolling windows (24, 36 months)...")

# Get raw V columns for extended rolling
raw_cols = [c for c in df_raw.columns if c.startswith('V') and c[1:].isdigit()]

for col in raw_cols[:20]:  # Top 20 V columns to avoid too many features
    series = df_raw[col]
    if series.isna().all():
        continue
    
    # 24-month rolling
    df_features[f'{col}_roll_mean_24'] = series.rolling(24, min_periods=12).mean()
    df_features[f'{col}_roll_std_24'] = series.rolling(24, min_periods=12).std()
    
    # 36-month rolling
    df_features[f'{col}_roll_mean_36'] = series.rolling(36, min_periods=18).mean()
    df_features[f'{col}_roll_std_36'] = series.rolling(36, min_periods=18).std()
    
    # YoY momentum (12-month rate of change)
    if (series > 0).all():
        df_features[f'{col}_yoy_roc'] = np.log(series).diff(12)

print(f"   Added features for {len(raw_cols[:20])} raw columns")

# =============================================================================
# 2. TREND FEATURES (Rolling Regression Slope)
# =============================================================================
print("2. Adding trend features (rolling regression slopes)...")

def rolling_slope(series, window):
    """Calculate rolling regression slope (trend strength)"""
    def calc_slope(x):
        if len(x) < window // 2:
            return np.nan
        y = x.values
        t = np.arange(len(y))
        # Simple linear regression slope
        slope = np.polyfit(t, y, 1)[0]
        return slope
    return series.rolling(window, min_periods=window//2).apply(calc_slope, raw=False)

# CPI trend (V105 = CPIAUCSL after renaming... actually we have inflation_rate)
log_cpi = np.log(df_raw['CPIAUCSL']) if 'CPIAUCSL' in df_raw.columns else monthly_inf.cumsum()

# 12-month and 24-month CPI trend
df_features['cpi_trend_12m'] = rolling_slope(monthly_inf, 12)
df_features['cpi_trend_24m'] = rolling_slope(monthly_inf, 24)

# Trend acceleration (change in trend)
df_features['cpi_trend_accel'] = df_features['cpi_trend_12m'].diff(6)

print("   Added CPI trend features")

# =============================================================================
# 3. AR TERMS (Lagged Inflation)
# =============================================================================
print("3. Adding AR terms (lagged inflation)...")

# CRITICAL LEAKAGE CHECK:
# At time t, we have features from t-1 (due to shift(1))
# We want to predict inf[t+4]
# Safe to use: inf[t-1], inf[t-2], inf[t-3], ... (all past)
# 
# These will be shifted with other features, so:
# AR feature at row t = inf[t-k].shift(1) = inf[t-k-1]... wait, no.
#
# Let me think more carefully:
# df_features is NOT shifted yet. Shift happens later: X_full_all = df_features.shift(1)
# So if we add inf.shift(k) to df_features, after the global shift(1):
#   inf.shift(k).shift(1) = inf.shift(k+1)
# At row t: inf[t-k-1]
#
# For AR terms, we want at prediction time t to use inf[t-1], inf[t-2], etc.
# So we need inf.shift(0), inf.shift(1), inf.shift(2) BEFORE the global shift
# After global shift(1): inf.shift(1), inf.shift(2), inf.shift(3)
# At row t: inf[t-1], inf[t-2], inf[t-3] ✓

# Actually wait - we need to match the feature shift logic
# Current features in df_features are at their original index
# Then X_full_all = df_features.shift(1) shifts everything by 1
# So if I add monthly_inf (which is inf[t] at row t) to df_features,
# after shift(1), at row t we get inf[t-1] ✓

# AR terms: past inflation values
df_features['ar_inf_lag1'] = monthly_inf  # After shift(1): inf[t-1]
df_features['ar_inf_lag2'] = monthly_inf.shift(1)  # After shift(1): inf[t-2]
df_features['ar_inf_lag3'] = monthly_inf.shift(2)  # After shift(1): inf[t-3]
df_features['ar_inf_lag4'] = monthly_inf.shift(3)  # After shift(1): inf[t-4]
df_features['ar_inf_lag6'] = monthly_inf.shift(5)  # After shift(1): inf[t-6]
df_features['ar_inf_lag12'] = monthly_inf.shift(11)  # After shift(1): inf[t-12]

# Rolling sums of past inflation
df_features['ar_inf_sum_3m'] = monthly_inf.rolling(3).sum()  # After shift: sum of t-1 to t-3
df_features['ar_inf_sum_6m'] = monthly_inf.rolling(6).sum()  # After shift: sum of t-1 to t-6
df_features['ar_inf_sum_12m'] = monthly_inf.rolling(12).sum()  # After shift: sum of t-1 to t-12

print("   Added 9 AR features")

# =============================================================================
# FINAL FEATURE COUNT
# =============================================================================
print(f"\nTotal features: {len(df_features.columns)}")

# =============================================================================
# CREATE TARGET (Point Forecast)
# =============================================================================
df_features['target_h'] = monthly_inf.shift(-HORIZON)  # inf[t+h]

# RW benchmark
rw_pred_series = monthly_inf.shift(1)

df_final = df_features.dropna(subset=['target_h']).fillna(0)
y_full = df_final['target_h']
X_full_all = df_final.drop('target_h', axis=1).shift(1)
X_full_all = X_full_all.replace([np.inf, -np.inf], np.nan).fillna(0)

TEST_START = '1990-01-01'
TEST_END = '2025-07-01'
test_indices = y_full.index[(y_full.index >= TEST_START) & (y_full.index <= TEST_END)]

print(f"\nHorizon: {HORIZON} months")
print(f"Test: {TEST_START} to {TEST_END} ({len(test_indices)} predictions)")
print("\n" + "="*70)

results = []
for i, idx in enumerate(test_indices):
    # CRITICAL: Prevent future target leakage
    train_end = idx - pd.DateOffset(months=HORIZON)
    train_mask = y_full.index < train_end
    if np.sum(train_mask) < 36:  # Need more data with extended windows
        continue
    
    X_train_all = X_full_all.loc[train_mask]
    y_train = y_full.loc[train_mask]
    
    # Rolling feature selection
    correlations = X_train_all.corrwith(y_train).abs().sort_values(ascending=False).dropna()
    top_features = correlations.head(N_FEATURES).index.tolist()
    
    X_train = X_train_all[top_features]
    X_test = X_full_all.loc[[idx], top_features]
    actual = y_full.loc[idx]
    
    # Random Walk
    pred_rw = rw_pred_series.loc[idx] if idx in rw_pred_series.index else 0
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestRegressor(**RF_PARAMS)
    xgb = XGBRegressor(**XGB_PARAMS)
    rf.fit(X_train_s, y_train)
    xgb.fit(X_train_s, y_train)
    
    pred = RF_WEIGHT * rf.predict(X_test_s)[0] + (1-RF_WEIGHT) * xgb.predict(X_test_s)[0]
    results.append({'date': idx, 'actual': actual, 'pred_model': pred, 'pred_rw': pred_rw})
    
    if (i+1) % 50 == 0:
        print(f"  Processed {i+1}/{len(test_indices)}...")

df_results = pd.DataFrame(results).set_index('date')
df_results['sq_error_model'] = (df_results['actual'] - df_results['pred_model']) ** 2
df_results['sq_error_rw'] = (df_results['actual'] - df_results['pred_rw']) ** 2
df_results.to_csv('predictions_h4_enhanced.csv')

rmse = np.sqrt(df_results['sq_error_model'].mean())
rmse_rw = np.sqrt(df_results['sq_error_rw'].mean())

print(f"\n{'='*70}")
print(f"H4 ENHANCED RESULTS")
print(f"RMSE: {rmse:.6f}, RW RMSE: {rmse_rw:.6f}, Ratio: {rmse/rmse_rw:.4f}")
print(f"{'='*70}")

def analyze(df, s, e, n):
    mask = (df.index >= s) & (df.index <= e)
    d = df.loc[mask]
    if len(d) == 0: return
    r_m = np.sqrt(d['sq_error_model'].mean())
    r_rw = np.sqrt(d['sq_error_rw'].mean())
    print(f"{n}: Ratio={r_m/r_rw:.4f}, N={len(d)}")

print("\nPeriod Analysis:")
analyze(df_results, '2016-01-01', '2022-09-01', '2016-2022')
analyze(df_results, '1990-01-01', '2015-12-01', '1990-2015')
analyze(df_results, '2020-01-01', '2022-12-01', '2020-2022')
print("="*70)
