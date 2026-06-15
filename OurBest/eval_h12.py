"""
H12 ENHANCED - Horizon-Specific Feature Engineering
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

HORIZON = 12
RF_PARAMS = {'n_estimators': 500, 'max_depth': 15, 'min_samples_leaf': 3, 'random_state': 42, 'n_jobs': -1}
XGB_PARAMS = {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'n_jobs': -1, 'verbosity': 0}
RF_WEIGHT = 0.5
N_FEATURES = 1000

print("="*70)
print(f"H12 ENHANCED - HORIZON-SPECIFIC FEATURES")
print("="*70)

# Load base features
fe = StationaryFeatureEngineer()
df_raw = fe.load_and_prepare_data("2025-12-MD.csv")
df_features = fe.get_all_stationary_features(df_raw)

monthly_inf = df_raw['inflation_rate']

print("\n=== ADDING HORIZON-SPECIFIC FEATURES ===")

# 1. LONGER ROLLING WINDOWS (24, 36 months)
print("1. Adding longer rolling windows (24, 36 months)...")
raw_cols = [c for c in df_raw.columns if c.startswith('V') and c[1:].isdigit()]

for col in raw_cols[:20]:
    series = df_raw[col]
    if series.isna().all():
        continue
    df_features[f'{col}_roll_mean_24'] = series.rolling(24, min_periods=12).mean()
    df_features[f'{col}_roll_std_24'] = series.rolling(24, min_periods=12).std()
    df_features[f'{col}_roll_mean_36'] = series.rolling(36, min_periods=18).mean()
    df_features[f'{col}_roll_std_36'] = series.rolling(36, min_periods=18).std()
    if (series > 0).all():
        df_features[f'{col}_yoy_roc'] = np.log(series).diff(12)

print(f"   Added features for {len(raw_cols[:20])} raw columns")

# 2. TREND FEATURES
print("2. Adding trend features...")
def rolling_slope(series, window):
    def calc_slope(x):
        if len(x) < window // 2:
            return np.nan
        y = x.values
        t = np.arange(len(y))
        slope = np.polyfit(t, y, 1)[0]
        return slope
    return series.rolling(window, min_periods=window//2).apply(calc_slope, raw=False)

df_features['cpi_trend_12m'] = rolling_slope(monthly_inf, 12)
df_features['cpi_trend_24m'] = rolling_slope(monthly_inf, 24)
df_features['cpi_trend_accel'] = df_features['cpi_trend_12m'].diff(6)
print("   Added CPI trend features")

# 3. AR TERMS
print("3. Adding AR terms...")
df_features['ar_inf_lag1'] = monthly_inf
df_features['ar_inf_lag2'] = monthly_inf.shift(1)
df_features['ar_inf_lag3'] = monthly_inf.shift(2)
df_features['ar_inf_lag4'] = monthly_inf.shift(3)
df_features['ar_inf_lag6'] = monthly_inf.shift(5)
df_features['ar_inf_lag12'] = monthly_inf.shift(11)
df_features['ar_inf_sum_3m'] = monthly_inf.rolling(3).sum()
df_features['ar_inf_sum_6m'] = monthly_inf.rolling(6).sum()
df_features['ar_inf_sum_12m'] = monthly_inf.rolling(12).sum()
print("   Added 9 AR features")

print(f"\nTotal features: {len(df_features.columns)}")

# TARGET & RW
df_features['target_h'] = monthly_inf.shift(-HORIZON)
rw_pred_series = monthly_inf.shift(1)

df_final = df_features.dropna(subset=['target_h']).fillna(0)
y_full = df_final['target_h']
X_full_all = df_final.drop('target_h', axis=1).shift(1)
X_full_all = X_full_all.replace([np.inf, -np.inf], np.nan).fillna(0)

TEST_START = '1990-01-01'
TEST_END = '2024-11-01'  # Earlier end due to longer horizon
test_indices = y_full.index[(y_full.index >= TEST_START) & (y_full.index <= TEST_END)]

print(f"\nHorizon: {HORIZON} months")
print(f"Test: {TEST_START} to {TEST_END} ({len(test_indices)} predictions)")
print("\n" + "="*70)

results = []
for i, idx in enumerate(test_indices):
    # CRITICAL: Prevent future target leakage
    train_end = idx - pd.DateOffset(months=HORIZON)
    train_mask = y_full.index < train_end
    if np.sum(train_mask) < 36: continue
    
    X_train_all = X_full_all.loc[train_mask]
    y_train = y_full.loc[train_mask]
    
    correlations = X_train_all.corrwith(y_train).abs().sort_values(ascending=False).dropna()
    top_features = correlations.head(N_FEATURES).index.tolist()
    
    X_train = X_train_all[top_features]
    X_test = X_full_all.loc[[idx], top_features]
    actual = y_full.loc[idx]
    
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
df_results.to_csv('predictions_h12.csv')

rmse = np.sqrt(df_results['sq_error_model'].mean())
rmse_rw = np.sqrt(df_results['sq_error_rw'].mean())

print(f"\n{'='*70}")
print(f"H12 ENHANCED RESULTS")
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
analyze(df_results, '1990-01-01', '2000-12-01', '1990-2000')
analyze(df_results, '2001-01-01', '2015-12-01', '2001-2015')
analyze(df_results, '2016-01-01', '2022-09-01', '2016-2022')
analyze(df_results, '2020-01-01', '2022-12-01', '2020-2022')
analyze(df_results, '1990-01-01', '2022-12-01', '1990-2022')
print("="*70)
