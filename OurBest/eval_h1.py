"""
CLEAN ENSEMBLE - Rolling Feature Selection
Academically pure: feature selection done for EACH prediction
using only data available at that time.

NO LEAKAGE - NO LOOK-AHEAD BIAS
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from feature_engineering import StationaryFeatureEngineer
import warnings
warnings.filterwarnings('ignore')

# Model parameters (default, not tuned)
RF_PARAMS = {'n_estimators': 500, 'max_depth': 15, 'min_samples_leaf': 3, 'random_state': 42, 'n_jobs': -1}
XGB_PARAMS = {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'n_jobs': -1, 'verbosity': 0}
RF_WEIGHT = 0.5
N_FEATURES = 1000

print("="*70)
print("CLEAN ENSEMBLE - ROLLING FEATURE SELECTION")
print("No leakage: feature selection per prediction")
print("="*70)

# Load data
fe = StationaryFeatureEngineer()
df_raw = fe.load_and_prepare_data("2025-12-MD.csv")
df_features = fe.get_all_stationary_features(df_raw)
df_features['target_inflation'] = df_raw['inflation_rate']
df_final = df_features.dropna(subset=['target_inflation']).fillna(0)

y_full = df_final['target_inflation']
X_full_all = df_final.drop('target_inflation', axis=1).shift(1)
X_full_all = X_full_all.replace([np.inf, -np.inf], np.nan).fillna(0)

TEST_START = '1990-01-01'
TEST_END = '2025-11-01'
test_indices = y_full.index[(y_full.index >= TEST_START) & (y_full.index <= TEST_END)]

print(f"\nFeatures available: {len(X_full_all.columns)}")
print(f"Features to select: {N_FEATURES} (per prediction)")
print(f"Test: {TEST_START} to {TEST_END} ({len(test_indices)} months)")
print("\nRunning with ROLLING feature selection...")

results = []
for i, idx in enumerate(test_indices):
    train_mask = y_full.index < idx
    if np.sum(train_mask) < 24: continue
    
    X_train_all = X_full_all.loc[train_mask]
    y_train = y_full.loc[train_mask]
    
    # ROLLING FEATURE SELECTION - Only use training data!
    correlations = X_train_all.corrwith(y_train).abs().sort_values(ascending=False).dropna()
    top_features = correlations.head(N_FEATURES).index.tolist()
    
    # Now extract selected features
    X_train = X_train_all[top_features]
    X_test = X_full_all.loc[[idx], top_features]
    actual = y_full.loc[idx]
    
    # Random Walk benchmark
    prev_idx = y_full.index[y_full.index < idx][-1]
    pred_rw = y_full.loc[prev_idx]
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train models
    rf = RandomForestRegressor(**RF_PARAMS)
    xgb = XGBRegressor(**XGB_PARAMS)
    rf.fit(X_train_s, y_train)
    xgb.fit(X_train_s, y_train)
    
    # Ensemble prediction
    pred = RF_WEIGHT * rf.predict(X_test_s)[0] + (1-RF_WEIGHT) * xgb.predict(X_test_s)[0]
    
    results.append({'date': idx, 'actual': actual, 'pred_model': pred, 'pred_rw': pred_rw})
    
    if (i+1) % 50 == 0:
        print(f"  Processed {i+1}/{len(test_indices)}...")

# Save results
df_results = pd.DataFrame(results).set_index('date')
df_results['sq_error_model'] = (df_results['actual'] - df_results['pred_model']) ** 2
df_results['sq_error_rw'] = (df_results['actual'] - df_results['pred_rw']) ** 2
df_results.to_csv('predictions_clean.csv')

# Calculate metrics
rmse = np.sqrt(df_results['sq_error_model'].mean())
rmse_rw = np.sqrt(df_results['sq_error_rw'].mean())

print(f"\n{'='*70}")
print(f"FULL PERIOD 1990-2025")
print(f"RMSE: {rmse:.6f}, Ratio: {rmse/rmse_rw:.4f}")
print(f"{'='*70}")

def analyze(df, s, e, n):
    mask = (df.index >= s) & (df.index <= e)
    d = df.loc[mask]
    if len(d) == 0: return
    r_m = np.sqrt(d['sq_error_model'].mean())
    r_rw = np.sqrt(d['sq_error_rw'].mean())
    print(f"{n}: RMSE={r_m:.6f}, Ratio={r_m/r_rw:.4f}, N={len(d)}")

print("\nPeriod Analysis:")
analyze(df_results, '2016-01-01', '2022-09-01', '2016-2022')
analyze(df_results, '1990-01-01', '2015-12-01', '1990-2015')
analyze(df_results, '2020-01-01', '2022-12-01', '2020-2022')
analyze(df_results, '2023-01-01', '2025-11-01', '2023-2025')
analyze(df_results, '1990-01-01', '2000-12-01', '1990-2000')
analyze(df_results, '2001-01-01', '2015-12-01', '2001-2015')

print("="*70)
print("DONE - Results saved to predictions_clean.csv")
print("="*70)
