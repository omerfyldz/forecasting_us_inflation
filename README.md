# Inflation Forecasting with Machine Learning

A comprehensive Python implementation for inflation forecasting using machine learning methods. This project implements the methodology from **"Forecasting Inflation in a Data-Rich Environment: The Benefits of Machine Learning Methods"** by Medeiros et al. (2021).

## Overview

- **21+ Forecasting Methods**: Random Forest, XGBoost, LSTM, LASSO, Ridge, and more
- **Two Sample Configurations**: `without_dummy` and `with_dummy` for different handling of the 2008 crisis
- **Dynamic Period Selection**: Each model can run on different sample periods
- **RMSE by Horizon (h1-h12)**: All models output RMSE for each forecast horizon
- **Rolling Window Evaluation**: Expanding window forecasts with comprehensive error metrics

## Project Structure

```
48E-Project-Files/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── utils.py                    # Shared utility functions (load_csv, calculate_errors, add_outlier_dummy)
├── fred_md_loader.py           # FRED-MD data loader and transformer
├── run_all_models.py           # Master script to run all models
├── comprehensive_comparison.py # Full model comparison with visualization
│
├── without_dummy/              # Periods WITHOUT 2008 crisis (no outlier dummy)
│   ├── rawdata_1990_2000.csv   # Great Moderation period
│   ├── rawdata_2016_2022.csv   # COVID/inflation period (default)
│   ├── rawdata_2020_2022.csv   # Pandemic subset
│   ├── functions/              # 20+ model implementations
│   │   ├── func_rf.py          # Random Forest
│   │   ├── func_ar.py          # Autoregressive
│   │   ├── func_lasso.py       # LASSO/Ridge/Elastic Net
│   │   ├── func_xgb.py         # XGBoost
│   │   ├── func_nn.py          # Neural Network
│   │   ├── func_lstm.py        # LSTM
│   │   └── ...                 # And many more
│   └── run/                    # 26 execution scripts
│       ├── rf.py, ar.py, lasso.py, xgb.py, nn.py, ...
│
├── with_dummy/                 # Periods WITH 2008 crisis (with outlier dummy)
│   ├── rawdata_2001_2015.csv   # Financial crisis period
│   ├── rawdata_1990_2022.csv   # Full sample (default)
│   ├── functions/              # Model implementations (with dummy support)
│   └── run/                    # 26 execution scripts
│
└── OurBest/                    # Custom best model configuration
```

## Sample Configuration

The project is organized into two main sample types based on whether they include the 2008 financial crisis:

### `without_dummy` (No Outlier Dummy Required)

For periods that **do not include** the 2008 financial crisis extreme values.

| Period | Date Range | Observations | nprev | Description |
|--------|------------|--------------|-------|-------------|
| 1990_2000 | Jan 1990 - Dec 2000 | ~132 | 60 | Great Moderation |
| **2016_2022** | Jan 2016 - Dec 2022 | ~84 | 48 | COVID/inflation (**default**) |
| 2020_2022 | Jan 2020 - Dec 2022 | ~36 | 24 | Pandemic subset |

### `with_dummy` (Outlier Dummy Applied)

For periods that **include** the 2008 financial crisis. The `add_outlier_dummy()` function automatically adds a dummy variable at the minimum CPI value (typically 2008).

| Period | Date Range | Observations | nprev | Description |
|--------|------------|--------------|-------|-------------|
| 2001_2015 | Jan 2001 - Dec 2015 | ~180 | 84 | Financial crisis & recovery |
| **1990_2022** | Jan 1990 - Dec 2022 | ~396 | 132 | Full sample (**default**) |

## Installation

```bash
pip install -r requirements.txt
```

### Optional Dependencies

Some models require additional packages:
- **XGBoost models**: `pip install xgboost`
- **LSTM models**: `pip install tensorflow`
- **Visualization**: `pip install matplotlib`

## Usage

### Running Individual Models

```bash
# Run Random Forest (without_dummy - 2016-2022, no outlier dummy)
python without_dummy/run/rf.py

# Run Random Forest (with_dummy - 1990-2022, with outlier dummy)
python with_dummy/run/rf.py
```

### Changing the Period

Edit `CURRENT_PERIOD` in any run file:

```python
# without_dummy periods
PERIOD_CONFIG = {
    '1990_2000': {'nprev': 60},
    '2016_2022': {'nprev': 48},
    '2020_2022': {'nprev': 24},
}
CURRENT_PERIOD = '2016_2022'  # Change this

# with_dummy periods
PERIOD_CONFIG = {
    '2001_2015': {'nprev': 84},
    '1990_2022': {'nprev': 132},
}
CURRENT_PERIOD = '1990_2022'  # Change this
```

### Running All Models

```bash
# List all available models
python run_all_models.py --list-models

# Run a specific model
python run_all_models.py --model rf

# Run a specific period
python run_all_models.py --period 1990_2000 --sample first
```

### Output Format

All models output RMSE by horizon (h1-h12):

```
============================================================
Random Forest - RMSE by Horizon (h1-h12)
============================================================
Horizon    CPI RMSE        PCE RMSE       
----------------------------------------
h=1        0.012345        0.011234       
h=2        0.013456        0.012345       
...
h=12       0.018765        0.017654       
----------------------------------------
Average    0.015678        0.014567       
============================================================
```

## Available Models

### Benchmark
- **Random Walk (RW)** - Naive forecast benchmark

### Linear Models
- **AR** - Autoregressive (fixed lag and BIC-selected)
- **Ridge** - L2 regularization
- **LASSO** - L1 regularization
- **Elastic Net** - Combined L1/L2
- **Adaptive LASSO** - Weighted LASSO
- **SCAD** - Smoothly Clipped Absolute Deviation

### Machine Learning
- **Random Forest (RF)** - Ensemble of decision trees
- **XGBoost** - Gradient boosted trees
- **Gradient Boosting** - Sequential tree boosting
- **Bagging** - Bootstrap aggregating
- **Neural Network (NN)** - Feedforward MLP
- **LSTM** - Long Short-Term Memory (deep learning)

### Factor Methods
- **Factor Models** - Principal component regression
- **Target Factors** - Targeted factor extraction
- **CSR** - Complete Subset Regression
- **RF-OLS** - Random Forest + OLS combination

### Special Methods
- **LBVAR** - Large Bayesian VAR
- **UC-SV** - Unobserved Components with Stochastic Volatility
- **Jackknife** - Leave-one-out ensemble
- **Adalasso-RF** - Adaptive LASSO with RF weights
- **Polynomial LASSO** - LASSO with polynomial features
- **RF-LASSO** - Random Forest + LASSO hybrid

## Key Functions

### `utils.py`

```python
# Load data
Y = load_csv('rawdata_1990_2022.csv')

# Calculate forecast errors
errors = calculate_errors(actual, predicted)
# Returns: {'rmse': 0.123, 'mae': 0.098, 'mse': 0.015}

# Add outlier dummy (for with_dummy models)
Y = add_outlier_dummy(Y, target_col=0)
```

## Model Function Interface

All model functions follow a consistent interface:

```python
def model_rolling_window(Y, nprev, indice=1, lag=1, **kwargs):
    """
    Parameters:
    -----------
    Y : ndarray
        Data matrix (observations x features)
    nprev : int
        Number of out-of-sample predictions
    indice : int
        Target variable (1=CPI, 2=PCE)
    lag : int
        Forecast horizon (1-12 months)
    
    Returns:
    --------
    dict with keys:
        'pred': ndarray of predictions
        'errors': dict with 'rmse', 'mae'
    """
```

## References

Medeiros, M. C., Vasconcelos, G., Veiga, A., & Zilberman, E. (2018). **Forecasting Inflation in a Data-Rich Environment: The Benefits of Machine Learning Methods.** Journal of Business & Economic Statistics.

## License

Academic use only. Please cite the original paper when using this code.
