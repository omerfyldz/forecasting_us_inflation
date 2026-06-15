# Configuration for Feature Engineering

# All 127 FRED-MD Variables
RAW_FEATURES = [f'V{i}' for i in range(1, 128)]

# Rolling Window Configurations
ROLLING_WINDOWS = [3, 6, 12]

# Momentum Configuration
MOMENTUM_HORIZONS = [3, 6, 12]
