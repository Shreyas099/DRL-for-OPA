import os
from pathlib import Path

# Paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Assets (11 S&P 500 Sector ETFs)
# XLRE (Real Estate) was created in 2015, so for a backtest starting from 2006,
# it might not have full data. We will fetch and handle missing data.
ASSETS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
MARKET_INDEX = "^GSPC"
VIX_INDEX = "^VIX"

# Dates
START_DATE = "2006-01-01"
END_DATE = "2021-12-31"

# Environment parameters
LOOKBACK_WINDOW = 60
INITIAL_BALANCE = 1_000_000
TRANSACTION_COST = 0.00  # Zero transaction costs as per the paper

# PPO Hyperparameters
PPO_PARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4, # linear decay will be handled in training
    "n_steps": 756,
    "batch_size": 1260, # Note: batch_size must be a factor of n_steps * n_envs (756 * 10 = 7560) -> 7560/1260 = 6
    "n_epochs": 16,
    "gamma": 0.9,
    "gae_lambda": 0.9,
    "clip_range": 0.25,
    "policy_kwargs": dict(net_arch=[64, 64])
}

# Walk-forward validation
NUM_WINDOWS = 10
WINDOW_TRAIN_YEARS = 5
WINDOW_VAL_YEARS = 1
WINDOW_TEST_YEARS = 1
NUM_SEEDS = 5
