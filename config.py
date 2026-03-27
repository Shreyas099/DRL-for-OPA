import os
from pathlib import Path
import torch

# Paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Assets (11 S&P 500 Sector ETFs)
ASSETS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
MARKET_INDEX = "^GSPC"
VIX_INDEX = "^VIX"

# Dates
START_DATE = "2006-01-01"
END_DATE = "2021-12-31"

# Environment parameters
LOOKBACK_WINDOW = 60
INITIAL_BALANCE = 100_000
TRANSACTION_COST = 0.00

# PPO Hyperparameters — exactly as in Table 1 of the paper
PPO_PARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,          # linear decay to 1e-5 applied in agent
    "n_steps": 756,
    "batch_size": 1260,
    "n_epochs": 16,
    "gamma": 0.9,
    "gae_lambda": 0.9,
    "clip_range": 0.25,
    "policy_kwargs": dict(
        net_arch=[64, 64],
        activation_fn=torch.nn.Tanh,  # paper Table 1: tanh activation
        log_std_init=-1,              # paper Table 1: log_std_init = -1
    )
}

# Walk-forward validation
NUM_WINDOWS = 10
WINDOW_TRAIN_YEARS = 5
WINDOW_VAL_YEARS = 1
WINDOW_TEST_YEARS = 1
NUM_SEEDS = 5
N_ENVS = 10                        # paper uses 10 parallel environments
