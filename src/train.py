import gc
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.env.portfolio_env import PortfolioEnv
from src.models.ppo_agent import PPOAgent

os.makedirs(config.MODELS_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def run_evaluation(env: PortfolioEnv, agent: PPOAgent):
    """
    Run one deterministic episode and return (Sharpe ratio, cumulative return).
    Used to select the best seed at the end of each training window.
    """
    state, _ = env.reset()
    done = truncated = False
    daily_returns = []

    while not (done or truncated):
        action = agent.predict(state, deterministic=True)
        state, _, done, truncated, info = env.step(action)
        daily_returns.append(info["return"])

    rets = np.array(daily_returns)
    if len(rets) < 2 or np.std(rets) == 0:
        sharpe = 0.0
    else:
        sharpe = np.sqrt(252) * rets.mean() / rets.std()

    cumret = (info["portfolio_value"] / config.INITIAL_BALANCE) - 1.0
    return sharpe, cumret


# --------------------------------------------------------------------------- #
# Walk-forward training pipeline                                                #
# --------------------------------------------------------------------------- #

def train_pipeline():
    data_path = config.DATA_DIR / "processed_features.csv"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run fetch_data.py first.")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    start_year = int(config.START_DATE[:4])

    # best agent from previous window is used to warm-start the next window
    best_agent_path = None

    for window in range(config.NUM_WINDOWS):
        train_start = f"{start_year + window}-01-01"
        train_end   = f"{start_year + window + config.WINDOW_TRAIN_YEARS - 1}-12-31"
        val_start   = f"{start_year + window + config.WINDOW_TRAIN_YEARS}-01-01"
        val_end     = f"{start_year + window + config.WINDOW_TRAIN_YEARS}-12-31"

        print(f"\n{'='*55}")
        print(f"Window {window + 1}/{config.NUM_WINDOWS}")
        print(f"  Train : {train_start} → {train_end}")
        print(f"  Val   : {val_start}   → {val_end}")
        print(f"{'='*55}")

        train_df = df.loc[train_start:train_end]
        val_df   = df.loc[val_start:val_end]

        if len(train_df) == 0 or len(val_df) == 0:
            print(f"  Skipping window {window + 1}: insufficient data.")
            continue

        best_val_sharpe          = -np.inf
        best_window_agent_path   = None

        for seed in range(config.NUM_SEEDS):
            print(f"\n  --- Seed {seed + 1}/{config.NUM_SEEDS} ---")

            agent = PPOAgent(df=train_df, seed=seed * 42)
            agent.train(
                total_timesteps=7_500_000,
                seed_model_path=best_agent_path,
            )

            # Evaluate on validation set
            val_env = PortfolioEnv(val_df)
            val_sharpe, val_ret = run_evaluation(val_env, agent)
            print(f"  Val Sharpe: {val_sharpe:.4f}  |  Val Return: {val_ret:.2%}")

            model_path = config.MODELS_DIR / f"ppo_window_{window}_seed_{seed}.zip"
            agent.save(model_path)

            if val_sharpe > best_val_sharpe:
                best_val_sharpe        = val_sharpe
                best_window_agent_path = model_path

            del agent
            gc.collect()

        print(f"\n  Best val Sharpe for window {window + 1}: {best_val_sharpe:.4f}")
        print(f"  Best model: {best_window_agent_path}")

        # Warm-start next window from best agent of this window
        best_agent_path = best_window_agent_path

    print("\nTraining complete.")


if __name__ == "__main__":
    train_pipeline()    return sharpe, cumulative_return

def train_pipeline():
    # Load processed data
    data_path = config.DATA_DIR / "processed_features.csv"
    if not data_path.exists():
        print(f"Error: Processed data not found at {data_path}. Please run fetch_data.py first.")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # 10 sliding windows from 2006 to 2021
    # Train: 5 years (e.g., 2006-2010), Val: 1 year (2011), Test: 1 year (2012)
    start_year = int(config.START_DATE[:4])

    # To pass weights from best agent of previous window to next window
    best_agent_path = None

    for window in range(config.NUM_WINDOWS):
        train_start = f"{start_year + window}-01-01"
        train_end = f"{start_year + window + config.WINDOW_TRAIN_YEARS - 1}-12-31"

        val_start = f"{start_year + window + config.WINDOW_TRAIN_YEARS}-01-01"
        val_end = f"{start_year + window + config.WINDOW_TRAIN_YEARS + config.WINDOW_VAL_YEARS - 1}-12-31"

        test_start = f"{start_year + window + config.WINDOW_TRAIN_YEARS + config.WINDOW_VAL_YEARS}-01-01"
        test_end = f"{start_year + window + config.WINDOW_TRAIN_YEARS + config.WINDOW_VAL_YEARS + config.WINDOW_TEST_YEARS - 1}-12-31"

        print(f"\n{'='*50}")
        print(f"Window {window + 1}/{config.NUM_WINDOWS}")
        print(f"Train: {train_start} to {train_end}")
        print(f"Val:   {val_start} to {val_end}")
        print(f"Test:  {test_start} to {test_end}")
        print(f"{'='*50}")

        train_df = df.loc[train_start:train_end]
        val_df = df.loc[val_start:val_end]

        if len(train_df) == 0 or len(val_df) == 0:
            print(f"Insufficient data for window {window+1}, skipping...")
            continue

        best_val_sharpe = -np.inf
        current_window_best_agent_path = None

        # Train with 5 different random seeds
        for seed in range(config.NUM_SEEDS):
            print(f"  Training Seed {seed + 1}/{config.NUM_SEEDS}...")
            agent = PPOAgent(df=train_df, seed=seed * 42)

            # Define timesteps
            # The paper mentions ~7.5M per round. For a small test, use 10,000.
            # Using 750,000 to keep it manageable locally, adjust as needed.
            timesteps = 7_500_000

            # If we have a best agent from the previous window, use its weights
            agent.train(total_timesteps=timesteps, seed_model_path=best_agent_path)

            # Evaluate on Validation set
            val_env = PortfolioEnv(val_df)
            val_sharpe, val_return = run_evaluation(val_env, agent)

            print(f"  Seed {seed + 1} - Val Sharpe: {val_sharpe:.4f}, Val Return: {val_return:.2%}")

            # Save the model
            model_path = config.MODELS_DIR / f"ppo_window_{window}_seed_{seed}.zip"
            agent.save(model_path)

            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                current_window_best_agent_path = model_path

            # Free memory
            del agent
            gc.collect()

        print(f"Best validation Sharpe for window {window+1}: {best_val_sharpe:.4f}")
        print(f"Best model saved at: {current_window_best_agent_path}")

        # Update best agent for next window's initialization
        best_agent_path = current_window_best_agent_path

if __name__ == "__main__":
    train_pipeline()
