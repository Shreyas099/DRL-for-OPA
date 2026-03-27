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
    Run one deterministic episode and return (mean episode reward, cumulative return).

    Paper Section 5.2: "we save the best performing agent (based on highest
    mean episode validation reward)". The reward is the DSR at each step.
    """
    state, _ = env.reset()
    done = truncated = False
    episode_rewards = []

    while not (done or truncated):
        action = agent.predict(state, deterministic=True)
        state, reward, done, truncated, info = env.step(action)
        episode_rewards.append(reward)

    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    cumret = (info["portfolio_value"] / config.INITIAL_BALANCE) - 1.0
    return mean_reward, cumret


# --------------------------------------------------------------------------- #
# Walk-forward training pipeline                                                #
# --------------------------------------------------------------------------- #

def train_pipeline():
    data_path = config.DATA_DIR / "processed_features.csv"
    prices_path = config.DATA_DIR / "prices.csv"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run fetch_data.py first.")
        return
    if not prices_path.exists():
        print(f"ERROR: {prices_path} not found. Run fetch_data.py first.")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
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

        best_val_reward          = -np.inf
        best_window_agent_path   = None

        for seed in range(config.NUM_SEEDS):
            print(f"\n  --- Seed {seed + 1}/{config.NUM_SEEDS} ---")

            agent = PPOAgent(df=train_df, seed=seed * 42, prices_df=prices_df)
            agent.train(
                total_timesteps=7_500_000,
                seed_model_path=best_agent_path,
            )

            # Evaluate on validation set — select by mean episode reward (paper §5.2)
            val_env = PortfolioEnv(val_df, prices_df=prices_df)
            val_reward, val_ret = run_evaluation(val_env, agent)
            print(f"  Val Mean Reward: {val_reward:.6f}  |  Val Return: {val_ret:.2%}")

            model_path = config.MODELS_DIR / f"ppo_window_{window}_seed_{seed}.zip"
            agent.save(model_path)

            if val_reward > best_val_reward:
                best_val_reward        = val_reward
                best_window_agent_path = model_path

            del agent
            gc.collect()

        print(f"\n  Best val reward for window {window + 1}: {best_val_reward:.6f}")
        print(f"  Best model: {best_window_agent_path}")

        # Warm-start next window from best agent of this window
        best_agent_path = best_window_agent_path

    print("\nTraining complete.")


if __name__ == "__main__":
    train_pipeline()

