"""
train_worker.py — Single-seed training worker.

Called by train.py for each seed in parallel via subprocess:

    python src/train_worker.py --window W --seed S [--seed_model PATH]

Trains one PPO agent for one walk-forward window, evaluates on the
validation set, saves the model and writes the result to a small JSON
file so the parent orchestrator can collect results and pick the winner.
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.env.portfolio_env import PortfolioEnv
from src.models.ppo_agent import PPOAgent

os.makedirs(config.MODELS_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Validation helper (same logic as train.py)                                   #
# --------------------------------------------------------------------------- #

def run_evaluation(env: PortfolioEnv, agent: PPOAgent):
    """
    Run one deterministic episode and return mean DSR reward.
    Paper §5.2: best agent selected by highest mean episode validation reward.
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
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train one PPO seed for one window.")
    parser.add_argument("--window",     type=int,   required=True,   help="Window index (0-based)")
    parser.add_argument("--seed",       type=int,   required=True,   help="Seed index (0-based)")
    parser.add_argument("--seed_model", type=str,   default=None,    help="Warm-start model path")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load data                                                            #
    # ------------------------------------------------------------------ #
    data_path   = config.DATA_DIR / "processed_features.csv"
    prices_path = config.DATA_DIR / "prices.csv"

    df        = pd.read_csv(data_path,   index_col=0, parse_dates=True)
    prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)

    start_year  = int(config.START_DATE[:4])
    window      = args.window
    seed_idx    = args.seed

    train_start = f"{start_year + window}-01-01"
    train_end   = f"{start_year + window + config.WINDOW_TRAIN_YEARS - 1}-12-31"
    val_start   = f"{start_year + window + config.WINDOW_TRAIN_YEARS}-01-01"
    val_end     = f"{start_year + window + config.WINDOW_TRAIN_YEARS}-12-31"

    train_df = df.loc[train_start:train_end]
    val_df   = df.loc[val_start:val_end]

    if len(train_df) == 0 or len(val_df) == 0:
        print(f"[W{window} S{seed_idx}] Insufficient data — skipping.")
        return

    # ------------------------------------------------------------------ #
    # Train                                                                #
    # ------------------------------------------------------------------ #
    actual_seed = seed_idx * 42
    print(f"\n[W{window} S{seed_idx}] Training (seed={actual_seed}) for 7.5M timesteps ...")

    agent = PPOAgent(df=train_df, seed=actual_seed, prices_df=prices_df)
    agent.train(
        total_timesteps=7_500_000,
        seed_model_path=args.seed_model,
    )

    # ------------------------------------------------------------------ #
    # Evaluate on validation set                                           #
    # ------------------------------------------------------------------ #
    val_env = PortfolioEnv(val_df, prices_df=prices_df)
    val_reward, val_ret = run_evaluation(val_env, agent)
    print(f"[W{window} S{seed_idx}] Val Mean Reward: {val_reward:.6f}  |  Val Return: {val_ret:.2%}")

    # ------------------------------------------------------------------ #
    # Save model                                                           #
    # ------------------------------------------------------------------ #
    model_path = config.MODELS_DIR / f"ppo_window_{window}_seed_{seed_idx}.zip"
    agent.save(model_path)
    agent.envs.close()

    # ------------------------------------------------------------------ #
    # Write result JSON for orchestrator (atomic write avoids partial reads)#
    # ------------------------------------------------------------------ #
    result = {
        "window":      window,
        "seed":        seed_idx,
        "val_reward":  val_reward,
        "val_return":  val_ret,
        "model_path":  str(model_path),
    }
    result_path = config.MODELS_DIR / f"result_window_{window}_seed_{seed_idx}.json"
    # Write to a temp file in the same directory, then atomically rename.
    # This guarantees the orchestrator never reads a partially-written file.
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=config.MODELS_DIR, prefix=f".tmp_result_w{window}_s{seed_idx}_", suffix=".json"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp_path, result_path)  # atomic on POSIX; near-atomic on Windows
    except Exception:
        os.unlink(tmp_path)  # clean up temp file on failure
        raise

    print(f"[W{window} S{seed_idx}] Done. Model → {model_path.name}")


if __name__ == "__main__":
    main()
