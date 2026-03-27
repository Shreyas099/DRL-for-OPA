import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from stable_baselines3 import PPO
from src.env.portfolio_env import PortfolioEnv
from src.models.mvo_agent import MVOAgent
from src.utils.metrics import calculate_metrics, print_metrics

os.makedirs(config.RESULTS_DIR, exist_ok=True)


def _run_drl_episode(test_df: pd.DataFrame, model_path, seed: int, prices_df=None):
    """
    Run a single DRL agent on the test data and return a list of daily returns.

    Uses a lightweight single-env loader for inference — no need to spin up 10
    SubprocVecEnv worker processes just to run one deterministic episode.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    test_env = PortfolioEnv(test_df, prices_df=prices_df)
    # Load model with a single DummyVecEnv — avoids spawning 10 subprocess workers
    vec_env = DummyVecEnv([lambda: PortfolioEnv(test_df, prices_df=prices_df)])
    model = PPO.load(str(model_path), env=vec_env)

    state, _ = test_env.reset()
    done = truncated = False
    daily_returns = []

    while not (done or truncated):
        action, _ = model.predict(state, deterministic=True)
        state, _, done, truncated, info = test_env.step(action)
        daily_returns.append(info["return"])

    vec_env.close()
    return daily_returns


def evaluate_pipeline():
    # ------------------------------------------------------------------ #
    # Load data                                                            #
    # ------------------------------------------------------------------ #
    features_path = config.DATA_DIR / "processed_features.csv"
    prices_path   = config.DATA_DIR / "prices.csv"

    if not features_path.exists():
        print(f"ERROR: {features_path} not found. Run fetch_data.py first.")
        return
    if not prices_path.exists():
        print(f"ERROR: {prices_path} not found. Run fetch_data.py first.")
        return

    df        = pd.read_csv(features_path, index_col=0, parse_dates=True)

    # MVO uses actual prices, not prices reconstructed from log returns
    prices_df = pd.read_csv(prices_path,   index_col=0, parse_dates=True)

    start_year = int(config.START_DATE[:4])
    mvo_agent  = MVOAgent()

    # Paper Section 6: "For DRL, we average the performance across the
    # 5 agents (each trained on a different seed) for each year"
    drl_returns_avg = []   # averaged across seeds
    mvo_returns     = []
    test_dates      = []

    # ------------------------------------------------------------------ #
    # Walk-forward evaluation                                              #
    # ------------------------------------------------------------------ #
    for window in range(config.NUM_WINDOWS):
        val_start  = f"{start_year + window + config.WINDOW_TRAIN_YEARS}-01-01"
        val_end    = f"{start_year + window + config.WINDOW_TRAIN_YEARS}-12-31"
        test_start = f"{start_year + window + config.WINDOW_TRAIN_YEARS + config.WINDOW_VAL_YEARS}-01-01"
        test_end   = f"{start_year + window + config.WINDOW_TRAIN_YEARS + config.WINDOW_VAL_YEARS}-12-31"

        test_df = df.loc[test_start:test_end]

        print(f"\nWindow {window + 1} — test: {test_start} → {test_end}")

        if len(test_df) == 0:
            print("  No test data, skipping.")
            continue

        # --- DRL: run ALL seeds and average (paper Section 6) ---
        seed_returns = []
        for seed in range(config.NUM_SEEDS):
            model_path = config.MODELS_DIR / f"ppo_window_{window}_seed_{seed}.zip"
            if not model_path.exists():
                print(f"  WARNING: {model_path.name} not found, skipping seed.")
                continue

            print(f"  Running seed {seed} ({model_path.name}) ...")
            rets = _run_drl_episode(test_df, model_path, seed=seed * 42, prices_df=prices_df)
            seed_returns.append(rets)

        if len(seed_returns) == 0:
            print("  WARNING: no saved models found for this window.")
            continue

        # Average daily returns across all seeds for this window
        min_seed_len = min(len(r) for r in seed_returns)
        avg_rets = np.mean(
            [r[:min_seed_len] for r in seed_returns], axis=0
        ).tolist()
        drl_returns_avg.extend(avg_rets)

        # --- MVO ---
        test_prices_dates = prices_df.index.intersection(test_df.index)
        mvo_daily, _ = mvo_agent.simulate(prices_df, test_prices_dates)
        mvo_returns.extend(mvo_daily)

        test_dates.extend(test_df.index.tolist())

    # ------------------------------------------------------------------ #
    # Align series and compute metrics                                     #
    # ------------------------------------------------------------------ #
    min_len = min(len(drl_returns_avg), len(mvo_returns), len(test_dates) - 1)

    common_dates = test_dates[1 : min_len + 1]
    drl_series   = pd.Series(drl_returns_avg[:min_len], index=common_dates)
    mvo_series   = pd.Series(mvo_returns[1:min_len+1],  index=common_dates)

    print("\n" + "=" * 55)
    print("Out-of-sample results  [2012 – 2021]")
    print("=" * 55)
    metrics_df = pd.DataFrame({
        "DRL": calculate_metrics(drl_series),
        "MVO": calculate_metrics(mvo_series),
    })
    print_metrics(metrics_df)
    metrics_df.to_csv(config.RESULTS_DIR / "evaluation_metrics.csv")

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #
    _plot_cumulative(drl_series, mvo_series)
    _plot_annual(drl_series, mvo_series)

    print(f"\nResults saved to {config.RESULTS_DIR}")


def _plot_cumulative(drl: pd.Series, mvo: pd.Series):
    fig, ax = plt.subplots(figsize=(12, 5))
    (1 + drl).cumprod().plot(ax=ax, label="DRL")
    (1 + mvo).cumprod().plot(ax=ax, label="MVO")
    ax.set_title("Cumulative returns: DRL vs MVO (out-of-sample 2012–2021)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (starting at 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "cumulative_returns.png", dpi=150)
    plt.close(fig)


def _plot_annual(drl: pd.Series, mvo: pd.Series):
    drl_annual = drl.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    mvo_annual = mvo.resample("YE").apply(lambda x: (1 + x).prod() - 1)

    annual_df         = pd.DataFrame({"DRL": drl_annual, "MVO": mvo_annual})
    annual_df.index   = annual_df.index.year

    fig, ax = plt.subplots(figsize=(12, 5))
    annual_df.plot.bar(ax=ax)
    ax.set_title("Annual returns: DRL vs MVO")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual return")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "annual_returns.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    evaluate_pipeline()
