import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from stable_baselines3 import PPO
from src.env.portfolio_env import PortfolioEnv
from src.models.mvo_agent import MVOAgent
from src.utils.metrics import calculate_metrics, print_metrics

os.makedirs(config.RESULTS_DIR, exist_ok=True)


def _run_drl_episode(test_df: pd.DataFrame, model_path, prices_df=None):
    """
    Run a single DRL agent on the test data.

    Returns a pd.Series of daily returns indexed by the TRADED dates.

    The env starts at current_step = lookback_window (60), so the first 60
    rows of test_df are consumed as history and never traded. The returned
    series is therefore indexed by test_df.index[60:] — the actual trading
    dates — not by test_df.index[0:].

    BUG FIXED: the original code returned a plain list and later used a
    global min_len to align dates, which silently discarded the last ~60
    days of every test year (600 trading days total = ~2.4 years missing
    from a 10-year backtest).
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    test_env = PortfolioEnv(test_df, prices_df=prices_df)
    vec_env  = DummyVecEnv([lambda: PortfolioEnv(test_df, prices_df=prices_df)])
    model    = PPO.load(str(model_path), env=vec_env)

    state, _ = test_env.reset()
    done = truncated = False
    daily_returns = []
    traded_steps  = []   # track which df rows were actually traded

    while not (done or truncated):
        action, _ = model.predict(state, deterministic=True)
        state, _, done, truncated, info = test_env.step(action)
        daily_returns.append(info["return"])
        traded_steps.append(test_env.current_step)   # index into test_df

    vec_env.close()

    # Map stepped indices back to dates
    traded_dates = test_df.index[traded_steps]
    return pd.Series(daily_returns, index=traded_dates)


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
    prices_df = pd.read_csv(prices_path,   index_col=0, parse_dates=True)

    start_year = int(config.START_DATE[:4])
    mvo_agent  = MVOAgent()

    # Accumulate per-window Series — concatenated at the end
    drl_pieces = []
    mvo_pieces = []

    # ------------------------------------------------------------------ #
    # Walk-forward evaluation                                              #
    # ------------------------------------------------------------------ #
    for window in range(config.NUM_WINDOWS):
        test_start = f"{start_year + window + config.WINDOW_TRAIN_YEARS + config.WINDOW_VAL_YEARS}-01-01"
        test_end   = f"{start_year + window + config.WINDOW_TRAIN_YEARS + config.WINDOW_VAL_YEARS}-12-31"

        test_df = df.loc[test_start:test_end]

        print(f"\nWindow {window + 1} — test: {test_start} → {test_end}")

        if len(test_df) == 0:
            print("  No test data, skipping.")
            continue

        # --- DRL: run ALL seeds and average (paper Section 6) ---
        seed_series = []
        for seed in range(config.NUM_SEEDS):
            model_path = config.MODELS_DIR / f"ppo_window_{window}_seed_{seed}.zip"
            if not model_path.exists():
                print(f"  WARNING: {model_path.name} not found, skipping seed.")
                continue

            print(f"  Running seed {seed} ({model_path.name}) ...")
            s = _run_drl_episode(test_df, model_path, prices_df=prices_df)
            seed_series.append(s)

        if not seed_series:
            print("  WARNING: no saved models found for this window.")
            continue

        # Align seeds on their common traded dates and average
        common_idx  = seed_series[0].index
        for s in seed_series[1:]:
            common_idx = common_idx.intersection(s.index)

        avg_returns = pd.Series(
            np.mean([s.loc[common_idx].values for s in seed_series], axis=0),
            index=common_idx,
        )
        drl_pieces.append(avg_returns)

        # --- MVO: simulate on the SAME traded dates ---
        # Use the full prices_df so MVO can look back before test_start
        mvo_daily, _ = mvo_agent.simulate(prices_df, common_idx)
        mvo_pieces.append(pd.Series(mvo_daily, index=common_idx))

    # ------------------------------------------------------------------ #
    # Concatenate across all windows                                       #
    # ------------------------------------------------------------------ #
    drl_series = pd.concat(drl_pieces).sort_index()
    mvo_series = pd.concat(mvo_pieces).sort_index()

    print(f"\nDRL series: {len(drl_series)} trading days  "
          f"({drl_series.index[0].date()} → {drl_series.index[-1].date()})")
    print(f"MVO series: {len(mvo_series)} trading days  "
          f"({mvo_series.index[0].date()} → {mvo_series.index[-1].date()})")

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #
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

    annual_df       = pd.DataFrame({"DRL": drl_annual, "MVO": mvo_annual})
    annual_df.index = annual_df.index.year

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
