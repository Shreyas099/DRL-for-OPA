import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.env.portfolio_env import PortfolioEnv
from src.models.mvo_agent import MVOAgent
from src.models.ppo_agent import PPOAgent
from src.utils.metrics import calculate_metrics, print_metrics

os.makedirs(config.RESULTS_DIR, exist_ok=True)


def _best_model_for_window(window: int, val_df: pd.DataFrame) -> str | None:
    """
    Identify the best seed model for a given window by re-evaluating saved
    models on the validation set and returning the path to the best one.
    """
    val_env     = PortfolioEnv(val_df)
    best_sharpe = -np.inf
    best_path   = None

    for seed in range(config.NUM_SEEDS):
        model_path = config.MODELS_DIR / f"ppo_window_{window}_seed_{seed}.zip"
        if not model_path.exists():
            continue

        agent = PPOAgent(df=val_df, seed=seed * 42)
        try:
            agent.load(model_path)
        except Exception as e:
            print(f"    Could not load {model_path}: {e}")
            continue

        state, _ = val_env.reset()
        done = truncated = False
        rets = []
        while not (done or truncated):
            action = agent.predict(state, deterministic=True)
            state, _, done, truncated, info = val_env.step(action)
            rets.append(info["return"])

        rets = np.array(rets)
        sharpe = (
            np.sqrt(252) * rets.mean() / (rets.std() + 1e-8)
            if len(rets) > 1
            else 0.0
        )
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_path   = model_path

    return best_path, best_sharpe


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

    # FIX: MVO now uses actual prices, not prices reconstructed from log returns
    prices_df = pd.read_csv(prices_path,   index_col=0, parse_dates=True)

    start_year = int(config.START_DATE[:4])
    mvo_agent  = MVOAgent()

    drl_returns  = []
    mvo_returns  = []
    test_dates   = []

    # ------------------------------------------------------------------ #
    # Walk-forward evaluation                                              #
    # ------------------------------------------------------------------ #
    for window in range(config.NUM_WINDOWS):
        val_start  = f"{start_year + window + config.WINDOW_TRAIN_YEARS}-01-01"
        val_end    = f"{start_year + window + config.WINDOW_TRAIN_YEARS}-12-31"
        test_start = f"{start_year + window + config.WINDOW_TRAIN_YEARS + config.WINDOW_VAL_YEARS}-01-01"
        test_end   = f"{start_year + window + config.WINDOW_TRAIN_YEARS + config.WINDOW_VAL_YEARS}-12-31"

        val_df  = df.loc[val_start:val_end]
        test_df = df.loc[test_start:test_end]

        print(f"\nWindow {window + 1} — test: {test_start} → {test_end}")

        if len(test_df) == 0:
            print("  No test data, skipping.")
            continue

        # --- DRL ---
        best_path, best_sharpe = _best_model_for_window(window, val_df)
        if best_path:
            print(f"  Best model: {best_path.name}  (val Sharpe={best_sharpe:.4f})")
        else:
            print("  WARNING: no saved models found for this window.")

        test_env = PortfolioEnv(test_df)
        agent    = PPOAgent(df=test_df, seed=42)
        if best_path:
            agent.load(best_path)

        state, _ = test_env.reset()
        done = truncated = False
        while not (done or truncated):
            action = agent.predict(state, deterministic=True)
            state, _, done, truncated, info = test_env.step(action)
            drl_returns.append(info["return"])

        # --- MVO ---
        # Use real prices for the test dates (FIX)
        test_prices_dates = prices_df.index.intersection(test_df.index)
        mvo_daily, _ = mvo_agent.simulate(prices_df, test_prices_dates)
        mvo_returns.extend(mvo_daily)

        test_dates.extend(test_df.index.tolist())

    # ------------------------------------------------------------------ #
    # Align series and compute metrics                                     #
    # ------------------------------------------------------------------ #
    # DRL produces T-1 returns (env advances one step before returning)
    min_len = min(len(drl_returns), len(mvo_returns), len(test_dates) - 1)

    common_dates = test_dates[1 : min_len + 1]
    drl_series   = pd.Series(drl_returns[:min_len],  index=common_dates)
    mvo_series   = pd.Series(mvo_returns[1:min_len+1], index=common_dates)

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
