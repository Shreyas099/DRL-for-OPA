"""
run_etf.py  —  DRL portfolio optimization on the 4-ETF dataset from
"Hopfield Networks for Asset Allocation" (Nicolini et al., 2024).

Assets : AGG, DBC, VTI, VIX  (2006-02-06 → 2023-12-29)
Method : walk-forward PPO — same algorithm as the main pipeline
Output : results_etf/etf_cumulative_returns_RL.png
         results_etf/etf_weights_RL.csv

Walk-forward windows (test years 2012–2023):
  Window 0 : train 2006–2010 | val 2011 | test 2012
  Window 1 : train 2007–2011 | val 2012 | test 2013
  ...
  Window 11: train 2017–2021 | val 2022 | test 2023

Usage:
  python run_etf.py            # full run (7.5 M steps × 3 seeds × 12 windows)
  python run_etf.py --quick    # 500 k steps × 1 seed  (sanity check)
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# ── Must patch config BEFORE importing env / agent ────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402  (imported for side-effects on path)

# ETF experiment settings
ETF_ASSETS  = ["AGG", "DBC", "VTI", "VIX"]   # clean column-name keys
ETF_TICKERS = ["AGG", "DBC", "VTI", "^VIX"]  # yfinance tickers (^VIX → VIX)

START_DATE         = "2006-02-06"
END_DATE           = "2023-12-29"
LOOKBACK_WINDOW    = 60
NUM_WINDOWS        = 12   # test years 2012–2023
WINDOW_TRAIN_YEARS = 5
WINDOW_VAL_YEARS   = 1
N_ENVS             = 4    # SubprocVecEnv workers per seed
NUM_SEEDS          = 3
TOTAL_TIMESTEPS    = 7_500_000

DATA_DIR    = config.BASE_DIR / "data_etf"
MODELS_DIR  = config.BASE_DIR / "saved_models_etf"
RESULTS_DIR = config.BASE_DIR / "results_etf"
for _d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Patch config so PortfolioEnv / PPOAgent see the ETF assets at runtime
config.ASSETS          = ETF_ASSETS
config.LOOKBACK_WINDOW = LOOKBACK_WINDOW
config.N_ENVS          = N_ENVS

# Now safe to import env / agent (they read config.ASSETS at runtime)
from stable_baselines3 import PPO                          # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from src.env.portfolio_env import PortfolioEnv             # noqa: E402
from src.models.ppo_agent  import PPOAgent                 # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_preprocess() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download (or load cached) prices for the four ETFs and build the
    features DataFrame expected by PortfolioEnv._get_state().

    Returns
    -------
    prices_df    : raw adjusted close prices  (for whole-share rebalancing)
    features_df  : daily log-returns per asset + 3 vol indicators
    """
    prices_path   = DATA_DIR / "prices.csv"
    features_path = DATA_DIR / "processed_features.csv"

    if prices_path.exists() and features_path.exists():
        print("Cached ETF data found — loading from disk.")
        return (
            pd.read_csv(prices_path,   index_col=0, parse_dates=True),
            pd.read_csv(features_path, index_col=0, parse_dates=True),
        )

    print(f"Fetching ETF data {START_DATE} → {END_DATE} ...")
    raw = yf.download(ETF_TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)["Close"]
    raw.columns = ETF_ASSETS          # rename ^VIX → VIX
    raw = raw.ffill().bfill()

    # --- raw prices (MVO / whole-share rebalancing) --------------------------
    prices_df = raw[ETF_ASSETS].copy()
    prices_df.to_csv(prices_path)
    print(f"  Prices saved to {prices_path}")

    # --- log returns ---------------------------------------------------------
    log_ret = np.log(raw / raw.shift(1))

    # --- vol indicators from VTI (total-market proxy ≈ S&P 500) -------------
    vti_ret   = log_ret["VTI"]
    vol20     = vti_ret.rolling(20).std() * np.sqrt(252)
    vol60     = vti_ret.rolling(60).std() * np.sqrt(252)
    vol_ratio = vol20 / (vol60 + 1e-10)
    vix_raw   = raw["VIX"]            # raw VIX level used as fear gauge

    def expanding_std(s: pd.Series, min_p: int = LOOKBACK_WINDOW) -> pd.Series:
        mu  = s.expanding(min_p).mean()
        std = s.expanding(min_p).std()
        return (s - mu) / (std + 1e-8)

    features_df = pd.DataFrame(index=raw.index)
    for asset in ETF_ASSETS:
        features_df[f"{asset}_daily_ret"] = log_ret[asset]
    features_df["vol20"]     = expanding_std(vol20)
    features_df["vol_ratio"] = expanding_std(vol_ratio)
    features_df["vix"]       = expanding_std(vix_raw)
    features_df = features_df.dropna()

    features_df.to_csv(features_path)
    print(f"  Features saved to {features_path}")
    print(f"  Date range : {features_df.index[0].date()} → {features_df.index[-1].date()}")
    print(f"  Rows / cols: {len(features_df)} / {list(features_df.columns)}")

    return prices_df, features_df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TRAINING  (sequential walk-forward, single process)
# ─────────────────────────────────────────────────────────────────────────────

def _quick_eval(
    model_path: str,
    val_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> tuple[float, float]:
    """Evaluate a saved model on val_df; return (mean_reward, cumulative_return)."""
    vec_env = DummyVecEnv([lambda: PortfolioEnv(val_df, prices_df=prices_df)])
    model   = PPO.load(str(model_path), env=vec_env)
    env     = PortfolioEnv(val_df, prices_df=prices_df)
    state, _ = env.reset()
    done = truncated = False
    rewards = []
    while not (done or truncated):
        action, _ = model.predict(state, deterministic=True)
        state, r, done, truncated, info = env.step(action)
        rewards.append(r)
    vec_env.close()
    mean_r = float(np.mean(rewards)) if rewards else 0.0
    cumret = info["portfolio_value"] / config.INITIAL_BALANCE - 1.0
    return mean_r, cumret


def train_pipeline(
    df: pd.DataFrame,
    prices_df: pd.DataFrame,
    total_timesteps: int,
    num_seeds: int,
) -> None:
    start_year      = int(START_DATE[:4])
    best_model_path = None   # warm-start from best agent of previous window

    for window in range(NUM_WINDOWS):
        train_start = f"{start_year + window}-01-01"
        train_end   = f"{start_year + window + WINDOW_TRAIN_YEARS - 1}-12-31"
        val_start   = f"{start_year + window + WINDOW_TRAIN_YEARS}-01-01"
        val_end     = f"{start_year + window + WINDOW_TRAIN_YEARS}-12-31"

        train_df = df.loc[train_start:train_end]
        val_df   = df.loc[val_start:val_end]

        if len(train_df) == 0 or len(val_df) == 0:
            print(f"\nWindow {window+1}: insufficient data — skipping.")
            continue

        print(f"\n{'='*55}")
        print(f"Window {window+1}/{NUM_WINDOWS}  "
              f"[train {train_start[:4]}–{train_end[:4]} | val {val_start[:4]}]")

        best_val, best_seed_path = -np.inf, None

        for seed_idx in range(num_seeds):
            actual_seed = seed_idx * 42
            model_path  = MODELS_DIR / f"ppo_window_{window}_seed_{seed_idx}.zip"

            if model_path.exists():
                print(f"  Seed {seed_idx}: cached model found — skipping training.")
            else:
                print(f"  Seed {seed_idx}: training for {total_timesteps:,} steps ...")
                agent = PPOAgent(df=train_df, seed=actual_seed, prices_df=prices_df)
                agent.train(
                    total_timesteps=total_timesteps,
                    seed_model_path=best_model_path,
                )
                agent.save(model_path)
                agent.envs.close()
                print(f"  Seed {seed_idx}: model saved → {model_path.name}")

            val_reward, val_ret = _quick_eval(str(model_path), val_df, prices_df)
            print(f"  Seed {seed_idx}: val_reward={val_reward:+.6f}  val_return={val_ret:+.2%}")

            if val_reward > best_val:
                best_val       = val_reward
                best_seed_path = model_path

        best_model_path = best_seed_path
        print(f"  Best: {best_seed_path.name}  (reward={best_val:+.6f})")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def _run_episode(
    test_df: pd.DataFrame,
    model_path: str,
    prices_df: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run one deterministic episode; return (daily_returns, weights_df)."""
    vec_env = DummyVecEnv([lambda: PortfolioEnv(test_df, prices_df=prices_df)])
    model   = PPO.load(str(model_path), env=vec_env)
    env     = PortfolioEnv(test_df, prices_df=prices_df)

    state, _ = env.reset()
    done = truncated = False
    daily_returns, weights_list, traded_steps = [], [], []

    while not (done or truncated):
        action, _ = model.predict(state, deterministic=True)
        state, _, done, truncated, info = env.step(action)
        daily_returns.append(info["return"])
        weights_list.append(info["weights"])
        traded_steps.append(env.current_step)

    vec_env.close()

    traded_dates = test_df.index[traded_steps]
    returns_s  = pd.Series(daily_returns, index=traded_dates)
    weights_df = pd.DataFrame(
        weights_list,
        index=traded_dates,
        columns=ETF_ASSETS + ["cash"],
    )
    return returns_s, weights_df


def evaluate_pipeline(
    df: pd.DataFrame,
    prices_df: pd.DataFrame,
    num_seeds: int,
) -> tuple[pd.Series, pd.DataFrame]:
    start_year    = int(START_DATE[:4])
    drl_pieces    = []
    weight_pieces = []

    for window in range(NUM_WINDOWS):
        test_start = (f"{start_year + window + WINDOW_TRAIN_YEARS + WINDOW_VAL_YEARS}"
                      f"-01-01")
        test_end   = (f"{start_year + window + WINDOW_TRAIN_YEARS + WINDOW_VAL_YEARS}"
                      f"-12-31")
        test_df    = df.loc[test_start:test_end]

        if len(test_df) == 0:
            continue

        seed_series, seed_weights = [], []
        for seed_idx in range(num_seeds):
            mp = MODELS_DIR / f"ppo_window_{window}_seed_{seed_idx}.zip"
            if not mp.exists():
                continue
            s, w = _run_episode(test_df, str(mp), prices_df)
            seed_series.append(s)
            seed_weights.append(w)

        if not seed_series:
            print(f"Window {window+1}: no models found — skipping.")
            continue

        # Align seeds on common traded dates, then average
        common_idx = seed_series[0].index
        for s in seed_series[1:]:
            common_idx = common_idx.intersection(s.index)

        avg_ret = pd.Series(
            np.mean([s.loc[common_idx].values for s in seed_series], axis=0),
            index=common_idx,
        )
        avg_wgt = sum(w.loc[common_idx] for w in seed_weights) / len(seed_weights)

        drl_pieces.append(avg_ret)
        weight_pieces.append(avg_wgt)
        print(f"Window {window+1}: test {test_start[:4]} — {len(avg_ret)} trading days")

    if not drl_pieces:
        raise RuntimeError("No test data found — check model paths and date ranges.")

    drl_series   = pd.concat(drl_pieces).sort_index()
    weights_full = pd.concat(weight_pieces).sort_index()
    return drl_series, weights_full


# ─────────────────────────────────────────────────────────────────────────────
# 4.  METRICS & PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def _avg_drawdown(returns: pd.Series) -> float:
    cum      = (1 + returns).cumprod()
    roll_max = cum.cummax()
    dd       = (cum - roll_max) / (roll_max + 1e-10)
    return float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0


def report_and_save(drl_series: pd.Series, weights_df: pd.DataFrame) -> None:
    import empyrical as ep

    ann_sharpe  = ep.sharpe_ratio(drl_series)
    ann_sortino = ep.sortino_ratio(drl_series)
    ann_return  = ep.annual_return(drl_series)
    max_dd      = ep.max_drawdown(drl_series)
    avg_dd      = _avg_drawdown(drl_series)

    print("\n" + "=" * 50)
    print("Out-of-sample  (ETF dataset, RL/PPO)")
    print(f"  Period  : {drl_series.index[0].date()} → {drl_series.index[-1].date()}")
    print("=" * 50)
    print(f"  {'Annualized return':<28} {ann_return:+.4f}")
    print(f"  {'Annualized Sharpe ratio':<28} {ann_sharpe:.4f}")
    print(f"  {'Sortino ratio':<28} {ann_sortino:.4f}")
    print(f"  {'Average drawdown':<28} {avg_dd:.4f}")
    print(f"  {'Max drawdown':<28} {max_dd:.4f}")
    print("=" * 50)

    # ── cumulative returns plot ───────────────────────────────────────────────
    cum = (1 + drl_series).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cum.index, cum.values, label="DRL (PPO)", color="steelblue", linewidth=1.5)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title(
        "Cumulative Returns — ETF Dataset (AGG, DBC, VTI, VIX)\n"
        f"DRL-PPO  |  Sharpe {ann_sharpe:.2f}  |  Ann. return {ann_return:.1%}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (starting at 1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "etf_cumulative_returns_RL.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot  saved → {plot_path}")

    # ── weights CSV ──────────────────────────────────────────────────────────
    weights_path = RESULTS_DIR / "etf_weights_RL.csv"
    weights_df.to_csv(weights_path)
    print(f"Weights saved → {weights_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DRL on the 4-ETF dataset.")
    p.add_argument(
        "--quick", action="store_true",
        help="Quick sanity-check run: 500 k timesteps, 1 seed.",
    )
    p.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; evaluate saved models and produce plots.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    total_timesteps = 500_000 if args.quick else TOTAL_TIMESTEPS
    num_seeds       = 1       if args.quick else NUM_SEEDS

    print("=" * 55)
    print("DRL Portfolio Optimization — ETF Dataset")
    print(f"  Assets      : {ETF_ASSETS}")
    print(f"  Period      : {START_DATE} → {END_DATE}")
    print(f"  Windows     : {NUM_WINDOWS}  (test years 2012–2023)")
    print(f"  Seeds       : {num_seeds}")
    print(f"  Timesteps   : {total_timesteps:,} per seed")
    print(f"  N_ENVS      : {N_ENVS}")
    print("=" * 55)

    prices_df, features_df = fetch_and_preprocess()

    if not args.eval_only:
        train_pipeline(features_df, prices_df, total_timesteps, num_seeds)

    drl_series, weights_df = evaluate_pipeline(features_df, prices_df, num_seeds)
    report_and_save(drl_series, weights_df)
