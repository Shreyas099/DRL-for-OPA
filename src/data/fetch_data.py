import yfinance as yf
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def fetch_data():
    """
    Fetch and preprocess data for portfolio allocation.

    Saves two files:
      - prices.csv            : raw adjusted close prices (used by MVO agent)
      - processed_features.csv: daily log returns per asset + 3 standardised
                                volatility indicators (used by the RL env)

    State matrix per the paper (Section 4.2):
      Rows 0-10 (assets): [w_i | r_{i,t-1}, ..., r_{i,t-59}]   (60-day history)
      Row  11   (cash):   [w_c | vol20_t, vol_ratio_t, vix_t, 0, ...]

    Volatility indicators (Section 5.1):
      - vol20      : 20-day rolling std of daily S&P 500 log returns
      - vol_ratio  : vol20 / vol60
      - vix        : VIX index value
      All three are standardised with an expanding window to prevent leakage.
      NOTE: the paper uses ONLY these three (vol20, vol_ratio, vix).
            The code previously stored vol60 separately — that is incorrect.
    """
    all_tickers = config.ASSETS + [config.MARKET_INDEX, config.VIX_INDEX]

    print(f"Fetching data from {config.START_DATE} to {config.END_DATE}...")
    raw = yf.download(all_tickers, start=config.START_DATE, end=config.END_DATE)["Close"]

    # Forward-fill then back-fill for newer ETFs (e.g. XLRE listed 2015)
    raw = raw.ffill().bfill()

    # ------------------------------------------------------------------ #
    # 1. Save raw prices for MVO (uses actual close prices, not log rtns) #
    # ------------------------------------------------------------------ #
    prices_df = raw[config.ASSETS].copy()
    prices_path = config.DATA_DIR / "prices.csv"
    prices_df.to_csv(prices_path)
    print(f"Raw prices saved to {prices_path}")

    # ------------------------------------------------------------------ #
    # 2. Daily log returns for every asset                                #
    # ------------------------------------------------------------------ #
    log_returns = np.log(raw / raw.shift(1))

    # ------------------------------------------------------------------ #
    # 3. Volatility indicators from S&P 500 (Section 5.1)                #
    #    Paper specifies EXACTLY three: vol20, vol20/vol60, VIX           #
    # ------------------------------------------------------------------ #
    sp500_ret = log_returns[config.MARKET_INDEX]

    vol20 = sp500_ret.rolling(window=20).std() * np.sqrt(252)
    vol60 = sp500_ret.rolling(window=60).std() * np.sqrt(252)
    vol_ratio = vol20 / vol60          # short-term vs long-term vol trend
    vix = raw[config.VIX_INDEX]

    # Expanding-window standardisation to prevent look-ahead bias
    def expanding_standardize(series, min_periods=config.LOOKBACK_WINDOW):
        mu  = series.expanding(min_periods=min_periods).mean()
        std = series.expanding(min_periods=min_periods).std()
        return (series - mu) / (std + 1e-8)

    vol20_std      = expanding_standardize(vol20)
    vol_ratio_std  = expanding_standardize(vol_ratio)
    vix_std        = expanding_standardize(vix)

    # ------------------------------------------------------------------ #
    # 4. Build processed_features DataFrame                               #
    #    Columns: {asset}_daily_ret (×11), vol20, vol_ratio, vix         #
    # ------------------------------------------------------------------ #
    features_df = pd.DataFrame(index=raw.index)

    for asset in config.ASSETS:
        features_df[f"{asset}_daily_ret"] = log_returns[asset]

    # FIX: paper uses only 3 vol indicators (was 4 in original code)
    features_df["vol20"]      = vol20_std
    features_df["vol_ratio"]  = vol_ratio_std
    features_df["vix"]        = vix_std

    # Drop rows with NaN (first ~60 rows due to rolling windows)
    features_df = features_df.dropna()

    output_path = config.DATA_DIR / "processed_features.csv"
    features_df.to_csv(output_path)
    print(f"Processed features saved to {output_path}")
    print(f"  Date range : {features_df.index[0].date()} → {features_df.index[-1].date()}")
    print(f"  Rows       : {len(features_df)}")
    print(f"  Columns    : {list(features_df.columns)}")

    return features_df


if __name__ == "__main__":
    fetch_data()
