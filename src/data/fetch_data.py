import yfinance as yf
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

def fetch_data():
    """Fetch and preprocess data for portfolio allocation."""
    all_tickers = config.ASSETS + [config.MARKET_INDEX, config.VIX_INDEX]

    print(f"Fetching data from {config.START_DATE} to {config.END_DATE}...")
    df = yf.download(all_tickers, start=config.START_DATE, end=config.END_DATE)['Close']

    # Fill missing values for newer ETFs like XLRE by filling backwards or using proxy (SPY)
    # Actually, backward fill might be bad for backtesting. Forward fill first, then backward fill if needed.
    df = df.ffill().bfill()

    print("Calculating features...")
    # Calculate daily log returns
    returns = np.log(df / df.shift(1))

    # We need 60-day log returns (which means rolling sum of daily log returns)
    returns_60d = returns.rolling(window=config.LOOKBACK_WINDOW).sum()

    # Volatility metrics based on S&P 500 (^GSPC)
    sp500_ret = returns[config.MARKET_INDEX]

    vol20 = sp500_ret.rolling(window=20).std() * np.sqrt(252)
    vol60 = sp500_ret.rolling(window=60).std() * np.sqrt(252)
    vol_ratio = vol20 / vol60

    vix = df[config.VIX_INDEX]

    # Standardize volatility metrics using an expanding window
    # To prevent data leakage, we use expanding mean and std
    def expanding_standardize(series, min_periods=config.LOOKBACK_WINDOW):
        expanding_mean = series.expanding(min_periods=min_periods).mean()
        expanding_std = series.expanding(min_periods=min_periods).std()
        return (series - expanding_mean) / (expanding_std + 1e-8)

    vol20_std = expanding_standardize(vol20)
    vol60_std = expanding_standardize(vol60)
    vol_ratio_std = expanding_standardize(vol_ratio)
    vix_std = expanding_standardize(vix)

    # Combine features into a single dataframe
    features_df = pd.DataFrame(index=df.index)

    # Store daily returns for the portfolio environment to calculate portfolio values
    for asset in config.ASSETS:
        features_df[f"{asset}_daily_ret"] = returns[asset]
        features_df[f"{asset}_60d_ret"] = returns_60d[asset]

    features_df['vol20'] = vol20_std
    features_df['vol60'] = vol60_std
    features_df['vol_ratio'] = vol_ratio_std
    features_df['vix'] = vix_std

    # Drop NaNs created by rolling windows
    features_df = features_df.dropna()

    # Save the processed data
    output_path = config.DATA_DIR / "processed_features.csv"
    features_df.to_csv(output_path)
    print(f"Data saved to {output_path}")

    return features_df

if __name__ == "__main__":
    fetch_data()
