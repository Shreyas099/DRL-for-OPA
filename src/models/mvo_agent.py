import pandas as pd
import numpy as np
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class MVOAgent:
    """Mean-Variance Optimization Agent."""

    def __init__(self, lookback_window=config.LOOKBACK_WINDOW):
        self.lookback_window = lookback_window
        self.assets = config.ASSETS

    def get_weights(self, historical_prices):
        """
        Calculate MVO weights given historical prices up to current timestep.
        historical_prices: DataFrame of shape (lookback_window, len(assets))
        """
        if len(historical_prices) < self.lookback_window:
             # Not enough data, return equal weights
             return np.ones(len(self.assets)) / len(self.assets)

        # Use only the last `lookback_window` days
        recent_prices = historical_prices.iloc[-self.lookback_window:]

        # Calculate expected returns (sample mean)
        mu = expected_returns.mean_historical_return(recent_prices, returns_data=False, frequency=252)

        # Calculate covariance matrix using Ledoit-Wolf Shrinkage
        S = risk_models.CovarianceShrinkage(recent_prices, returns_data=False, frequency=252).ledoit_wolf()

        # Optimize for Maximum Sharpe Ratio
        # Set bounds to (0, 1) for long-only portfolio
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            raw_weights = ef.max_sharpe(risk_free_rate=0.0) # Assume risk-free rate is 0 as in DRL
            cleaned_weights = ef.clean_weights()

            # Convert dict back to array
            weights = np.array([cleaned_weights.get(asset, 0.0) for asset in self.assets])
            return weights

        except Exception as e:
            # If optimization fails (e.g. all returns negative), return equal weights
            # print(f"Optimization failed: {e}")
            return np.ones(len(self.assets)) / len(self.assets)

    def simulate(self, prices_df, start_idx, end_idx):
        """
        Simulate MVO performance over a period.
        prices_df: DataFrame of all asset prices (unadjusted daily closes or adj closes)
        """
        portfolio_value = config.INITIAL_BALANCE
        values = [portfolio_value]

        # Weights: (N assets) + Cash. MVO will not hold cash unless forced, so cash=0.
        all_weights = []

        for t in range(start_idx, end_idx):
            # historical prices from (t - lookback) to t-1
            # Ensure we have enough history. If t < lookback, we can't do it, but start_idx should be >= lookback
            hist_prices = prices_df.iloc[t-self.lookback_window:t][self.assets]

            # Get weights for today
            w = self.get_weights(hist_prices)
            all_weights.append(w)

            # Calculate today's return
            # The return at time t is (Price_t - Price_{t-1}) / Price_{t-1}
            daily_returns = prices_df.iloc[t][self.assets] / prices_df.iloc[t-1][self.assets] - 1
            daily_returns = daily_returns.values

            portfolio_return = np.sum(w * daily_returns)
            portfolio_value *= (1 + portfolio_return)
            values.append(portfolio_value)

        return values, all_weights
