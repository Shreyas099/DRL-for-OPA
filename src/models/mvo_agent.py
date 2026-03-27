import numpy as np
import pandas as pd
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class MVOAgent:
    """
    Mean-Variance Optimisation agent (paper Section 5.3).

    At every timestep the agent:
      1. Takes the past 60 days of actual asset prices.
      2. Estimates expected returns (sample mean, annualised).
      3. Estimates the covariance matrix using Ledoit-Wolf shrinkage.
      4. Solves the max-Sharpe optimisation (risk-free rate = 0).
      5. Returns the cleaned long-only weights.

    FIX vs original code
    --------------------
    The original evaluate.py reconstructed prices from cumulative log returns
    starting at 1.0, introducing systematic drift error. The MVO agent now
    receives the real prices directly from prices.csv (saved by fetch_data.py).
    """

    def __init__(self, lookback_window: int = config.LOOKBACK_WINDOW):
        self.lookback_window = lookback_window
        self.assets          = config.ASSETS

    # -------------------------------------------------------------------- #
    # Core weight calculation                                                #
    # -------------------------------------------------------------------- #

    def get_weights(self, historical_prices: pd.DataFrame) -> np.ndarray:
        """
        Calculate MVO portfolio weights given a (lookback_window × n_assets)
        price DataFrame.

        Falls back to equal weights if the optimiser fails (e.g. all-negative
        expected returns, singular covariance matrix).
        """
        n = len(self.assets)

        if len(historical_prices) < self.lookback_window:
            return np.ones(n) / n

        recent = historical_prices.iloc[-self.lookback_window:]

        # Expected returns: sample mean (annualised)  — paper Section 5.3
        mu = expected_returns.mean_historical_return(
            recent, returns_data=False, frequency=252
        )

        # Covariance: Ledoit-Wolf shrinkage — paper Section 5.3
        S = risk_models.CovarianceShrinkage(
            recent, returns_data=False, frequency=252
        ).ledoit_wolf()

        # Enforce positive-semi-definite (paper Section 5.3):
        # "setting negative eigenvalues to 0, then rebuilding"
        eigenvalues, eigenvectors = np.linalg.eigh(S.values)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        S_psd = pd.DataFrame(
            eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T,
            index=S.index, columns=S.columns,
        )

        try:
            ef = EfficientFrontier(mu, S_psd, weight_bounds=(0, 1))
            ef.max_sharpe(risk_free_rate=0.0)          # same objective as DRL
            cleaned = ef.clean_weights()
            return np.array([cleaned.get(a, 0.0) for a in self.assets])
        except Exception:
            return np.ones(n) / n

    # -------------------------------------------------------------------- #
    # Simulation over a date range                                           #
    # -------------------------------------------------------------------- #

    def simulate(
        self,
        prices_df: pd.DataFrame,
        test_dates: pd.DatetimeIndex,
    ):
        """
        Simulate MVO portfolio over test_dates using real prices.

        Parameters
        ----------
        prices_df  : full prices DataFrame (all dates, indexed by date)
        test_dates : the dates to trade (out-of-sample test period)

        Returns
        -------
        daily_returns : list of float  (one per test date)
        all_weights   : list of np.ndarray
        """
        daily_returns = []
        all_weights   = []

        for date in test_dates:
            # Select the lookback window of prices ending just before `date`
            date_loc = prices_df.index.get_loc(date)
            if date_loc < self.lookback_window:
                w = np.ones(len(self.assets)) / len(self.assets)
            else:
                hist = prices_df.iloc[date_loc - self.lookback_window : date_loc]
                w    = self.get_weights(hist)

            all_weights.append(w)

            # Compute realised return on `date`
            if date_loc == 0:
                daily_returns.append(0.0)
                continue

            prev_prices  = prices_df.iloc[date_loc - 1][self.assets].values
            curr_prices  = prices_df.iloc[date_loc][self.assets].values
            asset_rets   = (curr_prices - prev_prices) / (prev_prices + 1e-10)
            daily_returns.append(float(np.dot(w, asset_rets)))

        return daily_returns, all_weights        # Optimize for Maximum Sharpe Ratio
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
