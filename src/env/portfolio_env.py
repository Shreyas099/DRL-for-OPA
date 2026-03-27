import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class PortfolioEnv(gym.Env):
    """
    Custom OpenAI Gym environment for portfolio allocation.

    Replicates the environment described in Section 4 of:
      'Deep Reinforcement Learning for Optimal Portfolio Allocation'

    State space (Section 4.2)
    -------------------------
    A flattened (n+1) × T matrix where n=11 assets, T=60 days:

        Row i  (asset i, i=0..10): [w_i, r_{i,t-1}, r_{i,t-2}, ..., r_{i,t-59}]
        Row 11 (cash)            : [w_c, vol20_t, vol_ratio_t, vix_t, 0, ..., 0]

    Total observation dimension: 12 × 60 = 720.

    Whole-share rebalancing (Section 4.5 & 5.4)
    --------------------------------------------
    The environment converts continuous portfolio weights into whole-share
    allocations: floor(w_i * portfolio_value / price_i) shares per asset.
    Remaining value is placed into cash. The resulting rebalanced weights
    are used in the state observation.

    Reward (Section 4.3) — Differential Sharpe Ratio
    -------------------------------------------------
    D_t = [B_{t-1}·ΔA_t − 0.5·A_{t-1}·ΔB_t] / (B_{t-1} − A_{t-1}²)^{3/2}

    where:
        ΔA_t = R_t − A_{t-1}
        ΔB_t = R_t² − B_{t-1}
        A_t  = A_{t-1} + η·ΔA_t
        B_t  = B_{t-1} + η·ΔB_t
        η    = 1/252   (one trading day on a yearly scale)

    A_0 = B_0 = 0.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        prices_df=None,
        initial_balance=config.INITIAL_BALANCE,
        lookback_window=config.LOOKBACK_WINDOW,
        transaction_cost=config.TRANSACTION_COST,
    ):
        super().__init__()

        self.df              = df.reset_index(drop=True)   # integer-indexed for speed
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window             # T = 60
        self.transaction_cost = transaction_cost

        # Prices for whole-share rebalancing (Section 4.5 & 5.4)
        # If prices_df is provided, align it to df's date index for rebalancing.
        if prices_df is not None:
            # Align prices to df's original date index before reset
            self.prices = prices_df[config.ASSETS].reindex(df.index).reset_index(drop=True)
        else:
            self.prices = None

        self.n_assets = len(config.ASSETS)                 # 11
        self.n_total  = self.n_assets + 1                  # 12  (11 ETFs + cash)

        # ---------------------------------------------------------------- #
        # Action space: continuous, softmax-normalised inside step()        #
        # ---------------------------------------------------------------- #
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_total,), dtype=np.float32
        )

        # ---------------------------------------------------------------- #
        # Observation space: flattened (n+1) × T = 12 × 60 = 720          #
        # ---------------------------------------------------------------- #
        self.state_dim = self.n_total * self.lookback_window   # 720
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # EMA update rate — paper: η ≈ 1/252
        self.eta = 1.0 / 252.0

        self.reset()

    # -------------------------------------------------------------------- #
    # Gym interface                                                          #
    # -------------------------------------------------------------------- #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start at step = lookback_window so we always have 60 days of history
        self.current_step    = self.lookback_window
        self.portfolio_value = self.initial_balance
        self.weights         = np.zeros(self.n_total, dtype=np.float32)
        self.weights[-1]     = 1.0   # start fully in cash

        # Differential Sharpe Ratio state (Section 4.3)
        # A_0 = B_0 = 0 as initialised in the paper
        self.A = 0.0
        self.B = 0.0

        return self._get_state(), {}

    def step(self, action):
        # 1. Convert raw action → portfolio weights via softmax
        new_weights = self._softmax(action)

        # Apply transaction cost (zero in paper, kept for extensibility)
        cost = np.sum(np.abs(new_weights - self.weights)) * self.transaction_cost
        self.portfolio_value *= (1.0 - cost)

        # 2. Whole-share rebalancing (paper Section 4.5 & 5.4)
        #    Convert continuous weights to whole shares, remainder → cash
        self.weights = self._rebalance_to_whole_shares(new_weights)

        # 3. Advance one day
        self.current_step += 1
        done      = self.current_step >= len(self.df) - 1
        truncated = False

        # 4. Realised portfolio return for this day
        row          = self.df.iloc[self.current_step]
        daily_log_r  = np.array(
            [row[f"{a}_daily_ret"] for a in config.ASSETS], dtype=np.float64
        )
        simple_r     = np.exp(daily_log_r) - 1.0          # simple returns
        portfolio_r  = float(np.dot(self.weights[:-1], simple_r))   # cash earns 0

        self.portfolio_value *= (1.0 + portfolio_r)

        # 5. Differential Sharpe Ratio reward (Section 4.3)
        reward = self._differential_sharpe(portfolio_r)

        # Bankruptcy guard
        if self.portfolio_value <= 0:
            done   = True
            reward = -1.0

        info = {
            "portfolio_value": self.portfolio_value,
            "return":          portfolio_r,
            "weights":         self.weights.copy(),
        }

        return self._get_state(), reward, done, truncated, info

    def render(self, mode="human"):
        print(f"Step {self.current_step} | Portfolio value: ${self.portfolio_value:,.2f}")

    # -------------------------------------------------------------------- #
    # Internal helpers                                                       #
    # -------------------------------------------------------------------- #

    def _rebalance_to_whole_shares(self, target_weights: np.ndarray) -> np.ndarray:
        """
        Whole-share rebalancing (paper Section 4.5 & 5.4).

        Converts continuous portfolio weights into whole-share allocations:
          shares_i = floor(w_i * portfolio_value / price_i)
        Remaining value (from rounding down) is placed into cash.
        Returns the rebalanced weight vector (n_total,).
        """
        if self.prices is None or self.current_step >= len(self.prices):
            # Fallback: use fractional weights if no price data available
            return target_weights

        prices_row = self.prices.iloc[self.current_step]
        asset_prices = prices_row.values.astype(np.float64)

        # Skip rebalancing if any price is zero/NaN
        if np.any(np.isnan(asset_prices)) or np.any(asset_prices <= 0):
            return target_weights

        # Allocate value to each asset based on target weights
        asset_weights = target_weights[:self.n_assets]
        allocated_values = asset_weights * self.portfolio_value

        # Floor to whole shares
        shares = np.floor(allocated_values / asset_prices)
        actual_values = shares * asset_prices

        # Remaining value goes to cash
        cash_value = self.portfolio_value - np.sum(actual_values)

        # Compute rebalanced weights
        rebalanced = np.zeros(self.n_total, dtype=np.float32)
        if self.portfolio_value > 0:
            rebalanced[:self.n_assets] = (actual_values / self.portfolio_value).astype(np.float32)
            rebalanced[-1] = float(cash_value / self.portfolio_value)

        return rebalanced

    def _get_state(self):
        """
        Build the (n+1) × T observation matrix and flatten it to 1D.

        Matrix layout (each row = 60 values):
          Row i  (asset i): [w_i, r_{i,t-1}, ..., r_{i,t-59}]
          Row 11 (cash)   : [w_c, vol20_t, vol_ratio_t, vix_t, 0, ..., 0]
        """
        t = self.current_step

        # Slice the past T rows of daily returns: shape (T, n_assets)
        window = self.df.iloc[t - self.lookback_window : t]

        state_matrix = np.zeros((self.n_total, self.lookback_window), dtype=np.float32)

        # Rows 0-10: asset return sequences
        for i, asset in enumerate(config.ASSETS):
            returns = window[f"{asset}_daily_ret"].values.astype(np.float32)
            # First element of each row = current portfolio weight
            state_matrix[i, 0]  = self.weights[i]
            # Elements 1..T-1 = daily log returns, most recent first
            # Paper §4.2: [r_{t-1}, r_{t-2}, ..., r_{t-T+1}]
            state_matrix[i, 1:] = returns[::-1][:self.lookback_window - 1]   # t-1, t-2, ..., t-59

        # Row 11: cash weight + 3 vol indicators (paper Section 4.2 & 5.1)
        latest = self.df.iloc[t]
        state_matrix[self.n_assets, 0] = self.weights[-1]          # w_cash
        state_matrix[self.n_assets, 1] = latest["vol20"]
        state_matrix[self.n_assets, 2] = latest["vol_ratio"]
        state_matrix[self.n_assets, 3] = latest["vix"]
        # Columns 4..59 remain zero (no additional indicators)

        return state_matrix.flatten()

    def _differential_sharpe(self, R_t: float) -> float:
        """
        Differential Sharpe Ratio as derived in Moody et al. (1998) and
        described in Section 4.3 of the paper.

        D_t = [B_{t-1}·ΔA_t − 0.5·A_{t-1}·ΔB_t] / (B_{t-1} − A_{t-1}²)^{3/2}

        with ΔA_t = R_t − A_{t-1}  and  ΔB_t = R_t² − B_{t-1}
        """
        A_prev = self.A
        B_prev = self.B

        delta_A = R_t      - A_prev
        delta_B = R_t**2   - B_prev

        # Update EMA estimates (η = 1/252)
        self.A = A_prev + self.eta * delta_A
        self.B = B_prev + self.eta * delta_B

        # Denominator: (B - A²)^(3/2)
        variance = B_prev - A_prev**2
        if variance <= 1e-10:
            # Not enough history yet; fall back to raw return
            return R_t

        denom = variance ** 1.5

        D_t = (B_prev * delta_A - 0.5 * A_prev * delta_B) / denom
        return float(D_t)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return (e_x / e_x.sum()).astype(np.float32)

