import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class PortfolioEnv(gym.Env):
    """A custom OpenAI Gym environment for portfolio allocation."""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=config.INITIAL_BALANCE, lookback_window=config.LOOKBACK_WINDOW,
                 transaction_cost=config.TRANSACTION_COST):
        super(PortfolioEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost

        # Assets: 11 ETFs + 1 Cash
        self.n_assets = len(config.ASSETS)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets + 1,), dtype=np.float32)

        # State size: 60-day log returns (11), previous weights (12), market indicators (4) = 27
        self.state_dim = self.n_assets + (self.n_assets + 1) + 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.asset_values = np.zeros(self.n_assets)
        self.weights = np.zeros(self.n_assets + 1)
        self.weights[-1] = 1.0  # Start fully in cash

        # Differential Sharpe Ratio variables
        self.eta = 0.99 # EMA decay parameter, common in DSR
        self.A = 0.0
        self.B = 0.0

        self.history = [self.portfolio_value]
        self.portfolio_returns = []

        return self._get_state(), {}

    def _get_state(self):
        # Current data point
        obs = self.df.iloc[self.current_step]

        # 60-day returns of 11 assets
        ret_60d = [obs[f"{asset}_60d_ret"] for asset in config.ASSETS]

        # 4 market indicators
        market_indicators = [
            obs['vol20'],
            obs['vol60'],
            obs['vol_ratio'],
            obs['vix']
        ]

        # State = 60d returns + prev weights + market indicators
        state = np.concatenate((ret_60d, self.weights, market_indicators))
        return state.astype(np.float32)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def step(self, action):
        # 1. Action -> Weights (Softmax)
        new_weights = self._softmax(action)

        # Calculate transaction costs (if applicable, though paper says 0)
        cost = np.sum(np.abs(new_weights - self.weights)) * self.transaction_cost
        self.portfolio_value *= (1 - cost)
        self.weights = new_weights

        # 2. Advance time to next day
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False

        # 3. Get today's daily returns for the assets (return for day t+1)
        obs = self.df.iloc[self.current_step]
        daily_returns = np.array([obs[f"{asset}_daily_ret"] for asset in config.ASSETS])

        # Convert log returns to simple returns for portfolio value calculation
        simple_returns = np.exp(daily_returns) - 1

        # Portfolio return
        # Weight on cash does not generate return (or we could use a risk-free rate, but let's assume 0)
        portfolio_return = np.sum(self.weights[:-1] * simple_returns)

        # 3. Update Portfolio Value
        prev_portfolio_value = self.portfolio_value
        self.portfolio_value = prev_portfolio_value * (1 + portfolio_return)

        self.history.append(self.portfolio_value)
        self.portfolio_returns.append(portfolio_return)

        # 4. Calculate Differential Sharpe Ratio Reward
        # D_t = (R_t - A_{t-1}) / sqrt(B_{t-1} - A_{t-1}^2 + eps)
        variance = self.B - self.A**2
        std_dev = np.sqrt(max(variance, 1e-8))

        if self.current_step == 0:
            reward = portfolio_return # Initial step approximation
        else:
            reward = (portfolio_return - self.A) / std_dev

        # Update EMA variables for DSR
        self.A = self.eta * self.A + (1 - self.eta) * portfolio_return
        self.B = self.eta * self.B + (1 - self.eta) * (portfolio_return**2)

        # 6. Add constraints (e.g., bankruptcy)
        if self.portfolio_value <= 0:
            done = True
            reward = -1.0 # Heavy penalty

        info = {
            'portfolio_value': self.portfolio_value,
            'return': portfolio_return,
            'weights': self.weights
        }

        return self._get_state(), reward, done, truncated, info

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}')
