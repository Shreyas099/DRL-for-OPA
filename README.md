# Deep Reinforcement Learning for Optimal Portfolio Allocation 📈

This repository contains a full PyTorch/Stable-Baselines3 implementation designed to recreate the experiments from the research paper **"Deep Reinforcement Learning for Optimal Portfolio Allocation: A Comparative Study with Mean-Variance Optimization"**.

This project trains a Proximal Policy Optimization (PPO) agent to autonomously manage a continuous, long-only portfolio of 11 S&P 500 Sector ETFs over a 16-year period (2006-2021) using a Walk-Forward validation methodology. Its performance is benchmarked against a traditional Mean-Variance Optimization (MVO) portfolio strategy.

## 🚀 Features
- **Custom OpenAI Gym Environment**: A highly configurable trading environment built to simulate continuous portfolio reallocation with zero transaction costs.
- **Differential Sharpe Ratio Reward**: Implements an iterative, step-by-step Exponential Moving Average (EMA) Sharpe Ratio approximation to provide dense, risk-adjusted rewards to the reinforcement learning agent.
- **Walk-Forward Validation**: A robust 10-window sliding backtest (5 years train, 1 year validation, 1 year test) designed to prevent data leakage and evaluate true out-of-sample performance.
- **Mean-Variance Optimization Baseline**: Uses `PyPortfolioOpt` to calculate expected returns and Ledoit-Wolf Shrinkage covariance matrices for the MVO benchmark.

## 📂 Project Structure
```text
drpm/
├── requirements.txt
├── config.py                 # Centralized hyperparameters & configuration
└── src/
    ├── data/
    │   └── fetch_data.py     # Yahoo Finance ETF downloader & feature engineering
    ├── env/
    │   └── portfolio_env.py  # Gym Environment & Differential Sharpe Ratio logic
    ├── models/
    │   ├── ppo_agent.py      # Stable-Baselines3 PPO wrapper & linear LR scheduler
    │   └── mvo_agent.py      # PyPortfolioOpt Mean-Variance Optimization baseline
    ├── train.py              # Walk-forward training pipeline (10 windows, 5 seeds)
    ├── evaluate.py           # Out-of-sample backtest & metric generation
    └── utils/
        └── metrics.py        # Financial indicators (Sharpe, Calmar, Max Drawdown)
```

## 🛠️ Installation
Ensure you have Python 3.8+ installed. It is highly recommended to run this inside a virtual environment.

```bash
git clone https://github.com/Shreyas099/DRL-for-OPA.git
cd DRL-for-OPA

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 How to Run

### 1. Fetch & Preprocess Data
Download daily historical prices for the 11 sector ETFs, S&P 500, and VIX. This script will compute 60-day log returns and standardize 4 market volatility indicators.
```bash
python src/data/fetch_data.py
```

### 2. Train the PPO Agents
Run the 10-window walk-forward training methodology. The pipeline trains 5 random seeds per window for 7.5 million timesteps each, saving the best models.
```bash
python src/train.py
```
*(Note: Full-scale training involves 37.5 million timesteps and may take several hours on a CPU).*

### 3. Evaluate and Compare
Run the simulated out-of-sample backtest for the 10-year test period (2012-2021). This will print a comparative table of financial metrics (Sharpe, Sortino, Max Drawdown) and save plots to `results/`.
```bash
python src/evaluate.py
```

## 🧠 Methodology Overview
**State Space:**
- 60-day historical log-returns of 11 assets
- Previous day's portfolio weights
- 4 Standardized market volatility indicators (`vol20`, `vol60`, `vol20/vol60`, `vix`)

**Action Space:**
- A continuous vector of length 12 (11 assets + Cash).
- Passed through a Softmax function to ensure weights are strictly long-only (non-negative) and sum to 1.0.

**PPO Hyperparameters:**
- Network Architecture: MLP `[64, 64]`
- `n_steps`: 756
- `batch_size`: 1260
- `learning_rate`: Linear decay from 3e-4 to 1e-5.

---
