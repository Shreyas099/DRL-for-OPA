# Deep Reinforcement Learning for Optimal Portfolio Allocation

A faithful PyTorch / Stable-Baselines3 reproduction of the experiments from:

> **"Deep Reinforcement Learning for Optimal Portfolio Allocation: A Comparative Study with Mean-Variance Optimization"**
> Sood, Papasotiriou, Vaiciulis, Balch — J.P. Morgan AI Research (FinPlan'23 @ ICAPS)

This repository trains a Proximal Policy Optimization (PPO) agent to manage a continuous, long-only portfolio of 11 S&P 500 sector ETFs over a 16-year period (2006–2021) using a walk-forward validation methodology, and benchmarks it against Mean-Variance Optimization (MVO).

---

## Project structure

```text
DRL-for-OPA/
├── config.py                   # Centralised hyperparameters & paths
├── requirements.txt
└── src/
    ├── data/
    │   └── fetch_data.py       # Downloads ETF/VIX/SPX data; saves prices.csv
    │                           # and processed_features.csv
    ├── env/
    │   └── portfolio_env.py    # Gym environment — 720-dim state,
    │                           # DSR reward, η = 1/252
    ├── models/
    │   ├── ppo_agent.py        # SB3 PPO wrapper — SubprocVecEnv, log_std_init=-1
    │   └── mvo_agent.py        # PyPortfolioOpt MVO — real prices + whole-shares
    ├── train.py                # Parallel orchestrator (10 windows × 5 seeds)
    ├── train_worker.py         # Single-seed worker called by train.py
    ├── evaluate.py             # Out-of-sample backtest & plots
    └── utils/
        └── metrics.py          # Sharpe, Sortino, Calmar, Max Drawdown
```

**Generated at runtime** (not committed):
```text
data/
├── prices.csv                  # Raw adjusted closes (MVO input)
└── processed_features.csv      # Daily log returns + 3 vol indicators (RL input)
saved_models/
├── ppo_window_{w}_seed_{s}.zip # Trained model per window/seed
└── result_window_{w}_seed_{s}.json  # Validation rewards (read by orchestrator)
results/
├── evaluation_metrics.csv
├── cumulative_returns.png
└── annual_returns.png
```

---

## Installation

Python 3.8+ required. A virtual environment is strongly recommended.

```bash
git clone https://github.com/Shreyas099/DRL-for-OPA.git
cd DRL-for-OPA

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## How to run

### 1. Fetch and preprocess data

Downloads daily adjusted close prices for the 11 sector ETFs, S&P 500, and VIX from Yahoo Finance (2006–2021). Saves two files: `prices.csv` for MVO and `processed_features.csv` for the RL environment.

```bash
python src/data/fetch_data.py
```

> If you previously ran the original code you must re-run this step — the feature columns have changed (3 vol indicators instead of 4).

### 2. Train the PPO agents

Runs the 10-window walk-forward pipeline. For each window, **all 5 seeds are trained simultaneously** as independent subprocesses (`train_worker.py`), each using 10 parallel environments via `SubprocVecEnv`. The best-performing agent (by mean episode validation reward, per §5.2) is selected and used to warm-start the next window.

```bash
python src/train.py
```

To keep training running after closing your SSH session:

```bash
nohup python src/train.py > training.log 2>&1 &
tail -f training.log          # watch progress live
```

> **Estimated runtime:**
> - Sequential (1 seed at a time): ~54 hours on a Xeon server
> - **Parallel (5 seeds at once): ~11–13 hours** — requires ≥55 CPU cores
>
> For a quick sanity check, set `NUM_SEEDS = 1` in `config.py` (~11 hrs) and confirm the pipeline runs end-to-end before the full run.

### 3. Evaluate and compare

Loads the saved models, runs deterministic inference over the 10-year out-of-sample test period (2012–2021), and compares against MVO. Prints a metrics table and saves plots to `results/`.

```bash
python src/evaluate.py
```

---

## Methodology

### State space (Section 4.2)

A flattened `(n+1) × T` matrix where `n = 11` assets and `T = 60` days:

- **Rows 0–10** (each asset): `[w_i, r_{i,t-1}, r_{i,t-2}, ..., r_{i,t-59}]`
- **Row 11** (cash): `[w_cash, vol20_t, vol_ratio_t, vix_t, 0, ..., 0]`

Total observation dimension: **720**.

### Action space (Section 4.1)

A continuous vector of length 12 (11 ETFs + cash). Passed through softmax to enforce long-only weights that sum to 1.

### Reward — Differential Sharpe Ratio (Section 4.3)

```
D_t = [B_{t-1}·ΔA_t − 0.5·A_{t-1}·ΔB_t] / (B_{t-1} − A_{t-1}²)^{3/2}

ΔA_t = R_t − A_{t-1}        A_t = A_{t-1} + η·ΔA_t
ΔB_t = R_t² − B_{t-1}       B_t = B_{t-1} + η·ΔB_t

η = 1/252,   A_0 = B_0 = 0
```

### Volatility indicators (Section 5.1)

Three standardised (expanding-window) features from the S&P 500:

| Feature | Description |
|---------|-------------|
| `vol20` | 20-day rolling std of daily log returns (annualised) |
| `vol_ratio` | `vol20 / vol60` — short- vs long-term vol regime indicator |
| `vix` | VIX index value |

### PPO hyperparameters (Table 1)

| Parameter | Value |
|-----------|-------|
| `n_envs` | 10 (SubprocVecEnv) |
| `n_steps` | 756 |
| `batch_size` | 1260 |
| `n_epochs` | 16 |
| `gamma` | 0.9 |
| `gae_lambda` | 0.9 |
| `clip_range` | 0.25 |
| `learning_rate` | 3e-4 → 1e-5 (linear decay) |
| `net_arch` | [64, 64] MLP, tanh |
| `log_std_init` | −1 |
| `total_timesteps` | 7,500,000 per seed |

### Walk-forward validation (Section 5.2)

10 sliding windows, each shifted 1 year:

```
Window 1:  Train 2006–2010  |  Val 2011  |  Test 2012
Window 2:  Train 2007–2011  |  Val 2012  |  Test 2013
...
Window 10: Train 2015–2019  |  Val 2020  |  Test 2021
```

5 seeds are trained per window; the best by mean episode validation reward (DSR) is selected for testing and used to warm-start the next window.

### MVO baseline (Section 5.3)

At every timestep the MVO agent uses the past 60 days of real adjusted close prices to estimate expected returns (sample mean) and the covariance matrix (Ledoit-Wolf shrinkage), then solves the max-Sharpe optimisation problem via `PyPortfolioOpt` with `risk_free_rate = 0`.

---

## Expected results (paper Table 2, averaged over 2012–2021)

| Metric | DRL | MVO |
|--------|-----|-----|
| Annual return | 0.1211 | 0.0653 |
| Sharpe ratio | 1.1662 | 0.6776 |
| Calmar ratio | 2.3133 | 1.1608 |
| Max drawdown | −0.3296 | −0.3303 |
| Sortino ratio | 1.7208 | 1.0060 |
| Annual volatility | 0.1249 | 0.1460 |

---

## Citation

```bibtex
@inproceedings{sood2023drl,
  title     = {Deep Reinforcement Learning for Optimal Portfolio Allocation:
               A Comparative Study with Mean-Variance Optimization},
  author    = {Sood, Srijan and Papasotiriou, Kassiani and
               Vaiciulis, Marius and Balch, Tucker},
  booktitle = {FinPlan Workshop, ICAPS 2023},
  year      = {2023}
}
```
