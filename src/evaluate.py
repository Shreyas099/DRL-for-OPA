import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.env.portfolio_env import PortfolioEnv
from src.models.ppo_agent import PPOAgent
from src.models.mvo_agent import MVOAgent
from src.utils.metrics import calculate_metrics, print_metrics

os.makedirs(config.RESULTS_DIR, exist_ok=True)

def evaluate_pipeline():
    # Load processed data
    data_path = config.DATA_DIR / "processed_features.csv"
    if not data_path.exists():
        print(f"Error: Processed data not found at {data_path}. Please run fetch_data.py first.")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # 10 sliding windows from 2006 to 2021
    start_year = int(config.START_DATE[:4])

    # Store results for 2012-2021 (the test period)
    drl_returns = []
    mvo_returns = []
    test_dates = []

    mvo_agent = MVOAgent()

    # Pre-calculate daily prices for MVO from features df
    # We reconstruct prices assuming start price = 1 for all
    prices_df = pd.DataFrame(index=df.index, columns=config.ASSETS)
    for asset in config.ASSETS:
        # log returns
        prices_df[asset] = np.exp(df[f"{asset}_daily_ret"].cumsum())

    for window in range(config.NUM_WINDOWS):
        # We assume the best model from the validation set was saved as ppo_window_{window}_seed_{best_seed}.zip
        # We need to find the best seed for this window
        best_sharpe = -np.inf
        best_model_path = None

        # Just to simplify the evaluation, we check all saved seeds for this window, load them, run on validation,
        # and select the best. Or we can assume train.py saved a record of the best seed.
        # For this script, we'll quickly evaluate the saved models on validation to find the best,
        # or if train.py created a symlink or naming convention like ppo_window_{window}_best.zip, we'd use that.

        # Let's write a simple logic: evaluate on val_df
        val_start = f"{start_year + window + config.WINDOW_TRAIN_YEARS}-01-01"
        val_end = f"{start_year + window + config.WINDOW_TRAIN_YEARS + config.WINDOW_VAL_YEARS - 1}-12-31"
        val_df = df.loc[val_start:val_end]

        test_start = f"{start_year + window + config.WINDOW_TRAIN_YEARS + config.WINDOW_VAL_YEARS}-01-01"
        test_end = f"{start_year + window + config.WINDOW_TRAIN_YEARS + config.WINDOW_VAL_YEARS + config.WINDOW_TEST_YEARS - 1}-12-31"
        test_df = df.loc[test_start:test_end]

        print(f"Evaluating Window {window+1} (Test Period: {test_start} to {test_end})")

        if len(test_df) == 0:
            print(f"No test data for window {window+1}")
            continue

        # Find best model
        val_env = PortfolioEnv(val_df)
        for seed in range(config.NUM_SEEDS):
            model_path = config.MODELS_DIR / f"ppo_window_{window}_seed_{seed}.zip"
            if not model_path.exists():
                continue

            # Load agent
            agent = PPOAgent(df=val_df, seed=seed * 42)
            try:
                agent.load(model_path)
            except Exception as e:
                print(f"Failed to load {model_path}: {e}")
                continue

            # Evaluate on validation
            state, _ = val_env.reset()
            done = False
            truncated = False
            rewards = []
            while not (done or truncated):
                action = agent.predict(state)
                state, reward, done, truncated, info = val_env.step(action)
                rewards.append(info['return'])

            sharpe = np.sqrt(252) * np.mean(rewards) / (np.std(rewards) + 1e-8)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_model_path = model_path

        # Run DRL on Test Set
        test_env_drl = PortfolioEnv(test_df)
        agent = PPOAgent(df=test_df, seed=42)
        if best_model_path:
            agent.load(best_model_path)
            print(f"Loaded {best_model_path} with Val Sharpe {best_sharpe:.4f}")

        state, _ = test_env_drl.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.predict(state)
            state, reward, done, truncated, info = test_env_drl.step(action)
            drl_returns.append(info['return'])

        # Run MVO on Test Set
        # Get integer indices for the test period in the main df
        start_idx = df.index.get_loc(test_df.index[0])
        end_idx = df.index.get_loc(test_df.index[-1]) + 1

        _, mvo_weights = mvo_agent.simulate(prices_df, start_idx, end_idx)

        # Calculate MVO daily returns based on weights and actual daily returns
        for t, w in enumerate(mvo_weights):
            actual_date = test_df.index[t]
            daily_returns = np.exp(test_df.iloc[t][[f"{asset}_daily_ret" for asset in config.ASSETS]].values.astype(float)) - 1
            portfolio_return = np.sum(w * daily_returns)
            mvo_returns.append(portfolio_return)

        test_dates.extend(test_df.index.tolist())

    # Create Series, aligning dates
    # DRL has T-1 returns because it makes a decision at t=0 for return at t=1
    drl_series = pd.Series(drl_returns, index=test_dates[1:])
    # MVO has T returns, but to compare fairly, we evaluate on the same dates
    mvo_series = pd.Series(mvo_returns[1:], index=test_dates[1:])

    # Calculate Metrics
    print("\nCalculating Final Metrics for Out-of-Sample Period (2012-2021)")
    metrics_df = pd.DataFrame({
        'DRL': calculate_metrics(drl_series),
        'MVO': calculate_metrics(mvo_series)
    })
    print_metrics(metrics_df)
    metrics_df.to_csv(config.RESULTS_DIR / 'evaluation_metrics.csv')

    # Plot Cumulative Returns
    plt.figure(figsize=(12, 6))
    (1 + drl_series).cumprod().plot(label='DRL')
    (1 + mvo_series).cumprod().plot(label='MVO')
    plt.title('Cumulative Returns: DRL vs MVO (Out-of-Sample)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'cumulative_returns.png')

    # Plot Annual Returns
    drl_annual = drl_series.resample('YE').apply(lambda x: (1+x).prod() - 1)
    mvo_annual = mvo_series.resample('YE').apply(lambda x: (1+x).prod() - 1)

    annual_df = pd.DataFrame({'DRL': drl_annual, 'MVO': mvo_annual})
    annual_df.index = annual_df.index.year

    plt.figure(figsize=(12, 6))
    annual_df.plot.bar()
    plt.title('Annual Returns: DRL vs MVO')
    plt.xlabel('Year')
    plt.ylabel('Return')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'annual_returns.png')

    print(f"\nResults saved to {config.RESULTS_DIR}")

if __name__ == "__main__":
    evaluate_pipeline()
