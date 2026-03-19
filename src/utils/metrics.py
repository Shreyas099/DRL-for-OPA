import pandas as pd
import numpy as np
import pyfolio as pf
import empyrical as ep

def calculate_metrics(returns, benchmark_returns=None):
    """
    Calculate financial metrics from daily returns.
    """
    metrics = {
        'Annual Return': ep.annual_return(returns),
        'Cumulative Returns': ep.cum_returns_final(returns),
        'Annual Volatility': ep.annual_volatility(returns),
        'Sharpe Ratio': ep.sharpe_ratio(returns),
        'Max Drawdown': ep.max_drawdown(returns),
        'Calmar Ratio': ep.calmar_ratio(returns),
        'Omega Ratio': ep.omega_ratio(returns),
        'Sortino Ratio': ep.sortino_ratio(returns),
    }

    if benchmark_returns is not None:
        metrics['Information Ratio'] = ep.information_ratio(returns, benchmark_returns)
        metrics['Alpha'] = ep.alpha(returns, benchmark_returns)
        metrics['Beta'] = ep.beta(returns, benchmark_returns)

    return pd.Series(metrics)

def print_metrics(metrics_df):
    """Print the calculated metrics."""
    print(f"{'Metric':<25} {'DRL':<15} {'MVO':<15}")
    print("-" * 55)
    for index, row in metrics_df.iterrows():
        print(f"{index:<25} {row['DRL']:<15.4f} {row['MVO']:<15.4f}")
