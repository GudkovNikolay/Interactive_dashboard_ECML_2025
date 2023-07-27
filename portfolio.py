import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

from constants import N_TRADING_DAYS, CONFIDENCE_INTERVAL


def plot_tickers_performance(df_returns: pd.DataFrame):
    tickers = df_returns.columns

    # Compute mean, std, sharpe for each ticker return
    means = df_returns.mean() * N_TRADING_DAYS * 100
    stds = df_returns.std() * np.sqrt(N_TRADING_DAYS) * 100
    sharpe = means / stds

    # Sort tickers by mean return
    sorted_ind = np.argsort(means)

    # Plot mean, std, sharpe
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    ax1.set_title('Mean')
    ax1.set_ylabel('%')
    sns.barplot(x=tickers[sorted_ind], y=means[sorted_ind], ax=ax1)

    ax2.set_title('Std')
    ax2.set_ylabel('%')
    sns.barplot(x=tickers[sorted_ind], y=stds[sorted_ind], ax=ax2)

    ax3.set_title('Sharpe')
    sns.barplot(x=tickers[sorted_ind], y=sharpe[sorted_ind], ax=ax3)

    plt.tight_layout()
    plt.show()


def print_statistics(w: pd.Series | pd.DataFrame, df_returns: pd.DataFrame):
    """
    Print statistics of the portfolio w
    """
    if isinstance(w, pd.Series):
        assert np.all(w.index == df_returns.columns)
        assert np.isclose(w.sum(), 1), w.sum()
        returns = df_returns @ w
    elif isinstance(w, pd.DataFrame):
        assert np.all(df_returns.columns == w.columns)
        df_returns = df_returns[df_returns.index >= w.index[0]]
        assert np.all(df_returns.index == w.index)
        assert np.allclose(w.sum(axis=1), 1), w.sum(axis=1)
        returns = (df_returns * w).sum(axis=1)
    else:
        assert False, 'Unreachable'
    mean = returns.mean() * N_TRADING_DAYS
    std = returns.std() * np.sqrt(N_TRADING_DAYS)
    sharpe = mean / std
    print(f'Annual Return (mean ± std): {mean:.1%} ± {std:.1%}')
    print(f'Sharpe: {sharpe:.2f}')
    print()
    mean_estimation = returns.mean() * N_TRADING_DAYS
    mean_error_estimation = returns.sem() * N_TRADING_DAYS
    t_score = scipy.stats.t.ppf(1 - (1 - CONFIDENCE_INTERVAL) / 2, len(returns) - 1)
    print(f'Annual Mean Return {CONFIDENCE_INTERVAL:.0%} confidence interval: {mean_estimation:.2%}±{mean_error_estimation * t_score:.2%}')

    plt.figure(figsize=(12, 6))
    plt.plot(returns.cumsum())
    plt.title('Cumulative returns')
    plt.ylabel('Cumulative log-return')
    plt.show()
