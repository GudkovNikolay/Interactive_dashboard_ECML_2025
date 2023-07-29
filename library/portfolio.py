import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from dataclasses import dataclass

from library.constants import N_TRADING_DAYS, CONFIDENCE_INTERVAL, get_shifted_returns


def plot_tickers_performance(df_returns: pd.DataFrame):
    """
    Plot performance of each stock
    """
    assert df_returns.isna().sum().sum() == 0

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


def get_returns(w: pd.Series | pd.DataFrame, df_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Return returns of the portfolio
    """

    # Shift returns
    df_returns = get_shifted_returns(df_returns)

    if isinstance(w, pd.Series):
        # Portfolio weights are constant
        assert np.all(w.index == df_returns.columns)
        assert np.isclose(w.sum(), 1), w.sum()
        returns = df_returns @ w
    elif isinstance(w, pd.DataFrame):
        assert np.all(df_returns.columns == w.columns) and np.all(df_returns.index == w.index)
        assert np.all(np.isclose(w.sum(axis=1), 1) | w.isna().all(axis=1)), w.sum(axis=1)
        returns = (df_returns * w).dropna(axis=0, how='any').sum(axis=1)
    else:
        assert False, 'Unreachable'
    # Drop NaNs from returns (for them return or weight is not defined)
    return returns.dropna()


@dataclass
class Statistics:
    mean_return_annual: float
    std_return_annual: float
    sharpe_annual: float
    mean_return_se_annual: float


def get_statistics(returns: pd.Series) -> Statistics:
    """
    Compute statistics for returns
    """
    # Compute Annual Return and Sharpe
    mean = returns.mean() * N_TRADING_DAYS
    std = returns.std() * np.sqrt(N_TRADING_DAYS)
    sharpe = mean / std

    # Compute confidence interval for Mean Annual Return
    mean_error_estimation = returns.sem() * N_TRADING_DAYS
    t_score = scipy.stats.t.ppf(1 - (1 - CONFIDENCE_INTERVAL) / 2, len(returns) - 1)

    # Return statistics
    return Statistics(
        mean_return_annual=mean,
        std_return_annual=std,
        sharpe_annual=sharpe,
        mean_return_se_annual=mean_error_estimation * t_score
    )


def print_statistics(returns: pd.Series):
    """
    Print statistics of the portfolio w
    """
    # Get returns and statistics
    stat = get_statistics(returns)

    # Print statistics
    print(f'Annual Return (mean ± std): {stat.mean_return_annual:.1%} ± {stat.std_return_annual:.1%}')
    print(f'Sharpe: {stat.sharpe_annual:.2f}')
    print(f'Annual Mean Return {CONFIDENCE_INTERVAL:.0%} confidence interval: {stat.mean_return_annual:.2%}±{stat.mean_return_se_annual:.2%}')


def plot_cumulative_returns(returns: pd.Series, title: str):
    plt.figure(figsize=(12, 6))
    plt.plot(returns.cumsum())
    plt.title(title)
    plt.ylabel('Cumulative log-return')
    plt.show()
