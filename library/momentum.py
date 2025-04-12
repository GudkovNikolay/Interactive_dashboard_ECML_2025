import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from library.constants import N_TRADING_DAYS, CONFIDENCE_INTERVAL, get_shifted_returns
from tqdm.notebook import tqdm


def get_equal_portfolio(tickers: list[str]) -> pd.Series:
    """
    Return equally weighted portfolio
    """
    return pd.Series(np.full(len(tickers), 1 / len(tickers)), index=tickers)


def get_momentum_signal(df_returns: pd.DataFrame, n_finish: int, n_start: int) -> pd.DataFrame:
    """
    Return equally weighted portfolio
    """
    from_min_periods = 20
    to_min_periods = 10
    assert n_finish >= 2 * from_min_periods and n_start >= 2 * to_min_periods
    from_returns_sum = df_returns.rolling(pd.Timedelta(days=n_finish), min_periods=from_min_periods).sum()
    to_returns_sum = df_returns.rolling(pd.Timedelta(days=n_start), min_periods=to_min_periods).sum()
    return (from_returns_sum - to_returns_sum)


def get_portfolio_from_signal(df_signal: pd.DataFrame) -> pd.DataFrame:
    """
    Weight portfolio by signal using ranks
    """
    df_ranks = scipy.stats.rankdata(df_signal, axis=1) - 1
    df_weights = df_ranks / df_ranks.sum(axis=1).reshape(-1, 1)
    return pd.DataFrame(df_weights, index=df_signal.index, columns=df_signal.columns)


def _get_return_by_rank_from_signal(df_signal: pd.DataFrame, df_returns: pd.DataFrame, verbose: bool = False):
    """
    Compute returns for each rank
    """
    # Shift returns
    df_returns = get_shifted_returns(df_returns)
    assert np.all(df_returns.columns == df_signal.columns)

    # Iterate over signal
    result_rows = []
    result_index = []
    iterator = df_signal.iterrows()
    if verbose:
        iterator = tqdm(list(iterator))
    for (date, signal) in iterator:
        returns = df_returns.loc[date]
        # Add returns by each rank of signal
        if signal.notna().all() and returns.notna().all():
            result_rows.append(returns.values[np.argsort(signal)])
            result_index.append(date)

    # Construct result DataFrame
    df_returns_ranks = pd.DataFrame(result_rows, columns=range(df_returns.shape[1]), index=result_index)
    assert df_returns_ranks.isna().sum().sum() == 0
    return df_returns_ranks


def plot_signal_ranks(df_signal: pd.DataFrame, df_returns: pd.DataFrame, title: str):
    """
    Plot mean return and its confidence interval for each rank of signal
    """
    assert df_signal.index.isin(df_returns.index).all()

    # Get returns by rank (does not contain NaNs)
    df_signal_return_by_rank = _get_return_by_rank_from_signal(df_signal, df_returns)

    # Compute Mean returns for each rank and its confidence interval
    mean_returns = df_signal_return_by_rank.mean(axis=0) * N_TRADING_DAYS * 100
    t_score = scipy.stats.t.ppf(1 - (1 - CONFIDENCE_INTERVAL) / 2, len(df_signal_return_by_rank) - 1)
    std_returns = df_signal_return_by_rank.sem(axis=0) * N_TRADING_DAYS * 100

    # Plot Mean returns confidence intervals for each rank
    x = range(1, len(mean_returns) + 1)
    plt.errorbar(x, mean_returns, yerr=std_returns * t_score, linestyle=None, marker='o', capsize=5)
    plt.xticks(x, x)

    plt.ylabel(f'Annual Mean Return {CONFIDENCE_INTERVAL:.0%} confidence interval')
    plt.xlabel('Signal rank')
    plt.title(title)

    plt.grid(axis='y')
    plt.show()
