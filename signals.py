import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from constants import SHIFT_RETURNS, N_TRADING_DAYS, CONFIDENCE_INTERVAL


def get_equal_portfolio(tickers: list[str]) -> pd.Series:
    """
    Return equally weighted portfolio
    """
    return pd.Series(np.full(len(tickers), 1 / len(tickers)), index=tickers)


def get_momentum_signal(df_returns: pd.DataFrame, days_for_signal: pd.Timedelta, n_min_observations: int) -> pd.DataFrame:
    return df_returns.rolling(days_for_signal, min_periods=n_min_observations).mean().dropna()


def get_portfolio_from_signal(df_signal: pd.DataFrame) -> pd.DataFrame:
    df_ranks = np.argsort(df_signal, axis=1)
    df_weights = df_ranks / df_ranks.sum(axis=1).values.reshape(-1, 1)
    return df_weights


def _get_return_by_rank_from_signal(df_signal: pd.DataFrame, df_returns: pd.DataFrame):
    df_returns = df_returns.shift(-SHIFT_RETURNS).dropna()
    df_returns = df_returns[df_returns.index >= df_signal.index[0]]
    df_signal = df_signal.iloc[:-SHIFT_RETURNS]
    assert np.all(df_returns.index == df_signal.index) and np.all(df_returns.columns == df_signal.columns)
    result = []
    for (i, signal), (i, returns) in zip(df_signal.iterrows(), df_returns.iterrows()):
        result.append(returns.values[np.argsort(signal)])
    return pd.DataFrame(result, columns=range(df_returns.shape[1]), index=df_returns.index)


def plot_signal_ranks(df_signal: pd.DataFrame, df_returns: pd.DataFrame, title: str):
    df_signal_return_by_rank = _get_return_by_rank_from_signal(df_signal, df_returns)
    mean_returns = df_signal_return_by_rank.mean(axis=0) * N_TRADING_DAYS * 100
    t_score = scipy.stats.t.ppf(1 - (1 - CONFIDENCE_INTERVAL) / 2, len(df_returns) - 1)
    std_returns = df_signal_return_by_rank.sem(axis=0) * N_TRADING_DAYS * 100
    plt.axhline(0, linestyle='--')
    x = range(1, len(mean_returns) + 1)
    plt.errorbar(x, mean_returns, yerr=std_returns * t_score, linestyle=None, marker='o', capsize=5)
    plt.ylabel(f'Annual Mean Return {CONFIDENCE_INTERVAL:.0%} confidence interval')
    plt.xticks(x, x)
    plt.xlabel('Signal rank')
    plt.title(title)
    plt.show()
