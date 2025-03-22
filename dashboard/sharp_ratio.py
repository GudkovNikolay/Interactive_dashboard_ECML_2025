from library.portfolio import get_statistics, Statistics
from library.momentum import get_momentum_signal, plot_signal_ranks
from library.momentum import get_portfolio_from_signal
from library.portfolio import get_returns, print_statistics, plot_cumulative_returns
from library.momentum import get_equal_portfolio
import pandas as pd
import numpy as np

def get_momentum_signal_with_params(df_returns: pd.DataFrame, n_finish: int, n_start: int) -> pd.DataFrame:
    df_momentum_signal = get_momentum_signal(df_returns, n_finish=n_finish, n_start=n_start)
    df_momentum_weights = get_portfolio_from_signal(df_momentum_signal)
    return get_returns(df_momentum_weights, df_returns)


def get_momentum_signal_statistics_with_params_list(df_returns_list: list[pd.DataFrame], n_finish: int, n_start: int) -> Statistics:
    df_returns = pd.concat([get_momentum_signal_with_params(df_returns, n_finish=n_finish, n_start=n_start) for df_returns in df_returns_list], axis=0).reset_index(drop=True)
    return get_statistics(df_returns)

def sharp_grid(df_returns_fake_list):
    N_START_VALUES = [20, 40, 60, 80, 100]
    N_FINISH_VALUES = [150, 200, 250, 300, 350, 400]
    
    stats_fake_by_params: dict[tuple[int, int], Statistics] = {}
    
    for n_start in N_START_VALUES:
        for n_finish in N_FINISH_VALUES:
            stats_fake_by_params[n_start, n_finish] = get_momentum_signal_statistics_with_params_list([df_returns_fake_list], n_finish=n_finish, n_start=n_start)
    return np.array([[stats_fake_by_params[n_start, n_finish].sharpe_annual for n_finish in N_FINISH_VALUES] for n_start in N_START_VALUES])