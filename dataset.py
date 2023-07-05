import pandas as pd
import numpy as np
from pathlib import Path

N_TICKERS = 10
N_TRADING_DAYS = 252


def get_dataset(n_tickers: int = N_TICKERS) -> pd.DataFrame:
    data_folder = Path('data/')
    tickers_file = data_folder / 'tickers.csv'
    close_folder = data_folder / 'close'
    assert data_folder.exists() and close_folder.exists() and tickers_file.exists()

    tickers = pd.read_csv(tickers_file)['SECID'].tolist()
    close_series = []
    for ticker in tickers:
        ticker_df = pd.read_csv(close_folder / f'{ticker}.csv')
        assert np.all(ticker_df['BOARDID'] == 'TQBR')
        ticker_df = ticker_df.rename(columns={'TRADEDATE': 'date', 'CLOSE': ticker})
        ticker_df['date'] = pd.to_datetime(ticker_df['date'], format='%Y-%m-%d')
        close_series.append(ticker_df.set_index('date')[ticker])
    df = pd.concat(close_series, axis=1)
    n_tickers = 10
    chosen_tickers = df.notna().sum().sort_values().iloc[-n_tickers:].index.tolist()
    return df[chosen_tickers].dropna()


if __name__ == '__main__':
    print(get_dataset())
