import pandas as pd
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from download_data import DATA_FOLDER, CLOSE_FOLDER
from constants import N_ASSETS



def get_prices(n_assets: int = N_ASSETS) -> pd.DataFrame:
    """
    Load prices DataFrame without NaNs
    """
    # Check that data exists
    tickers_file = DATA_FOLDER / 'tickers.csv'
    assert DATA_FOLDER.exists() and CLOSE_FOLDER.exists() and tickers_file.exists()

    # Read tickers list
    tickers = pd.read_csv(tickers_file)['SECID'].tolist()

    # Load close prices for each ticker
    close_series = []
    for ticker in tickers:
        # Load ticker
        ticker_df = pd.read_csv(CLOSE_FOLDER / f'{ticker}.csv', parse_dates=['TRADEDATE'])
        # Do some checks
        assert np.all(ticker_df['BOARDID'] == 'TQBR')
        assert ticker_df['TRADEDATE'].is_monotonic_increasing
        # Create Series
        ticker_df = ticker_df.rename(columns={'TRADEDATE': 'date', 'CLOSE': ticker}).set_index('date')[ticker]
        close_series.append(ticker_df)
    # Merge all tickers together
    df = pd.concat(close_series, axis=1)
    # Choose tickers with the longest history
    chosen_tickers = df.notna().sum().sort_values().index[:-n_assets - 1:-1].tolist()
    print(f'Chosen tickers: {chosen_tickers}')
    print(f'Length before dropping NaNs: {len(df)}')
    # Drop NaNs
    df = df[chosen_tickers].dropna()
    print(f'Length after dropping NaNs: {len(df)}')
    return df


def get_log_returns(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Return log-returns
    """
    # Check that prices do not contain NaNs
    assert df_prices.isna().sum().sum() == 0
    return np.log(1 + df_prices.pct_change().dropna())


class ReturnsDataset(Dataset):
    """
    PyTorch dataset with log-returns
    """

    def __init__(self, df_prices: pd.DataFrame, window_size: int):
        # Compute log-returns
        df_returns = get_log_returns(df_prices)

        # Compute sizes
        self.assets = df_returns.columns.tolist()
        self.n_assets = len(self.assets)
        self.window_size = window_size
        self.length = df_returns.shape[0] - window_size + 1

        # Convert to float-32
        returns_values = df_returns.values.astype(np.float32)

        # Fill data
        self.data = [torch.from_numpy(returns_values[ind:ind+self.window_size].T) for ind in range(self.length)]
        for i in range(self.length):
            assert self.data[i].size() == (self.n_assets, self.window_size)

    def __getitem__(self, ind: int) -> torch.tensor:
        return self.data[ind]

    def __len__(self) -> int:
        return self.length


def get_pytorch_datataset(window_size: int, batch_size: int, subset_columns: list[str] | None = None) -> tuple[Dataset, DataLoader, int]:
    """
    Get DataSet, DataLoader, number of assets
    """
    # Load prices
    df_prices = get_prices()

    # Get subset of columns
    if subset_columns is not None:
        df_prices = df_prices[subset_columns]

    # Create pytorch dataset and dataloader
    dataset = ReturnsDataset(df_prices, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    assert dataset[0].size() == dataset[len(dataset) - 1].size() == (dataset.n_assets, window_size)
    return dataset, dataloader, dataset.n_assets


if __name__ == '__main__':
    print(get_prices())
