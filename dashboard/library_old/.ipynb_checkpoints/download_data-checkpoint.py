import asyncio
from pathlib import Path

# Define directories
DATA_FOLDER = Path('data/')
CLOSE_FOLDER = DATA_FOLDER / 'close'

# Make directories
DATA_FOLDER.mkdir(exist_ok=True)
CLOSE_FOLDER.mkdir(exist_ok=True)


async def main():
    import asyncio
    import aiohttp
    import aiomoex
    import pandas as pd

    async with aiohttp.ClientSession() as session:
        # Download tickers list
        tickers_df = pd.DataFrame(await aiomoex.get_board_securities(session))
        tickers_df.to_csv(DATA_FOLDER / 'tickers.csv', index=False)

        tickers = tickers_df['SECID'].tolist()
        n_tickers = len(tickers)
        print(f'Found tickers: {n_tickers}')

        success_counter = 0

        async def download_ticker(ticker: str):
            print(f'Download: {ticker}')
            ticker_close_df = pd.DataFrame(await aiomoex.get_board_history(session, ticker))

            # Check non-empty
            n_observations = len(ticker_close_df)
            assert n_observations != 0, ticker

            # Save ticker
            ticker_close_df.to_csv(CLOSE_FOLDER / f'{ticker}.csv', index=False)

            # Log download
            nonlocal success_counter
            success_counter += 1
            print(f'Success ({success_counter} / {n_tickers}): {ticker} - {n_observations} observations')

        # Download all tickers
        await asyncio.gather(*[download_ticker(ticker) for ticker in tickers])


if __name__ == '__main__':
    asyncio.run(main())
