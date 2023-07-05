import asyncio
import aiohttp
import aiomoex
import pandas as pd

from pathlib import Path


async def download_tickers(session: aiohttp.ClientSession) -> list[str]:
    df = pd.DataFrame(await aiomoex.get_board_securities(session))
    return df


async def main():
    async with aiohttp.ClientSession() as session:
        tickers_df = await download_tickers(session)

        data_folder = Path('data/')
        data_folder.mkdir(exist_ok=True)
        close_folder = data_folder / 'close'
        close_folder.mkdir(exist_ok=True)

        tickers_df.to_csv(data_folder / 'tickers.csv', index=False)

        tickers = tickers_df['SECID'].tolist()
        n_tickers = len(tickers)
        print(f'Found tickers: {n_tickers}')

        success_counter = 0

        async def download_ticker(ticker: str):
            print(f'Download: {ticker}')

            ticker_close = pd.DataFrame(await aiomoex.get_board_history(session, ticker))

            n_observations = len(ticker_close)
            assert n_observations != 0, ticker

            ticker_close.to_csv(close_folder / f'{ticker}.csv', index=False)

            nonlocal success_counter
            success_counter += 1
            print(f'Success ({success_counter} / {n_tickers}): {ticker} - {n_observations} observations')

        await asyncio.gather(*[download_ticker(ticker) for ticker in tickers])


if __name__ == '__main__':
    asyncio.run(main())
