from typing import Tuple, Dict, Any

import yfinance as yf
import pandas as pd

from data_ingest.utils import get_logger

logger = get_logger(__name__)


def stock_data_from_package(
    symbol: str, start_date: str, end_date: str, interval: str = "1d"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract stock data from Yahoo Finance API.
    Args:
        symbol: Stock symbol.
        start_date: Start date.
        end_date: End date.
        interval: Data interval.
    Returns:
        Tuple containing the extracted data and metadata.
    """

    logger.info(
        f"Extracting stock data from {start_date} to {end_date} for {symbol} at {interval} interval"
    )
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, interval=interval)

    metadata = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
        "df_start_date": data.index[0].to_pydatetime(),
        "df_end_date": data.index[-1].to_pydatetime(),
        "missing data rate": data.isnull().mean().mean(),
        "data_shape": data.shape,
    }

    return data, metadata
