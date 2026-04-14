"""
data_fetcher.py
---------------
Handles all data acquisition from Yahoo Finance using the yfinance library.
Provides caching to avoid redundant API calls during repeated runs.
"""

import os
import hashlib
import yfinance as yf
import pandas as pd
from datetime import datetime


def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    cache_dir: str = "data",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given ticker symbol.

    Downloads data from Yahoo Finance and caches it locally as a CSV to speed
    up repeated requests for the same ticker/date range.

    Args:
        ticker:     Stock ticker symbol (e.g. "AAPL", "TSLA").
        start_date: Start date string in "YYYY-MM-DD" format.
        end_date:   End date string in "YYYY-MM-DD" format.
        cache_dir:  Directory where cached CSV files are stored.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume.
        Index is a DatetimeIndex.

    Raises:
        ValueError: If the ticker is invalid or no data is returned.
    """
    ticker = ticker.upper().strip()

    # Build a deterministic cache key from the query parameters
    cache_key = hashlib.md5(f"{ticker}_{start_date}_{end_date}".encode()).hexdigest()[:8]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{ticker}_{cache_key}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' between {start_date} and {end_date}. "
            "Check that the symbol is valid and the date range contains trading days."
        )

    # Keep only the core OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"

    # Flatten any multi-level columns yfinance might return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(cache_path)
    return df


def get_company_info(ticker: str) -> dict:
    """
    Retrieve basic company metadata from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dictionary with keys: 'name', 'sector', 'market_cap', 'currency'.
        Values default to 'N/A' if unavailable.
    """
    ticker = ticker.upper().strip()
    try:
        info = yf.Ticker(ticker).info
        return {
            "name":       info.get("longName", ticker),
            "sector":     info.get("sector", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "currency":   info.get("currency", "USD"),
        }
    except Exception:
        return {"name": ticker, "sector": "N/A", "market_cap": "N/A", "currency": "USD"}


def validate_ticker(ticker: str) -> bool:
    """
    Perform a lightweight check to verify a ticker symbol exists.

    Args:
        ticker: Stock ticker symbol to validate.

    Returns:
        True if Yahoo Finance returns any data, False otherwise.
    """
    try:
        t = yf.Ticker(ticker.upper().strip())
        hist = t.history(period="5d")
        return not hist.empty
    except Exception:
        return False
