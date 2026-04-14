"""
preprocessor.py
---------------
Transforms raw OHLCV data into a flat feature matrix ready for scikit-learn.

Pipeline:
    1. Compute technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands).
    2. Add lag features (previous 1–5 days' Close).
    3. Drop NaN rows produced by rolling windows.
    4. Build feature matrix X and target vector y (next-day Close).
    5. Split chronologically into train / test (80/20) — no shuffling.
    6. Scale X with MinMaxScaler (fit on train only to prevent leakage).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing), clamped to [0, 100]."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    """MACD line, signal line, and histogram."""
    ema_fast    = _ema(series, fast)
    ema_slow    = _ema(series, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def _bollinger(series: pd.Series, window=20, num_std=2.0):
    """Upper and lower Bollinger Bands."""
    mid   = _sma(series, window)
    std   = series.rolling(window=window).std()
    return mid + num_std * std, mid - num_std * std


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StockPreprocessor:
    """
    Converts a raw OHLCV DataFrame into train/test arrays for scikit-learn.

    Attributes:
        train_ratio:    Fraction of rows used for training.
        scaler:         Fitted MinMaxScaler for the feature matrix.
        feature_cols:   Column names used as model inputs.
    """

    def __init__(self, train_ratio: float = 0.80):
        self.train_ratio  = train_ratio
        self.scaler       = MinMaxScaler(feature_range=(0, 1))
        self.feature_cols: list[str] = []

    # ------------------------------------------------------------------

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append technical indicators and lag features to df in-place.

        Indicators:
            SMA_20, SMA_50, EMA_12, EMA_26, RSI_14,
            MACD, MACD_Signal, BB_Upper, BB_Lower,
            Daily_Return, Lag_1 … Lag_5
        """
        close = df["Close"]

        df["SMA_20"]       = _sma(close, 20)
        df["SMA_50"]       = _sma(close, 50)
        df["EMA_12"]       = _ema(close, 12)
        df["EMA_26"]       = _ema(close, 26)
        df["RSI_14"]       = _rsi(close)

        macd, sig, hist    = _macd(close)
        df["MACD"]         = macd
        df["MACD_Signal"]  = sig

        bb_up, bb_lo       = _bollinger(close)
        df["BB_Upper"]     = bb_up
        df["BB_Lower"]     = bb_lo

        df["Daily_Return"] = close.pct_change()

        # Lag features: price N days ago
        for lag in range(1, 6):
            df[f"Lag_{lag}"] = close.shift(lag)

        return df

    # ------------------------------------------------------------------

    def prepare(self, df: pd.DataFrame) -> dict:
        """
        Run the full preprocessing pipeline.

        Args:
            df: Raw OHLCV DataFrame with DatetimeIndex.

        Returns:
            Dict with keys:
                X_train, X_test   : scaled feature arrays (2-D)
                y_train, y_test   : target arrays (next-day Close, unscaled)
                train_size        : int — number of training rows
                dates             : DatetimeIndex after NaN drop
                df_clean          : cleaned DataFrame (indicators added)
        """
        df = df.copy()
        df = self.add_technical_indicators(df)

        # Target: next-day closing price (shift -1 so row i predicts row i+1)
        df["Target"] = df["Close"].shift(-1)
        df.dropna(inplace=True)

        dates = df.index

        # Feature columns = everything except Target
        self.feature_cols = [c for c in df.columns if c != "Target"]

        X = df[self.feature_cols].values   # (n_samples, n_features)
        y = df["Target"].values            # (n_samples,)

        # Chronological split
        n          = len(X)
        train_size = int(n * self.train_ratio)

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Scale only on training data to avoid leakage
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test  = self.scaler.transform(X_test)

        return {
            "X_train":    X_train,
            "X_test":     X_test,
            "y_train":    y_train,
            "y_test":     y_test,
            "train_size": train_size,
            "dates":      dates,
            "df_clean":   df,
        }

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale new feature rows using the already-fitted scaler."""
        return self.scaler.transform(X)
