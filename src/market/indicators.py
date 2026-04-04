"""Technical indicators — computed manually (no pandas-ta dependency)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Where avg_loss is 0, RSI is 100 (all gains)
    rs = pd.Series(np.where(avg_loss == 0, np.inf, avg_gain / avg_loss), index=series.index)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Upper band, middle (SMA), lower band."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    vwap = cum_tp_vol / cum_vol
    return vwap


def detect_volume_anomaly(volume: pd.Series, window: int = 20, threshold: float = 2.0) -> pd.Series:
    """Detect volume spikes above threshold * rolling average."""
    avg_vol = volume.rolling(window=window).mean()
    return volume > (threshold * avg_vol)


def compute_momentum_score(df: pd.DataFrame) -> float:
    """Composite momentum score from RSI + MACD + Bollinger position.

    Returns: float in [-1, 1]
    """
    if df is None or len(df) < 30:
        return 0.0

    close = df["close"]

    # RSI component: 50 = neutral, >70 = overbought, <30 = oversold
    rsi = compute_rsi(close)
    latest_rsi = rsi.iloc[-1]
    if np.isnan(latest_rsi):
        rsi_score = 0.0
    else:
        rsi_score = (latest_rsi - 50) / 50  # -1 to +1

    # MACD component: histogram sign and magnitude
    _, _, histogram = compute_macd(close)
    latest_hist = histogram.iloc[-1]
    if np.isnan(latest_hist):
        macd_score = 0.0
    else:
        # Normalize by recent range
        hist_std = histogram.std()
        macd_score = float(np.clip(latest_hist / (hist_std + 1e-10), -1, 1))

    # Bollinger position: where is price relative to bands
    upper, middle, lower = compute_bollinger_bands(close)
    band_width = upper.iloc[-1] - lower.iloc[-1]
    if np.isnan(band_width) or band_width < 1e-10:
        bb_score = 0.0
    else:
        bb_score = float(np.clip(
            (close.iloc[-1] - middle.iloc[-1]) / (band_width / 2), -1, 1
        ))

    # Weighted composite
    score = 0.4 * rsi_score + 0.35 * macd_score + 0.25 * bb_score
    return float(np.clip(score, -1, 1))


def compute_volume_score(df: pd.DataFrame) -> float:
    """Volume anomaly score.

    Returns: float in [0, 1] — higher means more unusual volume.
    """
    if df is None or len(df) < 25:
        return 0.0

    volume = df["volume"]
    avg_vol = volume.rolling(window=20).mean().iloc[-1]

    if np.isnan(avg_vol) or avg_vol < 1:
        return 0.0

    ratio = volume.iloc[-1] / avg_vol
    # Map ratio to [0, 1]: ratio=1 -> 0, ratio=3+ -> 1
    score = float(np.clip((ratio - 1) / 2, 0, 1))
    return score
