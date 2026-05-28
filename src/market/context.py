"""Market context feature computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.market.indicators import compute_atr, compute_vwap


@dataclass(frozen=True)
class MarketContext:
    """Point-in-time market context features for a ticker."""

    ticker: str
    last_price: float | None
    prev_close: float | None
    today_open: float | None
    day_high: float | None
    day_low: float | None
    session_vwap: float | None
    atr_14: float | None
    gap_pct: float | None
    gap_atr: float | None
    day_range_pos: float | None
    price_vs_vwap_pct: float | None
    return_5m: float | None
    return_15m: float | None
    return_60m: float | None


def compute_market_context(
    ticker: str,
    intraday_df: pd.DataFrame | None,
    daily_df: pd.DataFrame | None,
    quote: Any | None = None,
) -> MarketContext:
    """Compute market context features from intraday bars, daily bars, and an optional quote."""
    intraday = _sorted_frame(intraday_df)
    daily = _sorted_frame(daily_df)

    last_price = _quote_price(quote)
    if last_price is None and not intraday.empty:
        last_price = _finite_float(intraday["close"].iloc[-1])

    prev_close = None
    if not daily.empty:
        # Daily bars include the current in-progress trading day during market hours;
        # gap math needs the previous completed close when available.
        prev_close_idx = -2 if len(daily) >= 2 else -1
        prev_close = _finite_float(daily["close"].iloc[prev_close_idx])

    today_open = day_high = day_low = session_vwap = None
    if not intraday.empty:
        today_open = _finite_float(intraday["open"].iloc[0])
        day_high = _finite_float(intraday["high"].max())
        day_low = _finite_float(intraday["low"].min())
        session_vwap = _last_valid(compute_vwap(intraday))

    atr_14 = None
    if not daily.empty:
        atr_14 = _last_valid(compute_atr(daily, period=14))

    gap_pct = _safe_pct(today_open, prev_close)
    gap = None if today_open is None or prev_close is None else today_open - prev_close
    gap_atr = _safe_div(gap, atr_14)
    day_range_pos = _safe_div(
        None if last_price is None or day_low is None else last_price - day_low,
        None if day_high is None or day_low is None else day_high - day_low,
    )
    price_vs_vwap_pct = _safe_pct(last_price, session_vwap)

    return MarketContext(
        ticker=ticker,
        last_price=last_price,
        prev_close=prev_close,
        today_open=today_open,
        day_high=day_high,
        day_low=day_low,
        session_vwap=session_vwap,
        atr_14=atr_14,
        gap_pct=gap_pct,
        gap_atr=gap_atr,
        day_range_pos=day_range_pos,
        price_vs_vwap_pct=price_vs_vwap_pct,
        return_5m=_return_over_minutes(intraday, last_price, 5),
        return_15m=_return_over_minutes(intraday, last_price, 15),
        return_60m=_return_over_minutes(intraday, last_price, 60),
    )


def _sorted_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "timestamp" in df.columns:
        return df.sort_values("timestamp").reset_index(drop=True)
    return df.reset_index(drop=True)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _last_valid(series: pd.Series) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return _finite_float(valid.iloc[-1])


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return _finite_float(numerator / denominator)


def _safe_pct(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or base == 0:
        return None
    return _finite_float((value - base) / base)


def _quote_price(quote: Any | None) -> float | None:
    if quote is None:
        return None

    for name in ("last_price", "price", "mark", "close"):
        value = quote.get(name) if isinstance(quote, dict) else getattr(quote, name, None)
        price = _finite_float(value)
        if price is not None and price > 0:
            return price

    bid = quote.get("bid_price") if isinstance(quote, dict) else getattr(quote, "bid_price", None)
    ask = quote.get("ask_price") if isinstance(quote, dict) else getattr(quote, "ask_price", None)
    bid_price = _finite_float(bid)
    ask_price = _finite_float(ask)
    if bid_price is not None and ask_price is not None and bid_price > 0 and ask_price > 0:
        return (bid_price + ask_price) / 2
    if ask_price is not None and ask_price > 0:
        return ask_price
    if bid_price is not None and bid_price > 0:
        return bid_price
    return None


def _return_over_minutes(df: pd.DataFrame, last_price: float | None, minutes: int) -> float | None:
    if df.empty or last_price is None:
        return None

    if "timestamp" in df.columns and len(df) >= 2:
        timestamps = pd.to_datetime(df["timestamp"])
        target_time = timestamps.iloc[-1] - pd.Timedelta(minutes=minutes)
        candidates = df.loc[timestamps <= target_time]
        base = candidates["close"].iloc[-1] if not candidates.empty else df["close"].iloc[0]
    elif len(df) > minutes:
        base = df["close"].iloc[-minutes - 1]
    else:
        base = df["close"].iloc[0]

    return _safe_pct(last_price, _finite_float(base))
