"""Tests for market context feature generation."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.market.context import MarketContext, compute_market_context
from src.market.indicators import compute_atr


def _intraday_df() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-05-21 13:30:00", tz="UTC")
    # 13 one-minute bars; last close is 112, 5/15/60-minute returns are measurable/safe.
    closes = [100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113, 112]
    for i, close in enumerate(closes):
        rows.append(
            {
                "timestamp": start + pd.Timedelta(minutes=i),
                "open": 100.0 if i == 0 else float(closes[i - 1]),
                "high": float(close + 1),
                "low": float(close - 1),
                "close": float(close),
                "volume": 100.0,
            }
        )
    return pd.DataFrame(rows)


def _daily_df() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-05-01", tz="UTC")
    for i in range(15):
        close = 90.0 + i
        rows.append(
            {
                "timestamp": start + pd.Timedelta(days=i),
                "open": close - 0.5,
                "high": close + 3.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000_000.0,
            }
        )
    # Previous completed daily close used by gap math; also participates in ATR.
    rows.append(
        {
            "timestamp": start + pd.Timedelta(days=15),
            "open": 99.5,
            "high": 103.0,
            "low": 98.0,
            "close": 100.0,
            "volume": 1_000_000.0,
        }
    )
    # Current in-progress day should not be treated as previous close.
    rows.append(
        {
            "timestamp": start + pd.Timedelta(days=16),
            "open": 100.5,
            "high": 114.0,
            "low": 99.0,
            "close": 112.0,
            "volume": 1_000_000.0,
        }
    )
    return pd.DataFrame(rows)


def test_compute_atr_uses_true_range_and_rolling_mean() -> None:
    df = pd.DataFrame(
        {
            "high": [10.0, 12.0, 13.0],
            "low": [8.0, 9.0, 10.0],
            "close": [9.0, 11.0, 12.0],
        }
    )

    atr = compute_atr(df, period=2)

    assert math.isnan(atr.iloc[0])
    assert atr.iloc[1] == pytest.approx(2.5)  # mean(TR: 2, 3)
    assert atr.iloc[2] == pytest.approx(3.0)  # mean(TR: 3, 3)


def test_compute_market_context_core_metrics() -> None:
    intraday = _intraday_df()
    daily = _daily_df()

    context = compute_market_context("TEST", intraday, daily)

    expected_vwap = (((intraday["high"] + intraday["low"] + intraday["close"]) / 3) * intraday[
        "volume"
    ]).sum() / intraday["volume"].sum()
    expected_atr = compute_atr(daily, period=14).iloc[-1]

    assert isinstance(context, MarketContext)
    assert context.ticker == "TEST"
    assert context.last_price == pytest.approx(112.0)
    assert context.prev_close == pytest.approx(100.0)
    assert context.today_open == pytest.approx(100.0)
    assert context.day_high == pytest.approx(114.0)
    assert context.day_low == pytest.approx(99.0)
    assert context.session_vwap == pytest.approx(expected_vwap)
    assert context.atr_14 == pytest.approx(expected_atr)
    assert context.gap_pct == pytest.approx(0.0)
    assert context.gap_atr == pytest.approx(0.0)
    assert context.day_range_pos == pytest.approx((112.0 - 99.0) / (114.0 - 99.0))
    assert context.price_vs_vwap_pct == pytest.approx((112.0 - expected_vwap) / expected_vwap)
    assert context.return_5m == pytest.approx((112.0 - 108.0) / 108.0)
    assert context.return_15m == pytest.approx((112.0 - 100.0) / 100.0)
    assert context.return_60m == pytest.approx((112.0 - 100.0) / 100.0)


def test_quote_last_price_overrides_intraday_close() -> None:
    intraday = _intraday_df()
    daily = _daily_df()

    context = compute_market_context("TEST", intraday, daily, quote={"last_price": 120.0})

    assert context.last_price == pytest.approx(120.0)
    assert context.day_range_pos == pytest.approx((120.0 - 99.0) / (114.0 - 99.0))
    assert context.price_vs_vwap_pct > 0


def test_market_context_handles_empty_data_safely() -> None:
    context = compute_market_context("EMPTY", pd.DataFrame(), pd.DataFrame())

    assert context.ticker == "EMPTY"
    assert context.last_price is None
    assert context.prev_close is None
    assert context.today_open is None
    assert context.day_high is None
    assert context.day_low is None
    assert context.session_vwap is None
    assert context.atr_14 is None
    assert context.gap_pct is None
    assert context.gap_atr is None
    assert context.day_range_pos is None
    assert context.price_vs_vwap_pct is None
    assert context.return_5m is None
    assert context.return_15m is None
    assert context.return_60m is None
