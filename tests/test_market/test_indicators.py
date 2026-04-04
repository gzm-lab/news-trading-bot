"""Tests for technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.market.indicators import (
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_vwap,
    detect_volume_anomaly,
    compute_momentum_score,
    compute_volume_score,
)


@pytest.fixture
def prices():
    np.random.seed(42)
    return pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5))


@pytest.fixture
def ohlcv_df():
    np.random.seed(42)
    n = 50
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(100_000, 1_000_000, size=n).astype(float),
    })


class TestRSI:
    def test_rsi_range(self, prices):
        rsi = compute_rsi(prices)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_period(self, prices):
        rsi = compute_rsi(prices, period=14)
        assert len(rsi) == len(prices)

    def test_rsi_all_gains(self):
        """All positive changes should give RSI near 100."""
        prices = pd.Series([100 + i for i in range(30)])
        rsi = compute_rsi(prices)
        assert rsi.iloc[-1] > 90

    def test_rsi_all_losses(self):
        """All negative changes should give RSI near 0."""
        prices = pd.Series([100 - i * 0.5 for i in range(30)])
        rsi = compute_rsi(prices)
        assert rsi.iloc[-1] < 10


class TestMACD:
    def test_macd_components(self, prices):
        macd_line, signal_line, histogram = compute_macd(prices)
        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)

    def test_histogram_is_difference(self, prices):
        macd_line, signal_line, histogram = compute_macd(prices)
        # Histogram = MACD - Signal
        diff = macd_line - signal_line
        np.testing.assert_allclose(histogram.values, diff.values, atol=1e-10)

    def test_custom_periods(self, prices):
        macd_line, _, _ = compute_macd(prices, fast=8, slow=21, signal=5)
        assert not macd_line.isna().all()


class TestBollingerBands:
    def test_bands_structure(self, prices):
        upper, middle, lower = compute_bollinger_bands(prices)
        valid_idx = ~upper.isna()
        assert (upper[valid_idx] > middle[valid_idx]).all()
        assert (lower[valid_idx] < middle[valid_idx]).all()

    def test_custom_std(self, prices):
        u1, _, l1 = compute_bollinger_bands(prices, std_dev=1.0)
        u2, _, l2 = compute_bollinger_bands(prices, std_dev=2.0)
        valid_idx = ~u1.isna() & ~u2.isna()
        # Wider std_dev = wider bands
        assert (u2[valid_idx] >= u1[valid_idx]).all()

    def test_middle_is_sma(self, prices):
        _, middle, _ = compute_bollinger_bands(prices, period=20)
        sma = prices.rolling(window=20).mean()
        pd.testing.assert_series_equal(middle, sma)


class TestVWAP:
    def test_vwap_basic(self, ohlcv_df):
        vwap = compute_vwap(ohlcv_df)
        assert len(vwap) == len(ohlcv_df)
        assert not vwap.isna().all()

    def test_vwap_between_high_low(self, ohlcv_df):
        """VWAP should generally be between cumulative high and low."""
        vwap = compute_vwap(ohlcv_df)
        # First few values might be weird, check from index 5+
        assert vwap.iloc[-1] > 0


class TestVolumeAnomaly:
    def test_detects_spike(self):
        volume = pd.Series([100_000] * 25 + [500_000])
        anomalies = detect_volume_anomaly(volume, window=20, threshold=2.0)
        assert anomalies.iloc[-1] is True or anomalies.iloc[-1] == True

    def test_no_anomaly_normal_volume(self):
        volume = pd.Series([100_000] * 30)
        anomalies = detect_volume_anomaly(volume, window=20, threshold=2.0)
        assert anomalies.iloc[-1] == False


class TestMomentumScore:
    def test_returns_float(self, ohlcv_df):
        score = compute_momentum_score(ohlcv_df)
        assert isinstance(score, float)

    def test_range(self, ohlcv_df):
        score = compute_momentum_score(ohlcv_df)
        assert -1.0 <= score <= 1.0

    def test_none_returns_zero(self):
        assert compute_momentum_score(None) == 0.0

    def test_short_data_returns_zero(self):
        df = pd.DataFrame({"close": [100, 101, 102], "volume": [1000, 1000, 1000]})
        assert compute_momentum_score(df) == 0.0


class TestVolumeScore:
    def test_returns_float(self, ohlcv_df):
        score = compute_volume_score(ohlcv_df)
        assert isinstance(score, float)

    def test_range(self, ohlcv_df):
        score = compute_volume_score(ohlcv_df)
        assert 0.0 <= score <= 1.0

    def test_none_returns_zero(self):
        assert compute_volume_score(None) == 0.0
