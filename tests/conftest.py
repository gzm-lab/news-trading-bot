"""Shared fixtures for all tests."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import numpy as np
import pytest

from src.config import StrategySettings
from src.broker.interface import (
    Account,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from src.news.base import NewsItem
from src.storage.database import Database


# ──────────────────────────────────────────────
# Database fixture (temp SQLite)
# ──────────────────────────────────────────────
@pytest.fixture
def tmp_db(tmp_path):
    """In-memory SQLite database for tests."""
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.init()
    return db


# ──────────────────────────────────────────────
# Broker data fixtures
# ──────────────────────────────────────────────
@pytest.fixture
def sample_account():
    return Account(
        equity=100_000.0,
        cash=80_000.0,
        buying_power=160_000.0,
        portfolio_value=100_000.0,
        daily_pnl=500.0,
        daily_pnl_pct=0.005,
    )


@pytest.fixture
def sample_positions():
    return [
        Position(
            ticker="AAPL",
            qty=50,
            avg_entry_price=180.0,
            current_price=185.0,
            market_value=9250.0,
            unrealized_pnl=250.0,
            unrealized_pnl_pct=0.0278,
        ),
        Position(
            ticker="MSFT",
            qty=30,
            avg_entry_price=400.0,
            current_price=390.0,
            market_value=11700.0,
            unrealized_pnl=-300.0,
            unrealized_pnl_pct=-0.025,
        ),
    ]


@pytest.fixture
def sample_order():
    return Order(
        ticker="AAPL",
        side=OrderSide.BUY,
        qty=10,
        order_type=OrderType.MARKET,
        id="order-123",
        status=OrderStatus.FILLED,
        filled_price=182.50,
    )


# ──────────────────────────────────────────────
# News fixtures
# ──────────────────────────────────────────────
@pytest.fixture
def sample_news_items():
    now = datetime.now(timezone.utc)
    return [
        NewsItem(
            source="finnhub",
            title="Apple reports record quarterly earnings beating expectations",
            summary="Apple Inc. reported Q4 earnings of $1.46 per share, beating estimates.",
            url="https://example.com/apple-earnings",
            tickers=["AAPL"],
            published_at=now - timedelta(minutes=10),
        ),
        NewsItem(
            source="rss_yahoo",
            title="Tesla recalls 500,000 vehicles over safety concerns",
            summary="Tesla is recalling vehicles due to a potential safety issue.",
            url="https://example.com/tesla-recall",
            tickers=["TSLA"],
            published_at=now - timedelta(minutes=20),
        ),
        NewsItem(
            source="finnhub",
            title="Microsoft Azure revenue grows 30% year-over-year",
            summary="Microsoft cloud business continues strong growth trajectory.",
            url="https://example.com/msft-cloud",
            tickers=["MSFT"],
            published_at=now - timedelta(minutes=5),
        ),
    ]


# ──────────────────────────────────────────────
# Market data fixture (OHLCV DataFrame)
# ──────────────────────────────────────────────
@pytest.fixture
def sample_ohlcv():
    """Generate 50 bars of realistic OHLCV data."""
    np.random.seed(42)
    n = 50
    dates = pd.date_range(end=datetime.now(), periods=n, freq="h")

    close = 180.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(100_000, 1_000_000, size=n).astype(float)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "timestamp": dates,
    })


# ──────────────────────────────────────────────
# Strategy config fixture
# ──────────────────────────────────────────────
@pytest.fixture
def strategy_config():
    return StrategySettings(
        w_sentiment=0.4,
        w_news_velocity=0.2,
        w_technical=0.25,
        w_volume=0.15,
        buy_threshold=0.3,
        sell_threshold=-0.2,
        max_position_pct=0.05,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        max_positions=10,
        max_daily_drawdown_pct=0.05,
        cooldown_minutes=30,
    )
