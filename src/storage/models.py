"""SQLAlchemy models for trade logging and persistence."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Text,
    Boolean,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    pass


class NewsArticle(Base):
    """Stored news articles with sentiment scores."""

    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False)  # "finnhub", "rss", etc.
    title = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    url = Column(String(500), nullable=True)
    tickers = Column(String(200), nullable=True)  # comma-separated
    published_at = Column(DateTime, nullable=True)
    fetched_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    sentiment_score = Column(Float, nullable=True)  # -1.0 to +1.0
    sentiment_label = Column(String(20), nullable=True)  # positive/negative/neutral
    fingerprint = Column(String(64), unique=True, nullable=False)  # dedup hash


class TradeLog(Base):
    """Every order placed by the bot."""

    __tablename__ = "trade_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(100), nullable=True)
    ticker = Column(String(10), nullable=False)
    side = Column(String(4), nullable=False)  # buy/sell
    qty = Column(Integer, nullable=False)
    order_type = Column(String(10), nullable=False)
    limit_price = Column(Float, nullable=True)
    filled_price = Column(Float, nullable=True)
    status = Column(String(20), nullable=False)
    signal_score = Column(Float, nullable=True)
    reason = Column(Text, nullable=True)  # why the trade was made
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class CycleLog(Base):
    """One row per main loop iteration — for debugging and analysis."""

    __tablename__ = "cycle_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    news_count = Column(Integer, default=0)
    signals_generated = Column(Integer, default=0)
    orders_placed = Column(Integer, default=0)
    portfolio_value = Column(Float, nullable=True)
    daily_pnl = Column(Float, nullable=True)
    cycle_duration_ms = Column(Integer, nullable=True)


class PortfolioSnapshot(Base):
    """Periodic portfolio snapshots for performance tracking."""

    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    positions_count = Column(Integer, default=0)
    daily_pnl = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
