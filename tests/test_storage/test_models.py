"""Tests for storage models and database."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.storage.database import Database
from src.storage.models import (
    Base,
    NewsArticle,
    TradeLog,
    CycleLog,
    PortfolioSnapshot,
)


class TestNewsArticle:
    def test_create_and_save(self, tmp_db):
        article = NewsArticle(
            source="finnhub",
            title="Apple earnings beat",
            summary="Revenue up 15%",
            url="https://example.com/apple",
            tickers="AAPL",
            sentiment_score=0.85,
            sentiment_label="positive",
            fingerprint="abc123",
        )
        tmp_db.save(article)

        with tmp_db.get_session() as session:
            saved = session.query(NewsArticle).first()
            assert saved.title == "Apple earnings beat"
            assert saved.source == "finnhub"
            assert saved.sentiment_score == 0.85
            assert saved.fingerprint == "abc123"

    def test_unique_fingerprint(self, tmp_db):
        a1 = NewsArticle(
            source="test", title="Article 1", fingerprint="unique1"
        )
        a2 = NewsArticle(
            source="test", title="Article 2", fingerprint="unique1"
        )
        tmp_db.save(a1)

        with pytest.raises(Exception):
            tmp_db.save(a2)


class TestTradeLog:
    def test_create_and_save(self, tmp_db):
        trade = TradeLog(
            order_id="order-abc",
            ticker="AAPL",
            side="buy",
            qty=10,
            order_type="market",
            filled_price=185.0,
            status="filled",
            signal_score=0.65,
            reason="Strong positive sentiment",
        )
        tmp_db.save(trade)

        with tmp_db.get_session() as session:
            saved = session.query(TradeLog).first()
            assert saved.ticker == "AAPL"
            assert saved.side == "buy"
            assert saved.filled_price == 185.0
            assert saved.signal_score == 0.65


class TestCycleLog:
    def test_create_and_save(self, tmp_db):
        cycle = CycleLog(
            news_count=5,
            signals_generated=3,
            orders_placed=1,
            portfolio_value=100_500.0,
            daily_pnl=500.0,
            cycle_duration_ms=1234,
        )
        tmp_db.save(cycle)

        with tmp_db.get_session() as session:
            saved = session.query(CycleLog).first()
            assert saved.news_count == 5
            assert saved.cycle_duration_ms == 1234


class TestPortfolioSnapshot:
    def test_create_and_save(self, tmp_db):
        snap = PortfolioSnapshot(
            equity=100_000.0,
            cash=80_000.0,
            positions_count=3,
            daily_pnl=250.0,
            total_pnl=1500.0,
        )
        tmp_db.save(snap)

        with tmp_db.get_session() as session:
            saved = session.query(PortfolioSnapshot).first()
            assert saved.equity == 100_000.0
            assert saved.positions_count == 3


class TestDatabase:
    def test_init_creates_tables(self, tmp_path):
        db_path = str(tmp_path / "new_test.db")
        db = Database(db_path)
        db.init()

        # Should be able to query empty tables
        with db.get_session() as session:
            assert session.query(NewsArticle).count() == 0
            assert session.query(TradeLog).count() == 0

    def test_save_all(self, tmp_db):
        articles = [
            NewsArticle(source="test", title=f"Article {i}", fingerprint=f"fp{i}")
            for i in range(3)
        ]
        tmp_db.save_all(articles)

        with tmp_db.get_session() as session:
            assert session.query(NewsArticle).count() == 3

    def test_not_initialized_raises(self):
        db = Database("unused.db")
        with pytest.raises(AssertionError):
            db.get_session()
