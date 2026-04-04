"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from src.config import (
    BrokerSettings,
    NewsSettings,
    SentimentSettings,
    StrategySettings,
    AlertSettings,
    Settings,
)


class TestBrokerSettings:
    def test_defaults(self):
        cfg = BrokerSettings()
        assert cfg.api_key == ""
        assert cfg.secret_key == ""
        assert "paper" in cfg.base_url

    def test_env_override(self):
        with patch.dict(os.environ, {"ALPACA_API_KEY": "test-key", "ALPACA_SECRET_KEY": "test-secret"}):
            cfg = BrokerSettings()
            assert cfg.api_key == "test-key"
            assert cfg.secret_key == "test-secret"


class TestNewsSettings:
    def test_defaults(self):
        cfg = NewsSettings()
        assert cfg.fetch_interval == 120
        assert cfg.max_age_minutes == 60
        assert len(cfg.rss_feeds) >= 2

    def test_finnhub_key_alias(self):
        with patch.dict(os.environ, {"FINNHUB_API_KEY": "fk-123"}):
            cfg = NewsSettings()
            assert cfg.finnhub_api_key == "fk-123"


class TestSentimentSettings:
    def test_defaults(self):
        cfg = SentimentSettings()
        assert cfg.model_name == "ProsusAI/finbert"
        assert cfg.batch_size == 16
        assert cfg.max_length == 512


class TestStrategySettings:
    def test_weights_sum_to_one(self):
        cfg = StrategySettings()
        total = cfg.w_sentiment + cfg.w_news_velocity + cfg.w_technical + cfg.w_volume
        assert total == pytest.approx(1.0)

    def test_thresholds(self):
        cfg = StrategySettings()
        assert cfg.buy_threshold > 0
        assert cfg.sell_threshold < 0

    def test_risk_defaults(self):
        cfg = StrategySettings()
        assert cfg.stop_loss_pct > 0
        assert cfg.take_profit_pct > cfg.stop_loss_pct
        assert cfg.max_positions > 0
        assert cfg.max_daily_drawdown_pct > 0


class TestAlertSettings:
    def test_defaults(self):
        cfg = AlertSettings()
        assert cfg.webhook_url == ""
        assert cfg.enabled is True


class TestSettings:
    def test_creates_root(self):
        cfg = Settings()
        assert isinstance(cfg.broker, BrokerSettings)
        assert isinstance(cfg.news, NewsSettings)
        assert isinstance(cfg.strategy, StrategySettings)
        assert len(cfg.universe) >= 10

    def test_universe_contains_known_tickers(self):
        cfg = Settings()
        assert "AAPL" in cfg.universe
        assert "MSFT" in cfg.universe
        assert "TSLA" in cfg.universe

    def test_db_path(self):
        cfg = Settings()
        assert "trading.db" in cfg.db_path

    def test_cycle_interval(self):
        cfg = Settings()
        assert cfg.cycle_interval > 0
