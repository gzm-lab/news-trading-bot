"""Centralized configuration using pydantic-settings."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BrokerSettings(BaseSettings):
    """Alpaca / IBKR broker configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ALPACA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"


class NewsSettings(BaseSettings):
    """News sources configuration."""

    model_config = SettingsConfigDict(env_prefix="NEWS_")

    finnhub_api_key: str = Field(default="", alias="FINNHUB_API_KEY")
    rss_feeds: list[str] = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # Top News
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",  # Markets
    ]
    fetch_interval: int = 120  # seconds between news fetches
    max_age_minutes: int = 60  # ignore news older than this


class SentimentSettings(BaseSettings):
    """Sentiment analysis configuration."""

    model_config = SettingsConfigDict(env_prefix="SENTIMENT_")

    model_name: str = "ProsusAI/finbert"
    batch_size: int = 16
    max_length: int = 512


class StrategySettings(BaseSettings):
    """Trading strategy parameters."""

    model_config = SettingsConfigDict(env_prefix="STRATEGY_")

    # Signal weights
    w_sentiment: float = 0.40
    w_news_velocity: float = 0.20
    w_technical: float = 0.25
    w_volume: float = 0.15

    # Thresholds
    buy_threshold: float = 0.15
    sell_threshold: float = -0.1

    # Risk management
    max_position_pct: float = 0.05  # 5% of portfolio per position
    stop_loss_pct: float = 0.02  # -2%
    take_profit_pct: float = 0.04  # +4%
    max_positions: int = 10
    max_daily_drawdown_pct: float = 0.05  # -5% -> stop trading
    cooldown_minutes: int = 30
    blackout_minutes: int = 15



class AlertSettings(BaseSettings):
    """Discord alert configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DISCORD_",
        env_file=".env",
        extra="ignore"
    )

    webhook_url: str = ""
    enabled: bool = True


class Settings(BaseSettings):
    """Root configuration — aggregates all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-configs
    broker: BrokerSettings = BrokerSettings()
    news: NewsSettings = NewsSettings()
    sentiment: SentimentSettings = SentimentSettings()
    strategy: StrategySettings = StrategySettings()
    alerts: AlertSettings = AlertSettings()

    # Global
    log_level: str = "INFO"
    cycle_interval: int = 120  # seconds
    db_path: str = "data/trading.db"

    # Universe of tickers to track
    universe: list[str] = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "NVDA", "META", "NFLX", "AMD", "CRM",
        "JPM", "V", "JNJ", "UNH", "PG",
    ]


# Singleton
settings = Settings()
