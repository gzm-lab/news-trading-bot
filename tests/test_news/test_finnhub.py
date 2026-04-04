"""Tests for Finnhub news source — mocked API."""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.news.finnhub_source import FinnhubSource


@pytest.fixture
def source():
    with patch("src.news.finnhub_source.finnhub.Client"):
        return FinnhubSource(api_key="test-key", max_age_minutes=60)


class TestFinnhubSource:
    @pytest.mark.asyncio
    async def test_fetch_empty_tickers(self, source):
        result = await source.fetch(tickers=[])
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_none_tickers(self, source):
        result = await source.fetch(tickers=None)
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_returns_news_items(self, source):
        now = datetime.now(timezone.utc)
        mock_articles = [
            {
                "headline": "Apple beats earnings",
                "summary": "Revenue up 15%",
                "url": "https://example.com/1",
                "datetime": int(now.timestamp()) - 300,  # 5 min ago
            },
            {
                "headline": "New iPhone announced",
                "summary": "iPhone 16 features",
                "url": "https://example.com/2",
                "datetime": int(now.timestamp()) - 600,  # 10 min ago
            },
        ]
        source._client.company_news = MagicMock(return_value=mock_articles)

        items = await source.fetch(tickers=["AAPL"])

        assert len(items) == 2
        assert items[0].source == "finnhub"
        assert items[0].title == "Apple beats earnings"
        assert items[0].tickers == ["AAPL"]

    @pytest.mark.asyncio
    async def test_skips_old_news(self, source):
        very_old = int((datetime.now(timezone.utc) - timedelta(hours=5)).timestamp())
        mock_articles = [
            {
                "headline": "Old news",
                "summary": "",
                "url": "https://example.com/old",
                "datetime": very_old,
            },
        ]
        source._client.company_news = MagicMock(return_value=mock_articles)

        items = await source.fetch(tickers=["AAPL"])
        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_handles_api_error(self, source):
        source._client.company_news = MagicMock(side_effect=Exception("API error"))

        # Should not raise
        items = await source.fetch(tickers=["AAPL"])
        assert items == []

    @pytest.mark.asyncio
    async def test_multiple_tickers(self, source):
        now_ts = int(datetime.now(timezone.utc).timestamp()) - 60
        source._client.company_news = MagicMock(return_value=[
            {"headline": "News", "summary": "", "url": "https://example.com/x", "datetime": now_ts},
        ])

        items = await source.fetch(tickers=["AAPL", "MSFT", "TSLA"])
        # 1 article per ticker call
        assert len(items) == 3
