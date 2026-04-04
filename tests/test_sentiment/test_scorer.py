"""Tests for sentiment scorer — aggregation per ticker."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.news.base import NewsItem
from src.sentiment.finbert import SentimentResult
from src.sentiment.scorer import SentimentScorer, TickerSentiment


@pytest.fixture
def mock_analyzer():
    analyzer = MagicMock()
    analyzer.analyze = AsyncMock()
    return analyzer


@pytest.fixture
def scorer(mock_analyzer):
    return SentimentScorer(analyzer=mock_analyzer, decay_minutes=30)


def _make_news(title, tickers, minutes_ago=5):
    return NewsItem(
        source="test",
        title=title,
        tickers=tickers,
        published_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
    )


def _make_sentiment(score, label="positive"):
    return SentimentResult(
        text="",
        label=label,
        score=score,
        confidence=0.9,
        probabilities={"positive": 0.8, "negative": 0.1, "neutral": 0.1},
    )


class TestSentimentScorer:
    @pytest.mark.asyncio
    async def test_empty_news(self, scorer, mock_analyzer):
        result = await scorer.score_news([])
        assert result == {}
        mock_analyzer.analyze.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_ticker(self, scorer, mock_analyzer):
        news = [_make_news("Apple earnings beat", ["AAPL"])]
        mock_analyzer.analyze.return_value = [_make_sentiment(0.8)]

        result = await scorer.score_news(news)

        assert "AAPL" in result
        assert isinstance(result["AAPL"], TickerSentiment)
        assert result["AAPL"].avg_score > 0
        assert result["AAPL"].news_count == 1

    @pytest.mark.asyncio
    async def test_multiple_tickers(self, scorer, mock_analyzer):
        news = [
            _make_news("Apple up", ["AAPL"]),
            _make_news("Tesla down", ["TSLA"]),
        ]
        mock_analyzer.analyze.return_value = [
            _make_sentiment(0.7),
            _make_sentiment(-0.6, label="negative"),
        ]

        result = await scorer.score_news(news)
        assert "AAPL" in result
        assert "TSLA" in result
        assert result["AAPL"].avg_score > 0
        assert result["TSLA"].avg_score < 0

    @pytest.mark.asyncio
    async def test_shared_tickers(self, scorer, mock_analyzer):
        """One article mentioning multiple tickers should score both."""
        news = [_make_news("Tech rally boosts AAPL and MSFT", ["AAPL", "MSFT"])]
        mock_analyzer.analyze.return_value = [_make_sentiment(0.6)]

        result = await scorer.score_news(news)
        assert "AAPL" in result
        assert "MSFT" in result

    @pytest.mark.asyncio
    async def test_accumulates_history(self, scorer, mock_analyzer):
        """Multiple cycles should accumulate history."""
        news1 = [_make_news("First article", ["AAPL"])]
        mock_analyzer.analyze.return_value = [_make_sentiment(0.5)]
        await scorer.score_news(news1)

        news2 = [_make_news("Second article", ["AAPL"])]
        mock_analyzer.analyze.return_value = [_make_sentiment(0.8)]
        result = await scorer.score_news(news2)

        assert result["AAPL"].news_count == 2

    @pytest.mark.asyncio
    async def test_news_velocity(self, scorer, mock_analyzer):
        """Velocity counts news in the last hour."""
        news = [
            _make_news("Article 1", ["AAPL"], minutes_ago=10),
            _make_news("Article 2", ["AAPL"], minutes_ago=20),
            _make_news("Article 3", ["AAPL"], minutes_ago=30),
        ]
        mock_analyzer.analyze.return_value = [
            _make_sentiment(0.5),
            _make_sentiment(0.3),
            _make_sentiment(0.4),
        ]

        result = await scorer.score_news(news)
        assert result["AAPL"].news_velocity == 3  # All within 1 hour

    @pytest.mark.asyncio
    async def test_headlines_stored(self, scorer, mock_analyzer):
        news = [_make_news("Headline to remember", ["AAPL"])]
        mock_analyzer.analyze.return_value = [_make_sentiment(0.5)]

        result = await scorer.score_news(news)
        assert "Headline to remember" in result["AAPL"].headlines
