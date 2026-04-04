"""Tests for RSS news source."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.news.rss_source import RSSSource


def _make_feed(entries):
    """Create a mock feedparser result."""
    feed = MagicMock()
    feed.feed = {"title": "Test Feed"}
    feed.entries = entries
    return feed


def _make_entry(title, summary="", link="https://example.com", published=""):
    entry = MagicMock()
    entry.get = lambda key, default="": {
        "title": title,
        "summary": summary,
        "description": summary,
        "link": link,
        "published": published,
        "updated": "",
    }.get(key, default)
    return entry


class TestRSSSource:
    @pytest.mark.asyncio
    async def test_fetch_basic(self):
        source = RSSSource(
            feed_urls=["https://example.com/feed"],
            known_tickers=["AAPL", "MSFT"],
        )
        feed = _make_feed([
            _make_entry("Apple stock rises after earnings", "AAPL beats estimates"),
        ])
        with patch("src.news.rss_source.feedparser.parse", return_value=feed):
            items = await source.fetch()

        assert len(items) == 1
        assert "AAPL" in items[0].tickers

    @pytest.mark.asyncio
    async def test_extracts_tickers_from_text(self):
        source = RSSSource(
            feed_urls=["https://example.com/feed"],
            known_tickers=["AAPL", "MSFT", "TSLA"],
        )
        feed = _make_feed([
            _make_entry("AAPL and MSFT both surge", "Tech rally"),
        ])
        with patch("src.news.rss_source.feedparser.parse", return_value=feed):
            items = await source.fetch()

        assert "AAPL" in items[0].tickers
        assert "MSFT" in items[0].tickers

    @pytest.mark.asyncio
    async def test_no_tickers_matched(self):
        source = RSSSource(
            feed_urls=["https://example.com/feed"],
            known_tickers=["AAPL"],
        )
        feed = _make_feed([
            _make_entry("General market news", "No specific tickers"),
        ])
        with patch("src.news.rss_source.feedparser.parse", return_value=feed):
            items = await source.fetch()

        assert len(items) == 1
        assert items[0].tickers == []

    @pytest.mark.asyncio
    async def test_handles_parse_error(self):
        source = RSSSource(feed_urls=["https://bad.url/feed"])
        with patch("src.news.rss_source.feedparser.parse", side_effect=Exception("Parse error")):
            items = await source.fetch()
            assert items == []

    @pytest.mark.asyncio
    async def test_multiple_feeds(self):
        source = RSSSource(
            feed_urls=["https://feed1.com", "https://feed2.com"],
            known_tickers=["AAPL"],
        )
        feed = _make_feed([_make_entry("AAPL news")])
        with patch("src.news.rss_source.feedparser.parse", return_value=feed):
            items = await source.fetch()

        assert len(items) == 2  # One per feed
