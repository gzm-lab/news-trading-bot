"""Tests for RSS news source."""

from __future__ import annotations

import json
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
        feed = _make_feed(
            [
                _make_entry("Apple stock rises after earnings", "AAPL beats estimates"),
            ]
        )
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
        feed = _make_feed(
            [
                _make_entry("AAPL and MSFT both surge", "Tech rally"),
            ]
        )
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
        feed = _make_feed(
            [
                _make_entry("General market news", "No specific tickers"),
            ]
        )
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

    @pytest.mark.asyncio
    async def test_falls_back_to_text_extraction_when_llm_fails(self):
        source = RSSSource(
            feed_urls=["https://example.com/feed"],
            known_tickers=["AAPL", "MSFT", "TSLA"],
        )
        feed = _make_feed(
            [
                _make_entry("AAPL and MSFT both surge", "Tech rally"),
            ]
        )

        with (
            patch("src.news.rss_source.feedparser.parse", return_value=feed),
            patch("src.news.rss_source.urllib.request.urlopen", side_effect=Exception("down")),
        ):
            items = await source.fetch()

        assert "AAPL" in items[0].tickers
        assert "MSFT" in items[0].tickers
        assert items[0].ticker_scores["AAPL"] == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_parses_legacy_tickers_schema(self):
        source = RSSSource(feed_urls=["https://example.com/feed"], known_tickers=["AAPL"])
        feed = _make_feed([_make_entry("Apple news", "")])
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"tickers": ["AAPL"]}
        ).encode()

        with (
            patch("src.news.rss_source.feedparser.parse", return_value=feed),
            patch("src.news.rss_source.urllib.request.urlopen", return_value=response),
        ):
            items = await source.fetch()

        assert items[0].tickers == ["AAPL"]
        assert items[0].ticker_scores == {"AAPL": 0.7}

    @pytest.mark.asyncio
    async def test_parses_impacts_schema(self):
        source = RSSSource(feed_urls=["https://example.com/feed"], known_tickers=["AAPL"])
        feed = _make_feed([_make_entry("Apple news", "")])
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"impacts": [{"ticker": "AAPL", "score": 0.82}]}
        ).encode()

        with (
            patch("src.news.rss_source.feedparser.parse", return_value=feed),
            patch("src.news.rss_source.urllib.request.urlopen", return_value=response),
        ):
            items = await source.fetch()

        assert items[0].tickers == ["AAPL"]
        assert items[0].ticker_scores == {"AAPL": 0.82}

    @pytest.mark.asyncio
    async def test_seen_cache_skips_llm_for_duplicate_fingerprint(self):
        source = RSSSource(feed_urls=["https://example.com/feed"], known_tickers=["AAPL"])
        seen = {}
        source.inject_seen_cache(seen)
        feed = _make_feed([_make_entry("AAPL news", link="https://example.com/a")])

        with (
            patch("src.news.rss_source.feedparser.parse", return_value=feed),
            patch.object(source, "_extract_tickers_via_llm", return_value={"AAPL": 0.8}) as extract,
        ):
            first = await source.fetch()
            seen[first[0].fingerprint] = True
            second = await source.fetch()

        assert len(first) == 1
        assert second == []
        assert extract.call_count == 1
