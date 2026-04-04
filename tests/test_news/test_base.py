"""Tests for news base types."""

from datetime import datetime, timezone

import pytest

from src.news.base import NewsItem


class TestNewsItem:
    def test_create(self):
        item = NewsItem(
            source="test",
            title="Test headline",
            summary="A test summary",
            url="https://example.com/article",
            tickers=["AAPL"],
        )
        assert item.source == "test"
        assert item.title == "Test headline"
        assert item.tickers == ["AAPL"]

    def test_fingerprint_deterministic(self):
        item1 = NewsItem(source="a", title="Same title", url="https://example.com")
        item2 = NewsItem(source="b", title="Same title", url="https://example.com")
        assert item1.fingerprint == item2.fingerprint

    def test_fingerprint_different_for_different_content(self):
        item1 = NewsItem(source="test", title="Title A", url="https://example.com/a")
        item2 = NewsItem(source="test", title="Title B", url="https://example.com/b")
        assert item1.fingerprint != item2.fingerprint

    def test_text_for_analysis_with_summary(self):
        item = NewsItem(source="test", title="Headline", summary="Full summary text")
        assert "Headline" in item.text_for_analysis
        assert "Full summary text" in item.text_for_analysis

    def test_text_for_analysis_without_summary(self):
        item = NewsItem(source="test", title="Just a headline")
        assert item.text_for_analysis == "Just a headline"

    def test_fetched_at_auto_set(self):
        item = NewsItem(source="test", title="Test")
        assert item.fetched_at is not None
        assert item.fetched_at.tzinfo == timezone.utc

    def test_default_empty_tickers(self):
        item = NewsItem(source="test", title="No tickers")
        assert item.tickers == []
