"""Tests for news aggregator — deduplication + multi-source."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.news.aggregator import NewsAggregator
from src.news.base import NewsItem


def _make_item(title, url="https://example.com"):
    return NewsItem(source="test", title=title, url=url)


class TestNewsAggregator:
    @pytest.mark.asyncio
    async def test_fetch_from_single_source(self):
        source = MagicMock()
        source.fetch = AsyncMock(return_value=[_make_item("Article 1")])
        agg = NewsAggregator(sources=[source])

        items = await agg.fetch_latest(tickers=["AAPL"])
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_deduplicates_same_article(self):
        item = _make_item("Same Article", "https://example.com/same")
        source = MagicMock()
        source.fetch = AsyncMock(return_value=[item, item])
        agg = NewsAggregator(sources=[source])

        items = await agg.fetch_latest()
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_dedup_across_sources(self):
        item1 = _make_item("Breaking News", "https://example.com/breaking")
        item2 = _make_item("Breaking News", "https://example.com/breaking")

        source1 = MagicMock()
        source1.fetch = AsyncMock(return_value=[item1])
        source2 = MagicMock()
        source2.fetch = AsyncMock(return_value=[item2])

        agg = NewsAggregator(sources=[source1, source2])
        items = await agg.fetch_latest()
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_different_articles_kept(self):
        source = MagicMock()
        source.fetch = AsyncMock(return_value=[
            _make_item("Article A", "https://example.com/a"),
            _make_item("Article B", "https://example.com/b"),
        ])
        agg = NewsAggregator(sources=[source])

        items = await agg.fetch_latest()
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_dedup_across_cycles(self):
        """Items from a previous cycle should not appear again."""
        item = _make_item("Repeated News", "https://example.com/repeated")
        source = MagicMock()
        source.fetch = AsyncMock(return_value=[item])
        agg = NewsAggregator(sources=[source])

        items1 = await agg.fetch_latest()
        items2 = await agg.fetch_latest()

        assert len(items1) == 1
        assert len(items2) == 0

    @pytest.mark.asyncio
    async def test_source_error_handled(self):
        bad_source = MagicMock()
        bad_source.fetch = AsyncMock(side_effect=Exception("Failed"))
        good_source = MagicMock()
        good_source.fetch = AsyncMock(return_value=[_make_item("Good article")])

        agg = NewsAggregator(sources=[bad_source, good_source])
        items = await agg.fetch_latest()
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_stores_to_db(self, tmp_db):
        source = MagicMock()
        source.fetch = AsyncMock(return_value=[
            _make_item("Stored Article", "https://example.com/stored"),
        ])
        agg = NewsAggregator(sources=[source], db=tmp_db)

        await agg.fetch_latest()

        from src.storage.models import NewsArticle
        with tmp_db.get_session() as session:
            count = session.query(NewsArticle).count()
            assert count == 1
