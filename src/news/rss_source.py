"""RSS feed news source."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser
import structlog

from src.news.base import NewsItem, NewsSource

log = structlog.get_logger()


class RSSSource(NewsSource):
    """Fetches news from RSS/Atom feeds."""

    def __init__(self, feed_urls: list[str], known_tickers: list[str] | None = None):
        self._feed_urls = feed_urls
        self._known_tickers = set(known_tickers or [])
        # Build regex for ticker extraction
        if self._known_tickers:
            pattern = r"\b(" + "|".join(re.escape(t) for t in self._known_tickers) + r")\b"
            self._ticker_regex = re.compile(pattern)
        else:
            self._ticker_regex = None

    async def fetch(self, tickers: list[str] | None = None) -> list[NewsItem]:
        all_items: list[NewsItem] = []

        for url in self._feed_urls:
            try:
                feed = await asyncio.to_thread(feedparser.parse, url)
                source_name = f"rss_{feed.feed.get('title', url)[:20]}"

                for entry in feed.entries[:20]:  # Limit per feed
                    title = entry.get("title", "")
                    summary = entry.get("summary", entry.get("description", ""))
                    link = entry.get("link", "")

                    # Parse published date
                    published = None
                    pub_str = entry.get("published", entry.get("updated", ""))
                    if pub_str:
                        try:
                            published = parsedate_to_datetime(pub_str)
                            if published.tzinfo is None:
                                published = published.replace(tzinfo=timezone.utc)
                        except Exception:
                            published = None

                    # Extract tickers from title + summary
                    found_tickers = self._extract_tickers(f"{title} {summary}")

                    item = NewsItem(
                        source=source_name,
                        title=title,
                        summary=summary[:500],
                        url=link,
                        tickers=found_tickers,
                        published_at=published,
                    )
                    all_items.append(item)

            except Exception as e:
                log.warning("rss.fetch_failed", url=url, error=str(e))

        log.info("rss.fetched", items=len(all_items), feeds=len(self._feed_urls))
        return all_items

    def _extract_tickers(self, text: str) -> list[str]:
        """Extract known ticker symbols from text."""
        if not self._ticker_regex:
            return []
        matches = self._ticker_regex.findall(text)
        return list(set(matches))
