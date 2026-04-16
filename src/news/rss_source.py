"""RSS feed news source."""

from __future__ import annotations

import asyncio
import json
import urllib.request
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

                    # Extract tickers and sentiment via the new LLM service
                    found_impacts = await self._extract_tickers_via_llm(title, summary)
                    tickers_list = list(found_impacts.keys())

                    item = NewsItem(
                        source=source_name,
                        title=title,
                        summary=summary[:500],
                        url=link,
                        tickers=tickers_list,
                        ticker_scores=found_impacts,
                        published_at=published,
                    )
                    all_items.append(item)

            except Exception as e:
                log.warning("rss.fetch_failed", url=url, error=str(e))

        log.info("rss.fetched", items=len(all_items), feeds=len(self._feed_urls))
        return all_items

    async def _extract_tickers_via_llm(self, title: str, summary: str) -> dict[str, float]:
        """Extract tickers and their sentiment scores using the local Hermes LLM Docker service."""
        def do_request():
            url = "http://localhost:8000/extract_tickers"
            data = json.dumps({"title": title, "summary": summary}).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    impacts = res.get("impacts", [])
                    # Convert list of dicts to dict[ticker, score]
                    return {item["ticker"].upper(): float(item["score"]) for item in impacts if "ticker" in item and "score" in item}
            except Exception as e:
                log.error("llm_extraction.failed", error=str(e), title=title[:50])
                return {}
        
        return await asyncio.to_thread(do_request)
