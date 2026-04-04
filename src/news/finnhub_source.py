"""Finnhub news source."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta

import finnhub
import structlog

from src.news.base import NewsItem, NewsSource

log = structlog.get_logger()


class FinnhubSource(NewsSource):
    """Fetches company news from Finnhub API."""

    def __init__(self, api_key: str, max_age_minutes: int = 60):
        self._client = finnhub.Client(api_key=api_key)
        self._max_age = timedelta(minutes=max_age_minutes)

    async def fetch(self, tickers: list[str] | None = None) -> list[NewsItem]:
        if not tickers:
            return []

        all_items: list[NewsItem] = []
        now = datetime.now(timezone.utc)
        date_from = (now - self._max_age).strftime("%Y-%m-%d")
        date_to = now.strftime("%Y-%m-%d")

        for ticker in tickers:
            try:
                news = await asyncio.to_thread(
                    self._client.company_news, ticker, _from=date_from, to=date_to
                )

                for article in news[:10]:  # Limit per ticker
                    published = datetime.fromtimestamp(
                        article.get("datetime", 0), tz=timezone.utc
                    )

                    # Skip old news
                    if now - published > self._max_age:
                        continue

                    item = NewsItem(
                        source="finnhub",
                        title=article.get("headline", ""),
                        summary=article.get("summary", ""),
                        url=article.get("url", ""),
                        tickers=[ticker],
                        published_at=published,
                    )
                    all_items.append(item)

            except Exception as e:
                log.warning("finnhub.fetch_failed", ticker=ticker, error=str(e))

        log.info("finnhub.fetched", items=len(all_items), tickers=len(tickers))
        return all_items
