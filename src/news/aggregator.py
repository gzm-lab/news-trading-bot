"""News aggregator — combines multiple sources with deduplication."""

from __future__ import annotations

from cachetools import TTLCache
import structlog

from src.news.base import NewsItem, NewsSource
from src.storage.database import Database
from src.storage.models import NewsArticle

log = structlog.get_logger()


class NewsAggregator:
    """Aggregates news from multiple sources with dedup."""

    def __init__(self, sources: list[NewsSource], db: Database | None = None, ttl: int = 3600):
        self._sources = sources
        self._db = db
        self._seen: TTLCache = TTLCache(maxsize=10_000, ttl=ttl)

    async def fetch_latest(self, tickers: list[str] | None = None) -> list[NewsItem]:
        """Fetch from all sources, deduplicate, return new items."""
        all_items: list[NewsItem] = []

        for source in self._sources:
            try:
                items = await source.fetch(tickers=tickers)
                all_items.extend(items)
            except Exception as e:
                log.warning("aggregator.source_failed", source=type(source).__name__, error=str(e))

        # Deduplicate
        new_items: list[NewsItem] = []
        for item in all_items:
            fp = item.fingerprint
            if fp not in self._seen:
                self._seen[fp] = True
                new_items.append(item)

        # Persist to DB
        if self._db and new_items:
            self._store(new_items)

        log.info(
            "aggregator.fetched",
            total_fetched=len(all_items),
            new_items=len(new_items),
            duplicates=len(all_items) - len(new_items),
        )
        return new_items

    def _store(self, items: list[NewsItem]) -> None:
        """Store news items in the database."""
        try:
            articles = []
            for item in items:
                article = NewsArticle(
                    source=item.source,
                    title=item.title,
                    summary=item.summary,
                    url=item.url,
                    tickers=",".join(item.tickers) if item.tickers else None,
                    published_at=item.published_at,
                    fetched_at=item.fetched_at,
                    fingerprint=item.fingerprint,
                )
                articles.append(article)
            self._db.save_all(articles)
        except Exception as e:
            log.warning("aggregator.store_failed", error=str(e), count=len(items))
