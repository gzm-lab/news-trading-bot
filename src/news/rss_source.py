"""RSS feed news source."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import urllib.request
from datetime import UTC
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
        self._seen_cache = None

    def inject_seen_cache(self, cache):
        self._seen_cache = cache

    def _fingerprint(self, title: str, url: str) -> str:
        """Return the same fingerprint as NewsItem without creating the item yet."""
        content = f"{title}:{url}".lower().strip()
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _extract_tickers_from_text(self, title: str, summary: str) -> dict[str, float]:
        """Cheap fallback extraction when the local LLM service is unavailable."""
        if not self._known_tickers:
            return {}

        text = f" {title} {summary} ".upper()
        impacts: dict[str, float] = {}
        for ticker in self._known_tickers:
            normalized = ticker.upper()
            if f"${normalized}" in text or re.search(
                rf"(?<![A-Z]){re.escape(normalized)}(?![A-Z])", text
            ):
                impacts[normalized] = 0.7
        return impacts

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

                    # On génère le fingerprint en amont
                    fp = self._fingerprint(title, link)

                    # On évite d'appeler l'API OpenAI si on l'a déjà vu (évite la surfacturation !)
                    if self._seen_cache is not None and fp in self._seen_cache:
                        continue

                    # Parse published date
                    published = None
                    pub_str = entry.get("published", entry.get("updated", ""))
                    if pub_str:
                        try:
                            published = parsedate_to_datetime(pub_str)
                            if published.tzinfo is None:
                                published = published.replace(tzinfo=UTC)
                        except Exception:
                            published = None

                    # Extract tickers and sentiment via the local LLM service.
                    # The service originally returned {"tickers": [...]}; newer builds
                    # return {"impacts": [{"ticker": "AAPL", "score": 0.8}]}.
                    # Keep both formats working so a stale container cannot silently
                    # zero out all signals again.
                    found_impacts = await self._extract_tickers_via_llm(title, summary)
                    if not found_impacts:
                        found_impacts = self._extract_tickers_from_text(title, summary)
                    if self._known_tickers:
                        allowed_tickers = self._known_tickers | {
                            "SPY",
                            "QQQ",
                            "DIA",
                            "IWM",
                            "USO",
                            "GLD",
                            "SLV",
                            "VXX",
                            "TLT",
                            "XLE",
                            "XLF",
                            "XLK",
                            "XLV",
                            "XLY",
                            "XLI",
                            "XLP",
                            "XLU",
                            "SMH",
                        }
                        found_impacts = {
                            ticker: score
                            for ticker, score in found_impacts.items()
                            if ticker in allowed_tickers
                        }
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
            req = urllib.request.Request(
                url, data=data, method="POST", headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    res = json.loads(response.read().decode("utf-8"))

                    # Preferred schema: {"impacts": [{"ticker": "AAPL", "score": 0.8}]}
                    impacts = res.get("impacts")
                    if isinstance(impacts, list):
                        return {
                            item["ticker"].upper(): float(item["score"])
                            for item in impacts
                            if isinstance(item, dict) and "ticker" in item and "score" in item
                        }

                    # Backward-compatible schema from the running Docker image:
                    # {"tickers": ["AAPL"], "raw_response": "AAPL"}
                    tickers = res.get("tickers")
                    if isinstance(tickers, list):
                        return {
                            str(ticker).upper(): 0.7 for ticker in tickers if str(ticker).strip()
                        }

                    return {}
            except Exception as e:
                log.error("llm_extraction.failed", error=str(e), title=title[:50])
                return {}

        return await asyncio.to_thread(do_request)
