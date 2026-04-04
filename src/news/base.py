"""Base news data structures."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class NewsItem:
    """A single news article/headline."""

    source: str
    title: str
    summary: str = ""
    url: str = ""
    tickers: list[str] = field(default_factory=list)
    published_at: datetime | None = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def fingerprint(self) -> str:
        """Unique content hash for deduplication."""
        content = f"{self.title}:{self.url}".lower().strip()
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def text_for_analysis(self) -> str:
        """Preferred text for sentiment analysis."""
        if self.summary:
            return f"{self.title}. {self.summary}"
        return self.title


class NewsSource(ABC):
    """Abstract base for news sources."""

    @abstractmethod
    async def fetch(self, tickers: list[str] | None = None) -> list[NewsItem]: ...
