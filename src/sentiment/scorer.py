"""Sentiment scorer — aggregates per-ticker sentiment from multiple news."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

import structlog

from src.news.base import NewsItem
from src.sentiment.finbert import FinBERTAnalyzer, SentimentResult

log = structlog.get_logger()


@dataclass
class TickerSentiment:
    """Aggregated sentiment for a single ticker."""

    ticker: str
    avg_score: float  # -1.0 to +1.0
    news_count: int
    news_velocity: float  # news per hour (acceleration)
    latest_score: float
    scores: list[float] = field(default_factory=list)
    headlines: list[str] = field(default_factory=list)


class SentimentScorer:
    """Aggregates sentiment scores per ticker from analyzed news."""

    def __init__(self, analyzer: FinBERTAnalyzer, decay_minutes: int = 30):
        self._analyzer = analyzer
        self._decay_minutes = decay_minutes
        # History: ticker -> list of (timestamp, score, headline)
        self._history: dict[str, list[tuple[datetime, float, str]]] = defaultdict(list)

    async def score_news(self, news_items: list[NewsItem]) -> dict[str, TickerSentiment]:
        """Analyze news items and return per-ticker aggregated sentiment."""
        if not news_items:
            return {}

        # Extract texts and analyze with FinBERT
        texts = [item.text_for_analysis for item in news_items]
        results = await self._analyzer.analyze(texts)

        # Update news article sentiment in-place (for DB storage)
        for item, result in zip(news_items, results):
            item._sentiment = result  # attach for later use

        # Group by ticker
        ticker_scores: dict[str, list[tuple[datetime, float, str]]] = defaultdict(list)

        for item, result in zip(news_items, results):
            timestamp = item.published_at or item.fetched_at
            for ticker in item.tickers:
                ticker_scores[ticker].append((timestamp, result.score, item.title))

        # Update history
        now = datetime.now(timezone.utc)
        for ticker, entries in ticker_scores.items():
            self._history[ticker].extend(entries)
            # Prune old entries
            cutoff = now - timedelta(minutes=self._decay_minutes * 3)
            self._history[ticker] = [
                (t, s, h) for t, s, h in self._history[ticker] if t > cutoff
            ]

        # Build aggregated sentiment per ticker
        result_map: dict[str, TickerSentiment] = {}

        for ticker in set(list(ticker_scores.keys()) + list(self._history.keys())):
            history = self._history.get(ticker, [])
            if not history:
                continue

            # Time-weighted average (recent news counts more)
            scores = []
            weights = []
            for ts, score, _ in history:
                age_minutes = (now - ts).total_seconds() / 60
                # Exponential decay
                weight = 2.0 ** (-age_minutes / self._decay_minutes)
                scores.append(score)
                weights.append(weight)

            total_weight = sum(weights)
            if total_weight > 0:
                avg_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            else:
                avg_score = 0.0

            # News velocity: count of news in last hour
            one_hour_ago = now - timedelta(hours=1)
            recent_count = sum(1 for t, _, _ in history if t > one_hour_ago)

            result_map[ticker] = TickerSentiment(
                ticker=ticker,
                avg_score=avg_score,
                news_count=len(history),
                news_velocity=recent_count,  # per hour
                latest_score=history[-1][1] if history else 0.0,
                scores=scores,
                headlines=[h for _, _, h in history[-5:]],  # last 5
            )

        log.info(
            "sentiment.scored",
            tickers=len(result_map),
            total_news=len(news_items),
        )
        return result_map
