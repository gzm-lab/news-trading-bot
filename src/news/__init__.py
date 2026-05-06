"""News package."""

from src.news.aggregator import NewsAggregator
from src.news.base import NewsItem, NewsSource
from src.news.finnhub_source import FinnhubSource
from src.news.rss_source import RSSSource

__all__ = ["NewsItem", "NewsSource", "FinnhubSource", "RSSSource", "NewsAggregator"]
