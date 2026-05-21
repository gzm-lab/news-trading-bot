"""Tests for signal generator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.sentiment.scorer import TickerSentiment
from src.strategy.signals import SignalGenerator


def _make_sentiment(ticker, avg_score=0.5, news_count=3, news_velocity=2.0, latest_score=0.6):
    return TickerSentiment(
        ticker=ticker,
        avg_score=avg_score,
        news_count=news_count,
        news_velocity=news_velocity,
        latest_score=latest_score,
    )


def _make_ohlcv(n=50):
    np.random.seed(42)
    close = 180.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.2,
            "high": close + np.abs(np.random.randn(n) * 0.3),
            "low": close - np.abs(np.random.randn(n) * 0.3),
            "close": close,
            "volume": np.random.randint(100_000, 1_000_000, size=n).astype(float),
        }
    )


def _make_bullish_ohlcv(n=50):
    close = np.linspace(100.0, 130.0, n)
    volume = np.concatenate([np.full(n - 1, 100_000.0), np.array([500_000.0])])
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": volume,
        }
    )


class TestSignalGenerator:
    def test_no_sentiments_no_signals(self, strategy_config):
        gen = SignalGenerator(config=strategy_config)
        signals = gen.evaluate({}, {}, set())
        assert signals == []

    def test_strong_positive_generates_buy(self, strategy_config):
        gen = SignalGenerator(config=strategy_config)
        sentiments = {
            "AAPL": _make_sentiment("AAPL", avg_score=0.9, news_count=3, news_velocity=5.0),
        }
        market_data = {"AAPL": _make_bullish_ohlcv()}

        signals = gen.evaluate(sentiments, market_data, current_positions=set())

        assert len(signals) == 1
        assert signals[0].action == "buy"
        assert signals[0].ticker == "AAPL"
        assert signals[0].score > strategy_config.buy_threshold

    def test_strong_negative_generates_sell_if_holding(self, strategy_config):
        gen = SignalGenerator(config=strategy_config)
        sentiments = {
            "TSLA": _make_sentiment(
                "TSLA", avg_score=-0.9, news_velocity=0.0, latest_score=-0.9
            ),
        }
        market_data = {"TSLA": _make_ohlcv()}

        signals = gen.evaluate(sentiments, market_data, current_positions={"TSLA"})

        sell_signals = [s for s in signals if s.action == "sell"]
        assert len(sell_signals) == 1
        assert sell_signals[0].ticker == "TSLA"

    def test_no_buy_if_already_holding(self, strategy_config):
        gen = SignalGenerator(config=strategy_config)
        sentiments = {
            "AAPL": _make_sentiment("AAPL", avg_score=0.9, news_velocity=5.0),
        }
        market_data = {"AAPL": _make_ohlcv()}

        signals = gen.evaluate(sentiments, market_data, current_positions={"AAPL"})

        # Should be hold, not buy (already in position)
        buy_signals = [s for s in signals if s.action == "buy"]
        assert len(buy_signals) == 0

    def test_no_sell_if_not_holding(self, strategy_config):
        gen = SignalGenerator(config=strategy_config)
        sentiments = {
            "TSLA": _make_sentiment("TSLA", avg_score=-0.8, news_velocity=3.0),
        }

        signals = gen.evaluate(sentiments, {}, current_positions=set())

        sell_signals = [s for s in signals if s.action == "sell"]
        assert len(sell_signals) == 0

    def test_neutral_sentiment_holds(self, strategy_config):
        gen = SignalGenerator(config=strategy_config)
        sentiments = {
            "MSFT": _make_sentiment("MSFT", avg_score=0.1, news_velocity=1.0),
        }

        signals = gen.evaluate(sentiments, {}, current_positions=set())
        assert all(s.action == "hold" for s in signals)

    def test_signals_sorted_by_strength(self, strategy_config):
        gen = SignalGenerator(config=strategy_config)
        sentiments = {
            "AAPL": _make_sentiment("AAPL", avg_score=0.9, news_velocity=5.0),
            "MSFT": _make_sentiment("MSFT", avg_score=0.5, news_velocity=2.0),
            "NVDA": _make_sentiment("NVDA", avg_score=0.7, news_velocity=3.0),
        }
        market_data = {t: _make_ohlcv() for t in sentiments}

        signals = gen.evaluate(sentiments, market_data, current_positions=set())
        scores = [abs(s.score) for s in signals]
        assert scores == sorted(scores, reverse=True)


    def test_strong_positive_without_market_data_holds(self, strategy_config):
        gen = SignalGenerator(config=strategy_config)
        sentiments = {
            "AAPL": _make_sentiment("AAPL", avg_score=1.0, news_count=3, news_velocity=3.0),
        }

        signals = gen.evaluate(sentiments, {}, current_positions=set())

        assert signals[0].action == "hold"
        assert signals[0].technical_score == 0.0
        assert signals[0].volume_score == 0.0

    def test_single_article_does_not_buy_even_with_market_confirmation(self, strategy_config):
        gen = SignalGenerator(config=strategy_config)
        sentiments = {
            "AAPL": _make_sentiment("AAPL", avg_score=1.0, news_count=1, news_velocity=1.0),
        }
        market_data = {"AAPL": _make_bullish_ohlcv()}

        signals = gen.evaluate(sentiments, market_data, current_positions=set())

        assert signals[0].action == "hold"

    def test_positive_sentiment_with_market_confirmation_buys(self, strategy_config):
        gen = SignalGenerator(config=strategy_config)
        sentiments = {
            "AAPL": _make_sentiment("AAPL", avg_score=1.0, news_count=3, news_velocity=3.0),
        }
        market_data = {"AAPL": _make_bullish_ohlcv()}

        signals = gen.evaluate(sentiments, market_data, current_positions=set())

        assert signals[0].technical_score >= 0.10 or signals[0].volume_score >= 0.20
        assert signals[0].action == "buy"

    def test_mild_negative_sentiment_does_not_sell(self, strategy_config):
        gen = SignalGenerator(config=strategy_config)
        sentiments = {
            "AAPL": _make_sentiment("AAPL", avg_score=-0.4, news_count=3, news_velocity=1.0, latest_score=-0.2),
        }
        market_data = {"AAPL": _make_ohlcv()}

        signals = gen.evaluate(sentiments, market_data, current_positions={"AAPL"})

        assert signals[0].action == "hold"

    def test_signal_components_stored(self, strategy_config):
        gen = SignalGenerator(config=strategy_config)
        sentiments = {
            "AAPL": _make_sentiment("AAPL", avg_score=0.5, news_velocity=2.0),
        }
        market_data = {"AAPL": _make_ohlcv()}

        signals = gen.evaluate(sentiments, market_data, current_positions=set())
        sig = signals[0]

        assert sig.sentiment_score == 0.5
        assert sig.news_velocity == 2.0
        assert isinstance(sig.technical_score, float)
        assert isinstance(sig.volume_score, float)
        assert isinstance(sig.reason, str)

    def test_no_market_data_still_works(self, strategy_config):
        gen = SignalGenerator(config=strategy_config)
        sentiments = {
            "AAPL": _make_sentiment("AAPL", avg_score=0.9, news_velocity=5.0),
        }

        signals = gen.evaluate(sentiments, {}, current_positions=set())
        # Technical and volume scores should be 0 without market data
        assert signals[0].technical_score == 0.0
        assert signals[0].volume_score == 0.0
