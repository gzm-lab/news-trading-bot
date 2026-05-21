"""Signal generation — combines sentiment + technicals into trade signals."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import structlog

from src.config import StrategySettings
from src.market.indicators import compute_momentum_score, compute_volume_score
from src.sentiment.scorer import TickerSentiment

log = structlog.get_logger()


@dataclass
class Signal:
    """Trade signal for a single ticker."""

    ticker: str
    score: float  # -1.0 to +1.0
    action: str  # "buy", "sell", "hold"
    # Components (for logging / debugging)
    sentiment_score: float
    news_velocity: float
    technical_score: float
    volume_score: float
    reason: str


class SignalGenerator:
    """Combines sentiment + market data into trading signals."""

    def __init__(self, config: StrategySettings):
        self._config = config

    def evaluate(
        self,
        sentiments: dict[str, TickerSentiment],
        market_data: dict[str, pd.DataFrame],
        current_positions: set[str],
    ) -> list[Signal]:
        """Evaluate all tickers and generate signals.

        Args:
            sentiments: per-ticker sentiment (from SentimentScorer)
            market_data: per-ticker OHLCV DataFrame (from broker.get_bars)
            current_positions: set of tickers we already hold
        """
        signals: list[Signal] = []
        cfg = self._config

        # Evaluate tickers that have sentiment data
        for ticker, sentiment in sentiments.items():
            # Technical component
            bars = market_data.get(ticker)
            has_market_data = bars is not None and not bars.empty
            tech_score = compute_momentum_score(bars) if has_market_data else 0.0
            vol_score = compute_volume_score(bars) if has_market_data else 0.0

            # Composite signal (weighted)
            composite = (
                cfg.w_sentiment * sentiment.avg_score
                + cfg.w_news_velocity * min(sentiment.news_velocity / 5.0, 1.0)
                + cfg.w_technical * tech_score
                + cfg.w_volume * vol_score
            )

            # Determine action. Be deliberately conservative: the bot previously
            # traded almost entirely on LLM sentiment because market data was missing.
            # Buys now need fresh news plus market confirmation; sells need a materially
            # negative latest score, leaving stop-loss/trailing exits to the risk manager.
            enough_news = sentiment.news_count >= 2 or sentiment.news_velocity >= 2
            market_confirmed = has_market_data and (tech_score >= 0.10 or vol_score >= 0.20)
            if (
                composite > cfg.buy_threshold
                and ticker not in current_positions
                and enough_news
                and market_confirmed
            ):
                action = "buy"
                reason = (
                    f"Signal {composite:.3f} > {cfg.buy_threshold} | "
                    f"Sent={sentiment.avg_score:.2f} Tech={tech_score:.2f} "
                    f"Vol={vol_score:.2f} News={sentiment.news_count}"
                )
            elif (
                composite < cfg.sell_threshold
                and ticker in current_positions
                and sentiment.latest_score <= cfg.sell_threshold
            ):
                action = "sell"
                reason = (
                    f"Signal {composite:.3f} < {cfg.sell_threshold} | "
                    f"Sent={sentiment.avg_score:.2f} Latest={sentiment.latest_score:.2f} "
                    f"Tech={tech_score:.2f}"
                )
            else:
                action = "hold"
                reason = f"Signal {composite:.3f} in hold zone"

            signal = Signal(
                ticker=ticker,
                score=composite,
                action=action,
                sentiment_score=sentiment.avg_score,
                news_velocity=sentiment.news_velocity,
                technical_score=tech_score,
                volume_score=vol_score,
                reason=reason,
            )
            signals.append(signal)

        # Sort by absolute signal strength
        signals.sort(key=lambda s: abs(s.score), reverse=True)

        buy_count = sum(1 for s in signals if s.action == "buy")
        sell_count = sum(1 for s in signals if s.action == "sell")
        log.info("signals.generated", total=len(signals), buys=buy_count, sells=sell_count)

        return signals
