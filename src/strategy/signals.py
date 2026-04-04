"""Signal generation — combines sentiment + technicals into trade signals."""

from __future__ import annotations

from dataclasses import dataclass

import structlog
import pandas as pd

from src.config import StrategySettings
from src.sentiment.scorer import TickerSentiment
from src.market.indicators import compute_momentum_score, compute_volume_score

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
            tech_score = compute_momentum_score(bars) if bars is not None else 0.0
            vol_score = compute_volume_score(bars) if bars is not None else 0.0

            # Composite signal (weighted)
            composite = (
                cfg.w_sentiment * sentiment.avg_score
                + cfg.w_news_velocity * min(sentiment.news_velocity / 5.0, 1.0)
                + cfg.w_technical * tech_score
                + cfg.w_volume * vol_score
            )

            # Determine action
            if composite > cfg.buy_threshold and ticker not in current_positions:
                action = "buy"
                reason = (
                    f"Signal {composite:.3f} > {cfg.buy_threshold} | "
                    f"Sent={sentiment.avg_score:.2f} Tech={tech_score:.2f} "
                    f"Vol={vol_score:.2f} News={sentiment.news_count}"
                )
            elif composite < cfg.sell_threshold and ticker in current_positions:
                action = "sell"
                reason = (
                    f"Signal {composite:.3f} < {cfg.sell_threshold} | "
                    f"Sent={sentiment.avg_score:.2f} Tech={tech_score:.2f}"
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
