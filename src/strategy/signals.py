"""Signal generation — combines sentiment + technicals into trade signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

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
    market_context: Any | None = None
    features: dict[str, Any] = field(default_factory=dict)
    reject_reason: str | None = None


class SignalGenerator:
    """Combines sentiment + market data into trading signals."""

    def __init__(self, config: StrategySettings):
        self._config = config

    def evaluate(
        self,
        sentiments: dict[str, TickerSentiment],
        market_data: dict[str, pd.DataFrame],
        current_positions: set[str],
        market_contexts: dict[str, Any] | None = None,
    ) -> list[Signal]:
        """Evaluate all tickers and generate signals.

        Args:
            sentiments: per-ticker sentiment (from SentimentScorer)
            market_data: per-ticker OHLCV DataFrame (from broker.get_bars)
            current_positions: set of tickers we already hold
            market_contexts: optional per-ticker MarketContext with anti-chase features
        """
        signals: list[Signal] = []
        cfg = self._config
        market_contexts = market_contexts or {}

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
            market_context = market_contexts.get(ticker)
            features = _context_features(market_context)
            reject_reason = _market_context_reject_reason(sentiment, features)
            context_acceptable = reject_reason is None
            if (
                composite > cfg.buy_threshold
                and ticker not in current_positions
                and enough_news
                and market_confirmed
                and context_acceptable
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
                if reject_reason is not None:
                    reason = f"{reason} | Rejected: {reject_reason}"

            signal = Signal(
                ticker=ticker,
                score=composite,
                action=action,
                sentiment_score=sentiment.avg_score,
                news_velocity=sentiment.news_velocity,
                technical_score=tech_score,
                volume_score=vol_score,
                reason=reason,
                market_context=market_context,
                features=features,
                reject_reason=reject_reason,
            )
            signals.append(signal)

        # Sort by absolute signal strength
        signals.sort(key=lambda s: abs(s.score), reverse=True)

        buy_count = sum(1 for s in signals if s.action == "buy")
        sell_count = sum(1 for s in signals if s.action == "sell")
        log.info("signals.generated", total=len(signals), buys=buy_count, sells=sell_count)

        return signals


def _context_features(market_context: Any | None) -> dict[str, Any]:
    """Return a plain feature mapping from a MarketContext-like object."""
    if market_context is None:
        return {}

    raw_features = getattr(market_context, "features", None)
    if isinstance(raw_features, dict):
        return dict(raw_features)

    if is_dataclass(market_context):
        return {k: v for k, v in asdict(market_context).items() if k != "ticker"}

    if isinstance(market_context, dict):
        return {k: v for k, v in market_context.items() if k != "ticker"}

    return {
        name: getattr(market_context, name)
        for name in (
            "last_price",
            "prev_close",
            "today_open",
            "day_high",
            "day_low",
            "session_vwap",
            "atr_14",
            "gap_pct",
            "gap_atr",
            "day_range_pos",
            "price_vs_vwap_pct",
            "return_5m",
            "return_15m",
            "return_60m",
        )
        if hasattr(market_context, name)
    }


def _market_context_reject_reason(
    sentiment: TickerSentiment, features: dict[str, Any]
) -> str | None:
    """Return anti-chase rejection reason for positive buy candidates, if any."""
    if sentiment.avg_score <= 0:
        return None

    gap_pct = _as_float(features.get("gap_pct"))
    day_range_pos = _as_float(features.get("day_range_pos"))
    price_vs_vwap_pct = _as_float(features.get("price_vs_vwap_pct"))

    if gap_pct is not None and gap_pct > 0.05:
        return "gap_pct_above_5pct"
    if (
        gap_pct is not None
        and day_range_pos is not None
        and gap_pct > 0.03
        and day_range_pos > 0.80
    ):
        return "large_gap_near_day_high"
    if price_vs_vwap_pct is not None and price_vs_vwap_pct > 0.01:
        return "price_above_vwap_chase"
    if price_vs_vwap_pct is not None and price_vs_vwap_pct < 0:
        return "positive_news_below_vwap"

    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
