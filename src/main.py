"""Main orchestrator — the trading bot loop."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import structlog
from dotenv import load_dotenv

# NYSE timezone and schedule
_ET = ZoneInfo("America/New_York")
_NYSE_OPEN  = (9, 30)   # 09:30 ET — real market open
_NYSE_CLOSE = (16, 0)   # 16:00 ET
_PRE_MARKET_START = (4, 0)   # 04:00 ET — start news/signal collection
_PRE_MARKET_STOP  = (9, 30)  # 09:30 ET — hand off to live trading


def _market_phase(now_et: datetime | None = None) -> str:
    """Return the current market phase:
      'closed'     — weekend / off-hours (sleep until 04:00 ET next weekday)
      'premarket'  — 04:00–09:30 ET weekdays (news fetch + signal warmup, no orders)
      'open'       — 09:30–16:00 ET weekdays (full trading)
    """
    if now_et is None:
        now_et = datetime.now(_ET)

    from datetime import time as dt_time
    wd = now_et.weekday()   # Mon=0 … Sun=6
    t  = now_et.time()

    pre_t  = dt_time(*_PRE_MARKET_START)
    open_t = dt_time(*_NYSE_OPEN)
    close_t = dt_time(*_NYSE_CLOSE)

    if wd >= 5:
        return "closed"
    if t < pre_t or t >= close_t:
        return "closed"
    if pre_t <= t < open_t:
        return "premarket"
    return "open"


def _seconds_until_active(now_et: datetime | None = None) -> int:
    """Seconds until the next active window starts (pre-market 04:00 ET).
    Returns 0 if already in premarket or open phase."""
    if now_et is None:
        now_et = datetime.now(_ET)

    phase = _market_phase(now_et)
    if phase in ("premarket", "open"):
        return 0

    from datetime import time as dt_time
    wd = now_et.weekday()
    t  = now_et.time()
    pre_t = dt_time(*_PRE_MARKET_START)

    # Same weekday but before 04:00 (unlikely but possible)
    if wd < 5 and t < pre_t:
        next_start = now_et.replace(
            hour=_PRE_MARKET_START[0], minute=_PRE_MARKET_START[1],
            second=0, microsecond=0,
        )
    else:
        # Days until next Monday (wraps correctly for Fri/Sat/Sun)
        days_ahead = {4: 3, 5: 2, 6: 1}.get(wd, 1)  # Fri→Mon, Sat→Mon, Sun→Mon, else +1
        next_start = (now_et + timedelta(days=days_ahead)).replace(
            hour=_PRE_MARKET_START[0], minute=_PRE_MARKET_START[1],
            second=0, microsecond=0,
        )

    return max(1, int((next_start - now_et).total_seconds()))


# Keep old name as alias for tests
def _seconds_until_market(now_et: datetime | None = None) -> int:
    """Alias — returns 0 if market is open OR in pre-market warmup."""
    return _seconds_until_active(now_et)

from src.config import Settings
from src.broker.alpaca_broker import AlpacaBroker
from src.news.finnhub_source import FinnhubSource
from src.news.rss_source import RSSSource
from src.news.aggregator import NewsAggregator
from src.sentiment.finbert import FinBERTAnalyzer
from src.sentiment.scorer import SentimentScorer
from src.strategy.signals import SignalGenerator
from src.strategy.risk_manager import RiskManager
from src.alerts.discord import DiscordAlerter
from src.storage.database import Database
from src.storage.models import TradeLog, CycleLog, PortfolioSnapshot

log = structlog.get_logger()


class TradingBot:
    """News-driven trading bot — main orchestrator."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings()
        self._broker: AlpacaBroker | None = None
        self._aggregator: NewsAggregator | None = None
        self._scorer: SentimentScorer | None = None
        self._signal_gen: SignalGenerator | None = None
        self._risk_mgr: RiskManager | None = None
        self._alerter: DiscordAlerter | None = None
        self._db: Database | None = None
        self._running = False

    async def setup(self) -> None:
        """Initialize all components."""
        cfg = self._settings
        log.info("bot.setup.start")

        # Database
        self._db = Database(cfg.db_path)
        self._db.init()

        # Broker
        self._broker = AlpacaBroker(
            api_key=cfg.broker.api_key,
            secret_key=cfg.broker.secret_key,
            paper="paper" in cfg.broker.base_url,
        )
        await self._broker.connect()

        # News sources
        sources = []
        if cfg.news.finnhub_api_key:
            sources.append(
                FinnhubSource(
                    api_key=cfg.news.finnhub_api_key,
                    max_age_minutes=cfg.news.max_age_minutes,
                )
            )
        sources.append(
            RSSSource(
                feed_urls=cfg.news.rss_feeds,
                known_tickers=cfg.universe,
            )
        )
        self._aggregator = NewsAggregator(sources=sources, db=self._db)

        # Sentiment
        analyzer = FinBERTAnalyzer(
            model_name=cfg.sentiment.model_name,
            batch_size=cfg.sentiment.batch_size,
            max_length=cfg.sentiment.max_length,
        )
        await analyzer.load()
        self._scorer = SentimentScorer(analyzer=analyzer)

        # Strategy
        self._signal_gen = SignalGenerator(config=cfg.strategy)
        self._risk_mgr = RiskManager(config=cfg.strategy)

        # Alerts
        self._alerter = DiscordAlerter(
            webhook_url=cfg.alerts.webhook_url,
            enabled=cfg.alerts.enabled,
        )

        # Startup notification
        account = await self._broker.get_account()
        self._risk_mgr.set_daily_baseline(account.equity)
        await self._alerter.notify_startup(account)

        log.info("bot.setup.done", equity=account.equity, universe=len(cfg.universe))

    async def run(self) -> None:
        """Main loop — runs until stopped."""
        self._running = True
        self._daily_summary_sent = False   # reset each trading day
        log.info("bot.run.start", interval=self._settings.cycle_interval)

        while self._running:
            try:
                # Calculate qty for buys
                if order.side.value == "buy" and hasattr(order, "_max_value"):
                    price = await self._broker.get_latest_price(order.ticker)
                    if price > 0:
                        order.qty = int(order._max_value / price)
                        if order.order_type.value == "limit":
                            order.limit_price = round(price * 1.002, 2)  # Slippage protection +0.2%
                    if order.qty <= 0:
                        continue
                elif order.side.value == "sell":
                    pos = next((p for p in positions if p.ticker == order.ticker), None)
                    if pos:
                        order.qty = pos.qty
                    if order.qty <= 0:
                        continue
                    price = await self._broker.get_latest_price(order.ticker)
                    if price > 0:
                        if order.order_type.value == "limit":
                            order.limit_price = round(price * 0.998, 2)  # Slippage protection -0.2%

                result = await self._broker.place_order(order)
                signal = getattr(order, "_signal", None)
                await self._alerter.notify_trade(
                    result,
                    reason=signal.reason if signal else "",
                )
                self._log_trade(
                    result,
                    signal_score=signal.score if signal else 0.0,
                    reason=signal.reason if signal else "",
                )
                filled_count += 1

            except Exception as e:
                log.error("bot.order_failed", ticker=order.ticker, error=str(e))

        # 9. Log cycle
        duration_ms = int((time.monotonic() - t0) * 1000)
        self._log_cycle(
            len(news_items), len(signals), filled_count,
            account.portfolio_value, account.daily_pnl, t0,
        )

        log.info(
            "bot.cycle.done",
            news=len(news_items),
            signals=len(signals),
            orders=filled_count,
            duration_ms=duration_ms,
        )

    def _log_trade(self, order, signal_score: float, reason: str) -> None:
        if self._db:
