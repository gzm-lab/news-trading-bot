"""Main orchestrator — the trading bot loop."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import structlog
from dotenv import load_dotenv

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
        log.info("bot.run.start", interval=self._settings.cycle_interval)

        while self._running:
            try:
                # Check market hours
                if not await self._broker.is_market_open():
                    log.info("bot.market_closed")
                    await asyncio.sleep(60)
                    continue

                await self._cycle()

            except KeyboardInterrupt:
                log.info("bot.interrupted")
                break
            except Exception as e:
                log.error("bot.cycle_error", error=str(e), exc_info=True)

            await asyncio.sleep(self._settings.cycle_interval)

        log.info("bot.run.stopped")

    async def _cycle(self) -> None:
        """One iteration of the trading loop."""
        t0 = time.monotonic()
        cfg = self._settings

        # 1. Get account & positions
        account = await self._broker.get_account()
        positions = await self._broker.get_positions()
        current_tickers = {p.ticker for p in positions}

        # Update risk state
        self._risk_mgr.update_daily_pnl(account)
        if self._risk_mgr.state.trading_halted:
            await self._alerter.notify_halt(self._risk_mgr.state.halt_reason)
            return

        # 2. Check stop-loss / take-profit on existing positions
        exits = self._risk_mgr.check_exits(positions)
        for ticker in exits:
            pos = next((p for p in positions if p.ticker == ticker), None)
            result = await self._broker.close_position(ticker)
            if result and pos:
                exit_type = "stop_loss" if pos.unrealized_pnl < 0 else "take_profit"
                await self._alerter.notify_exit(
                    ticker, exit_type,
                    pnl=pos.unrealized_pnl,
                    pnl_pct=pos.unrealized_pnl_pct,
                )
                self._log_trade(result, signal_score=0.0, reason=f"Auto-exit: {exit_type}")

        # 3. Fetch latest news
        news_items = await self._aggregator.fetch_latest(tickers=cfg.universe)

        if not news_items:
            log.debug("bot.no_new_news")
            self._log_cycle(0, 0, 0, account.portfolio_value, account.daily_pnl, t0)
            return

        # 4. Score sentiment
        sentiments = await self._scorer.score_news(news_items)

        # 5. Get market data for tickers with sentiment
        market_data = {}
        for ticker in sentiments:
            try:
                bars = await self._broker.get_bars(ticker, timeframe="1Hour", limit=50)
                if not bars.empty:
                    market_data[ticker] = bars
            except Exception as e:
                log.debug("bot.bars_failed", ticker=ticker, error=str(e))

        # 6. Generate signals
        signals = self._signal_gen.evaluate(sentiments, market_data, current_tickers)

        # 7. Risk filter → orders
        orders = self._risk_mgr.filter_signals(signals, account, positions)

        # 8. Execute orders
        filled_count = 0
        for order in orders:
            try:
                # Calculate qty for buys
                if order.side.value == "buy" and hasattr(order, "_max_value"):
                    price = await self._broker.get_latest_price(order.ticker)
                    if price > 0:
                        order.qty = int(order._max_value / price)
                    if order.qty <= 0:
                        continue

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
            try:
                self._db.save(TradeLog(
                    order_id=order.id,
                    ticker=order.ticker,
                    side=order.side.value if hasattr(order.side, "value") else str(order.side),
                    qty=order.qty,
                    order_type=order.order_type.value if hasattr(order.order_type, "value") else str(order.order_type),
                    filled_price=order.filled_price,
                    status=order.status.value if hasattr(order.status, "value") else str(order.status),
                    signal_score=signal_score,
                    reason=reason,
                ))
            except Exception as e:
                log.warning("bot.log_trade_failed", error=str(e))

    def _log_cycle(
        self, news_count, signals, orders, portfolio_value, daily_pnl, t0
    ) -> None:
        if self._db:
            try:
                duration_ms = int((time.monotonic() - t0) * 1000)
                self._db.save(CycleLog(
                    news_count=news_count,
                    signals_generated=signals,
                    orders_placed=orders,
                    portfolio_value=portfolio_value,
                    daily_pnl=daily_pnl,
                    cycle_duration_ms=duration_ms,
                ))
            except Exception as e:
                log.warning("bot.log_cycle_failed", error=str(e))

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        log.info("bot.stopping")

        if self._broker:
            account = await self._broker.get_account()
            positions = await self._broker.get_positions()
            await self._alerter.notify_daily_summary(account, positions)

        log.info("bot.stopped")


def main():
    """Entry point."""
    load_dotenv()

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
    )

    bot = TradingBot()

    async def _run():
        await bot.setup()
        try:
            await bot.run()
        except KeyboardInterrupt:
            pass
        finally:
            await bot.stop()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
