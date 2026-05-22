"""Tests for TradingBot risk exits and phase gating."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.broker.interface import Account, Order, OrderSide, OrderStatus, OrderType, Position
from src.main import TradingBot
from src.market.context import MarketContext
from src.storage.models import CycleLog, PortfolioSnapshot, SignalLog, TradeLog
from src.strategy.signals import Signal


def _position(ticker="AAPL", pnl_pct=-0.03):
    return Position(
        ticker=ticker,
        qty=10,
        avg_entry_price=100.0,
        current_price=100.0 * (1 + pnl_pct),
        market_value=1000.0 * (1 + pnl_pct),
        unrealized_pnl=1000.0 * pnl_pct,
        unrealized_pnl_pct=pnl_pct,
    )


def _account():
    return Account(
        equity=100_000.0, cash=50_000.0, buying_power=100_000.0, portfolio_value=100_000.0
    )


def _signal(ticker="MSFT", action="buy"):
    return Signal(
        ticker=ticker,
        score=0.8,
        action=action,
        sentiment_score=0.8,
        news_velocity=1.0,
        technical_score=0.0,
        volume_score=0.0,
        reason="test signal",
    )


def _market_context(ticker="MSFT"):
    return MarketContext(
        ticker=ticker,
        last_price=420.0,
        prev_close=400.0,
        today_open=410.0,
        day_high=425.0,
        day_low=405.0,
        session_vwap=415.0,
        atr_14=8.0,
        gap_pct=0.025,
        gap_atr=1.25,
        day_range_pos=0.75,
        price_vs_vwap_pct=0.012,
        return_5m=0.003,
        return_15m=0.007,
        return_60m=0.018,
    )


def _intraday_bars(start_price=410.0) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-05-21 13:30:00", tz="UTC")
    for i in range(61):
        close = start_price + i
        rows.append(
            {
                "timestamp": start + pd.Timedelta(minutes=i),
                "open": start_price if i == 0 else start_price + i - 1,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000.0,
            }
        )
    return pd.DataFrame(rows)


def _daily_bars(prev_close=400.0) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-05-01", tz="UTC")
    for i in range(29):
        close = prev_close - 30 + i
        rows.append(
            {
                "timestamp": start + pd.Timedelta(days=i),
                "open": close - 1.0,
                "high": close + 2.0,
                "low": close - 2.0,
                "close": close,
                "volume": 1_000_000.0,
            }
        )
    rows.append(
        {
            "timestamp": start + pd.Timedelta(days=29),
            "open": prev_close - 1.0,
            "high": prev_close + 2.0,
            "low": prev_close - 2.0,
            "close": prev_close,
            "volume": 1_000_000.0,
        }
    )
    return pd.DataFrame(rows)


@pytest.fixture
def bot(strategy_config):
    bot = TradingBot.__new__(TradingBot)
    bot._settings = MagicMock(cycle_interval=1)
    bot._aggregator = MagicMock()
    bot._aggregator.fetch_latest = AsyncMock(return_value=[MagicMock()])
    bot._scorer = MagicMock()
    bot._scorer.score_news = AsyncMock(return_value={"MSFT": MagicMock()})
    bot._signal_gen = MagicMock()
    bot._risk_mgr = MagicMock()
    bot._broker = MagicMock()
    bot._alerter = MagicMock()
    bot._alerter.notify_trade = AsyncMock()
    bot._db = None
    return bot


@pytest.mark.asyncio
async def test_run_cycle_passes_computed_market_contexts_to_signal_generator(bot):
    intraday = _intraday_bars(start_price=410.0)
    daily = _daily_bars(prev_close=400.0)
    fifteen_minute = pd.DataFrame(
        {"open": [410.0], "high": [421.0], "low": [409.0], "close": [420.0], "volume": [1000.0]}
    )

    async def get_bars(ticker, timeframe, limit):
        assert ticker == "MSFT"
        if timeframe == "15Min":
            assert limit == 50
            return fifteen_minute
        if timeframe == "1Min":
            assert limit == 390
            return intraday
        if timeframe == "1Day":
            assert limit == 30
            return daily
        raise AssertionError(f"unexpected timeframe {timeframe}")

    bot._broker.get_positions = AsyncMock(return_value=[])
    bot._broker.get_account = AsyncMock()
    bot._broker.get_bars = AsyncMock(side_effect=get_bars)
    bot._broker.close_position = AsyncMock()
    bot._broker.place_order = AsyncMock()
    bot._signal_gen.evaluate.return_value = []

    metrics = await bot._run_cycle(phase="premarket")

    assert metrics == {"news": 1, "signals": 0, "orders": 0, "exits": 0}
    bot._signal_gen.evaluate.assert_called_once()
    _, kwargs = bot._signal_gen.evaluate.call_args
    market_contexts = kwargs["market_contexts"]
    assert set(market_contexts) == {"MSFT"}
    context = market_contexts["MSFT"]
    assert isinstance(context, MarketContext)
    assert context.ticker == "MSFT"
    assert context.last_price == 470.0
    assert context.prev_close == 400.0
    assert context.today_open == 410.0
    assert context.day_high == 471.0
    assert context.day_low == 409.0
    bot._broker.get_bars.assert_any_await("MSFT", timeframe="15Min", limit=50)
    bot._broker.get_bars.assert_any_await("MSFT", timeframe="1Min", limit=390)
    bot._broker.get_bars.assert_any_await("MSFT", timeframe="1Day", limit=30)


@pytest.mark.asyncio
async def test_run_cycle_context_failure_for_one_ticker_does_not_break_cycle(bot):
    bot._scorer.score_news = AsyncMock(return_value={"MSFT": MagicMock(), "AAPL": MagicMock()})
    fifteen_minute = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1000.0]}
    )
    intraday = _intraday_bars(start_price=100.0)
    daily = _daily_bars(prev_close=95.0)

    async def get_bars(ticker, timeframe, limit):
        if timeframe == "15Min":
            return fifteen_minute
        if ticker == "AAPL" and timeframe == "1Min":
            raise RuntimeError("context data unavailable")
        if timeframe == "1Min":
            return intraday
        if timeframe == "1Day":
            return daily
        raise AssertionError(f"unexpected timeframe {timeframe}")

    bot._broker.get_positions = AsyncMock(return_value=[])
    bot._broker.get_account = AsyncMock()
    bot._broker.get_bars = AsyncMock(side_effect=get_bars)
    bot._broker.close_position = AsyncMock()
    bot._broker.place_order = AsyncMock()
    bot._signal_gen.evaluate.return_value = []

    metrics = await bot._run_cycle(phase="premarket")

    assert metrics == {"news": 1, "signals": 0, "orders": 0, "exits": 0}
    bot._signal_gen.evaluate.assert_called_once()
    _, kwargs = bot._signal_gen.evaluate.call_args
    assert set(kwargs["market_contexts"]) == {"MSFT"}


@pytest.mark.asyncio
async def test_run_cycle_closes_risk_exits_before_orders(bot):
    close_result = Order(
        ticker="AAPL",
        side=OrderSide.SELL,
        qty=10,
        order_type=OrderType.MARKET,
        id="close-1",
        status=OrderStatus.PENDING,
    )
    bot._broker.get_positions = AsyncMock(side_effect=[[_position("AAPL", -0.03)], []])
    bot._broker.get_account = AsyncMock(return_value=_account())
    bot._broker.close_position = AsyncMock(return_value=close_result)
    bot._broker.place_order = AsyncMock()
    bot._risk_mgr.ensure_daily_baseline = MagicMock()
    bot._risk_mgr.update_daily_pnl = MagicMock()
    bot._risk_mgr.check_exits = MagicMock(return_value=["AAPL"])
    bot._risk_mgr.filter_signals = MagicMock(return_value=[])
    bot._signal_gen.evaluate.return_value = []

    metrics = await bot._run_cycle(phase="open")

    assert metrics["exits"] == 1
    assert metrics["orders"] == 0
    bot._broker.close_position.assert_awaited_once_with("AAPL")
    bot._broker.place_order.assert_not_called()
    bot._alerter.notify_trade.assert_awaited_once_with(close_result, reason="risk exit")


@pytest.mark.asyncio
async def test_run_cycle_persists_cycle_and_trade_logs(bot, tmp_db):
    placed_order = Order(
        ticker="MSFT",
        side=OrderSide.BUY,
        qty=10,
        order_type=OrderType.LIMIT,
        limit_price=420.0,
        id="order-1",
        status=OrderStatus.PENDING_NEW,
    )
    bot._db = tmp_db
    bot._broker.get_positions = AsyncMock(return_value=[])
    bot._broker.get_account = AsyncMock(return_value=_account())
    bot._broker.close_position = AsyncMock()
    bot._broker.get_latest_price = AsyncMock(return_value=420.0)
    bot._broker.get_bars = AsyncMock(return_value=MagicMock(empty=True))
    bot._broker.place_order = AsyncMock(return_value=placed_order)
    bot._risk_mgr.ensure_daily_baseline = MagicMock()
    bot._risk_mgr.update_daily_pnl = MagicMock()
    bot._risk_mgr.check_exits = MagicMock(return_value=[])
    order = Order(ticker="MSFT", side=OrderSide.BUY, qty=10, order_type=OrderType.LIMIT)
    buy_signal = _signal("MSFT", "buy")
    buy_signal.features = {"custom_feature": 123}
    buy_signal.market_context = _market_context("MSFT")
    order._signal = buy_signal
    bot._risk_mgr.filter_signals = MagicMock(return_value=[order])
    bot._signal_gen.evaluate.return_value = [buy_signal]

    metrics = await bot._run_cycle(phase="open")

    assert metrics["orders"] == 1
    with tmp_db.get_session() as session:
        cycle = session.query(CycleLog).one()
        trade = session.query(TradeLog).one()
        signal_log = session.query(SignalLog).one()
        snapshot = session.query(PortfolioSnapshot).one()
    assert cycle.news_count == 1
    assert cycle.signals_generated == 1
    assert cycle.orders_placed == 1
    assert cycle.portfolio_value == 100_000.0
    assert snapshot.equity == 100_000.0
    assert snapshot.cash == 50_000.0
    assert snapshot.positions_count == 0
    assert snapshot.daily_pnl == 0.0
    assert snapshot.total_pnl == 0.0
    assert trade.order_id == "order-1"
    assert trade.ticker == "MSFT"
    assert trade.side == "buy"
    assert trade.qty == 10
    assert trade.status == "pending_new"
    assert trade.reason == "test signal"
    assert signal_log.ticker == "MSFT"
    assert signal_log.action == "buy"
    assert signal_log.score == 0.8
    assert signal_log.reject_reason is None
    features = json.loads(signal_log.features_json)
    assert features["custom_feature"] == 123
    assert features["market_context"]["last_price"] == 420.0
    assert features["market_context"]["price_vs_vwap_pct"] == 0.012
    bot._broker.get_bars.assert_any_await("MSFT", timeframe="15Min", limit=50)


@pytest.mark.asyncio
async def test_run_cycle_persists_hold_and_rejected_signal_logs_before_premarket_return(
    bot, tmp_db
):
    hold = _signal("MSFT", "hold")
    hold.score = 0.2
    hold.reason = "Signal 0.200 in hold zone"
    hold.features = {"news_count": 1}
    rejected = _signal("AAPL", "hold")
    rejected.score = 0.61
    rejected.reason = "Signal 0.610 in hold zone | Rejected: price_above_vwap_chase"
    rejected.reject_reason = "price_above_vwap_chase"
    rejected.features = {"gap_pct": 0.02}
    rejected.market_context = {"ticker": "AAPL", "last_price": 191.5, "day_range_pos": 0.9}
    bot._db = tmp_db
    bot._broker.get_positions = AsyncMock(return_value=[])
    bot._broker.get_account = AsyncMock()
    bot._broker.close_position = AsyncMock()
    bot._broker.place_order = AsyncMock()
    bot._broker.get_bars = AsyncMock(return_value=MagicMock(empty=True))
    bot._signal_gen.evaluate.return_value = [hold, rejected]

    metrics = await bot._run_cycle(phase="premarket")

    assert metrics == {"news": 1, "signals": 2, "orders": 0, "exits": 0}
    with tmp_db.get_session() as session:
        logs = {row.ticker: row for row in session.query(SignalLog).all()}
    assert set(logs) == {"MSFT", "AAPL"}
    assert logs["MSFT"].action == "hold"
    assert logs["MSFT"].reject_reason is None
    assert json.loads(logs["MSFT"].features_json) == {"news_count": 1}
    assert logs["AAPL"].action == "hold"
    assert logs["AAPL"].reject_reason == "price_above_vwap_chase"
    rejected_features = json.loads(logs["AAPL"].features_json)
    assert rejected_features["gap_pct"] == 0.02
    assert rejected_features["market_context"]["last_price"] == 191.5
    assert rejected_features["market_context"]["day_range_pos"] == 0.9
    bot._broker.get_account.assert_not_called()
    bot._broker.place_order.assert_not_called()
    with tmp_db.get_session() as session:
        assert session.query(PortfolioSnapshot).count() == 0


@pytest.mark.asyncio
async def test_sync_pending_trades_updates_filled_orders(bot, tmp_db):
    pending_order = Order(
        ticker="MSFT",
        side=OrderSide.BUY,
        qty=10,
        order_type=OrderType.LIMIT,
        limit_price=420.0,
        id="order-1",
        status=OrderStatus.PENDING_NEW,
    )
    filled_order = Order(
        ticker="MSFT",
        side=OrderSide.BUY,
        qty=10,
        order_type=OrderType.LIMIT,
        limit_price=420.0,
        id="order-1",
        status=OrderStatus.FILLED,
        filled_price=419.25,
    )
    bot._db = tmp_db
    bot._persist_trade(pending_order, signal_score=0.8, reason="test signal")
    bot._broker.get_order = AsyncMock(return_value=filled_order)

    updated = await bot._sync_pending_trades()

    assert updated == 1
    with tmp_db.get_session() as session:
        trade = session.query(TradeLog).one()
    assert trade.status == "filled"
    assert trade.filled_price == 419.25
    bot._broker.get_order.assert_awaited_once_with("order-1")


@pytest.mark.asyncio
async def test_run_cycle_blocks_orders_in_premarket(bot):
    bot._broker.get_positions = AsyncMock(return_value=[])
    bot._broker.get_account = AsyncMock()
    bot._broker.close_position = AsyncMock()
    bot._broker.place_order = AsyncMock()
    bot._signal_gen.evaluate.return_value = [_signal("MSFT", "buy")]

    metrics = await bot._run_cycle(phase="premarket")

    assert metrics == {"news": 1, "signals": 1, "orders": 0, "exits": 0}
    bot._broker.get_account.assert_not_called()
    bot._broker.close_position.assert_not_called()
    bot._broker.place_order.assert_not_called()
