"""Tests for TradingBot risk exits and phase gating."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.broker.interface import Account, Order, OrderSide, OrderStatus, OrderType, Position
from src.main import TradingBot
from src.storage.models import CycleLog, TradeLog
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


@pytest.fixture
def bot(strategy_config):
    bot = TradingBot.__new__(TradingBot)
    bot._settings = MagicMock(cycle_interval=1)
    bot._aggregator = MagicMock()
    bot._aggregator.fetch_latest = AsyncMock(return_value=[MagicMock()])
    bot._scorer = MagicMock()
    bot._scorer.score_news = AsyncMock(return_value={})
    bot._signal_gen = MagicMock()
    bot._risk_mgr = MagicMock()
    bot._broker = MagicMock()
    bot._alerter = MagicMock()
    bot._alerter.notify_trade = AsyncMock()
    bot._db = None
    return bot


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
    bot._broker.place_order = AsyncMock(return_value=placed_order)
    bot._risk_mgr.ensure_daily_baseline = MagicMock()
    bot._risk_mgr.update_daily_pnl = MagicMock()
    bot._risk_mgr.check_exits = MagicMock(return_value=[])
    order = Order(ticker="MSFT", side=OrderSide.BUY, qty=10, order_type=OrderType.LIMIT)
    order._signal = _signal("MSFT", "buy")
    bot._risk_mgr.filter_signals = MagicMock(return_value=[order])
    bot._signal_gen.evaluate.return_value = [_signal("MSFT", "buy")]

    metrics = await bot._run_cycle(phase="open")

    assert metrics["orders"] == 1
    with tmp_db.get_session() as session:
        cycle = session.query(CycleLog).one()
        trade = session.query(TradeLog).one()
    assert cycle.news_count == 1
    assert cycle.signals_generated == 1
    assert cycle.orders_placed == 1
    assert cycle.portfolio_value == 100_000.0
    assert trade.order_id == "order-1"
    assert trade.ticker == "MSFT"
    assert trade.side == "buy"
    assert trade.qty == 10
    assert trade.status == "pending_new"
    assert trade.reason == "test signal"


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
