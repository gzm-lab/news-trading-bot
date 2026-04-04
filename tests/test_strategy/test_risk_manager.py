"""Tests for risk manager — drawdowns, stops, cooldowns."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from src.broker.interface import Account, Order, OrderSide, OrderType, Position
from src.strategy.risk_manager import RiskManager, RiskState
from src.strategy.signals import Signal


def _make_signal(ticker, action="buy", score=0.5):
    return Signal(
        ticker=ticker,
        action=action,
        score=score,
        sentiment_score=score,
        news_velocity=2.0,
        technical_score=0.3,
        volume_score=0.2,
        reason=f"Test {action} signal for {ticker}",
    )


def _make_account(equity=100_000, cash=80_000, buying_power=160_000):
    return Account(
        equity=float(equity),
        cash=float(cash),
        buying_power=float(buying_power),
        portfolio_value=float(equity),
    )


def _make_position(ticker, pnl_pct=0.0):
    return Position(
        ticker=ticker,
        qty=10,
        avg_entry_price=100.0,
        current_price=100.0 * (1 + pnl_pct),
        market_value=1000.0 * (1 + pnl_pct),
        unrealized_pnl=1000.0 * pnl_pct,
        unrealized_pnl_pct=pnl_pct,
    )


class TestRiskState:
    def test_defaults(self):
        state = RiskState()
        assert state.daily_start_equity == 0.0
        assert state.trading_halted is False
        assert state.halt_reason == ""
        assert state.cooldowns == {}


class TestRiskManagerBaseline:
    def test_set_daily_baseline(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        rm.set_daily_baseline(100_000.0)
        assert rm.state.daily_start_equity == 100_000.0
        assert rm.state.trading_halted is False

    def test_reset_after_halt(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        rm.state.trading_halted = True
        rm.state.halt_reason = "test halt"
        rm.set_daily_baseline(100_000.0)
        assert rm.state.trading_halted is False
        assert rm.state.halt_reason == ""


class TestDailyPnL:
    def test_positive_pnl(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        rm.set_daily_baseline(100_000.0)

        account = _make_account(equity=101_000)
        rm.update_daily_pnl(account)

        assert rm.state.daily_pnl == 1_000.0
        assert rm.state.daily_pnl_pct == pytest.approx(0.01)
        assert rm.state.trading_halted is False

    def test_negative_pnl_within_limit(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        rm.set_daily_baseline(100_000.0)

        # -3% loss, limit is -5%
        account = _make_account(equity=97_000)
        rm.update_daily_pnl(account)

        assert rm.state.daily_pnl == -3_000.0
        assert rm.state.trading_halted is False

    def test_drawdown_triggers_halt(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        rm.set_daily_baseline(100_000.0)

        # -6% loss, exceeds -5% limit
        account = _make_account(equity=94_000)
        rm.update_daily_pnl(account)

        assert rm.state.trading_halted is True
        assert "drawdown" in rm.state.halt_reason.lower()
        assert "-5.0%" in rm.state.halt_reason or "5.0%" in rm.state.halt_reason


class TestFilterSignals:
    def test_halted_returns_empty(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        rm.state.trading_halted = True

        signals = [_make_signal("AAPL", "buy")]
        account = _make_account()
        orders = rm.filter_signals(signals, account, [])

        assert orders == []

    def test_buy_signal_creates_order(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        signals = [_make_signal("AAPL", "buy", score=0.6)]
        account = _make_account()

        orders = rm.filter_signals(signals, account, [])

        assert len(orders) == 1
        assert orders[0].ticker == "AAPL"
        assert orders[0].side == OrderSide.BUY
        assert orders[0].order_type == OrderType.MARKET

    def test_sell_signal_creates_order(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        signals = [_make_signal("TSLA", "sell", score=-0.5)]
        account = _make_account()

        orders = rm.filter_signals(signals, account, [])

        assert len(orders) == 1
        assert orders[0].ticker == "TSLA"
        assert orders[0].side == OrderSide.SELL

    def test_hold_signal_skipped(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        signals = [_make_signal("AAPL", "hold", score=0.1)]
        account = _make_account()

        orders = rm.filter_signals(signals, account, [])
        assert orders == []

    def test_max_positions_enforced(self, strategy_config):
        # max_positions = 10
        rm = RiskManager(config=strategy_config)
        existing = [_make_position(f"TICK{i}") for i in range(10)]
        signals = [_make_signal("NEW", "buy")]
        account = _make_account()

        orders = rm.filter_signals(signals, account, existing)
        buy_orders = [o for o in orders if o.side == OrderSide.BUY]
        assert len(buy_orders) == 0

    def test_cooldown_blocks_ticker(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        # Set cooldown for AAPL 30 minutes from now
        rm.state.cooldowns["AAPL"] = datetime.now(timezone.utc) + timedelta(minutes=30)

        signals = [_make_signal("AAPL", "buy")]
        account = _make_account()
        orders = rm.filter_signals(signals, account, [])

        assert orders == []

    def test_expired_cooldown_allows_trade(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        # Cooldown expired 5 min ago
        rm.state.cooldowns["AAPL"] = datetime.now(timezone.utc) - timedelta(minutes=5)

        signals = [_make_signal("AAPL", "buy")]
        account = _make_account()
        orders = rm.filter_signals(signals, account, [])

        assert len(orders) == 1

    def test_multiple_signals_processed(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        signals = [
            _make_signal("AAPL", "buy", score=0.7),
            _make_signal("MSFT", "buy", score=0.5),
            _make_signal("TSLA", "sell", score=-0.6),
        ]
        account = _make_account()

        orders = rm.filter_signals(signals, account, [])
        assert len(orders) == 3


class TestCheckExits:
    def test_stop_loss_triggered(self, strategy_config):
        # stop_loss_pct = 0.02 (2%)
        rm = RiskManager(config=strategy_config)
        positions = [_make_position("AAPL", pnl_pct=-0.025)]  # -2.5%

        exits = rm.check_exits(positions)
        assert "AAPL" in exits

    def test_take_profit_triggered(self, strategy_config):
        # take_profit_pct = 0.04 (4%)
        rm = RiskManager(config=strategy_config)
        positions = [_make_position("MSFT", pnl_pct=0.05)]  # +5%

        exits = rm.check_exits(positions)
        assert "MSFT" in exits

    def test_no_exit_within_range(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        positions = [_make_position("AAPL", pnl_pct=0.01)]  # +1%, within range

        exits = rm.check_exits(positions)
        assert exits == []

    def test_exit_sets_cooldown(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        positions = [_make_position("AAPL", pnl_pct=-0.03)]

        exits = rm.check_exits(positions)
        assert "AAPL" in exits
        assert "AAPL" in rm.state.cooldowns
        assert rm.state.cooldowns["AAPL"] > datetime.now(timezone.utc)

    def test_multiple_exits(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        positions = [
            _make_position("AAPL", pnl_pct=-0.03),   # stop-loss
            _make_position("MSFT", pnl_pct=0.05),     # take-profit
            _make_position("GOOGL", pnl_pct=0.01),    # no exit
        ]

        exits = rm.check_exits(positions)
        assert "AAPL" in exits
        assert "MSFT" in exits
        assert "GOOGL" not in exits
