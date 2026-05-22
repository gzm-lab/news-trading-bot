"""Tests for risk manager — drawdowns, stops, cooldowns."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from src.broker.interface import Account, OrderSide, OrderType, Position
from src.strategy.risk_manager import RiskManager, RiskState, calculate_risk_position_size
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
        assert state.baseline_date is None
        assert state.trading_halted is False
        assert state.halt_reason == ""
        assert state.cooldowns == {}
        assert state.position_highs == {}
        assert state.symbol_buy_counts_by_utc_date == {}
        assert state.last_symbol_buy_at == {}


class TestRiskPositionSizing:
    def test_calculates_quantity_from_dollars_at_risk(self):
        qty = calculate_risk_position_size(
            equity=100_000,
            entry_price=100,
            stop_price=95,
            risk_pct=0.0025,
            max_notional=10_000,
        )

        assert qty == 50

    def test_caps_quantity_by_max_notional(self):
        qty = calculate_risk_position_size(
            equity=100_000,
            entry_price=100,
            stop_price=99,
            risk_pct=0.01,
            max_notional=5_000,
        )

        assert qty == 50

    @pytest.mark.parametrize(
        ("equity", "entry_price", "stop_price", "risk_pct", "max_notional"),
        [
            (0, 100, 95, 0.0025, 10_000),
            (100_000, 0, 95, 0.0025, 10_000),
            (100_000, 100, 100, 0.0025, 10_000),
            (100_000, 100, 95, 0, 10_000),
            (100_000, 100, 95, 0.0025, 0),
        ],
    )
    def test_invalid_or_zero_risk_inputs_return_zero(
        self, equity, entry_price, stop_price, risk_pct, max_notional
    ):
        assert (
            calculate_risk_position_size(
                equity=equity,
                entry_price=entry_price,
                stop_price=stop_price,
                risk_pct=risk_pct,
                max_notional=max_notional,
            )
            == 0
        )


class TestRiskManagerBaseline:
    def test_set_daily_baseline(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        rm.set_daily_baseline(100_000.0)
        assert rm.state.daily_start_equity == 100_000.0
        assert rm.state.baseline_date is not None
        assert rm.state.trading_halted is False

    def test_reset_after_halt(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        rm.state.trading_halted = True
        rm.state.halt_reason = "test halt"
        rm.set_daily_baseline(100_000.0)
        assert rm.state.trading_halted is False
        assert rm.state.halt_reason == ""

    def test_ensure_daily_baseline_sets_missing_baseline(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        account = _make_account(equity=100_000)
        today = date(2026, 5, 6)

        rm.ensure_daily_baseline(account, today=today)

        assert rm.state.daily_start_equity == 100_000.0
        assert rm.state.baseline_date == today

    def test_ensure_daily_baseline_same_day_does_not_reset_halt(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        today = date(2026, 5, 6)
        rm.set_daily_baseline(100_000.0, today)
        rm.state.trading_halted = True
        rm.state.halt_reason = "drawdown"

        rm.ensure_daily_baseline(_make_account(equity=101_000), today=today)

        assert rm.state.daily_start_equity == 100_000.0
        assert rm.state.trading_halted is True
        assert rm.state.halt_reason == "drawdown"

    def test_ensure_daily_baseline_new_day_resets_halt(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        rm.set_daily_baseline(100_000.0, date(2026, 5, 6))
        rm.state.trading_halted = True
        rm.state.halt_reason = "drawdown"

        rm.ensure_daily_baseline(_make_account(equity=99_000), today=date(2026, 5, 7))

        assert rm.state.daily_start_equity == 99_000.0
        assert rm.state.baseline_date == date(2026, 5, 7)
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
        assert orders[0].order_type == OrderType.LIMIT
        assert getattr(orders[0], "_max_value") == pytest.approx(
            account.equity * strategy_config.max_position_pct
        )
        assert getattr(orders[0], "_risk_qty") is None
        assert getattr(orders[0], "_signal") is signals[0]

    def test_atr_features_attach_risk_limited_quantity(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        signal = _make_signal("AAPL", "buy", score=0.7)
        signal.features = {"last_price": 100.0, "atr_14": 20.0}

        orders = rm.filter_signals([signal], _make_account(equity=100_000), [])

        assert len(orders) == 1
        assert getattr(orders[0], "_max_value") == pytest.approx(2_000.0)
        assert getattr(orders[0], "_risk_qty") == 12

    def test_atr_sizing_cannot_exceed_notional_cap(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        signal = _make_signal("AAPL", "buy", score=0.7)
        signal.features = {"last_price": 100.0, "atr_14": 0.5}

        orders = rm.filter_signals([signal], _make_account(equity=100_000), [])

        assert len(orders) == 1
        assert getattr(orders[0], "_risk_qty") == 20

    def test_sell_signal_creates_order(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        signals = [_make_signal("TSLA", "sell", score=-0.5)]
        account = _make_account()

        orders = rm.filter_signals(signals, account, [])

        assert len(orders) == 1
        assert orders[0].ticker == "TSLA"
        assert orders[0].side == OrderSide.SELL
        assert orders[0].order_type == OrderType.LIMIT
        assert getattr(orders[0], "_signal") is signals[0]

    def test_hold_signal_skipped(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        signals = [_make_signal("AAPL", "hold", score=0.1)]
        account = _make_account()

        orders = rm.filter_signals(signals, account, [])
        assert orders == []

    def test_max_positions_enforced(self, strategy_config):
        # max_positions = 5
        rm = RiskManager(config=strategy_config)
        existing = [_make_position(f"TICK{i}") for i in range(strategy_config.max_positions)]
        signals = [_make_signal("NEW", "buy")]
        account = _make_account()

        orders = rm.filter_signals(signals, account, existing)
        buy_orders = [o for o in orders if o.side == OrderSide.BUY]
        assert len(buy_orders) == 0

    def test_cooldown_blocks_ticker(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        # Set cooldown for AAPL 30 minutes from now
        rm.state.cooldowns["AAPL"] = datetime.now(UTC) + timedelta(minutes=30)

        signals = [_make_signal("AAPL", "buy")]
        account = _make_account()
        orders = rm.filter_signals(signals, account, [])

        assert orders == []

    def test_expired_cooldown_allows_trade(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        # Cooldown expired 5 min ago
        rm.state.cooldowns["AAPL"] = datetime.now(UTC) - timedelta(minutes=5)

        signals = [_make_signal("AAPL", "buy")]
        account = _make_account()
        orders = rm.filter_signals(signals, account, [])

        assert len(orders) == 1

    def test_multiple_signals_processed_until_order_limit(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        signals = [
            _make_signal("AAPL", "buy", score=0.7),
            _make_signal("MSFT", "buy", score=0.5),
            _make_signal("TSLA", "sell", score=-0.6),
        ]
        account = _make_account()

        orders = rm.filter_signals(signals, account, [])
        assert len(orders) == strategy_config.max_orders_per_cycle

    def test_max_buys_per_cycle_enforced(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        signals = [
            _make_signal("AAPL", "buy", score=0.7),
            _make_signal("MSFT", "buy", score=0.6),
            _make_signal("GOOGL", "buy", score=0.5),
        ]

        orders = rm.filter_signals(signals, _make_account(), [])

        buy_orders = [o for o in orders if o.side == OrderSide.BUY]
        assert len(buy_orders) == strategy_config.max_buys_per_cycle

    def test_approved_order_sets_cooldown(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        orders = rm.filter_signals([_make_signal("AAPL", "buy")], _make_account(), [])

        assert len(orders) == 1
        assert "AAPL" in rm.state.cooldowns
        assert rm.state.cooldowns["AAPL"] > datetime.now(UTC)

    def test_cooldown_blocks_duplicate_next_cycle(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        signal = _make_signal("AAPL", "buy")

        first = rm.filter_signals([signal], _make_account(), [])
        second = rm.filter_signals([signal], _make_account(), [])

        assert len(first) == 1
        assert second == []

    def test_max_buys_per_symbol_per_day_blocks_second_buy_after_cooldown(
        self, strategy_config
    ):
        rm = RiskManager(config=strategy_config)
        signal = _make_signal("AAPL", "buy")

        first = rm.filter_signals([signal], _make_account(), [])
        rm.state.cooldowns["AAPL"] = datetime.now(UTC) - timedelta(minutes=1)
        rm.state.last_symbol_buy_at["AAPL"] = datetime.now(UTC) - timedelta(
            minutes=strategy_config.min_minutes_between_symbol_buys + 1
        )
        second = rm.filter_signals([signal], _make_account(), [])

        assert len(first) == 1
        assert second == []
        today_counts = rm.state.symbol_buy_counts_by_utc_date[datetime.now(UTC).date()]
        assert today_counts["AAPL"] == 1

    def test_per_symbol_daily_buy_limit_allows_different_symbol(self, strategy_config):
        rm = RiskManager(config=strategy_config)

        first = rm.filter_signals([_make_signal("AAPL", "buy")], _make_account(), [])
        second = rm.filter_signals([_make_signal("MSFT", "buy")], _make_account(), [])

        assert [order.ticker for order in first] == ["AAPL"]
        assert [order.ticker for order in second] == ["MSFT"]

    def test_daily_buy_count_is_keyed_by_utc_date(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        yesterday = datetime.now(UTC).date() - timedelta(days=1)
        rm.state.symbol_buy_counts_by_utc_date[yesterday] = {"AAPL": 1}
        rm.state.last_symbol_buy_at["AAPL"] = datetime.now(UTC) - timedelta(days=1)

        orders = rm.filter_signals([_make_signal("AAPL", "buy")], _make_account(), [])

        assert len(orders) == 1
        assert rm.state.symbol_buy_counts_by_utc_date[datetime.now(UTC).date()]["AAPL"] == 1

    def test_min_minutes_between_symbol_buys_blocks_after_generic_cooldown(
        self, strategy_config
    ):
        strategy_config.max_buys_per_symbol_per_day = 2
        rm = RiskManager(config=strategy_config)
        signal = _make_signal("AAPL", "buy")

        first = rm.filter_signals([signal], _make_account(), [])
        rm.state.cooldowns["AAPL"] = datetime.now(UTC) - timedelta(minutes=1)
        rm.state.symbol_buy_counts_by_utc_date[datetime.now(UTC).date()]["AAPL"] = 1
        rm.state.last_symbol_buy_at["AAPL"] = datetime.now(UTC) - timedelta(
            minutes=strategy_config.min_minutes_between_symbol_buys - 1
        )
        second = rm.filter_signals([signal], _make_account(), [])

        assert len(first) == 1
        assert second == []


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
        assert rm.state.cooldowns["AAPL"] > datetime.now(UTC)

    def test_multiple_exits(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        positions = [
            _make_position("AAPL", pnl_pct=-0.03),  # stop-loss
            _make_position("MSFT", pnl_pct=0.05),  # take-profit
            _make_position("GOOGL", pnl_pct=0.01),  # no exit
        ]

        exits = rm.check_exits(positions)
        assert "AAPL" in exits
        assert "MSFT" in exits
        assert "GOOGL" not in exits

    def test_trailing_stop_triggers_after_drop_from_high(self, strategy_config):
        strategy_config.trailing_stop_pct = 0.01
        rm = RiskManager(config=strategy_config)

        rm.check_exits([_make_position("AAPL", pnl_pct=0.03)])
        exits = rm.check_exits([_make_position("AAPL", pnl_pct=0.015)])

        assert "AAPL" in exits
        assert "AAPL" in rm.state.cooldowns

    def test_trailing_stop_ignores_small_pullback(self, strategy_config):
        strategy_config.trailing_stop_pct = 0.01
        rm = RiskManager(config=strategy_config)

        rm.check_exits([_make_position("AAPL", pnl_pct=0.03)])
        exits = rm.check_exits([_make_position("AAPL", pnl_pct=0.025)])

        assert exits == []
        assert rm.state.position_highs["AAPL"] == pytest.approx(103.0)

    def test_stale_position_high_removed(self, strategy_config):
        rm = RiskManager(config=strategy_config)
        rm.check_exits([_make_position("AAPL", pnl_pct=0.03)])

        rm.check_exits([])

        assert "AAPL" not in rm.state.position_highs
