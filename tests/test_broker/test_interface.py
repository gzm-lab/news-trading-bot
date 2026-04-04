"""Tests for broker interface data classes."""

import pytest

from src.broker.interface import (
    Account,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


class TestOrderSide:
    def test_values(self):
        assert OrderSide.BUY == "buy"
        assert OrderSide.SELL == "sell"

    def test_is_string_enum(self):
        assert isinstance(OrderSide.BUY, str)


class TestOrderType:
    def test_values(self):
        assert OrderType.MARKET == "market"
        assert OrderType.LIMIT == "limit"


class TestOrderStatus:
    def test_all_statuses(self):
        statuses = {s.value for s in OrderStatus}
        assert "pending" in statuses
        assert "filled" in statuses
        assert "cancelled" in statuses
        assert "rejected" in statuses
        assert "partially_filled" in statuses


class TestOrder:
    def test_create_market_order(self):
        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            qty=10,
            order_type=OrderType.MARKET,
        )
        assert order.ticker == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.qty == 10
        assert order.order_type == OrderType.MARKET
        assert order.limit_price is None
        assert order.id is None
        assert order.status == OrderStatus.PENDING
        assert order.filled_price is None
        assert order.filled_at is None

    def test_create_limit_order(self):
        order = Order(
            ticker="TSLA",
            side=OrderSide.SELL,
            qty=5,
            order_type=OrderType.LIMIT,
            limit_price=250.0,
        )
        assert order.limit_price == 250.0
        assert order.order_type == OrderType.LIMIT


class TestPosition:
    def test_create(self):
        pos = Position(
            ticker="MSFT",
            qty=20,
            avg_entry_price=400.0,
            current_price=420.0,
            market_value=8400.0,
            unrealized_pnl=400.0,
            unrealized_pnl_pct=0.05,
        )
        assert pos.ticker == "MSFT"
        assert pos.unrealized_pnl == 400.0
        assert pos.unrealized_pnl_pct == 0.05

    def test_negative_pnl(self):
        pos = Position(
            ticker="META",
            qty=10,
            avg_entry_price=500.0,
            current_price=480.0,
            market_value=4800.0,
            unrealized_pnl=-200.0,
            unrealized_pnl_pct=-0.04,
        )
        assert pos.unrealized_pnl < 0
        assert pos.unrealized_pnl_pct < 0


class TestAccount:
    def test_create(self):
        account = Account(
            equity=100_000.0,
            cash=50_000.0,
            buying_power=100_000.0,
            portfolio_value=100_000.0,
        )
        assert account.equity == 100_000.0
        assert account.daily_pnl == 0.0
        assert account.daily_pnl_pct == 0.0

    def test_with_pnl(self, sample_account):
        assert sample_account.daily_pnl == 500.0
        assert sample_account.daily_pnl_pct == 0.005
