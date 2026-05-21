"""Tests for AlpacaBroker — mocked API calls."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from alpaca.data.timeframe import TimeFrameUnit
from alpaca.trading.enums import OrderSide as AlpacaOrderSide

from src.broker.alpaca_broker import AlpacaBroker
from src.broker.interface import Order, OrderSide, OrderStatus, OrderType


@pytest.fixture
def broker():
    return AlpacaBroker(api_key="test-key", secret_key="test-secret", paper=True)


class TestAlpacaBrokerConnect:
    @pytest.mark.asyncio
    async def test_connect_creates_clients(self, broker):
        with (
            patch("src.broker.alpaca_broker.TradingClient") as mock_tc,
            patch("src.broker.alpaca_broker.StockHistoricalDataClient") as mock_hdc,
        ):
            await broker.connect()
            mock_tc.assert_called_once_with(
                api_key="test-key", secret_key="test-secret", paper=True
            )
            mock_hdc.assert_called_once_with(api_key="test-key", secret_key="test-secret")


class TestAlpacaBrokerAccount:
    @pytest.mark.asyncio
    async def test_get_account(self, broker):
        mock_account = MagicMock()
        mock_account.equity = MagicMock(__float__=lambda s: 100000.0)
        mock_account.cash = MagicMock(__float__=lambda s: 80000.0)
        mock_account.buying_power = MagicMock(__float__=lambda s: 160000.0)
        mock_account.portfolio_value = MagicMock(__float__=lambda s: 100000.0)

        with (
            patch("src.broker.alpaca_broker.TradingClient") as mock_tc,
            patch("src.broker.alpaca_broker.StockHistoricalDataClient"),
        ):
            mock_client = MagicMock()
            mock_tc.return_value = mock_client
            mock_client.get_account.return_value = mock_account

            await broker.connect()
            account = await broker.get_account()

            assert account.equity == 100000.0
            assert account.cash == 80000.0


class TestAlpacaBrokerPositions:
    @pytest.mark.asyncio
    async def test_get_positions_empty(self, broker):
        with (
            patch("src.broker.alpaca_broker.TradingClient") as mock_tc,
            patch("src.broker.alpaca_broker.StockHistoricalDataClient"),
        ):
            mock_client = MagicMock()
            mock_tc.return_value = mock_client
            mock_client.get_all_positions.return_value = []

            await broker.connect()
            positions = await broker.get_positions()
            assert positions == []


class TestAlpacaBrokerOrder:
    @pytest.mark.asyncio
    async def test_place_market_order(self, broker):
        mock_alpaca_order = MagicMock()
        mock_alpaca_order.id = "order-abc"
        mock_alpaca_order.status.value = "filled"
        mock_alpaca_order.filled_avg_price = 185.0
        mock_alpaca_order.filled_at = datetime.now(UTC)

        with (
            patch("src.broker.alpaca_broker.TradingClient") as mock_tc,
            patch("src.broker.alpaca_broker.StockHistoricalDataClient"),
        ):
            mock_client = MagicMock()
            mock_tc.return_value = mock_client
            mock_client.submit_order.return_value = mock_alpaca_order
            mock_client.get_orders.return_value = []

            await broker.connect()

            order = Order(
                ticker="AAPL",
                side=OrderSide.BUY,
                qty=10,
                order_type=OrderType.MARKET,
            )
            result = await broker.place_order(order)

            assert result.id == "order-abc"
            assert result.filled_price == 185.0
            mock_client.get_orders.assert_called_once()
            mock_client.submit_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_sell_order_cancels_conflicting_open_sell(self, broker):
        open_sell = MagicMock()
        open_sell.id = "old-sell"
        open_sell.side = AlpacaOrderSide.SELL

        mock_alpaca_order = MagicMock()
        mock_alpaca_order.id = "new-sell"
        mock_alpaca_order.status.value = "pending_new"
        mock_alpaca_order.filled_avg_price = None
        mock_alpaca_order.filled_at = None

        with (
            patch("src.broker.alpaca_broker.TradingClient") as mock_tc,
            patch("src.broker.alpaca_broker.StockHistoricalDataClient"),
        ):
            mock_client = MagicMock()
            mock_tc.return_value = mock_client
            mock_client.get_orders.return_value = [open_sell]
            mock_client.submit_order.return_value = mock_alpaca_order

            await broker.connect()
            order = Order(
                ticker="TSLA",
                side=OrderSide.SELL,
                qty=14,
                order_type=OrderType.LIMIT,
                limit_price=450.0,
            )
            result = await broker.place_order(order)

        assert result.id == "new-sell"
        mock_client.cancel_order_by_id.assert_called_once_with("old-sell")
        mock_client.submit_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_order_maps_canceled_status(self, broker):
        mock_alpaca_order = MagicMock()
        mock_alpaca_order.id = "old-buy"
        mock_alpaca_order.symbol = "AAPL"
        mock_alpaca_order.side.value = "buy"
        mock_alpaca_order.type.value = "limit"
        mock_alpaca_order.qty = "10"
        mock_alpaca_order.limit_price = "200.50"
        mock_alpaca_order.status.value = "canceled"
        mock_alpaca_order.filled_avg_price = None
        mock_alpaca_order.filled_at = None

        with (
            patch("src.broker.alpaca_broker.TradingClient") as mock_tc,
            patch("src.broker.alpaca_broker.StockHistoricalDataClient"),
        ):
            mock_client = MagicMock()
            mock_tc.return_value = mock_client
            mock_client.get_order_by_id.return_value = mock_alpaca_order

            await broker.connect()
            result = await broker.get_order("old-buy")

        assert result is not None
        assert result.id == "old-buy"
        assert result.ticker == "AAPL"
        assert result.status == OrderStatus.CANCELED
        assert result.limit_price == 200.50

    @pytest.mark.asyncio
    async def test_place_sell_order_keeps_opposite_side_open_order(self, broker):
        open_buy = MagicMock()
        open_buy.id = "old-buy"
        open_buy.side = AlpacaOrderSide.BUY

        mock_alpaca_order = MagicMock()
        mock_alpaca_order.id = "new-sell"
        mock_alpaca_order.status.value = "pending_new"
        mock_alpaca_order.filled_avg_price = None
        mock_alpaca_order.filled_at = None

        with (
            patch("src.broker.alpaca_broker.TradingClient") as mock_tc,
            patch("src.broker.alpaca_broker.StockHistoricalDataClient"),
        ):
            mock_client = MagicMock()
            mock_tc.return_value = mock_client
            mock_client.get_orders.return_value = [open_buy]
            mock_client.submit_order.return_value = mock_alpaca_order

            await broker.connect()
            order = Order(
                ticker="TSLA",
                side=OrderSide.SELL,
                qty=14,
                order_type=OrderType.MARKET,
            )
            await broker.place_order(order)

        mock_client.cancel_order_by_id.assert_not_called()
        mock_client.submit_order.assert_called_once()


class TestAlpacaBrokerBars:
    @pytest.mark.asyncio
    async def test_get_bars_uses_timeframe_units(self, broker):
        with (
            patch("src.broker.alpaca_broker.TradingClient"),
            patch("src.broker.alpaca_broker.StockHistoricalDataClient") as mock_hdc,
        ):
            mock_data_client = MagicMock()
            mock_hdc.return_value = mock_data_client
            mock_data_client.get_stock_bars.return_value = MagicMock(data={"AAPL": []})

            await broker.connect()
            bars = await broker.get_bars("AAPL", timeframe="15Min", limit=5)

        assert bars.empty
        request = mock_data_client.get_stock_bars.call_args.args[0]
        assert request.timeframe.amount == 15
        assert request.timeframe.unit == TimeFrameUnit.Minute
        assert str(request.timeframe) == "15Min"


class TestAlpacaBrokerLatestPrice:
    @pytest.mark.asyncio
    async def test_latest_price_uses_midpoint(self, broker):
        quote = MagicMock()
        quote.ask_price = 101.0
        quote.bid_price = 99.0

        with (
            patch("src.broker.alpaca_broker.TradingClient"),
            patch("src.broker.alpaca_broker.StockHistoricalDataClient") as mock_hdc,
        ):
            mock_data_client = MagicMock()
            mock_hdc.return_value = mock_data_client
            mock_data_client.get_stock_latest_quote.return_value = {"AAPL": quote}

            await broker.connect()
            price = await broker.get_latest_price("AAPL")

        assert price == 100.0

    @pytest.mark.asyncio
    async def test_latest_price_uses_ask_when_bid_missing(self, broker):
        quote = MagicMock()
        quote.ask_price = 101.0
        quote.bid_price = 0

        with (
            patch("src.broker.alpaca_broker.TradingClient"),
            patch("src.broker.alpaca_broker.StockHistoricalDataClient") as mock_hdc,
        ):
            mock_data_client = MagicMock()
            mock_hdc.return_value = mock_data_client
            mock_data_client.get_stock_latest_quote.return_value = {"AAPL": quote}

            await broker.connect()
            price = await broker.get_latest_price("AAPL")

        assert price == 101.0

    @pytest.mark.asyncio
    async def test_latest_price_returns_zero_when_ticker_missing(self, broker):
        with (
            patch("src.broker.alpaca_broker.TradingClient"),
            patch("src.broker.alpaca_broker.StockHistoricalDataClient") as mock_hdc,
        ):
            mock_data_client = MagicMock()
            mock_hdc.return_value = mock_data_client
            mock_data_client.get_stock_latest_quote.return_value = {}

            await broker.connect()
            price = await broker.get_latest_price("AAPL")

        assert price == 0.0


class TestAlpacaBrokerMarketStatus:
    @pytest.mark.asyncio
    async def test_is_market_open(self, broker):
        mock_clock = MagicMock()
        mock_clock.is_open = True

        with (
            patch("src.broker.alpaca_broker.TradingClient") as mock_tc,
            patch("src.broker.alpaca_broker.StockHistoricalDataClient"),
        ):
            mock_client = MagicMock()
            mock_tc.return_value = mock_client
            mock_client.get_clock.return_value = mock_clock

            await broker.connect()
            assert await broker.is_market_open() is True

    @pytest.mark.asyncio
    async def test_market_closed(self, broker):
        mock_clock = MagicMock()
        mock_clock.is_open = False

        with (
            patch("src.broker.alpaca_broker.TradingClient") as mock_tc,
            patch("src.broker.alpaca_broker.StockHistoricalDataClient"),
        ):
            mock_client = MagicMock()
            mock_tc.return_value = mock_client
            mock_client.get_clock.return_value = mock_clock

            await broker.connect()
            assert await broker.is_market_open() is False
