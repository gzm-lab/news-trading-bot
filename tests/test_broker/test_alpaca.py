"""Tests for AlpacaBroker — mocked API calls."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.broker.alpaca_broker import AlpacaBroker
from src.broker.interface import Order, OrderSide, OrderStatus, OrderType


@pytest.fixture
def broker():
    return AlpacaBroker(api_key="test-key", secret_key="test-secret", paper=True)


class TestAlpacaBrokerConnect:
    @pytest.mark.asyncio
    async def test_connect_creates_clients(self, broker):
        with patch("src.broker.alpaca_broker.TradingClient") as mock_tc, \
             patch("src.broker.alpaca_broker.StockHistoricalDataClient") as mock_hdc:
            await broker.connect()
            mock_tc.assert_called_once_with(api_key="test-key", secret_key="test-secret", paper=True)
            mock_hdc.assert_called_once_with(api_key="test-key", secret_key="test-secret")


class TestAlpacaBrokerAccount:
    @pytest.mark.asyncio
    async def test_get_account(self, broker):
        mock_account = MagicMock()
        mock_account.equity = MagicMock(__float__=lambda s: 100000.0)
        mock_account.cash = MagicMock(__float__=lambda s: 80000.0)
        mock_account.buying_power = MagicMock(__float__=lambda s: 160000.0)
        mock_account.portfolio_value = MagicMock(__float__=lambda s: 100000.0)

        with patch("src.broker.alpaca_broker.TradingClient") as mock_tc, \
             patch("src.broker.alpaca_broker.StockHistoricalDataClient"):
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
        with patch("src.broker.alpaca_broker.TradingClient") as mock_tc, \
             patch("src.broker.alpaca_broker.StockHistoricalDataClient"):
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
        mock_alpaca_order.filled_at = datetime.now(timezone.utc)

        with patch("src.broker.alpaca_broker.TradingClient") as mock_tc, \
             patch("src.broker.alpaca_broker.StockHistoricalDataClient"):
            mock_client = MagicMock()
            mock_tc.return_value = mock_client
            mock_client.submit_order.return_value = mock_alpaca_order

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
            mock_client.submit_order.assert_called_once()


class TestAlpacaBrokerMarketStatus:
    @pytest.mark.asyncio
    async def test_is_market_open(self, broker):
        mock_clock = MagicMock()
        mock_clock.is_open = True

        with patch("src.broker.alpaca_broker.TradingClient") as mock_tc, \
             patch("src.broker.alpaca_broker.StockHistoricalDataClient"):
            mock_client = MagicMock()
            mock_tc.return_value = mock_client
            mock_client.get_clock.return_value = mock_clock

            await broker.connect()
            assert await broker.is_market_open() is True

    @pytest.mark.asyncio
    async def test_market_closed(self, broker):
        mock_clock = MagicMock()
        mock_clock.is_open = False

        with patch("src.broker.alpaca_broker.TradingClient") as mock_tc, \
             patch("src.broker.alpaca_broker.StockHistoricalDataClient"):
            mock_client = MagicMock()
            mock_tc.return_value = mock_client
            mock_client.get_clock.return_value = mock_clock

            await broker.connect()
            assert await broker.is_market_open() is False
