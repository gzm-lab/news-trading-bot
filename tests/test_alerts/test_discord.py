"""Tests for Discord alerter — mocked webhook."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch, call
import asyncio

import pytest

from src.alerts.discord import DiscordAlerter
from src.broker.interface import Account, Order, OrderSide, OrderStatus, OrderType, Position


@pytest.fixture
def alerter():
    return DiscordAlerter(webhook_url="https://discord.com/api/webhooks/test/token")


@pytest.fixture
def disabled_alerter():
    return DiscordAlerter(webhook_url="https://discord.com/api/webhooks/test/token", enabled=False)


@pytest.fixture
def no_url_alerter():
    return DiscordAlerter(webhook_url="")


class TestDiscordAlerterInit:
    def test_enabled_with_url(self, alerter):
        assert alerter._enabled is True

    def test_disabled_explicitly(self, disabled_alerter):
        assert disabled_alerter._enabled is False

    def test_disabled_without_url(self, no_url_alerter):
        assert no_url_alerter._enabled is False


class TestNotifyTrade:
    @pytest.mark.asyncio
    async def test_buy_notification(self, alerter):
        order = Order(
            ticker="AAPL", side=OrderSide.BUY, qty=10,
            order_type=OrderType.MARKET, id="order-1",
            status=OrderStatus.FILLED, filled_price=185.0,
        )

        with patch.object(alerter, "_send", new_callable=AsyncMock) as mock_send:
            await alerter.notify_trade(order, reason="Strong positive sentiment")
            mock_send.assert_called_once()
            embed = mock_send.call_args.kwargs["embed"]
            assert "BUY" in embed.title
            assert "AAPL" in embed.title

    @pytest.mark.asyncio
    async def test_sell_notification(self, alerter):
        order = Order(
            ticker="TSLA", side=OrderSide.SELL, qty=5,
            order_type=OrderType.MARKET, status=OrderStatus.FILLED,
            filled_price=250.0,
        )

        with patch.object(alerter, "_send", new_callable=AsyncMock) as mock_send:
            await alerter.notify_trade(order)
            mock_send.assert_called_once()
            embed = mock_send.call_args.kwargs["embed"]
            assert "SELL" in embed.title

    @pytest.mark.asyncio
    async def test_disabled_does_nothing(self, disabled_alerter):
        order = Order(
            ticker="AAPL", side=OrderSide.BUY, qty=10,
            order_type=OrderType.MARKET,
        )
        with patch.object(disabled_alerter, "_send", new_callable=AsyncMock) as mock_send:
            await disabled_alerter.notify_trade(order)
            mock_send.assert_not_called()


class TestNotifyExit:
    @pytest.mark.asyncio
    async def test_stop_loss(self, alerter):
        with patch.object(alerter, "_send", new_callable=AsyncMock) as mock_send:
            await alerter.notify_exit("AAPL", "stop_loss", pnl=-150.0, pnl_pct=-0.02)
            mock_send.assert_called_once()
            embed = mock_send.call_args.kwargs["embed"]
            assert "STOP" in embed.title.upper()

    @pytest.mark.asyncio
    async def test_take_profit(self, alerter):
        with patch.object(alerter, "_send", new_callable=AsyncMock) as mock_send:
            await alerter.notify_exit("MSFT", "take_profit", pnl=300.0, pnl_pct=0.04)
            mock_send.assert_called_once()
            embed = mock_send.call_args.kwargs["embed"]
            assert "TAKE PROFIT" in embed.title.upper()


class TestNotifyDailySummary:
    @pytest.mark.asyncio
    async def test_summary_with_positions(self, alerter, sample_account, sample_positions):
        with patch.object(alerter, "_send", new_callable=AsyncMock) as mock_send:
            await alerter.notify_daily_summary(sample_account, sample_positions)
            mock_send.assert_called_once()
            embed = mock_send.call_args.kwargs["embed"]
            assert "Summary" in embed.title

    @pytest.mark.asyncio
    async def test_summary_no_positions(self, alerter, sample_account):
        with patch.object(alerter, "_send", new_callable=AsyncMock) as mock_send:
            await alerter.notify_daily_summary(sample_account, [])
            mock_send.assert_called_once()


class TestNotifyHalt:
    @pytest.mark.asyncio
    async def test_halt_notification(self, alerter):
        with patch.object(alerter, "_send", new_callable=AsyncMock) as mock_send:
            await alerter.notify_halt("Drawdown exceeded 5%")
            mock_send.assert_called_once()
            embed = mock_send.call_args.kwargs["embed"]
            assert "HALT" in embed.title.upper()


class TestNotifyStartup:
    @pytest.mark.asyncio
    async def test_startup(self, alerter, sample_account):
        with patch.object(alerter, "_send", new_callable=AsyncMock) as mock_send:
            await alerter.notify_startup(sample_account)
            mock_send.assert_called_once()
            embed = mock_send.call_args.kwargs["embed"]
            assert "Started" in embed.title


class TestSendWebhook:
    @pytest.mark.asyncio
    async def test_send_handles_error(self, alerter):
        """_send should log but not raise on failure."""
        with patch("src.alerts.discord.DiscordWebhook") as mock_wh_cls:
            mock_wh = MagicMock()
            mock_wh.execute.side_effect = Exception("Connection refused")
            mock_wh_cls.return_value = mock_wh

            # Should not raise
            await alerter._send(content="test")
