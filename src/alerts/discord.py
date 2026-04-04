"""Discord webhook alerts for trade notifications."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog
from discord_webhook import DiscordWebhook, DiscordEmbed

from src.broker.interface import Order, OrderSide, Account, Position

log = structlog.get_logger()


class DiscordAlerter:
    """Sends trading alerts to Discord via webhook."""

    def __init__(self, webhook_url: str, enabled: bool = True):
        self._webhook_url = webhook_url
        self._enabled = enabled and bool(webhook_url)

    async def notify_trade(self, order: Order, reason: str = "") -> None:
        """Send a trade execution alert."""
        if not self._enabled:
            return

        is_buy = order.side == OrderSide.BUY
        color = "03b2f4" if is_buy else "ff5555"
        emoji = "🟢" if is_buy else "🔴"
        side_str = "BUY" if is_buy else "SELL"

        embed = DiscordEmbed(
            title=f"{emoji} {side_str} {order.ticker}",
            color=color,
        )
        embed.add_embed_field(name="Qty", value=str(order.qty), inline=True)
        embed.add_embed_field(
            name="Price",
            value=f"${order.filled_price:.2f}" if order.filled_price else "Market",
            inline=True,
        )
        embed.add_embed_field(name="Status", value=order.status.value, inline=True)
        if reason:
            embed.add_embed_field(name="Reason", value=reason[:200], inline=False)
        embed.set_timestamp()

        await self._send(embed=embed)

    async def notify_exit(self, ticker: str, exit_type: str, pnl: float, pnl_pct: float) -> None:
        """Send a position exit alert (stop-loss or take-profit)."""
        if not self._enabled:
            return

        is_profit = pnl >= 0
        emoji = "🎯" if exit_type == "take_profit" else "🛑"
        color = "2ecc71" if is_profit else "e74c3c"

        embed = DiscordEmbed(
            title=f"{emoji} {exit_type.upper().replace('_', ' ')} — {ticker}",
            color=color,
        )
        embed.add_embed_field(name="P&L", value=f"${pnl:+.2f}", inline=True)
        embed.add_embed_field(name="P&L %", value=f"{pnl_pct:+.2%}", inline=True)
        embed.set_timestamp()

        await self._send(embed=embed)

    async def notify_daily_summary(self, account: Account, positions: list[Position]) -> None:
        """Send daily portfolio summary."""
        if not self._enabled:
            return

        pnl_emoji = "📈" if account.daily_pnl >= 0 else "📉"

        embed = DiscordEmbed(
            title=f"{pnl_emoji} Daily Summary",
            color="f39c12",
        )
        embed.add_embed_field(name="Equity", value=f"${account.equity:,.2f}", inline=True)
        embed.add_embed_field(name="Cash", value=f"${account.cash:,.2f}", inline=True)
        embed.add_embed_field(
            name="Daily P&L",
            value=f"${account.daily_pnl:+,.2f} ({account.daily_pnl_pct:+.2%})",
            inline=True,
        )
        embed.add_embed_field(name="Positions", value=str(len(positions)), inline=True)

        if positions:
            pos_lines = []
            for p in sorted(positions, key=lambda x: x.unrealized_pnl, reverse=True)[:10]:
                emoji = "🟢" if p.unrealized_pnl >= 0 else "🔴"
                pos_lines.append(
                    f"{emoji} **{p.ticker}** × {p.qty} → "
                    f"${p.unrealized_pnl:+.2f} ({p.unrealized_pnl_pct:+.2%})"
                )
            embed.add_embed_field(
                name="Open Positions",
                value="\n".join(pos_lines),
                inline=False,
            )

        embed.set_timestamp()
        await self._send(embed=embed)

    async def notify_halt(self, reason: str) -> None:
        """Send a trading halt alert."""
        if not self._enabled:
            return

        embed = DiscordEmbed(
            title="⚠️ TRADING HALTED",
            description=reason,
            color="e74c3c",
        )
        embed.set_timestamp()
        await self._send(embed=embed)

    async def notify_startup(self, account: Account) -> None:
        """Send bot startup notification."""
        if not self._enabled:
            return

        embed = DiscordEmbed(
            title="🚀 Trading Bot Started",
            color="2ecc71",
        )
        embed.add_embed_field(name="Equity", value=f"${account.equity:,.2f}", inline=True)
        embed.add_embed_field(name="Cash", value=f"${account.cash:,.2f}", inline=True)
        embed.add_embed_field(name="Mode", value="Paper Trading", inline=True)
        embed.set_timestamp()
        await self._send(embed=embed)

    async def _send(self, content: str = "", embed: DiscordEmbed | None = None) -> None:
        """Send a message via webhook."""
        try:
            webhook = DiscordWebhook(url=self._webhook_url, content=content)
            if embed:
                webhook.add_embed(embed)
            await asyncio.to_thread(webhook.execute)
        except Exception as e:
            log.warning("discord.send_failed", error=str(e))
