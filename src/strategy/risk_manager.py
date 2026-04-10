"""Risk manager — position sizing, stop-loss, drawdown limits."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

import structlog

from src.broker.interface import Order, OrderSide, OrderType, Position, Account
from src.config import StrategySettings
from src.strategy.signals import Signal

log = structlog.get_logger()


@dataclass
class RiskState:
    """Tracks risk management state across cycles."""

    daily_start_equity: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    trading_halted: bool = False
    halt_reason: str = ""
    # Cooldown: ticker -> earliest next trade time
    cooldowns: dict[str, datetime] = field(default_factory=dict)


class RiskManager:
    """Enforces risk rules before order execution."""

    def __init__(self, config: StrategySettings):
        self._config = config
        self.state = RiskState()

    def set_daily_baseline(self, equity: float) -> None:
        """Call at market open to set the daily starting equity."""
        self.state.daily_start_equity = equity
        self.state.trading_halted = False
        self.state.halt_reason = ""
        log.info("risk.daily_baseline", equity=equity)

    def update_daily_pnl(self, account: Account) -> None:
        """Update daily P&L tracking."""
        if self.state.daily_start_equity > 0:
            self.state.daily_pnl = account.equity - self.state.daily_start_equity
            self.state.daily_pnl_pct = self.state.daily_pnl / self.state.daily_start_equity

            # Check daily drawdown limit
            if self.state.daily_pnl_pct < -self._config.max_daily_drawdown_pct:
                self.state.trading_halted = True
                self.state.halt_reason = (
                    f"Daily drawdown {self.state.daily_pnl_pct:.1%} "
                    f"exceeds limit -{self._config.max_daily_drawdown_pct:.1%}"
                )
                log.warning("risk.trading_halted", reason=self.state.halt_reason)

    def filter_signals(
        self,
        signals: list[Signal],
        account: Account,
        positions: list[Position],
    ) -> list[Order]:
        """Apply risk rules to signals and return approved orders."""
        if self.state.trading_halted:
            log.warning("risk.halted_skip", reason=self.state.halt_reason)
            return []

        orders: list[Order] = []
        current_position_count = len(positions)
        now = datetime.now(timezone.utc)
        cfg = self._config

        for signal in signals:
            # Skip holds
            if signal.action == "hold":
                continue

            # Check cooldown
            if signal.ticker in self.state.cooldowns:
                if now < self.state.cooldowns[signal.ticker]:
                    log.debug("risk.cooldown_skip", ticker=signal.ticker)
                    continue

            if signal.action == "buy":
                # Max positions check
                if current_position_count >= cfg.max_positions:
                    log.debug("risk.max_positions", ticker=signal.ticker)
                    continue

                # Position sizing: max X% of portfolio
                max_value = account.equity * cfg.max_position_pct
                # We need a price to calculate qty — use buying power as sanity check
                if max_value > account.buying_power:
                    max_value = account.buying_power * 0.95  # 5% buffer

                order = Order(
                    ticker=signal.ticker,
                    side=OrderSide.BUY,
                    qty=0,  # Will be calculated with current price
                    order_type=OrderType.LIMIT,
                )
                order._max_value = max_value  # type: ignore[attr-defined]
                order._signal = signal  # type: ignore[attr-defined]
                orders.append(order)
                current_position_count += 1

            elif signal.action == "sell":
                order = Order(
                    ticker=signal.ticker,
                    side=OrderSide.SELL,
                    qty=0,  # close_position handles qty
                    order_type=OrderType.LIMIT,
                )
                order._signal = signal  # type: ignore[attr-defined]
                orders.append(order)

        log.info("risk.filtered", approved=len(orders))
        return orders

    def check_exits(
        self, positions: list[Position]
    ) -> list[str]:
        """Check stop-loss and take-profit for all positions.

        Returns list of tickers to close.
        """
        exits: list[str] = []
        cfg = self._config

        for pos in positions:
            pnl_pct = pos.unrealized_pnl_pct

            if pnl_pct <= -cfg.stop_loss_pct:
                log.warning(
                    "risk.stop_loss",
                    ticker=pos.ticker,
                    pnl_pct=f"{pnl_pct:.2%}",
                    threshold=f"-{cfg.stop_loss_pct:.2%}",
                )
                exits.append(pos.ticker)

            elif pnl_pct >= cfg.take_profit_pct:
                log.info(
                    "risk.take_profit",
                    ticker=pos.ticker,
                    pnl_pct=f"{pnl_pct:.2%}",
                    threshold=f"+{cfg.take_profit_pct:.2%}",
                )
                exits.append(pos.ticker)

        # Set cooldowns for exited positions
        now = datetime.now(timezone.utc)
        cooldown_until = now + timedelta(minutes=cfg.cooldown_minutes)
        for ticker in exits:
            self.state.cooldowns[ticker] = cooldown_until

        if exits:
            log.info("risk.exits", tickers=exits)

        return exits
