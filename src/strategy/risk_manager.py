"""Risk manager — position sizing, stop-loss, drawdown limits."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import structlog

from src.broker.interface import Account, Order, OrderSide, OrderType, Position
from src.config import StrategySettings
from src.strategy.signals import Signal

log = structlog.get_logger()


def calculate_risk_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float,
    max_notional: float,
) -> int:
    """Calculate whole-share quantity from account risk and notional limits.

    Returns 0 for invalid inputs or when even one share would violate constraints.
    """
    if equity <= 0 or entry_price <= 0 or risk_pct <= 0 or max_notional <= 0:
        return 0

    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return 0

    risk_budget = equity * risk_pct
    risk_limited_qty = int(risk_budget // risk_per_share)
    notional_limited_qty = int(max_notional // entry_price)
    return max(0, min(risk_limited_qty, notional_limited_qty))


@dataclass
class RiskState:
    """Tracks risk management state across cycles."""

    daily_start_equity: float = 0.0
    baseline_date: date | None = None
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    trading_halted: bool = False
    halt_reason: str = ""
    # Cooldown: ticker -> earliest next trade time
    cooldowns: dict[str, datetime] = field(default_factory=dict)
    # Trailing stop high-water mark: ticker -> highest observed current price
    position_highs: dict[str, float] = field(default_factory=dict)
    # UTC date -> ticker -> approved buy count
    symbol_buy_counts_by_utc_date: dict[date, dict[str, int]] = field(default_factory=dict)
    # Ticker -> most recent approved buy timestamp
    last_symbol_buy_at: dict[str, datetime] = field(default_factory=dict)


class RiskManager:
    """Enforces risk rules before order execution."""

    def __init__(self, config: StrategySettings):
        self._config = config
        self.state = RiskState()

    def set_daily_baseline(self, equity: float, baseline_date: date | None = None) -> None:
        """Call at market open to set the daily starting equity."""
        baseline_date = baseline_date or datetime.now(UTC).date()
        self.state.daily_start_equity = equity
        self.state.baseline_date = baseline_date
        self.state.daily_pnl = 0.0
        self.state.daily_pnl_pct = 0.0
        self.state.trading_halted = False
        self.state.halt_reason = ""
        log.info("risk.daily_baseline", equity=equity, baseline_date=str(baseline_date))

    def ensure_daily_baseline(self, account: Account, today: date | None = None) -> None:
        """Reset the baseline once per day before evaluating daily drawdown."""
        today = today or datetime.now(UTC).date()
        if self.state.baseline_date != today or self.state.daily_start_equity <= 0:
            self.set_daily_baseline(account.equity, today)

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
        now = datetime.now(UTC)
        today = now.date()
        cfg = self._config
        buy_count = 0
        today_symbol_buy_counts = self.state.symbol_buy_counts_by_utc_date.setdefault(today, {})

        for signal in signals:
            if len(orders) >= cfg.max_orders_per_cycle:
                log.debug("risk.max_orders_per_cycle", ticker=signal.ticker)
                break

            # Skip holds
            if signal.action == "hold":
                continue

            # Check cooldown
            if signal.ticker in self.state.cooldowns:
                if now < self.state.cooldowns[signal.ticker]:
                    log.debug("risk.cooldown_skip", ticker=signal.ticker)
                    continue

            if signal.action == "buy":
                if buy_count >= cfg.max_buys_per_cycle:
                    log.debug("risk.max_buys_per_cycle", ticker=signal.ticker)
                    continue

                if (
                    today_symbol_buy_counts.get(signal.ticker, 0)
                    >= cfg.max_buys_per_symbol_per_day
                ):
                    log.debug("risk.max_buys_per_symbol_per_day", ticker=signal.ticker)
                    continue

                last_symbol_buy_at = self.state.last_symbol_buy_at.get(signal.ticker)
                if last_symbol_buy_at is not None:
                    next_symbol_buy_at = last_symbol_buy_at + timedelta(
                        minutes=cfg.min_minutes_between_symbol_buys
                    )
                    if now < next_symbol_buy_at:
                        log.debug("risk.symbol_buy_cooldown", ticker=signal.ticker)
                        continue

                # Max positions check
                if current_position_count >= cfg.max_positions:
                    log.debug("risk.max_positions", ticker=signal.ticker)
                    continue

                # Position sizing: cap notional, then optionally tighten by ATR stop distance.
                max_value = account.equity * cfg.max_position_pct
                # We need a price to calculate qty — use buying power as sanity check
                if max_value > account.buying_power:
                    max_value = account.buying_power * 0.95  # 5% buffer

                features = getattr(signal, "features", {}) or {}
                entry_price = _as_positive_float(features.get("last_price"))
                atr = _as_positive_float(features.get("atr_14"))
                risk_qty = None
                if entry_price is not None and atr is not None and cfg.atr_stop_mult > 0:
                    stop_price = entry_price - (atr * cfg.atr_stop_mult)
                    risk_qty = calculate_risk_position_size(
                        equity=account.equity,
                        entry_price=entry_price,
                        stop_price=stop_price,
                        risk_pct=cfg.risk_per_trade_pct,
                        max_notional=max_value,
                    )

                order = Order(
                    ticker=signal.ticker,
                    side=OrderSide.BUY,
                    qty=0,  # Will be calculated with current price
                    order_type=OrderType.LIMIT,
                )
                order._max_value = max_value  # type: ignore[attr-defined]
                order._risk_qty = risk_qty  # type: ignore[attr-defined]
                order._signal = signal  # type: ignore[attr-defined]
                orders.append(order)
                current_position_count += 1
                buy_count += 1
                today_symbol_buy_counts[signal.ticker] = (
                    today_symbol_buy_counts.get(signal.ticker, 0) + 1
                )
                self.state.last_symbol_buy_at[signal.ticker] = now
                self.state.cooldowns[signal.ticker] = now + timedelta(minutes=cfg.cooldown_minutes)

            elif signal.action == "sell":
                order = Order(
                    ticker=signal.ticker,
                    side=OrderSide.SELL,
                    qty=0,  # close_position handles qty
                    order_type=OrderType.LIMIT,
                )
                order._signal = signal  # type: ignore[attr-defined]
                orders.append(order)
                self.state.cooldowns[signal.ticker] = now + timedelta(minutes=cfg.cooldown_minutes)

        log.info("risk.filtered", approved=len(orders))
        return orders

    def check_exits(self, positions: list[Position]) -> list[str]:
        """Check stop-loss and take-profit for all positions.

        Returns list of tickers to close.
        """
        exits: list[str] = []
        cfg = self._config

        current_tickers = {pos.ticker for pos in positions}
        for stale_ticker in set(self.state.position_highs) - current_tickers:
            self.state.position_highs.pop(stale_ticker, None)

        for pos in positions:
            pnl_pct = pos.unrealized_pnl_pct
            previous_high = self.state.position_highs.get(pos.ticker, pos.current_price)
            high = max(previous_high, pos.current_price)
            self.state.position_highs[pos.ticker] = high
            trailing_stop_price = high * (1 - cfg.trailing_stop_pct)

            if pnl_pct <= -cfg.stop_loss_pct:
                log.warning(
                    "risk.stop_loss",
                    ticker=pos.ticker,
                    pnl_pct=f"{pnl_pct:.2%}",
                    threshold=f"-{cfg.stop_loss_pct:.2%}",
                )
                exits.append(pos.ticker)

            elif pos.current_price <= trailing_stop_price:
                log.warning(
                    "risk.trailing_stop",
                    ticker=pos.ticker,
                    current_price=pos.current_price,
                    high=high,
                    threshold=trailing_stop_price,
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
        now = datetime.now(UTC)
        cooldown_until = now + timedelta(minutes=cfg.cooldown_minutes)
        for ticker in exits:
            self.state.cooldowns[ticker] = cooldown_until

        if exits:
            log.info("risk.exits", tickers=exits)

        return exits


def _as_positive_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None
