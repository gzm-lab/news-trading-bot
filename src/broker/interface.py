"""Abstract broker interface + shared data classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trade order."""

    ticker: str
    side: OrderSide
    qty: int
    order_type: OrderType
    limit_price: float | None = None
    id: str | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float | None = None
    filled_at: datetime | None = None


@dataclass
class Position:
    """Represents an open position."""

    ticker: str
    qty: int
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass
class Account:
    """Broker account summary."""

    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0


class BrokerInterface(ABC):
    """Abstract broker interface — all brokers must implement this."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def get_account(self) -> Account: ...

    @abstractmethod
    async def get_positions(self) -> list[Position]: ...

    @abstractmethod
    async def place_order(self, order: Order) -> Order: ...

    @abstractmethod
    async def close_position(self, ticker: str) -> Order | None: ...

    @abstractmethod
    async def get_bars(
        self, ticker: str, timeframe: str = "1Hour", limit: int = 50
    ) -> pd.DataFrame: ...

    @abstractmethod
    async def get_latest_price(self, ticker: str) -> float: ...

    @abstractmethod
    async def is_market_open(self) -> bool: ...
