"""Interactive Brokers implementation — placeholder for Phase 5."""

from __future__ import annotations

import pandas as pd

from src.broker.interface import (
    BrokerInterface,
    Account,
    Order,
    Position,
)


class IBKRBroker(BrokerInterface):
    """IBKR broker — not yet implemented."""

    async def connect(self) -> None:
        raise NotImplementedError("IBKR support coming in Phase 5")

    async def get_account(self) -> Account:
        raise NotImplementedError

    async def get_positions(self) -> list[Position]:
        raise NotImplementedError

    async def place_order(self, order: Order) -> Order:
        raise NotImplementedError

    async def close_position(self, ticker: str) -> Order | None:
        raise NotImplementedError

    async def get_bars(self, ticker: str, timeframe: str = "1Hour", limit: int = 50) -> pd.DataFrame:
        raise NotImplementedError

    async def get_latest_price(self, ticker: str) -> float:
        raise NotImplementedError

    async def is_market_open(self) -> bool:
        raise NotImplementedError
