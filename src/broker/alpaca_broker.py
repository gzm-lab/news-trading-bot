"""Alpaca paper trading broker implementation."""

from __future__ import annotations

import asyncio

import pandas as pd
import structlog
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaSide
from alpaca.trading.enums import TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest

from src.broker.interface import (
    Account,
    BrokerInterface,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

log = structlog.get_logger()


class AlpacaBroker(BrokerInterface):
    """Alpaca Markets broker — paper and live trading."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self._api_key = api_key
        self._secret_key = secret_key
        self._paper = paper
        self._trading_client: TradingClient | None = None
        self._data_client: StockHistoricalDataClient | None = None

    async def connect(self) -> None:
        """Initialize Alpaca API clients."""
        self._trading_client = TradingClient(
            api_key=self._api_key, secret_key=self._secret_key, paper=self._paper
        )
        self._data_client = StockHistoricalDataClient(
            api_key=self._api_key, secret_key=self._secret_key
        )
        log.info("alpaca.connected", paper=self._paper)

    async def get_account(self) -> Account:
        assert self._trading_client is not None
        raw = await asyncio.to_thread(self._trading_client.get_account)
        return Account(
            equity=float(raw.equity),
            cash=float(raw.cash),
            buying_power=float(raw.buying_power),
            portfolio_value=float(raw.portfolio_value),
        )

    async def get_positions(self) -> list[Position]:
        assert self._trading_client is not None
        raw_positions = await asyncio.to_thread(self._trading_client.get_all_positions)
        positions = []
        for rp in raw_positions:
            positions.append(
                Position(
                    ticker=rp.symbol,
                    qty=int(rp.qty),
                    avg_entry_price=float(rp.avg_entry_price),
                    current_price=float(rp.current_price),
                    market_value=float(rp.market_value),
                    unrealized_pnl=float(rp.unrealized_pl),
                    unrealized_pnl_pct=float(rp.unrealized_plpc),
                )
            )
        return positions

    async def place_order(self, order: Order) -> Order:
        assert self._trading_client is not None

        await self._cancel_conflicting_open_orders(order.ticker, order.side)

        side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL

        if order.order_type == OrderType.LIMIT and order.limit_price:
            request = LimitOrderRequest(
                symbol=order.ticker,
                qty=order.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=order.limit_price,
            )
        else:
            request = MarketOrderRequest(
                symbol=order.ticker,
                qty=order.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )

        raw = await asyncio.to_thread(self._trading_client.submit_order, request)

        return self._order_from_raw(raw, fallback=order)

    async def get_order(self, order_id: str) -> Order | None:
        """Fetch the latest broker state for an order by id."""
        assert self._trading_client is not None
        try:
            raw = await asyncio.to_thread(self._trading_client.get_order_by_id, order_id)
        except Exception as e:
            log.warning("alpaca.order_fetch_failed", order_id=order_id, error=str(e))
            return None
        return self._order_from_raw(raw)

    async def get_open_orders(self) -> list[Order]:
        """Fetch all open orders from Alpaca."""
        assert self._trading_client is not None
        try:
            request = GetOrdersRequest(status="open")
            raw_orders = await asyncio.to_thread(self._trading_client.get_orders, request)
        except Exception as e:
            log.warning("alpaca.open_orders_failed", error=str(e))
            return []
        return [self._order_from_raw(raw) for raw in raw_orders]

    def _order_from_raw(self, raw, fallback: Order | None = None) -> Order:
        """Convert an Alpaca SDK order model into the internal Order dataclass."""
        raw_side = getattr(raw, "side", None)
        side_value = raw_side.value if hasattr(raw_side, "value") else str(raw_side)
        if side_value not in {OrderSide.BUY.value, OrderSide.SELL.value} and fallback is not None:
            side = fallback.side
        else:
            side = OrderSide(side_value)

        raw_type = getattr(raw, "type", None) or getattr(raw, "order_type", None)
        type_value = raw_type.value if hasattr(raw_type, "value") else str(raw_type)
        if type_value not in {OrderType.MARKET.value, OrderType.LIMIT.value} and fallback is not None:
            order_type = fallback.order_type
        else:
            order_type = OrderType(type_value)

        raw_status = getattr(raw, "status", None)
        status_value = raw_status.value if hasattr(raw_status, "value") else str(raw_status)
        try:
            status = OrderStatus(status_value)
        except ValueError:
            log.warning("alpaca.unknown_order_status", status=status_value)
            status = OrderStatus.PENDING

        raw_symbol = getattr(raw, "symbol", None)
        raw_qty = getattr(raw, "qty", None)
        raw_limit_price = getattr(raw, "limit_price", None)

        return Order(
            ticker=str(raw_symbol or (fallback.ticker if fallback else "")),
            side=side,
            qty=int(float(raw_qty if raw_qty is not None else fallback.qty)),
            order_type=order_type,
            limit_price=float(raw_limit_price) if raw_limit_price else (fallback.limit_price if fallback else None),
            id=str(raw.id),
            status=status,
            filled_price=float(raw.filled_avg_price) if raw.filled_avg_price else None,
            filled_at=getattr(raw, "filled_at", None),
        )

    async def _cancel_conflicting_open_orders(self, ticker: str, side: OrderSide) -> None:
        """Cancel open orders for the same symbol/side before submitting a replacement.

        Alpaca reserves quantity for open sell orders. If the bot emits a fresh sell
        signal while a prior limit sell is still pending, submitting another sell for
        the full position fails with `insufficient qty available for order`.
        """
        assert self._trading_client is not None
        try:
            request = GetOrdersRequest(status="open", symbols=[ticker])
            open_orders = await asyncio.to_thread(self._trading_client.get_orders, request)
        except Exception as e:
            log.warning("alpaca.open_orders_failed", ticker=ticker, error=str(e))
            return

        for open_order in open_orders:
            raw_side = getattr(open_order, "side", None)
            open_side = raw_side.value if hasattr(raw_side, "value") else str(raw_side)
            if open_side != side.value:
                continue
            order_id = getattr(open_order, "id", None)
            try:
                await asyncio.to_thread(self._trading_client.cancel_order_by_id, order_id)
                log.info(
                    "alpaca.open_order_cancelled",
                    ticker=ticker,
                    side=side.value,
                    order_id=order_id,
                )
            except Exception as e:
                log.warning(
                    "alpaca.open_order_cancel_failed",
                    ticker=ticker,
                    side=side.value,
                    order_id=order_id,
                    error=str(e),
                )

    async def close_position(self, ticker: str) -> Order | None:
        assert self._trading_client is not None
        try:
            raw = await asyncio.to_thread(self._trading_client.close_position, ticker)
            return Order(
                ticker=ticker,
                side=OrderSide.SELL,
                qty=int(raw.qty) if hasattr(raw, "qty") else 0,
                order_type=OrderType.MARKET,
                id=str(raw.id),
                status=OrderStatus(raw.status.value)
                if hasattr(raw.status, "value")
                else OrderStatus.PENDING,
                filled_price=float(raw.filled_avg_price) if raw.filled_avg_price else None,
            )
        except Exception as e:
            log.warning("alpaca.close_position_failed", ticker=ticker, error=str(e))
            return None

    async def get_bars(
        self, ticker: str, timeframe: str = "1Hour", limit: int = 50
    ) -> pd.DataFrame:
        assert self._data_client is not None

        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        tf = tf_map.get(timeframe, TimeFrame.Hour)

        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=tf,
            limit=limit,
        )
        raw = await asyncio.to_thread(self._data_client.get_stock_bars, request)
        if not raw or not hasattr(raw, "data") or ticker not in raw.data:
            return pd.DataFrame()

        bars = raw.data[ticker]

        data = []
        for bar in bars:
            data.append(
                {
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                }
            )

        return pd.DataFrame(data)

    async def get_latest_price(self, ticker: str) -> float:
        assert self._data_client is not None
        request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        try:
            raw = await asyncio.to_thread(self._data_client.get_stock_latest_quote, request)
            quote = raw.get(ticker) if hasattr(raw, "get") else raw[ticker]
            ask = float(getattr(quote, "ask_price", 0) or 0)
            bid = float(getattr(quote, "bid_price", 0) or 0)
            if ask > 0 and bid > 0:
                return (ask + bid) / 2
            return ask or bid or 0.0
        except Exception as e:
            log.warning("alpaca.latest_price_failed", ticker=ticker, error=str(e))
            return 0.0

    async def is_market_open(self) -> bool:
        assert self._trading_client is not None
        clock = await asyncio.to_thread(self._trading_client.get_clock)
        return clock.is_open
