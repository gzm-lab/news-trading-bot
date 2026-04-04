"""Alpaca paper trading broker implementation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pandas as pd
import structlog

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from src.broker.interface import (
    BrokerInterface,
    Account,
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

        return Order(
            ticker=order.ticker,
            side=order.side,
            qty=order.qty,
            order_type=order.order_type,
            limit_price=order.limit_price,
            id=str(raw.id),
            status=OrderStatus(raw.status.value) if hasattr(raw.status, "value") else OrderStatus.PENDING,
            filled_price=float(raw.filled_avg_price) if raw.filled_avg_price else None,
            filled_at=raw.filled_at,
        )

    async def close_position(self, ticker: str) -> Order | None:
        assert self._trading_client is not None
        try:
            raw = await asyncio.to_thread(
                self._trading_client.close_position, ticker
            )
            return Order(
                ticker=ticker,
                side=OrderSide.SELL,
                qty=int(raw.qty) if hasattr(raw, "qty") else 0,
                order_type=OrderType.MARKET,
                id=str(raw.id),
                status=OrderStatus(raw.status.value) if hasattr(raw.status, "value") else OrderStatus.PENDING,
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
            "5Min": TimeFrame(5, "Min"),
            "15Min": TimeFrame(15, "Min"),
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
        bars = raw[ticker]

        data = []
        for bar in bars:
            data.append({
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            })

        return pd.DataFrame(data)

    async def get_latest_price(self, ticker: str) -> float:
        assert self._data_client is not None
        request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        raw = await asyncio.to_thread(self._data_client.get_stock_latest_quote, request)
        quote = raw[ticker]
        return float(quote.ask_price or quote.bid_price or 0.0)

    async def is_market_open(self) -> bool:
        assert self._trading_client is not None
        clock = await asyncio.to_thread(self._trading_client.get_clock)
        return clock.is_open
