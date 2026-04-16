import asyncio
from src.config import Settings
from src.broker.alpaca_broker import AlpacaBroker

async def main():
    settings = Settings()
    broker = AlpacaBroker(api_key=settings.broker.api_key, secret_key=settings.broker.secret_key)
    await broker.connect()
    pos = await broker.get_positions()
    print(f"Positions sur Alpaca: {len(pos)}")
    for p in pos:
        print(f"{p.ticker}: {p.qty} shares (PnL: {p.unrealized_pnl})")

asyncio.run(main())
