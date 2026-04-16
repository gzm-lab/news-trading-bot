import asyncio
from src.config import Settings
from src.broker.alpaca_broker import AlpacaBroker

async def main():
    settings = Settings()
    broker = AlpacaBroker(api_key=settings.broker.api_key, secret_key=settings.broker.secret_key)
    await broker.connect()
    try:
        bars = await broker.get_bars("AAPL", timeframe="1Hour", limit=50)
        print(f"Data for AAPL: {len(bars)} rows")
        if not bars.empty:
            print(bars.head())
    except Exception as e:
        print(f"ERROR: {e}")

asyncio.run(main())
