# 📈 News Trading Bot

Bot de trading automatisé sur actions US (NYSE/NASDAQ) basé sur l'analyse de sentiment des news financières.

## Architecture

```
News Sources (Finnhub + RSS)
    → Sentiment Analysis (FinBERT)
    → Signal Generation (sentiment + technicals)
    → Risk Management (stop-loss, position sizing)
    → Order Execution (Alpaca paper trading)
    → Discord Alerts
```

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env with your API keys

# 3. Run tests
pytest

# 4. Run
python -m src.main
# or: trading-bot
```

## Required API Keys

| Service | Purpose | Free? | Get it |
|---------|---------|:---:|--------|
| **Alpaca** | Paper trading broker | ✅ | [alpaca.markets](https://alpaca.markets) |
| **Finnhub** | Financial news | ✅ (60 calls/min) | [finnhub.io](https://finnhub.io) |
| **Discord Webhook** | Trade alerts | ✅ | Server Settings → Integrations |

## Configuration

All settings configurable via `.env` or environment variables:

- **Strategy**: signal weights, buy/sell thresholds
- **Risk**: stop-loss (-2%), take-profit (+4%), max positions (10), max drawdown (-5%)
- **Universe**: which tickers to track (default: 15 large caps)
- **Cycle interval**: how often to run (default: 2 minutes)
