# 📰 News Trading Bot

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-157%20passing-brightgreen?logo=pytest)
![License](https://img.shields.io/badge/license-MIT-green)
![Paper Trading](https://img.shields.io/badge/broker-Alpaca%20Paper-orange?logo=alpaca)
![FinBERT](https://img.shields.io/badge/NLP-FinBERT-purple?logo=huggingface)
![Discord](https://img.shields.io/badge/alerts-Discord-5865F2?logo=discord)

Automated algorithmic trading bot for US equities that combines **real-time news sentiment analysis** with **technical indicators** to generate trade signals. Runs on [Alpaca](https://alpaca.markets/) paper trading with zero commissions.

The bot fetches financial news every 2 minutes during NYSE market hours, runs [FinBERT](https://huggingface.co/ProsusAI/finbert) sentiment analysis, blends it with technical momentum, applies risk management rules, and executes trades — fully autonomous.

> **⚠️ Paper trading only.** This bot is designed for paper trading and educational purposes. The IBKR live broker integration is planned but not yet implemented.

---

## How It Works

```
                          ┌──────────────────┐
                          │   News Sources    │
                          │  Finnhub + RSS    │
                          └────────┬─────────┘
                                   │ headlines + summaries
                                   ▼
                          ┌──────────────────┐
                          │    FinBERT NLP    │
                          │  sentiment score  │
                          │   [-1.0 … +1.0]  │
                          └────────┬─────────┘
                                   │ per-ticker sentiment
                                   ▼
┌──────────────────┐      ┌──────────────────┐
│ Technical Indic. │─────▶│ Signal Generator │
│ RSI MACD Bolling.│      │ weighted composite│
│ VWAP Volume      │      │  buy / sell / hold│
└──────────────────┘      └────────┬─────────┘
                                   │ ranked signals
                                   ▼
                          ┌──────────────────┐
                          │  Risk Manager    │
                          │ sizing, stops,   │
                          │ drawdown limits  │
                          └────────┬─────────┘
                                   │ approved orders
                                   ▼
                          ┌──────────────────┐       ┌─────────┐
                          │  Alpaca Broker   │──────▶│ Discord │
                          │  paper trading   │       │ alerts  │
                          └──────────────────┘       └─────────┘
```

### Signal Formula

Each ticker gets a **composite score** from -1.0 to +1.0:

| Component | Weight | Source |
|-----------|--------|--------|
| Sentiment | 40% | FinBERT score (time-decayed, 30-min half-life) |
| News Velocity | 20% | Articles/hour (normalized, capped) |
| Technical Momentum | 25% | RSI + MACD histogram + Bollinger position |
| Volume Anomaly | 15% | Current volume vs 20-bar average |

**Thresholds:** Buy if score > 0.3 · Sell if score < -0.2 · Hold otherwise

---

## Quick Start

### Prerequisites

- Python ≥ 3.11
- [Alpaca](https://alpaca.markets/) account (free — paper trading)
- [Finnhub](https://finnhub.io/) API key (free tier, optional but recommended)
- Discord webhook URL (optional, for alerts)

### Install

```bash
git clone https://github.com/gzm-lab/news-trading-bot.git
cd news-trading-bot
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
chmod 600 .env   # restrict permissions — file contains secrets
```

Edit `.env` with your API keys:

```env
ALPACA_API_KEY=***
ALPACA_SECRET_KEY=your-a...cret
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2

FINNHUB_API_KEY=***        # optional — RSS still works without it

DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...   # optional
```

### Run

```bash
# Activate venv
source .venv/bin/activate

# Run the bot
python -m src.main
```

The bot will:
1. Connect to Alpaca and load FinBERT (~500MB model, downloaded on first run)
2. Check market hours — if NYSE is closed, sleep until next open
3. Start the 2-minute trading cycle

### Run Tests

```bash
python -m pytest tests/ -v
```

157 tests covering every component.

---

## Architecture

```
src/
├── main.py                  # Orchestrator — setup, run loop, cycle logic
├── config.py                # Pydantic-settings configuration (env + .env)
│
├── broker/
│   ├── interface.py         # Abstract broker + Order/Position/Account dataclasses
│   ├── alpaca_broker.py     # Alpaca implementation (paper + live)
│   └── ibkr_broker.py       # IBKR placeholder (future)
│
├── news/
│   ├── base.py              # NewsItem dataclass + NewsSource ABC
│   ├── finnhub_source.py    # Finnhub company news API
│   ├── rss_source.py        # RSS/Atom feeds (Yahoo Finance, CNBC)
│   └── aggregator.py        # Multi-source dedup (SHA256 fingerprints + TTL cache)
│
├── sentiment/
│   ├── finbert.py           # FinBERT wrapper (GPU/CPU, batch inference)
│   ├── preprocessor.py      # Text cleaning for financial text
│   └── scorer.py            # Per-ticker aggregation (time-decay, velocity)
│
├── market/
│   └── indicators.py        # RSI, MACD, Bollinger, VWAP, volume anomaly
│
├── strategy/
│   ├── signals.py           # Composite signal generation
│   └── risk_manager.py      # Position sizing, stops, drawdown circuit breaker
│
├── storage/
│   ├── database.py          # SQLite via SQLAlchemy
│   └── models.py            # NewsArticle, TradeLog, CycleLog, PortfolioSnapshot
│
└── alerts/
    └── discord.py           # Discord webhook notifications
```

### The Trading Cycle

Every 120 seconds during market hours, `_cycle()` runs these 9 steps:

1. **Account snapshot** — Get equity, cash, buying power from Alpaca
2. **Risk check** — Update daily P&L, halt trading if drawdown exceeds -5%
3. **Exit scan** — Check all positions for stop-loss (-2%) or take-profit (+4%), close + alert
4. **News fetch** — Pull latest articles from Finnhub + RSS, deduplicate
5. **Sentiment scoring** — Run FinBERT on new headlines, aggregate per ticker (time-decayed)
6. **Technical data** — Fetch 1-hour OHLCV bars for tickers with fresh sentiment
7. **Signal generation** — Compute composite score (sentiment + velocity + momentum + volume)
8. **Risk filtering** — Apply position limits, cooldowns, sizing (max 5% portfolio per trade)
9. **Execution** — Place orders on Alpaca, log to DB, send Discord alerts

### Market Hours Handling

The bot is smart about when to sleep:

| Scenario | Behavior |
|----------|----------|
| **Weekday 9:15–16:00 ET** | Active trading (pre-market buffer for news) |
| **Weekday before 9:15 ET** | Sleeps until 9:15 ET |
| **Weekday after 16:00 ET** | Sleeps until next day 9:15 ET |
| **Weekend** | Sleeps until Monday 9:15 ET |
| **NYSE holiday** | Detected via Alpaca clock API, sleeps 5 min and retries |

Sleep is computed locally (no API calls) and done in 5-minute interruptible chunks.

---

## Configuration Reference

All settings can be overridden via environment variables or `.env` file.

### Broker

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPACA_API_KEY` | — | Alpaca API key (required) |
| `ALPACA_SECRET_KEY` | — | Alpaca secret key (required) |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` | Paper or live endpoint |

### News Sources

| Variable | Default | Description |
|----------|---------|-------------|
| `FINNHUB_API_KEY` | — | Finnhub key (optional — RSS works without it) |
| `NEWS_FETCH_INTERVAL` | `120` | Seconds between news fetches |
| `NEWS_MAX_AGE_MINUTES` | `60` | Ignore articles older than this |

Three RSS feeds are configured by default: Yahoo Finance (top tickers), CNBC Top News, CNBC Markets.

### Sentiment

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTIMENT_MODEL_NAME` | `ProsusAI/finbert` | HuggingFace model |
| `SENTIMENT_BATCH_SIZE` | `16` | Inference batch size |
| `SENTIMENT_MAX_LENGTH` | `512` | Max token length |

### Strategy

| Variable | Default | Description |
|----------|---------|-------------|
| `STRATEGY_W_SENTIMENT` | `0.40` | Sentiment weight in composite |
| `STRATEGY_W_NEWS_VELOCITY` | `0.20` | News velocity weight |
| `STRATEGY_W_TECHNICAL` | `0.25` | Technical momentum weight |
| `STRATEGY_W_VOLUME` | `0.15` | Volume anomaly weight |
| `STRATEGY_BUY_THRESHOLD` | `0.3` | Min score to buy |
| `STRATEGY_SELL_THRESHOLD` | `-0.2` | Max score to sell |
| `STRATEGY_MAX_POSITION_PCT` | `0.05` | Max 5% of portfolio per position |
| `STRATEGY_STOP_LOSS_PCT` | `0.02` | Stop-loss at -2% |
| `STRATEGY_TAKE_PROFIT_PCT` | `0.04` | Take-profit at +4% |
| `STRATEGY_MAX_POSITIONS` | `10` | Max concurrent positions |
| `STRATEGY_MAX_DAILY_DRAWDOWN_PCT` | `0.05` | Halt trading at -5% daily loss |
| `STRATEGY_COOLDOWN_MINUTES` | `30` | Wait after exiting a ticker |
| `STRATEGY_BLACKOUT_MINUTES` | `15` | No trading near open/close |

### General

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level |
| `CYCLE_INTERVAL` | `120` | Seconds between trading cycles |
| `DB_PATH` | `data/trading.db` | SQLite database path |

### Default Universe

15 large-cap US stocks tracked by default:

```
AAPL  MSFT  GOOGL  AMZN  TSLA  NVDA  META  NFLX  AMD  CRM  JPM  V  JNJ  UNH  PG
```

Override via `UNIVERSE` env var (comma-separated).

---

## Risk Management

The bot implements multiple layers of protection:

### Position-Level
- **Stop-loss:** Auto-close at -2% unrealized P&L
- **Take-profit:** Auto-close at +4% unrealized P&L
- **Cooldown:** 30-minute wait after exiting a ticker (prevents churn)
- **Max sizing:** 5% of portfolio equity per position

### Portfolio-Level
- **Max positions:** 10 concurrent positions
- **Daily drawdown circuit breaker:** Trading halts if daily P&L drops below -5%
- **Buying power cap:** Orders limited to 95% of available buying power

### Discord Alerts

The bot sends real-time Discord notifications for every significant event:

| Event | Emoji | Details |
|-------|-------|---------|
| Buy execution | 🟢 | Ticker, price, shares, composite score, reasoning |
| Sell execution | 🔴 | Ticker, price, shares, P&L, reason |
| Stop-loss triggered | 🛑 | Ticker, loss %, exit price |
| Take-profit triggered | 🎯 | Ticker, gain %, exit price |
| Trading halted | ⚠️ | Daily drawdown threshold hit |
| End-of-day summary | 📊 | Equity, cash, open positions, daily P&L |

---

## Technical Indicators

All indicators are implemented from scratch in NumPy/Pandas (no `pandas-ta` dependency):

| Indicator | Parameters | Usage in Signal |
|-----------|-----------|-----------------| 
| **RSI** | 14-period EWM | 40% of momentum score (normalized from 50-center) |
| **MACD** | 12/26/9 | 35% of momentum score (histogram, normalized by σ) |
| **Bollinger Bands** | 20-period, 2σ | 25% of momentum score (position within bands) |
| **VWAP** | Session | Reference price level |
| **Volume Anomaly** | 20-bar SMA, 2× threshold | Volume score: ratio vs average (0→1 mapping) |

---

## Database Schema

SQLite stores all operational data in `data/trading.db`:

| Table | Purpose |
|-------|---------|
| `news_articles` | Every fetched article with fingerprint, sentiment score, source |
| `trade_log` | Every order with signal score, reasoning, fill price |
| `cycle_log` | Per-cycle metrics: news count, signals, orders, portfolio value, duration |
| `portfolio_snapshots` | Periodic equity/cash/position snapshots for performance tracking |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `alpaca-py` | Broker API (trading + market data) |
| `finnhub-python` | News source |
| `feedparser` | RSS parsing |
| `transformers` + `torch` | FinBERT NLP model |
| `pandas` + `numpy` | Data processing + indicators |
| `sqlalchemy` + `aiosqlite` | Async SQLite storage |
| `pydantic-settings` | Typed configuration |
| `discord-webhook` | Trade alerts |
| `structlog` | Structured logging |
| `cachetools` | News deduplication cache |
| `aiohttp` | Async HTTP |

Dev: `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`

---

## Roadmap

- [x] Phase 1 — Project scaffold + config
- [x] Phase 2 — News pipeline (Finnhub + RSS + dedup)
- [x] Phase 3 — Sentiment engine (FinBERT + scorer)
- [x] Phase 4 — Strategy (signals + risk management + indicators)
- [x] Phase 5 — Orchestrator (main loop + market hours + alerts)
- [ ] Phase 6 — Performance dashboard (web UI)
- [ ] Phase 7 — IBKR live broker integration
- [ ] Phase 8 — ML signal tuning (backtest + optimize weights)

---

## License

MIT
