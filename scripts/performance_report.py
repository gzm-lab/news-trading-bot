#!/usr/bin/env python3
"""Generate a lightweight performance/attribution report from the trading SQLite DB."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_DB = "data/trading.db"


def _rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return list(conn.execute(sql, params))


def _since_arg(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).replace(tzinfo=None).isoformat(sep=" ")


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bucket_score(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.70:
        return ">=0.70"
    if score >= 0.50:
        return "0.50-0.69"
    if score >= 0.35:
        return "0.35-0.49"
    if score <= -0.50:
        return "<=-0.50"
    if score <= -0.35:
        return "-0.49--0.35"
    return "mid"


def _parse_features(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _print_table(title: str, headers: list[str], rows: list[list[Any]]) -> None:
    print(f"\n## {title}")
    if not rows:
        print("No data")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = " | ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def build_report(db_path: str, days: int) -> None:
    path = Path(db_path)
    if not path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    since = _since_arg(days)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    print(f"# Trading Bot Performance Report — last {days} days")
    print(f"DB: {db_path}")
    print(f"Generated: {datetime.now(UTC).isoformat(timespec='seconds')}")

    snapshots = _rows(
        conn,
        """
        SELECT timestamp, equity, cash, positions_count, daily_pnl, total_pnl
        FROM portfolio_snapshots
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
        """,
        (since,),
    )
    if snapshots:
        first = snapshots[0]
        last = snapshots[-1]
        pnl = float(last["equity"]) - float(first["equity"])
        pnl_pct = pnl / float(first["equity"]) if float(first["equity"]) else None
        peak = max(float(s["equity"]) for s in snapshots)
        max_dd = min((float(s["equity"]) - peak) / peak for s in snapshots if peak)
        print("\n## Portfolio")
        print(f"Start equity: {_fmt_money(float(first['equity']))}")
        print(f"End equity:   {_fmt_money(float(last['equity']))}")
        print(f"PnL:          {_fmt_money(pnl)} ({_fmt_pct(pnl_pct)})")
        print(f"Latest cash:  {_fmt_money(float(last['cash']))}")
        print(f"Positions:    {last['positions_count']}")
        print(f"Max DD obs.:  {_fmt_pct(max_dd)}")
    else:
        print("\n## Portfolio\nNo snapshots yet in selected window")

    trades = _rows(
        conn,
        """
        SELECT ticker, side, qty, status, signal_score, reason, created_at, filled_price
        FROM trade_log
        WHERE created_at >= ?
        ORDER BY created_at ASC
        """,
        (since,),
    )
    by_ticker = Counter(t["ticker"] for t in trades)
    by_status = Counter(t["status"] for t in trades)
    by_hour = Counter(str(t["created_at"])[11:13] for t in trades if t["created_at"])
    by_bucket = Counter(_bucket_score(_safe_float(t["signal_score"])) for t in trades)

    print("\n## Trades")
    print(f"Trades: {len(trades)}")
    print(f"Buys:   {sum(1 for t in trades if t['side'] == 'buy')}")
    print(f"Sells:  {sum(1 for t in trades if t['side'] == 'sell')}")
    _print_table("Trades by status", ["status", "count"], [[k, v] for k, v in by_status.most_common()])
    _print_table("Trades by ticker", ["ticker", "count"], [[k, v] for k, v in by_ticker.most_common(15)])
    _print_table("Trades by hour UTC", ["hour", "count"], [[k, v] for k, v in sorted(by_hour.items())])
    _print_table("Trades by signal bucket", ["bucket", "count"], [[k, v] for k, v in by_bucket.items()])

    signals = _rows(
        conn,
        """
        SELECT ticker, action, score, sentiment_score, technical_score, volume_score,
               reject_reason, features_json, timestamp
        FROM signal_log
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
        """,
        (since,),
    )
    action_counts = Counter(s["action"] for s in signals)
    reject_counts = Counter(s["reject_reason"] for s in signals if s["reject_reason"])
    regime_counts: Counter[str] = Counter()
    rs15_values: list[float] = []
    tech_values: list[float] = []
    vol_values: list[float] = []
    for signal in signals:
        features = _parse_features(signal["features_json"])
        regime = features.get("market_regime")
        if regime:
            regime_counts[str(regime)] += 1
        rs15 = _safe_float(features.get("relative_strength_15m"))
        if rs15 is not None:
            rs15_values.append(rs15)
        tech = _safe_float(signal["technical_score"])
        vol = _safe_float(signal["volume_score"])
        if tech is not None:
            tech_values.append(tech)
        if vol is not None:
            vol_values.append(vol)

    print("\n## Signals")
    print(f"Signals evaluated: {len(signals)}")
    _print_table("Signals by action", ["action", "count"], [[k, v] for k, v in action_counts.items()])
    _print_table("Top reject reasons", ["reason", "count"], [[k, v] for k, v in reject_counts.most_common(12)])
    _print_table("Market regime counts", ["regime", "count"], [[k, v] for k, v in regime_counts.items()])
    print(f"Avg relative strength 15m: {_fmt_pct(mean(rs15_values)) if rs15_values else 'n/a'}")
    print(f"Avg technical score:       {mean(tech_values):.3f}" if tech_values else "Avg technical score:       n/a")
    print(f"Avg volume score:          {mean(vol_values):.3f}" if vol_values else "Avg volume score:          n/a")

    cycles = _rows(
        conn,
        """
        SELECT timestamp, news_count, signals_generated, orders_placed, cycle_duration_ms
        FROM cycle_log
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
        """,
        (since,),
    )
    if cycles:
        print("\n## Cycles")
        print(f"Cycles:          {len(cycles)}")
        print(f"News processed:  {sum(int(c['news_count'] or 0) for c in cycles)}")
        print(f"Signals:         {sum(int(c['signals_generated'] or 0) for c in cycles)}")
        print(f"Orders:          {sum(int(c['orders_placed'] or 0) for c in cycles)}")
        durations = [int(c["cycle_duration_ms"] or 0) for c in cycles if c["cycle_duration_ms"]]
        print(f"Avg duration ms: {int(mean(durations)) if durations else 'n/a'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB, help=f"SQLite DB path, default: {DEFAULT_DB}")
    parser.add_argument("--days", type=int, default=14, help="Lookback window in days")
    args = parser.parse_args()
    build_report(args.db, args.days)


if __name__ == "__main__":
    main()
