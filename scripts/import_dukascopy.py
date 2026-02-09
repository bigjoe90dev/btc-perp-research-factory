"""
Dukascopy historical data importer.

Downloads free tick/candle data from Dukascopy's public feed
and stores it in the quotes table for backtesting.

Dukascopy serves hourly .bi5 (LZMA-compressed) files at:
  https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YEAR}/{MONTH:00}/{DAY:02}/{HOUR}h_ticks.bi5

Months are 0-indexed (Jan=00, Feb=01, ..., Dec=11).
Each file is 20 bytes per tick: [ms_offset(i32), ask(i32), bid(i32), ask_vol(f32), bid_vol(f32)]

Usage:
    python -m scripts.import_dukascopy --symbol EURUSD --start 2025-01-01 --end 2025-02-01
    python -m scripts.import_dukascopy --symbol EURUSD --start 2025-01-01 --end 2025-02-01 --timeframe M1
"""

import argparse
import io
import lzma
import struct
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.db import connect

try:
    import requests
except ImportError:
    print("Error: requests is required. Run: pip install requests")
    sys.exit(1)


DUKASCOPY_BASE = "https://datafeed.dukascopy.com/datafeed"

# Point values for converting raw int prices to decimal
# EURUSD, GBPUSD, etc. with 5 decimal places use 100_000
# USDJPY, etc. with 3 decimal places use 1_000
POINT_VALUES = {
    "EURUSD": 100_000,
    "GBPUSD": 100_000,
    "AUDUSD": 100_000,
    "NZDUSD": 100_000,
    "USDCHF": 100_000,
    "USDCAD": 100_000,
    "EURGBP": 100_000,
    "EURJPY": 1_000,
    "USDJPY": 1_000,
    "GBPJPY": 1_000,
    "XAUUSD": 1_000,
}


def download_hour(symbol: str, dt_hour: datetime, point_value: int) -> list:
    """
    Download and parse one hour of tick data from Dukascopy.

    Returns list of dicts: [{"ts_utc": ..., "bid": ..., "ask": ...}, ...]
    """
    year = dt_hour.year
    month = dt_hour.month - 1  # 0-indexed!
    day = dt_hour.day
    hour = dt_hour.hour

    url = f"{DUKASCOPY_BASE}/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as e:
        return []

    if resp.status_code == 404:
        return []  # No data for this hour (weekend, holiday)
    if resp.status_code != 200:
        return []

    if len(resp.content) == 0:
        return []

    # Decompress LZMA
    try:
        raw = lzma.decompress(resp.content)
    except lzma.LZMAError:
        return []

    # Parse 20-byte records: [ms(i32), ask(i32), bid(i32), ask_vol(f32), bid_vol(f32)]
    record_size = 20
    n_ticks = len(raw) // record_size
    ticks = []

    for i in range(n_ticks):
        offset = i * record_size
        ms, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(
            ">iiiff", raw[offset:offset + record_size]
        )

        ts = dt_hour + timedelta(milliseconds=ms)
        ask = ask_raw / point_value
        bid = bid_raw / point_value

        ticks.append({
            "ts_utc": ts.isoformat(timespec="milliseconds"),
            "bid": round(bid, 6),
            "ask": round(ask, 6),
        })

    return ticks


def resample_to_m1(ticks: list) -> list:
    """
    Resample tick data to 1-minute OHLC bars using mid price.
    Returns list of dicts with ts_utc (minute start), bid (close), ask (close).
    """
    if not ticks:
        return []

    df = pd.DataFrame(ticks)
    df["ts"] = pd.to_datetime(df["ts_utc"])
    df = df.set_index("ts")

    # Use close of each minute as the representative price
    bars = df.resample("1min").agg({"bid": "last", "ask": "last"}).dropna()

    result = []
    for ts, row in bars.iterrows():
        result.append({
            "ts_utc": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "bid": round(row["bid"], 6),
            "ask": round(row["ask"], 6),
        })

    return result


def store_quotes(quotes: list, env: str, account_id: str, symbol_id: int) -> int:
    """Store quotes into the DB. Returns count inserted."""
    if not quotes:
        return 0

    db = connect()
    count = 0

    for q in quotes:
        # Skip duplicates
        existing = db.execute(
            "SELECT 1 FROM quotes WHERE ts_utc=? AND env=? AND account_id=? AND symbol_id=?",
            (q["ts_utc"], env, account_id, symbol_id),
        ).fetchone()
        if existing:
            continue

        db.execute(
            """INSERT INTO quotes (ts_utc, env, account_id, symbol_id, bid, ask)
               VALUES (?,?,?,?,?,?)""",
            (q["ts_utc"], env, account_id, symbol_id, q["bid"], q["ask"]),
        )
        count += 1

    db.commit()
    db.close()
    return count


def import_range(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str = "tick",
    env: str = "demo",
    account_id: str = "dukascopy",
    symbol_id: int = 1,
) -> int:
    """
    Import historical data from Dukascopy for a date range.

    Args:
        symbol: e.g. "EURUSD"
        start: Start datetime (UTC)
        end: End datetime (UTC)
        timeframe: "tick" for raw ticks, "M1" for 1-minute bars
        env: Environment label for DB
        account_id: Account label for DB
        symbol_id: Symbol ID for DB

    Returns:
        Total quotes stored
    """
    point_value = POINT_VALUES.get(symbol, 100_000)
    total_stored = 0
    current = start.replace(minute=0, second=0, microsecond=0)
    total_hours = int((end - current).total_seconds() / 3600)
    hours_done = 0

    print(f"Importing {symbol} from {start.date()} to {end.date()} ({timeframe})")
    print(f"Total hours to scan: {total_hours}")
    print()

    while current < end:
        ticks = download_hour(symbol, current, point_value)

        if ticks:
            if timeframe == "M1":
                quotes = resample_to_m1(ticks)
            else:
                quotes = ticks

            stored = store_quotes(quotes, env, account_id, symbol_id)
            total_stored += stored

            if stored > 0:
                print(f"  {current.strftime('%Y-%m-%d %H:%M')} -- {len(ticks)} ticks -> {stored} stored ({timeframe})")

        hours_done += 1
        if hours_done % 24 == 0:
            pct = (hours_done / total_hours * 100) if total_hours > 0 else 100
            print(f"  Progress: {hours_done}/{total_hours} hours ({pct:.0f}%) -- {total_stored:,} total stored")

        current += timedelta(hours=1)

        # Be polite to the server
        time.sleep(0.1)

    return total_stored


def main():
    parser = argparse.ArgumentParser(description="Import Dukascopy historical forex data")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol (default: EURUSD)")
    parser.add_argument("--start", required=True, help="Start date: YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date: YYYY-MM-DD")
    parser.add_argument("--timeframe", default="M1", choices=["tick", "M1"],
                        help="Timeframe (default: M1)")
    parser.add_argument("--env", default="demo", help="Environment label (default: demo)")
    parser.add_argument("--account-id", default="dukascopy", help="Account label (default: dukascopy)")
    parser.add_argument("--symbol-id", type=int, default=1, help="Symbol ID (default: 1)")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if start >= end:
        print("Error: start must be before end")
        sys.exit(1)

    total = import_range(
        symbol=args.symbol,
        start=start,
        end=end,
        timeframe=args.timeframe,
        env=args.env,
        account_id=args.account_id,
        symbol_id=args.symbol_id,
    )

    print(f"\nDone. Total quotes stored: {total:,}")


if __name__ == "__main__":
    main()
