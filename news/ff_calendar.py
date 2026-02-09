"""
Forex Factory economic calendar integration (free, no API key).

Fetches the current week's high-impact events from Forex Factory's
public JSON feed and stores them in the news_events table.

Usage:
    python -m news.ff_calendar
"""

import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests

from shared.db import connect


FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


def fetch_ff_calendar() -> List[dict]:
    """
    Fetch this week's economic calendar from Forex Factory.

    Returns list of event dicts with keys:
        title, country, date, impact, forecast, previous
    """
    resp = requests.get(FF_URL, timeout=30, headers={
        "User-Agent": "ict-bot/1.0",
    })
    resp.raise_for_status()
    return resp.json()


def filter_events(
    events: List[dict],
    currencies: List[str] = None,
    min_impact: str = "high",
) -> List[dict]:
    """Filter events by currency and minimum impact level."""
    impact_order = {"low": 0, "medium": 1, "high": 2, "holiday": -1}
    min_level = impact_order.get(min_impact.lower(), 2)

    filtered = []
    for ev in events:
        currency = (ev.get("country") or "").upper()
        impact = (ev.get("impact") or "").lower()
        event_level = impact_order.get(impact, 0)

        if currencies and currency not in [c.upper() for c in currencies]:
            continue
        if event_level < min_level:
            continue

        filtered.append(ev)

    return filtered


def _parse_ff_datetime(date_str: str) -> str:
    """Parse FF datetime string to ISO UTC format."""
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        else:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat(timespec="seconds")
    except (ValueError, TypeError):
        return date_str


def store_events(events: List[dict]) -> int:
    """Store filtered events into news_events table. Returns count stored."""
    db = connect()
    ts_now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    count = 0

    for ev in events:
        event_name = ev.get("title") or "Unknown"
        currency = (ev.get("country") or "").upper()
        impact = (ev.get("impact") or "").lower()
        dt_utc = _parse_ff_datetime(ev.get("date") or "")
        forecast = str(ev.get("forecast") or "")
        previous = str(ev.get("previous") or "")

        # Skip duplicates (same event + same datetime)
        existing = db.execute(
            "SELECT 1 FROM news_events WHERE event_name=? AND datetime_utc=?",
            (event_name, dt_utc),
        ).fetchone()
        if existing:
            continue

        db.execute(
            """INSERT INTO news_events
               (event_name, currency, impact, datetime_utc, actual, forecast, previous, fetched_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (event_name, currency, impact, dt_utc, "", forecast, previous, ts_now),
        )
        count += 1

    db.commit()
    db.close()
    return count


def refresh_calendar(
    currencies: List[str] = None,
    min_impact: str = "high",
) -> int:
    """
    Full pipeline: fetch from Forex Factory, filter, store.
    Returns count of events stored.
    """
    if not currencies:
        raw = os.getenv("NEWS_CURRENCIES", "EUR,USD")
        currencies = [c.strip() for c in raw.split(",")]

    if not min_impact:
        min_impact = os.getenv("NEWS_MIN_IMPACT", "high").strip()

    raw_events = fetch_ff_calendar()
    filtered = filter_events(raw_events, currencies, min_impact)
    count = store_events(filtered)
    return count


if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    count = refresh_calendar()
    print(f"Stored {count} upcoming high-impact events from Forex Factory")
