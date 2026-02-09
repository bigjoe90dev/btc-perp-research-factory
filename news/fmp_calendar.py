"""
FMP (Financial Modeling Prep) economic calendar integration.

Fetches upcoming economic events, filters by currency and impact,
stores in news_events table.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests

from shared.db import connect


FMP_BASE = "https://financialmodelingprep.com/api/v3"


def fetch_economic_calendar(
    api_key: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> List[dict]:
    """
    Fetch economic calendar events from FMP.

    Args:
        api_key: FMP API key
        from_date: Start date (YYYY-MM-DD). Default: today.
        to_date: End date (YYYY-MM-DD). Default: 7 days from now.

    Returns:
        List of event dicts from FMP API.
    """
    now = datetime.now(timezone.utc)
    if not from_date:
        from_date = now.strftime("%Y-%m-%d")
    if not to_date:
        to_date = (now + timedelta(days=7)).strftime("%Y-%m-%d")

    url = f"{FMP_BASE}/economic_calendar"
    params = {
        "from": from_date,
        "to": to_date,
        "apikey": api_key,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def filter_events(
    events: List[dict],
    currencies: List[str] = None,
    min_impact: str = "high",
) -> List[dict]:
    """
    Filter events by currency and minimum impact level.

    Args:
        events: Raw events from FMP
        currencies: List of currencies to keep (e.g., ["EUR", "USD"])
        min_impact: Minimum impact level ("low", "medium", "high")
    """
    impact_order = {"low": 0, "medium": 1, "high": 2}
    min_level = impact_order.get(min_impact.lower(), 2)

    filtered = []
    for ev in events:
        # FMP uses "country" or "currency" field
        currency = (ev.get("currency") or ev.get("country") or "").upper()
        impact = (ev.get("impact") or "").lower()
        event_level = impact_order.get(impact, 0)

        if currencies and currency not in [c.upper() for c in currencies]:
            continue
        if event_level < min_level:
            continue

        filtered.append(ev)

    return filtered


def store_events(events: List[dict]) -> int:
    """Store filtered events into news_events table. Returns count stored."""
    db = connect()
    ts_now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    count = 0

    for ev in events:
        event_name = ev.get("event") or ev.get("title") or "Unknown"
        currency = (ev.get("currency") or ev.get("country") or "").upper()
        impact = (ev.get("impact") or "").lower()
        dt_utc = ev.get("date") or ""
        actual = str(ev.get("actual") or "")
        forecast = str(ev.get("estimate") or ev.get("forecast") or "")
        previous = str(ev.get("previous") or "")

        db.execute(
            """INSERT INTO news_events
               (event_name, currency, impact, datetime_utc, actual, forecast, previous, fetched_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (event_name, currency, impact, dt_utc, actual, forecast, previous, ts_now),
        )
        count += 1

    db.commit()
    db.close()
    return count


def refresh_calendar(
    api_key: str = None,
    currencies: List[str] = None,
    min_impact: str = "high",
    days_ahead: int = 7,
) -> int:
    """
    Full pipeline: fetch, filter, store. Returns count of events stored.

    Uses .env values as defaults if not provided.
    """
    if not api_key:
        api_key = os.getenv("FMP_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing FMP_API_KEY")

    if not currencies:
        raw = os.getenv("NEWS_CURRENCIES", "EUR,USD")
        currencies = [c.strip() for c in raw.split(",")]

    if not min_impact:
        min_impact = os.getenv("NEWS_MIN_IMPACT", "high").strip()

    now = datetime.now(timezone.utc)
    from_date = now.strftime("%Y-%m-%d")
    to_date = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    raw_events = fetch_economic_calendar(api_key, from_date, to_date)
    filtered = filter_events(raw_events, currencies, min_impact)
    count = store_events(filtered)
    return count


if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    count = refresh_calendar()
    print(f"Stored {count} upcoming high-impact events")
