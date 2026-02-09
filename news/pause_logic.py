"""
Trade pause logic around high-impact economic events.

Checks the news_events table and determines whether trading should
be paused based on proximity to upcoming events.
"""

import os
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional

from shared.db import connect


@dataclass
class PauseStatus:
    should_pause: bool
    reason: str = ""
    event_name: str = ""
    event_time: str = ""
    minutes_until: float = 0.0
    minutes_since: float = 0.0


def should_pause_trading(
    currencies: list = None,
    pause_before_min: int = None,
    pause_after_min: int = None,
    now: datetime = None,
) -> PauseStatus:
    """
    Check if trading should be paused due to an upcoming/recent news event.

    Reads configuration from .env if not provided:
        NEWS_PAUSE_BEFORE_MIN (default 15)
        NEWS_PAUSE_AFTER_MIN (default 30)
        NEWS_CURRENCIES (default EUR,USD)

    Returns:
        PauseStatus with should_pause flag and details.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if pause_before_min is None:
        pause_before_min = int(os.getenv("NEWS_PAUSE_BEFORE_MIN", "15"))
    if pause_after_min is None:
        pause_after_min = int(os.getenv("NEWS_PAUSE_AFTER_MIN", "30"))
    if currencies is None:
        raw = os.getenv("NEWS_CURRENCIES", "EUR,USD")
        currencies = [c.strip().upper() for c in raw.split(",")]

    # Window: [now - pause_after_min, now + pause_before_min]
    window_start = now - timedelta(minutes=pause_after_min)
    window_end = now + timedelta(minutes=pause_before_min)

    db = connect()
    placeholders = ",".join(["?"] * len(currencies))
    rows = db.execute(
        f"""
        SELECT event_name, currency, impact, datetime_utc
        FROM news_events
        WHERE currency IN ({placeholders})
          AND impact = 'high'
          AND datetime_utc >= ?
          AND datetime_utc <= ?
        ORDER BY datetime_utc ASC
        LIMIT 1
        """,
        (*currencies, window_start.isoformat(), window_end.isoformat()),
    ).fetchall()
    db.close()

    if not rows:
        return PauseStatus(should_pause=False, reason="No high-impact events in window")

    row = rows[0]
    event_name = row["event_name"]
    event_time_str = row["datetime_utc"]

    try:
        event_time = datetime.fromisoformat(event_time_str)
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return PauseStatus(should_pause=False, reason="Could not parse event time")

    delta = (event_time - now).total_seconds() / 60

    if delta > 0:
        return PauseStatus(
            should_pause=True,
            reason=f"Pre-event pause: {event_name} in {delta:.0f} min",
            event_name=event_name,
            event_time=event_time_str,
            minutes_until=delta,
        )
    else:
        return PauseStatus(
            should_pause=True,
            reason=f"Post-event pause: {event_name} was {abs(delta):.0f} min ago",
            event_name=event_name,
            event_time=event_time_str,
            minutes_since=abs(delta),
        )
