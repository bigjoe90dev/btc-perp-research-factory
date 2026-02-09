"""Structured event logger -- writes to event_log table and optionally to console."""

import sys
from datetime import datetime, timezone
from shared.db import connect


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_event(level: str, source: str, message: str, *, echo: bool = True) -> None:
    """
    Log a structured event to the event_log table.

    Args:
        level:   INFO, WARN, ERROR, CRITICAL
        source:  Component name (e.g. "paper_engine", "market_data", "backtest")
        message: Human-readable message
        echo:    Also print to stdout (default True)
    """
    ts = _utc_now_iso()
    full_msg = f"[{source}] {message}"

    try:
        conn = connect()
        conn.execute(
            "INSERT INTO event_log (ts_utc, level, message) VALUES (?, ?, ?)",
            (ts, level.upper(), full_msg),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[logger] DB write failed: {e}", file=sys.stderr)

    if echo:
        print(f"[{ts}] {level.upper()} {full_msg}")
