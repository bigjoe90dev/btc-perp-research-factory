"""
Session classification with DST-aware rules.

Default windows (local time):
  Asia:      00:00-07:00 Europe/London
  Frankfurt: 07:00-08:00 Europe/London
  London:    08:00-12:00 Europe/London
  NewYork:   08:00-17:00 America/New_York

Rules can be overridden via BACKTEST_SESSION_RULES using:
  Name@TZ=HH:MM-HH:MM;Name@TZ=HH:MM-HH:MM
"""
from dataclasses import dataclass
from datetime import datetime, time
from typing import List

from dateutil import tz


@dataclass(frozen=True)
class SessionWindow:
    name: str
    tz_name: str
    start: time
    end: time
    priority: int = 0


def _parse_time(hhmm: str) -> time:
    h, m = hhmm.split(":")
    return time(int(h), int(m))


def parse_session_rules(raw: str) -> List[SessionWindow]:
    raw = (raw or "").strip()
    if not raw:
        return default_session_rules()
    windows = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        # Name@TZ=HH:MM-HH:MM
        if "=" not in part:
            continue
        left, rng = part.split("=", 1)
        if "@" in left:
            name, tz_name = left.split("@", 1)
        else:
            name, tz_name = left, "UTC"
        if "-" not in rng:
            continue
        start_s, end_s = rng.split("-", 1)
        windows.append(SessionWindow(
            name=name.strip(),
            tz_name=tz_name.strip(),
            start=_parse_time(start_s.strip()),
            end=_parse_time(end_s.strip()),
            priority=len(windows),
        ))
    return windows or default_session_rules()


def default_session_rules() -> List[SessionWindow]:
    return [
        SessionWindow("NewYork", "America/New_York", _parse_time("08:00"), _parse_time("17:00"), priority=0),
        SessionWindow("London", "Europe/London", _parse_time("08:00"), _parse_time("12:00"), priority=1),
        SessionWindow("Frankfurt", "Europe/London", _parse_time("07:00"), _parse_time("08:00"), priority=2),
        SessionWindow("Asia", "Europe/London", _parse_time("00:00"), _parse_time("07:00"), priority=3),
    ]


def _in_window(t_local: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= t_local < end
    # wraps midnight
    return t_local >= start or t_local < end


def classify_session(ts_utc: datetime, rules: List[SessionWindow]) -> str:
    for w in sorted(rules, key=lambda r: r.priority):
        tzinfo = tz.gettz(w.tz_name)
        if tzinfo is None:
            tzinfo = tz.UTC
        local = ts_utc.astimezone(tzinfo)
        if _in_window(local.time(), w.start, w.end):
            return w.name
    return "Off"
