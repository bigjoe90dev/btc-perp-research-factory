from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from research.data_tools.validate_dataset import validate_frame


@dataclass
class IntegrityResult:
    ok: bool
    kind: str
    summary: dict[str, Any]
    errors: list[str]


def validate_candles_frame(df: pd.DataFrame, expected_seconds: int) -> IntegrityResult:
    summary, errors = validate_frame(df.copy(), kind="ohlc", expected_seconds=expected_seconds)
    # Extra fail-closed check for NaN coverage in required cols.
    req = ["ts_utc", "open", "high", "low", "close", "volume"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    else:
        nulls = int(df[req].isna().sum().sum())
        if nulls > 0:
            errors.append(f"NaN values in required candle fields: {nulls}")

    summary["ok"] = len(errors) == 0
    summary["errors"] = errors
    return IntegrityResult(ok=len(errors) == 0, kind="ohlc", summary=summary, errors=errors)


def validate_funding_frame(df: pd.DataFrame) -> IntegrityResult:
    summary, errors = validate_frame(df.copy(), kind="series", expected_seconds=None)
    req = ["ts_utc", "funding_rate_raw"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    else:
        nulls = int(df[req].isna().sum().sum())
        if nulls > 0:
            errors.append(f"NaN values in required funding fields: {nulls}")

    summary["ok"] = len(errors) == 0
    summary["errors"] = errors
    return IntegrityResult(ok=len(errors) == 0, kind="series", summary=summary, errors=errors)
