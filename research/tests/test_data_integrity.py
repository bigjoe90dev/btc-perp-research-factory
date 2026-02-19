from __future__ import annotations

import pandas as pd

from research.data.integrity import validate_candles_frame


def test_data_integrity_detects_duplicates() -> None:
    ts = pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "ts_utc": [ts[0], ts[1], ts[1]],
            "open": [1.0, 2.0, 2.0],
            "high": [1.0, 2.0, 2.0],
            "low": [1.0, 2.0, 2.0],
            "close": [1.0, 2.0, 2.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    res = validate_candles_frame(df, expected_seconds=60)
    assert not res.ok
    assert any("Duplicate timestamps" in err for err in res.errors)
