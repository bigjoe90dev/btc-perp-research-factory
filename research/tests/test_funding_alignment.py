from __future__ import annotations

import pandas as pd

from research.data.funding import align_funding_to_bars, funding_cashflow


def test_funding_alignment_and_sign() -> None:
    bars = pd.Series(pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="1h", tz="UTC"))
    funding = pd.DataFrame(
        {
            "ts_utc": [pd.Timestamp("2026-01-01T01:00:00Z"), pd.Timestamp("2026-01-01T03:30:00Z")],
            "funding_rate_raw": [0.001, -0.002],
        }
    )
    vec = align_funding_to_bars(bars, funding)
    assert vec[1] == 0.001
    assert vec[4] == -0.002

    long_cf = funding_cashflow(position_qty=1.0, mark_price=100.0, funding_rate=0.001)
    short_cf = funding_cashflow(position_qty=-1.0, mark_price=100.0, funding_rate=0.001)
    assert long_cf < 0
    assert short_cf > 0
