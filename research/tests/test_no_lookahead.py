from __future__ import annotations

import pandas as pd
import pytest

from research.strategies.base import StrategyContext


def test_strategy_context_blocks_future_access() -> None:
    df = pd.DataFrame(
        {
            "ts_utc": pd.date_range("2026-01-01", periods=5, freq="min", tz="UTC"),
            "open": [1, 2, 3, 4, 5],
            "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
            "volume": [1, 1, 1, 1, 1],
        }
    )
    ctx = StrategyContext(df, idx=3)
    hist = ctx.history("close")
    assert len(hist) == 4
    assert hist[-1] == 4

    with pytest.raises(RuntimeError):
        _ = ctx.future("close")
