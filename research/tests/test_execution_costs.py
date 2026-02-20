from __future__ import annotations

import pandas as pd

from research.engine.execution import execute_market_order_next_open
from research.engine.types import Bar


def test_execution_applies_fee_and_slippage() -> None:
    bar = Bar(
        ts_utc=pd.Timestamp("2026-01-01T00:05:00Z"),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1.0,
    )
    fill, cash_delta = execute_market_order_next_open(
        next_bar=bar,
        qty_signed=1.0,
        reason="test",
        fee_bps=10.0,
        slippage_cfg={"bps_base": 0.0, "bps_per_vol": 0.0, "vol_proxy": "candle_range"},
        execution_cfg={},
        slippage_bar=bar,
    )
    assert fill.price == 100.0
    assert fill.fee_paid == 0.1
    assert cash_delta == -100.1


def test_slippage_can_use_decision_bar_not_future_bar_range() -> None:
    decision_bar = Bar(
        ts_utc=pd.Timestamp("2026-01-01T00:00:00Z"),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1.0,
    )
    next_bar = Bar(
        ts_utc=pd.Timestamp("2026-01-01T00:05:00Z"),
        open=100.0,
        high=150.0,
        low=50.0,
        close=100.0,
        volume=1.0,
    )

    cfg = {"bps_base": 0.0, "bps_per_vol": 100.0, "vol_proxy": "candle_range"}

    fill_decision, _ = execute_market_order_next_open(
        next_bar=next_bar,
        qty_signed=1.0,
        reason="test",
        fee_bps=0.0,
        slippage_cfg=cfg,
        execution_cfg={},
        slippage_bar=decision_bar,
    )
    fill_future, _ = execute_market_order_next_open(
        next_bar=next_bar,
        qty_signed=1.0,
        reason="test",
        fee_bps=0.0,
        slippage_cfg=cfg,
        execution_cfg={},
        slippage_bar=next_bar,
    )

    assert fill_decision.slippage_bps < fill_future.slippage_bps
