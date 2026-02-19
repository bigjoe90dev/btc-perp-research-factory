from __future__ import annotations

from research.engine.execution import round_price_to_tick, round_qty_to_step


def test_qty_rounding_step_floor() -> None:
    assert round_qty_to_step(0.1234, 0.01) == 0.12
    assert round_qty_to_step(0.009, 0.01) == 0.0


def test_price_rounding_conservative_by_side() -> None:
    assert round_price_to_tick(100.01, 0.1, side=1) == 100.1
    assert round_price_to_tick(100.09, 0.1, side=-1) == 100.0

