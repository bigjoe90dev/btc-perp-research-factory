from __future__ import annotations

import pandas as pd

from research.cli import _apply_correlation_gate


def _outcome(strategy_id: str, equity: list[float], score: float) -> dict:
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=len(equity), freq="1h", tz="UTC")
    eq = pd.DataFrame({"ts_utc": ts, "equity": equity})
    return {
        "strategy_id": strategy_id,
        "score_adjusted": score,
        "gates": {"passed_hard": True, "hard_reject_reasons": []},
        "equity_curve": eq,
    }


def test_correlation_gate_rejects_highly_correlated_candidate() -> None:
    ranked = [
        _outcome("a", [100.0, 101.0, 102.0, 103.0], score=10.0),
        _outcome("b", [200.0, 202.0, 204.0, 206.0], score=9.0),
    ]
    cfg = {
        "gates": {
            "enable_correlation_gate": True,
            "max_return_correlation": 0.7,
            "correlation_use_absolute": False,
            "correlation_min_points": 2,
        }
    }
    out = _apply_correlation_gate(ranked, backtest_cfg=cfg)

    assert out[0]["selected"] is True
    assert out[1]["selected"] is False
    assert out[1]["gates"]["passed_hard"] is False
    assert any("correlation>" in r for r in out[1]["gates"]["hard_reject_reasons"])


def test_correlation_gate_rejects_inverse_clone_when_absolute_enabled() -> None:
    ranked = [
        _outcome("a", [100.0, 101.0, 102.0, 103.0], score=10.0),
        _outcome("b", [100.0, 99.0, 98.0, 97.0], score=9.0),
    ]
    cfg = {
        "gates": {
            "enable_correlation_gate": True,
            "max_return_correlation": 0.7,
            "correlation_use_absolute": True,
            "correlation_min_points": 2,
        }
    }
    out = _apply_correlation_gate(ranked, backtest_cfg=cfg)

    assert out[0]["selected"] is True
    assert out[1]["selected"] is False
    assert out[1]["gates"]["passed_hard"] is False
    assert any("correlation>" in r for r in out[1]["gates"]["hard_reject_reasons"])
