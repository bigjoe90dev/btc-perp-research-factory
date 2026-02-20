from __future__ import annotations

from research.engine.scoring import raw_score


def test_raw_score_uses_turnover_ratio_when_present() -> None:
    metrics = {
        "sharpe": 1.0,
        "sortino": 1.0,
        "total_return": 0.10,
        "max_drawdown": -0.10,
        "turnover": 2_000_000.0,
        "turnover_ratio": 2.0,
        "return_skewness": 0.0,
        "return_excess_kurtosis": 0.0,
    }
    cfg = {
        "weight_sharpe": 0.40,
        "weight_sortino": 0.25,
        "weight_return": 0.20,
        "weight_drawdown": 0.15,
        "complexity_penalty_per_param": 0.0,
        "turnover_penalty": 0.001,
        "negative_skew_penalty": 0.0,
        "kurtosis_penalty": 0.0,
    }

    score = raw_score(metrics=metrics, n_params=0, cfg=cfg)
    assert score > 0.60
