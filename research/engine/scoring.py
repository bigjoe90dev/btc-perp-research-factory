from __future__ import annotations

import math


def raw_score(metrics: dict, n_params: int, cfg: dict) -> float:
    w_sh = float(cfg.get("weight_sharpe", 0.4))
    w_so = float(cfg.get("weight_sortino", 0.25))
    w_rt = float(cfg.get("weight_return", 0.2))
    w_dd = float(cfg.get("weight_drawdown", 0.15))

    sharpe = float(metrics.get("sharpe", 0.0))
    sortino = float(metrics.get("sortino", 0.0))
    ret = float(metrics.get("total_return", 0.0))
    dd = abs(float(metrics.get("max_drawdown", 0.0)))

    base = w_sh * sharpe + w_so * sortino + w_rt * ret - w_dd * dd

    c_pen = float(cfg.get("complexity_penalty_per_param", 0.02)) * max(n_params, 0)
    turnover_penalty = float(cfg.get("turnover_penalty", 0.0))
    turnover_signal = 0.0
    if turnover_penalty > 0.0:
        if "turnover_ratio_annualized" not in metrics:
            raise ValueError("Missing required metric: turnover_ratio_annualized")
        turnover_signal = float(metrics.get("turnover_ratio_annualized", 0.0))
    t_pen = float(cfg.get("turnover_penalty", 0.0)) * max(turnover_signal, 0.0)
    skew = float(metrics.get("return_skewness", 0.0))
    kurt = float(metrics.get("return_excess_kurtosis", 0.0))
    skew_pen = float(cfg.get("negative_skew_penalty", 0.0)) * max(-skew, 0.0)
    kurt_pen = float(cfg.get("kurtosis_penalty", 0.0)) * max(kurt, 0.0)
    return float(base - c_pen - t_pen - skew_pen - kurt_pen)


def adjusted_score_for_multiple_testing(raw: float, rank: int, total: int) -> float:
    if total <= 1:
        return raw
    # Conservative proxy adjustment: penalize by search breadth and rank.
    penalty = 0.05 * math.log1p(total) + 0.01 * rank
    return float(raw - penalty)
