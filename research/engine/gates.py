from __future__ import annotations


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate_gates(
    aggregate_metrics: dict,
    fold_metrics: list[dict],
    stress: dict,
    cfg: dict,
) -> dict:
    reasons: list[str] = []
    soft_flags: list[str] = []

    min_trades = int(cfg.get("min_trades", 0))
    max_dd_cap = float(cfg.get("max_drawdown_pct", 1.0))
    min_pos_fold_ratio = float(cfg.get("min_positive_fold_ratio", 0.0))
    trade_count_mode = str(cfg.get("trade_count_mode", "aggregate_mean")).strip().lower()
    positive_fold_use_active_only = bool(cfg.get("positive_fold_use_active_only", False))

    fold_trade_counts = [_to_float(f.get("trades_count", 0.0), 0.0) for f in fold_metrics]
    total_trades_across_folds = sum(max(0.0, x) for x in fold_trade_counts)
    active_fold_count = sum(1 for x in fold_trade_counts if x > 0.0)

    if trade_count_mode == "fold_total":
        trades_metric = total_trades_across_folds
    elif trade_count_mode == "active_fold_mean":
        active_trades = [x for x in fold_trade_counts if x > 0.0]
        trades_metric = (sum(active_trades) / len(active_trades)) if active_trades else 0.0
    else:
        trade_count_mode = "aggregate_mean"
        trades_metric = _to_float(aggregate_metrics.get("trades_count", 0.0), 0.0)

    if trades_metric < min_trades:
        reasons.append(f"trades<{min_trades}")

    if abs(_to_float(aggregate_metrics.get("max_drawdown", 0.0), 0.0)) > max_dd_cap:
        reasons.append(f"max_drawdown>{max_dd_cap}")

    positive_fold_ratio: float | None = None
    if fold_metrics:
        eval_folds = fold_metrics
        if positive_fold_use_active_only:
            eval_folds = [f for f in fold_metrics if _to_float(f.get("trades_count", 0.0), 0.0) > 0.0]
        if eval_folds:
            pos = sum(1 for f in eval_folds if _to_float(f.get("total_return", 0.0), 0.0) > 0.0)
            positive_fold_ratio = pos / len(eval_folds)
        else:
            positive_fold_ratio = 0.0
        if positive_fold_ratio < min_pos_fold_ratio:
            reasons.append(f"positive_fold_ratio<{min_pos_fold_ratio}")

    if bool(cfg.get("require_cost_robustness", True)):
        if not bool(stress.get("cost_sweep", {}).get("pass", False)):
            reasons.append("cost_robustness_failed")

    if bool(cfg.get("require_latency_robustness", False)):
        if not bool(stress.get("latency_sweep", {}).get("pass", False)):
            reasons.append("latency_robustness_failed")

    if bool(cfg.get("require_parameter_perturbation", True)):
        if not bool(stress.get("parameter_perturbation", {}).get("pass", False)):
            reasons.append("parameter_perturbation_failed")

    max_bootstrap_prob_negative = float(cfg.get("max_bootstrap_prob_negative", 0.6))
    if _to_float(stress.get("bootstrap", {}).get("prob_negative", 0.0), 0.0) > max_bootstrap_prob_negative:
        reasons.append("bootstrap_fragility_high")

    min_skew = cfg.get("min_return_skewness")
    if min_skew is not None and _to_float(aggregate_metrics.get("return_skewness", 0.0), 0.0) < float(min_skew):
        reasons.append(f"return_skewness<{float(min_skew)}")

    max_kurt = cfg.get("max_return_excess_kurtosis")
    if max_kurt is not None and _to_float(aggregate_metrics.get("return_excess_kurtosis", 0.0), 0.0) > float(max_kurt):
        reasons.append(f"return_excess_kurtosis>{float(max_kurt)}")

    min_wl = cfg.get("min_win_loss_ratio")
    if min_wl is not None and _to_float(aggregate_metrics.get("win_loss_ratio", 0.0), 0.0) < float(min_wl):
        reasons.append(f"win_loss_ratio<{float(min_wl)}")

    # Soft flags.
    if _to_float(aggregate_metrics.get("turnover", 0.0), 0.0) > 100_000:
        soft_flags.append("high_turnover")
    if _to_float(aggregate_metrics.get("tail_ratio", 0.0), 0.0) < 0.8:
        soft_flags.append("weak_tail_ratio")
    if _to_float(aggregate_metrics.get("return_skewness", 0.0), 0.0) < -1.0:
        soft_flags.append("negative_skew")
    if _to_float(aggregate_metrics.get("return_excess_kurtosis", 0.0), 0.0) > 6.0:
        soft_flags.append("fat_tails")

    return {
        "passed_hard": len(reasons) == 0,
        "hard_reject_reasons": reasons,
        "soft_flags": soft_flags,
        "diagnostics": {
            "trade_count_mode": trade_count_mode,
            "trades_metric": trades_metric,
            "total_trades_across_folds": total_trades_across_folds,
            "fold_count": len(fold_metrics),
            "active_fold_count": active_fold_count,
            "positive_fold_ratio": positive_fold_ratio,
            "positive_fold_use_active_only": positive_fold_use_active_only,
        },
    }
