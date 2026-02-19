from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _criterion_score_map(row: dict[str, Any], gates_cfg: dict[str, Any]) -> dict[str, bool | None]:
    metrics = row.get("aggregate_metrics", {}) or {}
    stress = row.get("stress", {}) or {}
    gates = row.get("gates", {}) or {}
    diag = gates.get("diagnostics", {}) or {}
    oos = row.get("oos_metrics")

    min_trades = int(gates_cfg.get("min_trades", 0))
    min_fold_ratio = float(gates_cfg.get("min_positive_fold_ratio", 0.0))
    max_dd_cap = float(gates_cfg.get("max_drawdown_pct", 1.0))
    max_bootstrap = float(gates_cfg.get("max_bootstrap_prob_negative", 1.0))
    max_kurt = gates_cfg.get("max_return_excess_kurtosis")
    min_wl = gates_cfg.get("min_win_loss_ratio")

    trades_metric = _to_float(diag.get("trades_metric", metrics.get("trades_count", 0.0)), 0.0)
    fold_ratio = diag.get("positive_fold_ratio")
    if fold_ratio is None:
        fold_ratio = 0.0
    fold_ratio = _to_float(fold_ratio, 0.0)

    checks: dict[str, bool | None] = {
        "enough_activity": trades_metric >= min_trades,
        "fold_consistency": fold_ratio >= min_fold_ratio,
        "profitable": _to_float(metrics.get("total_return", 0.0), 0.0) > 0.0,
        "sharpe_positive": _to_float(metrics.get("sharpe", 0.0), 0.0) > 0.0,
        "drawdown_within_cap": abs(_to_float(metrics.get("max_drawdown", 0.0), 0.0)) <= max_dd_cap,
        "cost_robust": bool(stress.get("cost_sweep", {}).get("pass", False)),
        "latency_robust": bool(stress.get("latency_sweep", {}).get("pass", False)),
        "parameter_stable": bool(stress.get("parameter_perturbation", {}).get("pass", False)),
        "bootstrap_ok": _to_float(stress.get("bootstrap", {}).get("prob_negative", 1.0), 1.0) <= max_bootstrap,
    }

    if max_kurt is None:
        checks["tail_kurtosis_ok"] = None
    else:
        checks["tail_kurtosis_ok"] = _to_float(metrics.get("return_excess_kurtosis", 0.0), 0.0) <= float(max_kurt)

    if min_wl is None:
        checks["win_loss_ok"] = None
    else:
        checks["win_loss_ok"] = _to_float(metrics.get("win_loss_ratio", 0.0), 0.0) >= float(min_wl)

    if isinstance(oos, dict):
        checks["oos_positive"] = _to_float(oos.get("total_return", 0.0), 0.0) > 0.0
    else:
        checks["oos_positive"] = None

    return checks


def build_checklist_scores(outcomes: list[dict[str, Any]], gates_cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in outcomes:
        checks = _criterion_score_map(row=row, gates_cfg=gates_cfg)
        active_keys = [k for k, v in checks.items() if v is not None]
        pass_keys = [k for k, v in checks.items() if v is True]
        fail_keys = [k for k, v in checks.items() if v is False]
        na_keys = [k for k, v in checks.items() if v is None]

        check_count = len(active_keys)
        pass_count = len(pass_keys)
        score_pct = (100.0 * pass_count / check_count) if check_count > 0 else 0.0

        rows.append(
            {
                "strategy_id": row.get("strategy_id"),
                "family": row.get("family"),
                "timeframe": row.get("timeframe"),
                "score_adjusted": _to_float(row.get("score_adjusted", 0.0), 0.0),
                "passed_hard": bool(row.get("gates", {}).get("passed_hard", False)),
                "checklist_score_pct": score_pct,
                "checklist_pass_count": pass_count,
                "checklist_check_count": check_count,
                "checklist_failed_items": ";".join(fail_keys),
                "checklist_not_evaluated": ";".join(na_keys),
                **{f"chk_{k}": checks[k] for k in sorted(checks.keys())},
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        ["checklist_score_pct", "score_adjusted", "passed_hard"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return out


def write_checklist_artifacts(run_dir: Path, outcomes: list[dict[str, Any]], gates_cfg: dict[str, Any]) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    scores = build_checklist_scores(outcomes=outcomes, gates_cfg=gates_cfg)

    parquet_path = run_dir / "checklist_scores.parquet"
    csv_path = run_dir / "checklist_scores.csv"
    md_path = run_dir / "checklist_summary.md"

    if scores.empty:
        scores.to_parquet(parquet_path, index=False)
        scores.to_csv(csv_path, index=False)
        md_path.write_text("# Strategy Checklist Summary\n\nNo outcomes available.\n", encoding="utf-8")
        return {
            "parquet": str(parquet_path),
            "csv": str(csv_path),
            "summary_md": str(md_path),
        }

    scores.to_parquet(parquet_path, index=False)
    scores.to_csv(csv_path, index=False)

    lines: list[str] = []
    lines.append("# Strategy Checklist Summary")
    lines.append("")
    lines.append(f"- candidates_scored: `{len(scores)}`")
    lines.append(f"- hard_gate_passed: `{int(scores['passed_hard'].sum())}`")
    lines.append("")
    lines.append("## Top Checklist Scores")
    lines.append("| strategy_id | family | tf | checklist_score | pass/checks | hard_gate | failed_items |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    top = scores.head(15)
    for _, r in top.iterrows():
        lines.append(
            "| {sid} | {fam} | {tf} | {score:.1f}% | {passed}/{total} | {gate} | {failed} |".format(
                sid=r.get("strategy_id"),
                fam=r.get("family"),
                tf=r.get("timeframe"),
                score=_to_float(r.get("checklist_score_pct", 0.0), 0.0),
                passed=int(_to_float(r.get("checklist_pass_count", 0.0), 0.0)),
                total=int(_to_float(r.get("checklist_check_count", 0.0), 0.0)),
                gate="PASS" if bool(r.get("passed_hard", False)) else "FAIL",
                failed=(r.get("checklist_failed_items") or "none"),
            )
        )
    lines.append("")
    lines.append("## Criteria Keys")
    lines.append("- `enough_activity`: enough trade activity for this run")
    lines.append("- `fold_consistency`: enough positive test slices")
    lines.append("- `profitable`: positive overall return")
    lines.append("- `sharpe_positive`: positive risk-adjusted return")
    lines.append("- `drawdown_within_cap`: drawdown under cap")
    lines.append("- `cost_robust`: survives higher costs")
    lines.append("- `latency_robust`: survives slower execution")
    lines.append("- `parameter_stable`: survives small parameter changes")
    lines.append("- `bootstrap_ok`: low fragility under bootstrap")
    lines.append("- `tail_kurtosis_ok`: no extreme tail concentration")
    lines.append("- `win_loss_ok`: average win/loss quality check")
    lines.append("- `oos_positive`: positive out-of-sample window (if evaluated)")
    lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "parquet": str(parquet_path),
        "csv": str(csv_path),
        "summary_md": str(md_path),
    }
