from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from research.data.funding import align_funding_to_bars
from research.data.integrity import validate_candles_frame, validate_funding_frame
from research.data.loader import DataBundle, detect_datasets_from_manifest, load_dataset_from_manifest
from research.engine.cv import walk_forward_splits
from research.engine.gates import evaluate_gates
from research.engine.metrics import compute_metrics
from research.engine.scoring import adjusted_score_for_multiple_testing, raw_score
from research.engine.simulator import run_backtest
from research.engine.stress import run_stress_suite
from research.engine.types import SimulationResult
from research.reports.charts import write_equity_snapshots
from research.reports.checklist import write_checklist_artifacts
from research.reports.daily import write_daily_report
from research.strategies.generator import generate_candidates
from research.strategies.registry import build_strategy


def _read_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def _run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def _expected_seconds_for_timeframe(tf: str) -> int:
    if tf == "1m":
        return 60
    if tf == "5m":
        return 300
    if tf == "1h":
        return 3600
    raise ValueError(f"Unsupported timeframe: {tf}")


def _aggregate_fold_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"ok": False, "reason": "no_fold_metrics"}
    keys = [
        "total_return",
        "max_drawdown",
        "sharpe",
        "sortino",
        "win_rate",
        "avg_trade",
        "trades_count",
        "exposure",
        "turnover",
        "turnover_ratio",
        "profit_factor",
        "expectancy",
        "avg_win",
        "avg_loss",
        "win_loss_ratio",
        "time_in_market",
        "worst_1d",
        "worst_1w",
        "tail_ratio",
        "drawdown_duration",
        "ulcer_index",
        "return_skewness",
        "return_excess_kurtosis",
        "trades_per_day",
        "avg_bar_range",
    ]
    out: dict[str, Any] = {"ok": True}
    for k in keys:
        vals = [float(x.get(k, 0.0)) for x in rows]
        out[k] = float(sum(vals) / len(vals))
    out["fold_count"] = len(rows)
    out["fold_total_return_std"] = float(pd.Series([x.get("total_return", 0.0) for x in rows]).std(ddof=0))
    return out


def _serialize_candidate(spec: Any) -> dict[str, Any]:
    return {
        "strategy_id": spec.strategy_id,
        "family": spec.family,
        "timeframe": spec.timeframe,
        "params": spec.params,
        "rules_version": spec.rules_version,
        "dataset_key": spec.dataset_key,
    }


def _parse_ts_or_none(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp value: {value}")
    return pd.Timestamp(ts)


def _slice_bundle_by_time(
    bundle: DataBundle,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
    label: str,
    funding_end_tolerance_hours: float,
) -> DataBundle:
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(f"{label}: start_ts is after end_ts")

    c = bundle.candles
    mask = pd.Series(True, index=c.index)
    if start_ts is not None:
        mask = mask & (c["ts_utc"] >= start_ts)
    if end_ts is not None:
        mask = mask & (c["ts_utc"] <= end_ts)
    sliced_c = c.loc[mask].reset_index(drop=True)
    if sliced_c.empty:
        raise RuntimeError(f"{label}: candle window is empty for timeframe={bundle.timeframe}")

    f = bundle.funding
    fmask = pd.Series(True, index=f.index)
    if start_ts is not None:
        fmask = fmask & (f["ts_utc"] >= start_ts)
    if end_ts is not None:
        fmask = fmask & (f["ts_utc"] <= end_ts)
    sliced_f = f.loc[fmask].reset_index(drop=True)
    if sliced_f.empty:
        raise RuntimeError(f"{label}: funding window is empty for timeframe={bundle.timeframe}")

    c_start = pd.Timestamp(sliced_c["ts_utc"].iloc[0])
    c_end = pd.Timestamp(sliced_c["ts_utc"].iloc[-1])
    f_start = pd.Timestamp(sliced_f["ts_utc"].iloc[0])
    f_end = pd.Timestamp(sliced_f["ts_utc"].iloc[-1])
    tol = pd.Timedelta(hours=max(float(funding_end_tolerance_hours), 0.0))
    if f_start > c_start or (f_end + tol) < c_end:
        raise RuntimeError(
            f"{label}: funding does not span candle window for timeframe={bundle.timeframe}: "
            f"candles=[{c_start},{c_end}] funding=[{f_start},{f_end}] tol={tol}"
        )

    return DataBundle(
        dataset=bundle.dataset,
        candles_1m=bundle.candles_1m,
        candles=sliced_c,
        funding=sliced_f,
        timeframe=bundle.timeframe,
    )


def _build_window_bundles(
    bundles: dict[str, DataBundle],
    window_cfg: dict[str, Any] | None,
    label: str,
    funding_end_tolerance_hours: float,
) -> dict[str, DataBundle] | None:
    if not isinstance(window_cfg, dict):
        return None
    start_ts = _parse_ts_or_none(window_cfg.get("start_ts_utc"))
    end_ts = _parse_ts_or_none(window_cfg.get("end_ts_utc"))
    if start_ts is None and end_ts is None:
        return None
    out: dict[str, DataBundle] = {}
    for tf, bundle in bundles.items():
        out[tf] = _slice_bundle_by_time(
            bundle=bundle,
            start_ts=start_ts,
            end_ts=end_ts,
            label=label,
            funding_end_tolerance_hours=funding_end_tolerance_hours,
        )
    return out


def _evaluate_window_metrics(
    family: str,
    params: dict[str, Any],
    bundle: DataBundle,
    timeframe: str,
    backtest_cfg: dict[str, Any],
) -> dict[str, Any]:
    funding_by_bar = align_funding_to_bars(bundle.candles["ts_utc"], bundle.funding)
    strat = build_strategy(family, params)
    sim = run_backtest(
        candles=bundle.candles,
        funding_rates_by_bar=funding_by_bar,
        strategy=strat,
        timeframe=timeframe,
        backtest_cfg=backtest_cfg,
    )
    return compute_metrics(
        equity_curve=sim.equity_curve,
        trades=sim.trades,
        bars_per_year=int(sim.summary.get("bars_per_year", 1)),
    )


def _equity_returns(equity_curve: pd.DataFrame) -> pd.Series:
    eq = equity_curve.sort_values("ts_utc").copy()
    out = eq.set_index("ts_utc")["equity"].pct_change().fillna(0.0)
    out.name = "ret"
    return out


def _apply_correlation_gate(
    ranked: list[dict[str, Any]],
    backtest_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    gates_cfg = backtest_cfg.get("gates", {})
    if not bool(gates_cfg.get("enable_correlation_gate", True)):
        for row in ranked:
            row["selected"] = bool(row.get("gates", {}).get("passed_hard", False))
        return ranked

    max_corr = float(gates_cfg.get("max_return_correlation", 0.75))
    use_abs = bool(gates_cfg.get("correlation_use_absolute", False))
    min_points = int(gates_cfg.get("correlation_min_points", 100))

    selected_returns: dict[str, pd.Series] = {}
    for row in ranked:
        row["selected"] = False
        gates = row.setdefault("gates", {})
        if not bool(gates.get("passed_hard", False)):
            continue

        cur = _equity_returns(row["equity_curve"])
        reject_reason = None
        for sid, baseline in selected_returns.items():
            joined = pd.concat([cur, baseline], axis=1, join="inner").dropna()
            if len(joined) < min_points:
                continue
            corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            cmp = abs(corr) if use_abs else corr
            if cmp > max_corr:
                reject_reason = f"correlation>{max_corr:.2f} vs {sid} ({corr:.3f})"
                break

        if reject_reason is not None:
            reasons = gates.setdefault("hard_reject_reasons", [])
            if reject_reason not in reasons:
                reasons.append(reject_reason)
            gates["passed_hard"] = False
            row["selected"] = False
            continue

        selected_returns[str(row.get("strategy_id"))] = cur
        row["selected"] = True

    return ranked


def _print_detected_datasets(data_cfg_path: str) -> list[dict[str, Any]]:
    rows = detect_datasets_from_manifest(data_cfg_path)
    print("Detected datasets from manifest:")
    for row in rows:
        print(
            "- {dataset_id} | candles={canonical_file} | funding={funding_file} | "
            "start={start_ts_utc} | end={end_ts_utc}".format(**row)
        )
    return rows


def _bars_per_year_from_cfg(timeframe: str, cfg: dict[str, Any]) -> int:
    metrics_cfg = cfg.get("metrics", {})
    if timeframe == "5m":
        return int(metrics_cfg.get("annualization_bars_5m", 105_120))
    if timeframe == "1h":
        return int(metrics_cfg.get("annualization_bars_1h", 8_760))
    return 525_600


def _stitch_fold_equity(fold_equity_curves: list[pd.DataFrame], initial_equity: float) -> pd.DataFrame:
    required_cols = ["ts_utc", "equity", "cash", "position_qty", "close", "high", "low", "gross_notional", "trade_notional"]
    stitched: list[pd.DataFrame] = []
    running_equity = float(initial_equity)

    for frame in fold_equity_curves:
        if frame.empty:
            continue
        eq = frame.sort_values("ts_utc").copy().reset_index(drop=True)
        for col in required_cols:
            if col not in eq.columns:
                eq[col] = pd.NaT if col == "ts_utc" else 0.0

        # Compound fold returns so combined equity is OOS-only and continuous.
        ret = eq["equity"].pct_change().fillna(0.0)
        eq["equity"] = running_equity * (1.0 + ret).cumprod()
        running_equity = float(eq["equity"].iloc[-1])

        stitched.append(eq[required_cols].copy())

    if not stitched:
        return pd.DataFrame(columns=required_cols)

    out = pd.concat(stitched, ignore_index=True)
    out = out.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last").reset_index(drop=True)
    return out


def _build_oos_simulation(fold_sims: list[SimulationResult], initial_equity: float) -> SimulationResult:
    equity_curve = _stitch_fold_equity([sim.equity_curve for sim in fold_sims], initial_equity=initial_equity)
    trades = sorted(
        [t for sim in fold_sims for t in sim.trades],
        key=lambda x: (x.entry_ts_utc, x.exit_ts_utc),
    )
    summary = {
        "bars": len(equity_curve),
        "trades": len(trades),
        "folds": len(fold_sims),
        "mode": "oos_stitched",
    }
    return SimulationResult(equity_curve=equity_curve, trades=trades, fills=[], funding_events=[], summary=summary)


def _runner_for_candidate(
    bundle,
    funding_by_bar,
    splits,
    family: str,
    timeframe: str,
    backtest_cfg: dict[str, Any],
):
    def _runner(params: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        fold_sims: list[SimulationResult] = []
        for sp in splits:
            test_df = bundle.candles.iloc[sp.test_idx].reset_index(drop=True)
            test_funding = funding_by_bar[sp.test_idx]
            strat = build_strategy(family, params)
            sim = run_backtest(
                candles=test_df,
                funding_rates_by_bar=test_funding,
                strategy=strat,
                timeframe=timeframe,
                backtest_cfg=cfg,
            )
            fold_sims.append(sim)

        sim = _build_oos_simulation(
            fold_sims=fold_sims,
            initial_equity=float(cfg.get("initial_equity", backtest_cfg.get("initial_equity", 10_000.0))),
        )
        metrics = compute_metrics(
            equity_curve=sim.equity_curve,
            trades=sim.trades,
            bars_per_year=_bars_per_year_from_cfg(timeframe=timeframe, cfg=cfg),
        )
        return metrics, sim

    return _runner


def _evaluate_candidate(
    spec,
    bundle,
    funding_by_bar,
    backtest_cfg: dict[str, Any],
) -> dict[str, Any]:
    cv_cfg = backtest_cfg.get("cv", {})
    splits = walk_forward_splits(
        n_bars=len(bundle.candles),
        train_bars=int(cv_cfg.get("train_bars", 400)),
        test_bars=int(cv_cfg.get("test_bars", 160)),
        step_bars=int(cv_cfg.get("step_bars", 160)),
        purge_bars=int(cv_cfg.get("purge_bars", 5)),
        embargo_bars=int(cv_cfg.get("embargo_bars", 5)),
    )
    if not splits:
        # Fallback for short histories.
        idx = pd.RangeIndex(start=0, stop=len(bundle.candles), step=1)
        splits = [
            type("Split", (), {"fold_id": 0, "train_idx": idx[:0].to_numpy(), "test_idx": idx.to_numpy()})
        ]

    fold_metrics: list[dict[str, Any]] = []
    fold_sims: list[SimulationResult] = []
    for sp in splits:
        test_df = bundle.candles.iloc[sp.test_idx].reset_index(drop=True)
        test_funding = funding_by_bar[sp.test_idx]

        strat = build_strategy(spec.family, spec.params)
        sim = run_backtest(
            candles=test_df,
            funding_rates_by_bar=test_funding,
            strategy=strat,
            timeframe=spec.timeframe,
            backtest_cfg=backtest_cfg,
        )
        fold_sims.append(sim)
        met = compute_metrics(
            equity_curve=sim.equity_curve,
            trades=sim.trades,
            bars_per_year=int(sim.summary.get("bars_per_year", 1)),
        )
        met["fold_id"] = int(sp.fold_id)
        fold_metrics.append(met)

    aggregate = _aggregate_fold_metrics(fold_metrics)
    oos_sim = _build_oos_simulation(
        fold_sims=fold_sims,
        initial_equity=float(backtest_cfg.get("initial_equity", 10_000.0)),
    )
    base_metrics = compute_metrics(
        equity_curve=oos_sim.equity_curve,
        trades=oos_sim.trades,
        bars_per_year=_bars_per_year_from_cfg(timeframe=spec.timeframe, cfg=backtest_cfg),
    )

    runner = _runner_for_candidate(
        bundle=bundle,
        funding_by_bar=funding_by_bar,
        splits=splits,
        family=spec.family,
        timeframe=spec.timeframe,
        backtest_cfg=backtest_cfg,
    )

    stress = run_stress_suite(
        base_params=spec.params,
        base_metrics=base_metrics,
        base_sim=oos_sim,
        backtest_cfg=backtest_cfg,
        runner=runner,
    )

    gates = evaluate_gates(
        aggregate_metrics=aggregate,
        fold_metrics=fold_metrics,
        stress=stress,
        cfg=backtest_cfg.get("gates", {}),
    )

    score_cfg = backtest_cfg.get("scoring", {})
    raw = raw_score(metrics=aggregate, n_params=len(spec.params), cfg=score_cfg)

    return {
        "strategy_id": spec.strategy_id,
        "family": spec.family,
        "timeframe": spec.timeframe,
        "params": spec.params,
        "rules_version": spec.rules_version,
        "dataset_key": spec.dataset_key,
        "fold_metrics": fold_metrics,
        "aggregate_metrics": aggregate,
        "base_metrics": base_metrics,
        "stress": stress,
        "gates": gates,
        "score_raw": raw,
        "score_adjusted": raw,
        "equity_curve": oos_sim.equity_curve,
        "trades": oos_sim.trades,
    }


def _write_run_artifacts(
    run_dir: Path,
    data_cfg_path: str,
    backtest_cfg_path: str,
    manifest_path: str,
    candidates: list[Any],
    outcomes: list[dict[str, Any]],
    report_path: Path,
    checklist_paths: dict[str, str] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    Path(run_dir / "config_snapshot.yml").write_text(Path(backtest_cfg_path).read_text(encoding="utf-8"), encoding="utf-8")
    Path(run_dir / "data_config_snapshot.yml").write_text(Path(data_cfg_path).read_text(encoding="utf-8"), encoding="utf-8")
    Path(run_dir / "manifest_snapshot.yml").write_text(Path(manifest_path).read_text(encoding="utf-8"), encoding="utf-8")

    cands = [_serialize_candidate(x) for x in candidates]
    (run_dir / "candidates.json").write_text(json.dumps(cands, indent=2), encoding="utf-8")

    folds = {
        x["strategy_id"]: x["fold_metrics"] for x in outcomes
    }
    (run_dir / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")

    gates = {
        x["strategy_id"]: x["gates"] for x in outcomes
    }
    (run_dir / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")

    rows = []
    for x in outcomes:
        m = x["aggregate_metrics"]
        rows.append(
            {
                "strategy_id": x["strategy_id"],
                "family": x["family"],
                "timeframe": x["timeframe"],
                "score_raw": x["score_raw"],
                "score_adjusted": x["score_adjusted"],
                "passed_hard": x["gates"].get("passed_hard", False),
                "selected": x.get("selected", False),
                "hard_reject_reasons": ";".join(x["gates"].get("hard_reject_reasons", [])),
                "total_return": m.get("total_return"),
                "sharpe": m.get("sharpe"),
                "sortino": m.get("sortino"),
                "max_drawdown": m.get("max_drawdown"),
                "trades_count": m.get("trades_count"),
                "turnover": m.get("turnover"),
                "turnover_ratio": m.get("turnover_ratio"),
                "win_loss_ratio": m.get("win_loss_ratio"),
                "return_skewness": m.get("return_skewness"),
                "return_excess_kurtosis": m.get("return_excess_kurtosis"),
                "oos_total_return": (x.get("oos_metrics") or {}).get("total_return"),
                "oos_sharpe": (x.get("oos_metrics") or {}).get("sharpe"),
                "oos_trades_count": (x.get("oos_metrics") or {}).get("trades_count"),
                "robustness_total_return": (x.get("robustness_metrics") or {}).get("total_return"),
                "robustness_sharpe": (x.get("robustness_metrics") or {}).get("sharpe"),
            }
        )
    pd.DataFrame(rows).to_parquet(run_dir / "results.parquet", index=False)

    trade_rows = []
    for x in outcomes:
        for t in x["trades"]:
            trade_rows.append(
                {
                    "strategy_id": x["strategy_id"],
                    "entry_ts_utc": t.entry_ts_utc,
                    "exit_ts_utc": t.exit_ts_utc,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "qty": t.qty,
                    "pnl": t.pnl,
                    "pnl_after_costs": t.pnl_after_costs,
                    "bars_held": t.bars_held,
                }
            )
    if trade_rows:
        pd.DataFrame(trade_rows).to_parquet(run_dir / "trades.parquet", index=False)

    logs = f"run_dir={run_dir}\nreport={report_path}\noutcomes={len(outcomes)}\n"
    if checklist_paths:
        logs += (
            f"checklist_summary={checklist_paths.get('summary_md')}\n"
            f"checklist_csv={checklist_paths.get('csv')}\n"
            f"checklist_parquet={checklist_paths.get('parquet')}\n"
        )
    (run_dir / "logs.txt").write_text(logs, encoding="utf-8")


def _execute_run(
    data_cfg_path: str,
    backtest_cfg_path: str,
    candidates_n: int,
    timeframes: list[str],
    families: list[str],
) -> tuple[str, Path]:
    data_cfg = _read_yaml(data_cfg_path)
    backtest_cfg = _read_yaml(backtest_cfg_path)
    dataset_key = str(data_cfg.get("primary_dataset_key", "BTC_HYPERLIQUID_PERP_1M"))

    _print_detected_datasets(data_cfg_path)

    bundles: dict[str, Any] = {}
    for tf in timeframes:
        bundle = load_dataset_from_manifest(data_config_path=data_cfg_path, timeframe=tf)
        c_val = validate_candles_frame(bundle.candles, expected_seconds=_expected_seconds_for_timeframe(tf))
        f_val = validate_funding_frame(bundle.funding)
        if not c_val.ok:
            raise RuntimeError(f"Candle integrity failed ({tf}): {c_val.errors}")
        if not f_val.ok:
            raise RuntimeError(f"Funding integrity failed ({tf}): {f_val.errors}")
        bundles[tf] = bundle

    windowing_cfg = backtest_cfg.get("windowing", {})
    windowing_enabled = bool(windowing_cfg.get("enabled", False))
    funding_tol_hours = float(data_cfg.get("loader", {}).get("funding_end_tolerance_hours", 12.0))

    optimization_bundles = None
    oos_bundles = None
    robustness_bundles = None
    if windowing_enabled:
        optimization_bundles = _build_window_bundles(
            bundles=bundles,
            window_cfg=windowing_cfg.get("optimization"),
            label="optimization_window",
            funding_end_tolerance_hours=funding_tol_hours,
        )
        oos_bundles = _build_window_bundles(
            bundles=bundles,
            window_cfg=windowing_cfg.get("out_of_sample"),
            label="oos_window",
            funding_end_tolerance_hours=funding_tol_hours,
        )
        robustness_bundles = _build_window_bundles(
            bundles=bundles,
            window_cfg=windowing_cfg.get("robustness"),
            label="robustness_window",
            funding_end_tolerance_hours=funding_tol_hours,
        )

    eval_bundles = optimization_bundles if (windowing_enabled and optimization_bundles is not None) else bundles

    rules_version = str(backtest_cfg.get("rules_version", "v1"))
    candidates = generate_candidates(
        families=families,
        timeframes=timeframes,
        count=candidates_n,
        rules_version=rules_version,
        dataset_key=dataset_key,
    )
    if not candidates:
        raise RuntimeError("No candidates generated")

    outcomes: list[dict[str, Any]] = []
    for i, spec in enumerate(candidates, start=1):
        bundle = eval_bundles[spec.timeframe]
        funding_by_bar = align_funding_to_bars(bundle.candles["ts_utc"], bundle.funding)
        outcome = _evaluate_candidate(spec=spec, bundle=bundle, funding_by_bar=funding_by_bar, backtest_cfg=backtest_cfg)
        outcomes.append(outcome)
        print(
            f"[{i}/{len(candidates)}] {spec.strategy_id} {spec.family} {spec.timeframe} "
            f"ret={outcome['aggregate_metrics'].get('total_return', 0.0):.4f} "
            f"sharpe={outcome['aggregate_metrics'].get('sharpe', 0.0):.3f} "
            f"gate={'PASS' if outcome['gates'].get('passed_hard') else 'FAIL'}"
        )

    ranked = sorted(outcomes, key=lambda x: float(x["score_raw"]), reverse=True)
    total = len(ranked)
    for rank, row in enumerate(ranked, start=1):
        row["score_adjusted"] = adjusted_score_for_multiple_testing(float(row["score_raw"]), rank=rank, total=total)

    ranked = sorted(ranked, key=lambda x: float(x["score_adjusted"]), reverse=True)

    if windowing_enabled and oos_bundles is not None:
        gates_cfg = backtest_cfg.get("gates", {})
        require_pos_oos = bool(gates_cfg.get("require_positive_oos_window", True))
        min_oos_trades = int(gates_cfg.get("min_oos_trades", 0))
        require_nonneg_robust = bool(gates_cfg.get("require_nonnegative_robustness_window", False))

        for row in ranked:
            row["oos_metrics"] = None
            row["robustness_metrics"] = None

            if not bool(row.get("gates", {}).get("passed_hard", False)):
                continue

            oos_bundle = oos_bundles.get(str(row["timeframe"]))
            if oos_bundle is not None:
                oos_m = _evaluate_window_metrics(
                    family=str(row["family"]),
                    params=dict(row["params"]),
                    bundle=oos_bundle,
                    timeframe=str(row["timeframe"]),
                    backtest_cfg=backtest_cfg,
                )
                row["oos_metrics"] = oos_m
                reasons = row["gates"].setdefault("hard_reject_reasons", [])
                if require_pos_oos and float(oos_m.get("total_return", 0.0)) <= 0.0:
                    reasons.append("oos_window_non_positive")
                if min_oos_trades > 0 and int(oos_m.get("trades_count", 0)) < min_oos_trades:
                    reasons.append(f"oos_trades<{min_oos_trades}")
                if reasons:
                    row["gates"]["passed_hard"] = False

            if row["gates"].get("passed_hard", False) and robustness_bundles is not None:
                rb_bundle = robustness_bundles.get(str(row["timeframe"]))
                if rb_bundle is not None:
                    rb_m = _evaluate_window_metrics(
                        family=str(row["family"]),
                        params=dict(row["params"]),
                        bundle=rb_bundle,
                        timeframe=str(row["timeframe"]),
                        backtest_cfg=backtest_cfg,
                    )
                    row["robustness_metrics"] = rb_m
                    if require_nonneg_robust and float(rb_m.get("total_return", 0.0)) < 0.0:
                        row["gates"].setdefault("hard_reject_reasons", []).append(
                            "robustness_window_negative"
                        )
                        row["gates"]["passed_hard"] = False

    ranked = _apply_correlation_gate(ranked, backtest_cfg=backtest_cfg)

    run_id = _run_id()
    run_dir = Path("research/artefacts/runs") / run_id

    primary = eval_bundles[timeframes[0]]
    primary_full = bundles[timeframes[0]]
    ds_summary = {
        "dataset_id": primary.dataset.dataset_id,
        "candles_path": str(primary.dataset.canonical_file),
        "funding_path": str(primary.dataset.funding_file),
        "candles_start": str(primary.candles["ts_utc"].iloc[0]),
        "candles_end": str(primary.candles["ts_utc"].iloc[-1]),
        "candle_rows": int(len(primary.candles)),
        "gaps_found": int(validate_candles_frame(primary.candles, _expected_seconds_for_timeframe(primary.timeframe)).summary.get("gaps_found") or 0),
        "duplicates_found": int(validate_candles_frame(primary.candles, _expected_seconds_for_timeframe(primary.timeframe)).summary.get("duplicates_found") or 0),
        "funding_start": str(primary.funding["ts_utc"].iloc[0]),
        "funding_end": str(primary.funding["ts_utc"].iloc[-1]),
        "funding_rows": int(len(primary.funding)),
        "full_candles_start": str(primary_full.candles["ts_utc"].iloc[0]),
        "full_candles_end": str(primary_full.candles["ts_utc"].iloc[-1]),
        "windowing_enabled": windowing_enabled,
        "optimization_window_start": str(windowing_cfg.get("optimization", {}).get("start_ts_utc")) if windowing_enabled else None,
        "optimization_window_end": str(windowing_cfg.get("optimization", {}).get("end_ts_utc")) if windowing_enabled else None,
        "oos_window_start": str(windowing_cfg.get("out_of_sample", {}).get("start_ts_utc")) if windowing_enabled else None,
        "oos_window_end": str(windowing_cfg.get("out_of_sample", {}).get("end_ts_utc")) if windowing_enabled else None,
        "robustness_window_start": str(windowing_cfg.get("robustness", {}).get("start_ts_utc")) if windowing_enabled else None,
        "robustness_window_end": str(windowing_cfg.get("robustness", {}).get("end_ts_utc")) if windowing_enabled else None,
    }

    assumptions = [
        "Bar-close decision with next-bar-open execution.",
        "Market orders use adverse slippage and taker fees.",
        "Execution enforces participation cap and exchange notional/precision constraints.",
        "Funding applied as discrete events aligned to first bar ts >= funding timestamp.",
        "No leverage above x1; risk_fraction controls position notional.",
    ]
    unknowns = [
        "Hyperliquid 1m candle history in this environment is currently limited to recent days.",
        "No orderbook-level simulation yet; slippage is parametric.",
        "Proxy liquidation signals are candle/volume based (no direct liquidation feed).",
    ]

    report_cfg = backtest_cfg.get("reporting", {})
    report_path = write_daily_report(
        run_dir=run_dir,
        dataset_summary=ds_summary,
        outcomes=ranked,
        assumptions=assumptions,
        unknowns=unknowns,
        top_overall=int(report_cfg.get("top_overall", 10)),
        top_per_family=int(report_cfg.get("top_per_family", 3)),
    )

    top_for_chart = ranked[: min(5, len(ranked))]
    write_equity_snapshots(run_dir=run_dir, top_rows=top_for_chart)

    checklist_paths = write_checklist_artifacts(
        run_dir=run_dir,
        outcomes=ranked,
        gates_cfg=backtest_cfg.get("gates", {}),
    )
    print(f"Checklist summary: {checklist_paths.get('summary_md')}")

    manifest_path = str(data_cfg.get("manifest_path", "research/data_manifest.yml"))
    _write_run_artifacts(
        run_dir=run_dir,
        data_cfg_path=data_cfg_path,
        backtest_cfg_path=backtest_cfg_path,
        manifest_path=manifest_path,
        candidates=candidates,
        outcomes=ranked,
        report_path=report_path,
        checklist_paths=checklist_paths,
    )

    print(f"Run complete: {run_id}")
    print(f"Report: {report_path}")
    return run_id, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC Hyperliquid research CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_smoke = sub.add_parser("smoke_test", help="Run small end-to-end vertical slice")
    p_smoke.add_argument("--data-config", default="research/config/data.yml")
    p_smoke.add_argument("--backtest-config", default="research/config/backtest.yml")

    p_run = sub.add_parser("run", help="Run candidate batch")
    p_run.add_argument("--data-config", default="research/config/data.yml")
    p_run.add_argument("--backtest-config", default="research/config/backtest.yml")
    p_run.add_argument("--candidates", type=int, default=300)
    p_run.add_argument("--timeframes", nargs="+", default=["5m", "1h"])

    p_report = sub.add_parser("report", help="Show report path for run id")
    p_report.add_argument("--run_id", required=True)

    p_detect = sub.add_parser("detect_datasets", help="Print datasets from manifest")
    p_detect.add_argument("--data-config", default="research/config/data.yml")

    args = parser.parse_args()

    if args.cmd == "detect_datasets":
        _print_detected_datasets(args.data_config)
        return

    if args.cmd == "smoke_test":
        back_cfg = _read_yaml(args.backtest_config)
        smoke_n = int(back_cfg.get("candidate_generation", {}).get("smoke_count", 5))
        fams = list(back_cfg.get("candidate_generation", {}).get("families", ["momentum_breakout"]))
        _execute_run(
            data_cfg_path=args.data_config,
            backtest_cfg_path=args.backtest_config,
            candidates_n=smoke_n,
            timeframes=["5m"],
            families=fams,
        )
        return

    if args.cmd == "run":
        back_cfg = _read_yaml(args.backtest_config)
        fams = list(back_cfg.get("candidate_generation", {}).get("families", ["momentum_breakout"]))
        _execute_run(
            data_cfg_path=args.data_config,
            backtest_cfg_path=args.backtest_config,
            candidates_n=int(args.candidates),
            timeframes=[str(x) for x in args.timeframes],
            families=fams,
        )
        return

    if args.cmd == "report":
        run_dir = Path("research/artefacts/runs") / str(args.run_id)
        report_path = run_dir / "report.md"
        if not report_path.exists():
            raise FileNotFoundError(f"Report not found: {report_path}")
        print(report_path)
        return


if __name__ == "__main__":
    main()
