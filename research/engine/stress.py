from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd


RunnerFn = Callable[[Dict[str, Any], Dict[str, Any]], Tuple[Dict[str, Any], Any]]


def run_stress_suite(
    base_params: dict[str, Any],
    base_metrics: dict[str, Any],
    base_sim: Any,
    backtest_cfg: dict[str, Any],
    runner: RunnerFn,
) -> dict[str, Any]:
    gates_cfg = backtest_cfg.get("gates", {})

    # Cost robustness sweep.
    levels = list(gates_cfg.get("cost_robustness_levels", [1.25, 1.5]))
    cost_rows: list[dict[str, Any]] = []
    cost_pass = True
    for mult in levels:
        cfg2 = deepcopy(backtest_cfg)
        exec_cfg = cfg2.setdefault("execution", {})
        exec_cfg["taker_fee_bps"] = float(exec_cfg.get("taker_fee_bps", 0.0)) * float(mult)

        slip = exec_cfg.setdefault("slippage", {})
        slip["bps_base"] = float(slip.get("bps_base", 0.0)) * float(mult)
        slip["bps_per_vol"] = float(slip.get("bps_per_vol", 0.0)) * float(mult)

        m, _ = runner(base_params, cfg2)
        cost_rows.append({"multiplier": mult, "total_return": m.get("total_return"), "sharpe": m.get("sharpe")})
        if float(m.get("total_return", 0.0)) <= 0.0:
            cost_pass = False

    # Latency/microstructure robustness.
    latency_levels = list(gates_cfg.get("latency_delay_seconds", [2, 5]))
    latency_rows: list[dict[str, Any]] = []
    latency_pass = True
    for sec in latency_levels:
        cfg2 = deepcopy(backtest_cfg)
        exec_cfg = cfg2.setdefault("execution", {})
        exec_cfg["execution_delay_seconds"] = float(sec)
        m, _ = runner(base_params, cfg2)
        latency_rows.append({"delay_seconds": sec, "total_return": m.get("total_return"), "sharpe": m.get("sharpe")})
        if float(m.get("total_return", 0.0)) <= 0.0:
            latency_pass = False

    # Parameter perturbation (+/- step for integer-like params).
    step = int(gates_cfg.get("parameter_perturbation_step", 1))
    perturb_rows: list[dict[str, Any]] = []
    for key, val in base_params.items():
        if isinstance(val, bool):
            continue
        if isinstance(val, int):
            for d in (-step, step):
                new_val = val + d
                if new_val <= 0:
                    continue
                p2 = dict(base_params)
                p2[key] = new_val
                m, _ = runner(p2, backtest_cfg)
                perturb_rows.append(
                    {
                        "param": key,
                        "value": new_val,
                        "total_return": m.get("total_return"),
                        "sharpe": m.get("sharpe"),
                    }
                )

    perturb_pass = True
    if perturb_rows:
        good = sum(1 for x in perturb_rows if float(x.get("total_return", 0.0)) > 0.0)
        perturb_pass = (good / len(perturb_rows)) >= 0.5

    # Bootstrap fragility from trade outcomes.
    pnls = np.array([float(t.pnl_after_costs) for t in base_sim.trades], dtype=float)
    seed = int(backtest_cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    prob_negative = 1.0
    if len(pnls) > 0:
        n_iter = 500
        neg = 0
        for _ in range(n_iter):
            sample = rng.choice(pnls, size=len(pnls), replace=True)
            if float(sample.sum()) <= 0.0:
                neg += 1
        prob_negative = float(neg / n_iter)

    # Regime split: high/low vol buckets on bar returns.
    eq = base_sim.equity_curve.sort_values("ts_utc").reset_index(drop=True)
    eq_ret = eq["equity"].pct_change().fillna(0.0)
    close_ret = eq["close"].pct_change().fillna(0.0)
    vol = close_ret.rolling(24, min_periods=8).std().fillna(0.0)
    med = float(vol.median())
    high_mask = vol >= med
    low_mask = ~high_mask

    high_ret = float(eq_ret[high_mask].sum()) if high_mask.any() else 0.0
    low_ret = float(eq_ret[low_mask].sum()) if low_mask.any() else 0.0

    return {
        "cost_sweep": {
            "pass": bool(cost_pass),
            "rows": cost_rows,
        },
        "latency_sweep": {
            "pass": bool(latency_pass),
            "rows": latency_rows,
        },
        "parameter_perturbation": {
            "pass": bool(perturb_pass),
            "rows": perturb_rows,
        },
        "bootstrap": {
            "prob_negative": prob_negative,
            "trades": int(len(pnls)),
        },
        "regimes": {
            "high_vol_return": high_ret,
            "low_vol_return": low_ret,
        },
        "base": {
            "total_return": base_metrics.get("total_return"),
            "sharpe": base_metrics.get("sharpe"),
        },
    }
