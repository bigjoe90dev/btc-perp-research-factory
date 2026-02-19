from __future__ import annotations

import hashlib
import json
from itertools import product
from typing import Any

from research.engine.types import CandidateSpec


def _strategy_id(family: str, timeframe: str, params: dict[str, Any], rules_version: str, dataset_key: str) -> str:
    payload = {
        "family": family,
        "timeframe": timeframe,
        "params": params,
        "rules_version": rules_version,
        "dataset_key": dataset_key,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _momentum_grid() -> list[dict[str, Any]]:
    lookbacks = [20, 50, 100]
    atr_lbs = [14, 20]
    atr_min = [0.0007, 0.0010]
    time_stops = [24, 48]
    trail = [1.5, 2.0]

    grid: list[dict[str, Any]] = []
    for a, b, c, d, e in product(lookbacks, atr_lbs, atr_min, time_stops, trail):
        grid.append(
            {
                "breakout_lookback": a,
                "atr_lookback": b,
                "atr_min_pct": c,
                "time_stop_bars": d,
                "trailing_stop_atr": e,
            }
        )
    return grid


def _trend_ma_grid() -> list[dict[str, Any]]:
    fast_slow = [(10, 40), (20, 80), (30, 120)]
    atr_lbs = [14, 20]
    atr_stop = [1.5, 2.0]
    buffers = [0.0, 0.0005]
    time_stop = [72]

    grid: list[dict[str, Any]] = []
    for (f, s), a, st, b, t in product(fast_slow, atr_lbs, atr_stop, buffers, time_stop):
        grid.append(
            {
                "fast_ma": f,
                "slow_ma": s,
                "atr_lookback": a,
                "atr_stop_mult": st,
                "trend_buffer_pct": b,
                "time_stop_bars": t,
            }
        )
    return grid


def _meanrev_vwap_grid() -> list[dict[str, Any]]:
    lookbacks = [24, 48, 96]
    entry = [0.0015, 0.0025]
    exit_levels = [0.0005, 0.0010]
    vol_max = [0.0030, 0.0050]

    grid: list[dict[str, Any]] = []
    for lb, en, ex, vm in product(lookbacks, entry, exit_levels, vol_max):
        if ex >= en:
            continue
        grid.append(
            {
                "vwap_lookback": lb,
                "entry_dev_pct": en,
                "exit_dev_pct": ex,
                "vol_lookback": 36,
                "max_vol_pct": vm,
                "time_stop_bars": 48,
            }
        )
    return grid


def _volatility_expansion_grid() -> list[dict[str, Any]]:
    range_lb = [20, 50]
    expansion = [1.8, 2.2]
    confirm = [0.0003, 0.0006]
    stop = [1.6, 2.0]

    grid: list[dict[str, Any]] = []
    for r, e, c, s in product(range_lb, expansion, confirm, stop):
        grid.append(
            {
                "range_lookback": r,
                "expansion_mult": e,
                "confirm_break_pct": c,
                "atr_lookback": 20,
                "atr_stop_mult": s,
                "time_stop_bars": 24,
            }
        )
    return grid


def _liquidation_reversal_grid() -> list[dict[str, Any]]:
    range_lb = [20, 40]
    vol_lb = [20, 40]
    range_mult = [2.0, 2.5]
    volume_mult = [2.0]
    reclaim = [0.60, 0.70]
    time_stop = [8, 12]

    grid: list[dict[str, Any]] = []
    for r, v, rm, vm, rc, ts in product(range_lb, vol_lb, range_mult, volume_mult, reclaim, time_stop):
        grid.append(
            {
                "range_lookback": r,
                "volume_lookback": v,
                "range_mult": rm,
                "volume_mult": vm,
                "reclaim_frac": rc,
                "time_stop_bars": ts,
                "adverse_stop_pct": 0.008,
            }
        )
    return grid


def generate_candidates(
    families: list[str],
    timeframes: list[str],
    count: int,
    rules_version: str,
    dataset_key: str,
) -> list[CandidateSpec]:
    pools: dict[str, list[dict[str, Any]]] = {
        "liquidation_reversal": _liquidation_reversal_grid(),
        "meanrev_vwap": _meanrev_vwap_grid(),
        "momentum_breakout": _momentum_grid(),
        "trend_ma_regime": _trend_ma_grid(),
        "volatility_expansion": _volatility_expansion_grid(),
    }

    specs: list[CandidateSpec] = []
    for family in families:
        if family not in pools:
            continue
        for timeframe in timeframes:
            for params in pools[family]:
                sid = _strategy_id(family, timeframe, params, rules_version, dataset_key)
                specs.append(
                    CandidateSpec(
                        strategy_id=sid,
                        family=family,
                        timeframe=timeframe,
                        params=params,
                        rules_version=rules_version,
                        dataset_key=dataset_key,
                    )
                )

    specs = sorted(specs, key=lambda x: (x.family, x.timeframe, x.strategy_id))

    target = max(0, int(count))
    if target == 0 or not specs:
        return []

    # Keep deterministic ordering while avoiding family/timeframe concentration
    # when the requested count is smaller than the full generated universe.
    buckets: dict[tuple[str, str], list[CandidateSpec]] = {}
    for spec in specs:
        key = (spec.family, spec.timeframe)
        buckets.setdefault(key, []).append(spec)

    keys = sorted(buckets.keys())
    selected: list[CandidateSpec] = []
    while len(selected) < target:
        added = False
        for key in keys:
            bucket = buckets[key]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            added = True
            if len(selected) >= target:
                break
        if not added:
            break
    return selected
