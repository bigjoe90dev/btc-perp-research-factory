from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research.engine.types import CandidateSpec
from research.strategies.candidate_ids import strategy_id

from .schema import ParsedIdea


@dataclass(frozen=True)
class CandidateDraft:
    spec: CandidateSpec
    idea: ParsedIdea


_FAMILY_LIMITS: dict[str, dict[str, tuple[float, float, type]]] = {
    "liquidation_reversal": {
        "range_lookback": (5, 500, int),
        "volume_lookback": (5, 500, int),
        "range_mult": (1.0, 20.0, float),
        "volume_mult": (1.0, 20.0, float),
        "reclaim_frac": (0.1, 0.99, float),
        "time_stop_bars": (1, 300, int),
        "adverse_stop_pct": (0.0001, 0.2, float),
    },
    "meanrev_vwap": {
        "vwap_lookback": (2, 500, int),
        "entry_dev_pct": (0.0001, 0.05, float),
        "exit_dev_pct": (0.00001, 0.05, float),
        "vol_lookback": (2, 500, int),
        "max_vol_pct": (0.0001, 0.2, float),
        "time_stop_bars": (1, 500, int),
    },
    "momentum_breakout": {
        "breakout_lookback": (2, 500, int),
        "atr_lookback": (2, 300, int),
        "atr_min_pct": (0.0, 0.1, float),
        "time_stop_bars": (1, 500, int),
        "trailing_stop_atr": (0.1, 20.0, float),
    },
    "trend_ma_regime": {
        "fast_ma": (2, 500, int),
        "slow_ma": (3, 1000, int),
        "atr_lookback": (2, 300, int),
        "atr_stop_mult": (0.1, 20.0, float),
        "trend_buffer_pct": (0.0, 0.05, float),
        "time_stop_bars": (1, 1000, int),
    },
    "volatility_expansion": {
        "range_lookback": (2, 500, int),
        "expansion_mult": (1.0, 20.0, float),
        "confirm_break_pct": (0.0, 0.05, float),
        "atr_lookback": (2, 300, int),
        "atr_stop_mult": (0.1, 20.0, float),
        "time_stop_bars": (1, 500, int),
    },
}


def _choose_scalar(v: Any, *, allow_param_ranges: bool) -> Any:
    if isinstance(v, list):
        if not allow_param_ranges:
            raise ValueError("param_ranges_not_allowed")
        if len(v) >= 2:
            return v[1]
        if v:
            return v[0]
    return v


def _coerce_value(raw: Any, out_type: type) -> float | int:
    if out_type is int:
        return int(round(float(raw)))
    return float(raw)


def _materialize_params(
    family: str,
    params: dict[str, Any],
    *,
    allow_param_ranges: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    limits = _FAMILY_LIMITS.get(family)
    if limits is None:
        return None, f"unsupported_family:{family}"

    unknown = sorted(set(params.keys()) - set(limits.keys()))
    if unknown:
        return None, f"unknown_param:{unknown[0]}"

    concrete: dict[str, Any] = {}
    for k, (lo, hi, out_type) in limits.items():
        if k not in params:
            return None, f"missing_param:{k}"
        try:
            raw = _choose_scalar(params[k], allow_param_ranges=allow_param_ranges)
        except ValueError:
            return None, f"param_range_not_allowed:{k}"
        try:
            val = _coerce_value(raw, out_type)
        except (TypeError, ValueError):
            return None, f"bad_param_type:{k}"
        if float(val) < lo or float(val) > hi:
            return None, f"param_out_of_range:{k}"
        concrete[k] = val

    if family == "trend_ma_regime" and int(concrete["fast_ma"]) >= int(concrete["slow_ma"]):
        return None, "invalid_ma_order"
    if family == "meanrev_vwap" and float(concrete["exit_dev_pct"]) >= float(concrete["entry_dev_pct"]):
        return None, "invalid_meanrev_dev_order"
    return concrete, None


def validate_family_params_for_candidate_file(
    family: str,
    params: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    return _materialize_params(family=family, params=params, allow_param_ranges=False)


def ideas_to_candidate_drafts(
    ideas: list[ParsedIdea],
    requested_timeframes: list[str],
    rules_version: str,
    dataset_key: str,
) -> tuple[list[CandidateDraft], list[str]]:
    out: list[CandidateDraft] = []
    rejects: list[str] = []

    req_tfs = [str(x) for x in requested_timeframes]
    for idx, idea in enumerate(ideas):
        params, err = _materialize_params(idea.family, idea.params, allow_param_ranges=True)
        if err is not None:
            rejects.append(f"idea_{idx}:{err}")
            continue

        suggested = set(idea.suggested_timeframes)
        candidate_tfs = [tf for tf in req_tfs if tf in suggested] or req_tfs
        for tf in candidate_tfs:
            sid = strategy_id(
                family=idea.family,
                timeframe=tf,
                params=params,
                rules_version=rules_version,
                dataset_key=dataset_key,
            )
            out.append(
                CandidateDraft(
                    spec=CandidateSpec(
                        strategy_id=sid,
                        family=idea.family,
                        timeframe=tf,
                        params=params,
                        rules_version=rules_version,
                        dataset_key=dataset_key,
                    ),
                    idea=idea,
                )
            )

    return out, rejects
