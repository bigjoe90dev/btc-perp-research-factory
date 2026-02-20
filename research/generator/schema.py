from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from research.strategies.registry import available_families


@dataclass(frozen=True)
class ParsedIdea:
    family: str
    name: str
    description: str
    params: dict[str, Any]
    expected_turnover: str
    confidence: float
    why_it_passes_gates: str
    suggested_timeframes: list[str]
    source_model: str
    source_lane: str


_ALLOWED_TURNOVER = {"low", "medium", "medium_high", "high"}


def _is_simple_value(value: Any) -> bool:
    return isinstance(value, (int, float, bool, str))


def _is_valid_param_value(value: Any) -> bool:
    if _is_simple_value(value):
        return True
    if isinstance(value, list):
        return len(value) <= 6 and all(_is_simple_value(v) for v in value)
    return False


def _payload_to_strategies(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("strategies"), list):
            return [x for x in payload.get("strategies", []) if isinstance(x, dict)]
        if isinstance(payload.get("candidates"), list):
            return [x for x in payload.get("candidates", []) if isinstance(x, dict)]
    return []


def parse_model_payload(
    content: str,
    model_id: str,
    lane_name: str,
) -> tuple[list[ParsedIdea], list[str]]:
    reasons: list[str] = []
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return [], [f"invalid_json:{exc}"]

    strategies = _payload_to_strategies(payload)
    if not strategies:
        return [], ["no_strategies_array"]

    allowed_families = set(available_families())
    parsed: list[ParsedIdea] = []

    for idx, row in enumerate(strategies):
        prefix = f"row_{idx}"
        family = str(row.get("family", "")).strip()
        if family not in allowed_families:
            reasons.append(f"{prefix}:unknown_family:{family}")
            continue

        params = row.get("params")
        if not isinstance(params, dict) or not params:
            reasons.append(f"{prefix}:invalid_params")
            continue
        bad_keys = [k for k, v in params.items() if not isinstance(k, str) or not _is_valid_param_value(v)]
        if bad_keys:
            reasons.append(f"{prefix}:invalid_param_values:{','.join([str(x) for x in bad_keys])}")
            continue

        name = str(row.get("name", "")).strip() or f"{family}_{idx}"
        description = str(row.get("description", "")).strip()
        why = str(row.get("why_it_passes_gates", "")).strip()
        if not description or not why:
            reasons.append(f"{prefix}:missing_description_or_reasoning")
            continue

        expected_turnover = str(row.get("expected_turnover", "medium")).strip().lower()
        if expected_turnover not in _ALLOWED_TURNOVER:
            reasons.append(f"{prefix}:invalid_turnover:{expected_turnover}")
            continue

        try:
            confidence = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            reasons.append(f"{prefix}:invalid_confidence")
            continue
        confidence = max(0.0, min(confidence, 10.0))

        tfs = row.get("suggested_timeframes", ["1h"])
        if not isinstance(tfs, list):
            reasons.append(f"{prefix}:invalid_suggested_timeframes")
            continue
        suggested_timeframes = [str(x) for x in tfs if str(x) in {"1m", "5m", "1h", "4h"}]
        if not suggested_timeframes:
            suggested_timeframes = ["1h"]

        parsed.append(
            ParsedIdea(
                family=family,
                name=name,
                description=description,
                params=params,
                expected_turnover=expected_turnover,
                confidence=confidence,
                why_it_passes_gates=why,
                suggested_timeframes=suggested_timeframes,
                source_model=model_id,
                source_lane=lane_name,
            )
        )

    if not parsed and not reasons:
        reasons.append("all_rows_filtered")
    return parsed, reasons
