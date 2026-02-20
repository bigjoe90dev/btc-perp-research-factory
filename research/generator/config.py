from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelLane:
    name: str
    model_id: str
    calls: int


@dataclass(frozen=True)
class BudgetConfig:
    max_total_calls: int = 12
    max_completion_tokens_per_call: int = 1800
    max_total_completion_tokens: int = 22000


@dataclass(frozen=True)
class PricingConfig:
    default_input_per_1k: float = 0.0
    default_output_per_1k: float = 0.0
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMGeneratorConfig:
    prompt_path: Path
    prompt_version: str
    openrouter_base_url: str
    timeout_seconds: float
    max_retries: int
    lanes: list[ModelLane]
    synthesis_model_id: str
    fail_closed_on_missing_grok: bool
    top_n_final: int
    strategies_per_call: int
    diversity_param_overlap_max: float
    improvement_round_enabled: bool
    improvement_model_id: str
    improvement_calls: int
    budget: BudgetConfig
    pricing: PricingConfig


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return default


def llm_config_from_backtest(backtest_cfg: dict[str, Any]) -> LLMGeneratorConfig:
    root = backtest_cfg.get("llm_generation", {}) if isinstance(backtest_cfg, dict) else {}

    prompt_cfg = root.get("prompt", {}) if isinstance(root.get("prompt"), dict) else {}
    prompt_path = Path(str(prompt_cfg.get("path", "research/generator/prompts/master_btc_perp_generator_prompt_v1.md")))
    prompt_version = str(prompt_cfg.get("version", "master_btc_perp_generator_prompt_v1"))

    api_cfg = root.get("api", {}) if isinstance(root.get("api"), dict) else {}
    base_url = str(api_cfg.get("base_url", "https://openrouter.ai/api/v1"))
    timeout_seconds = _as_float(api_cfg.get("timeout_seconds", 60.0), 60.0)
    max_retries = max(_as_int(api_cfg.get("max_retries", 3), 3), 0)

    lanes_cfg = root.get("model_lanes", {}) if isinstance(root.get("model_lanes"), dict) else {}

    def _lane(name: str, default_model: str, default_calls: int) -> ModelLane:
        lane_cfg = lanes_cfg.get(name, {}) if isinstance(lanes_cfg.get(name), dict) else {}
        model_id = str(lane_cfg.get("model_id", default_model))
        calls = max(_as_int(lane_cfg.get("calls", default_calls), default_calls), 0)
        return ModelLane(name=name, model_id=model_id, calls=calls)

    lanes = [
        _lane("minimax", "minimax/minimax-m2.5", 2),
        _lane("kimi", "moonshotai/kimi-k2.5", 2),
        _lane("deepseek", "deepseek/deepseek-v3.2", 2),
        _lane("grok_generate", "x-ai/grok-4.1-fast", 4),
    ]

    synth_cfg = root.get("synthesis", {}) if isinstance(root.get("synthesis"), dict) else {}
    synthesis_model_id = str(synth_cfg.get("model_id", "x-ai/grok-4.1-fast"))
    fail_closed_on_missing_grok = _as_bool(root.get("fail_closed_on_missing_grok", True), True)

    gen_cfg = root.get("generation", {}) if isinstance(root.get("generation"), dict) else {}
    top_n_final = max(_as_int(gen_cfg.get("top_n_final", 12), 12), 1)
    strategies_per_call = max(_as_int(gen_cfg.get("strategies_per_call", 8), 8), 1)
    diversity_param_overlap_max = min(max(_as_float(gen_cfg.get("diversity_param_overlap_max", 0.70), 0.70), 0.0), 1.0)

    improve_cfg = root.get("improvement_round", {}) if isinstance(root.get("improvement_round"), dict) else {}
    improvement_round_enabled = _as_bool(improve_cfg.get("enabled", False), False)
    improvement_model_id = str(improve_cfg.get("model_id", "deepseek/deepseek-v3.2"))
    improvement_calls = max(_as_int(improve_cfg.get("calls", 1), 1), 0)

    budget_cfg = root.get("limits", {}) if isinstance(root.get("limits"), dict) else {}
    budget = BudgetConfig(
        max_total_calls=max(_as_int(budget_cfg.get("max_total_calls", 12), 12), 1),
        max_completion_tokens_per_call=max(_as_int(budget_cfg.get("max_completion_tokens_per_call", 1800), 1800), 1),
        max_total_completion_tokens=max(_as_int(budget_cfg.get("max_total_completion_tokens", 22000), 22000), 1),
    )

    price_cfg = root.get("pricing", {}) if isinstance(root.get("pricing"), dict) else {}
    by_model = price_cfg.get("by_model", {}) if isinstance(price_cfg.get("by_model"), dict) else {}
    pricing = PricingConfig(
        default_input_per_1k=_as_float(price_cfg.get("default_input_per_1k", 0.0), 0.0),
        default_output_per_1k=_as_float(price_cfg.get("default_output_per_1k", 0.0), 0.0),
        by_model={str(k): v for k, v in by_model.items() if isinstance(v, dict)},
    )

    return LLMGeneratorConfig(
        prompt_path=prompt_path,
        prompt_version=prompt_version,
        openrouter_base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        lanes=lanes,
        synthesis_model_id=synthesis_model_id,
        fail_closed_on_missing_grok=fail_closed_on_missing_grok,
        top_n_final=top_n_final,
        strategies_per_call=strategies_per_call,
        diversity_param_overlap_max=diversity_param_overlap_max,
        improvement_round_enabled=improvement_round_enabled,
        improvement_model_id=improvement_model_id,
        improvement_calls=improvement_calls,
        budget=budget,
        pricing=pricing,
    )
