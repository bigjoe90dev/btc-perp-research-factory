from __future__ import annotations

import json
from pathlib import Path

from research.generator.data_summary import DataSummary
from research.generator.openrouter_client import LLMResponse
from research.generator.pipeline import generate_candidates_with_llm


def test_pipeline_writes_consistent_candidate_files(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Summary: {data_summary_json_block}", encoding="utf-8")

    monkeypatch.setattr(
        "research.generator.pipeline.build_data_summary",
        lambda *args, **kwargs: DataSummary(timeframe="1h", summary={"timeframe": "1h"}),
    )

    calls = {"n": 0}

    def _fake_chat_json(self, model_id, system_prompt, user_prompt, max_tokens, temperature=0.0, top_p=1.0):
        calls["n"] += 1
        payload = {
            "strategies": [
                {
                    "family": "momentum_breakout",
                    "name": "alpha",
                    "description": "desc",
                    "params": {
                        "breakout_lookback": 50,
                        "atr_lookback": 20,
                        "atr_min_pct": 0.001,
                        "time_stop_bars": 48,
                        "trailing_stop_atr": 2.0,
                    },
                    "expected_turnover": "medium",
                    "confidence": 8.0,
                    "why_it_passes_gates": "reason",
                    "suggested_timeframes": ["1h"],
                }
            ]
        }
        return LLMResponse(
            model_id=model_id,
            content=json.dumps(payload),
            prompt_tokens=100,
            completion_tokens=50,
            raw={"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
        )

    monkeypatch.setattr("research.generator.openrouter_client.OpenRouterClient.chat_json", _fake_chat_json)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    backtest_cfg = {
        "rules_version": "v1",
        "llm_generation": {
            "prompt": {"path": str(prompt_path), "version": "test_prompt_v1"},
            "model_lanes": {
                "minimax": {"model_id": "minimax/minimax-m2.5", "calls": 0},
                "kimi": {"model_id": "moonshotai/kimi-k2.5", "calls": 0},
                "deepseek": {"model_id": "deepseek/deepseek-v3.2", "calls": 0},
                "grok_generate": {"model_id": "x-ai/grok-4.1-fast", "calls": 1},
            },
            "synthesis": {"model_id": "x-ai/grok-4.1-fast"},
            "fail_closed_on_missing_grok": True,
            "limits": {
                "max_total_calls": 3,
                "max_completion_tokens_per_call": 400,
                "max_total_completion_tokens": 2000,
            },
            "generation": {
                "top_n_final": 1,
                "strategies_per_call": 1,
                "diversity_param_overlap_max": 0.7,
            },
            "improvement_round": {"enabled": False, "model_id": "deepseek/deepseek-v3.2", "calls": 0},
            "pricing": {"default_input_per_1k": 0.0, "default_output_per_1k": 0.0},
            "api": {"temperature": 0.0, "top_p": 1.0, "max_retries": 0, "timeout_seconds": 1},
        },
    }

    out_path = tmp_path / "candidates_out.json"
    result = generate_candidates_with_llm(
        data_config_path="unused.yml",
        backtest_cfg=backtest_cfg,
        count=1,
        timeframes=["1h"],
        dataset_key="BTC_BITMEX_PERP_1M",
        rules_version="v1",
        out_path=str(out_path),
        estimate_only=False,
    )

    assert calls["n"] == 2
    assert result.candidate_file == out_path

    out_payload = json.loads(out_path.read_text(encoding="utf-8"))
    artefact_payload = json.loads((result.artefact_dir / "candidates_validated.json").read_text(encoding="utf-8"))
    enriched_payload = json.loads((result.artefact_dir / "candidates_validated_enriched.json").read_text(encoding="utf-8"))

    assert out_payload == artefact_payload
    assert "source_model" not in out_payload["candidates"][0]
    assert "source_model" in enriched_payload["candidates"][0]
    assert "candidate_file_enriched" in result.manifest.get("outputs", {})


def test_pipeline_improvement_round_cannot_mutate_source_params(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Summary: {data_summary_json_block}", encoding="utf-8")

    monkeypatch.setattr(
        "research.generator.pipeline.build_data_summary",
        lambda *args, **kwargs: DataSummary(timeframe="1h", summary={"timeframe": "1h"}),
    )

    calls = {"n": 0}

    def _fake_chat_json(self, model_id, system_prompt, user_prompt, max_tokens, temperature=0.0, top_p=1.0):
        calls["n"] += 1
        if calls["n"] == 1:
            lookback = 50
        elif calls["n"] == 2:
            lookback = 300
        else:
            lookback = 400
        payload = {
            "strategies": [
                {
                    "family": "momentum_breakout",
                    "name": "alpha",
                    "description": "desc",
                    "params": {
                        "breakout_lookback": lookback,
                        "atr_lookback": 20,
                        "atr_min_pct": 0.001,
                        "time_stop_bars": 48,
                        "trailing_stop_atr": 2.0,
                    },
                    "expected_turnover": "medium",
                    "confidence": 8.0,
                    "why_it_passes_gates": "reason",
                    "suggested_timeframes": ["1h"],
                }
            ]
        }
        return LLMResponse(
            model_id=model_id,
            content=json.dumps(payload),
            prompt_tokens=100,
            completion_tokens=50,
            raw={"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
        )

    monkeypatch.setattr("research.generator.openrouter_client.OpenRouterClient.chat_json", _fake_chat_json)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    backtest_cfg = {
        "rules_version": "v1",
        "llm_generation": {
            "prompt": {"path": str(prompt_path), "version": "test_prompt_v1"},
            "model_lanes": {
                "minimax": {"model_id": "minimax/minimax-m2.5", "calls": 0},
                "kimi": {"model_id": "moonshotai/kimi-k2.5", "calls": 0},
                "deepseek": {"model_id": "deepseek/deepseek-v3.2", "calls": 0},
                "grok_generate": {"model_id": "x-ai/grok-4.1-fast", "calls": 1},
            },
            "synthesis": {"model_id": "x-ai/grok-4.1-fast"},
            "fail_closed_on_missing_grok": True,
            "limits": {
                "max_total_calls": 4,
                "max_completion_tokens_per_call": 400,
                "max_total_completion_tokens": 2000,
            },
            "generation": {
                "top_n_final": 1,
                "strategies_per_call": 1,
                "diversity_param_overlap_max": 0.7,
            },
            "improvement_round": {"enabled": True, "model_id": "deepseek/deepseek-v3.2", "calls": 1},
            "pricing": {"default_input_per_1k": 0.0, "default_output_per_1k": 0.0},
            "api": {"temperature": 0.0, "top_p": 1.0, "max_retries": 0, "timeout_seconds": 1},
        },
    }

    result = generate_candidates_with_llm(
        data_config_path="unused.yml",
        backtest_cfg=backtest_cfg,
        count=1,
        timeframes=["1h"],
        dataset_key="BTC_BITMEX_PERP_1M",
        rules_version="v1",
        out_path=str(tmp_path / "candidates_out.json"),
        estimate_only=False,
    )

    assert calls["n"] == 3
    assert result.candidates[0].params["breakout_lookback"] == 50
