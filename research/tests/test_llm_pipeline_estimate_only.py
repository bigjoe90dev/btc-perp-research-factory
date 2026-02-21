from __future__ import annotations

from pathlib import Path

from research.generator.data_summary import DataSummary
from research.generator.pipeline import generate_candidates_with_llm


def test_generate_candidates_estimate_only_without_data(monkeypatch, tmp_path: Path) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("no data")

    monkeypatch.setattr("research.generator.pipeline.build_data_summary", _boom)

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Summary: {data_summary_json_block}", encoding="utf-8")

    backtest_cfg = {
        "rules_version": "v1",
        "llm_generation": {
            "prompt": {
                "path": str(prompt_path),
                "version": "test_prompt_v1",
            }
        },
    }

    out_path = tmp_path / "candidates.json"
    result = generate_candidates_with_llm(
        data_config_path="missing.yml",
        backtest_cfg=backtest_cfg,
        count=5,
        timeframes=["1h"],
        dataset_key="BTC_BITMEX_PERP_1M",
        rules_version="v1",
        out_path=str(out_path),
        estimate_only=True,
    )

    assert result.candidates == []
    assert result.candidate_file.exists()
    assert "estimated_total_cost_usd" in result.manifest
    assert any("pricing_all_zero" in x for x in result.manifest.get("warnings", []))


def test_generate_candidates_estimate_only_builds_summary_for_each_timeframe(monkeypatch, tmp_path: Path) -> None:
    seen: list[str] = []

    def _fake_summary(*args, **kwargs):
        timeframe = str(kwargs.get("timeframe"))
        seen.append(timeframe)
        return DataSummary(timeframe=timeframe, summary={"timeframe": timeframe, "num_bars": 100})

    monkeypatch.setattr("research.generator.pipeline.build_data_summary", _fake_summary)

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Summary: {data_summary_json_block}", encoding="utf-8")

    backtest_cfg = {
        "rules_version": "v1",
        "llm_generation": {
            "prompt": {
                "path": str(prompt_path),
                "version": "test_prompt_v1",
            }
        },
    }

    result = generate_candidates_with_llm(
        data_config_path="unused.yml",
        backtest_cfg=backtest_cfg,
        count=5,
        timeframes=["1h", "5m"],
        dataset_key="BTC_BITMEX_PERP_1M",
        rules_version="v1",
        out_path=str(tmp_path / "candidates.json"),
        estimate_only=True,
    )

    assert result.candidates == []
    assert seen == ["1h", "5m"]
    tf_summaries = result.manifest.get("data_summary", {}).get("timeframe_summaries", {})
    assert set(tf_summaries.keys()) == {"1h", "5m"}
