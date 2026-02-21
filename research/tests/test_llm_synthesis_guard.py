from __future__ import annotations

from research.generator.pipeline import _restrict_ideas_to_source
from research.generator.schema import ParsedIdea


def _idea(name: str, family: str = "momentum_breakout") -> ParsedIdea:
    return ParsedIdea(
        family=family,
        name=name,
        description="desc",
        params={
            "breakout_lookback": 50,
            "atr_lookback": 20,
            "atr_min_pct": 0.001,
            "time_stop_bars": 48,
            "trailing_stop_atr": 2.0,
        },
        expected_turnover="medium",
        confidence=8.0,
        why_it_passes_gates="reason",
        suggested_timeframes=["1h"],
        source_model="x-ai/grok-4.1-fast",
        source_lane="lane",
    )


def test_restrict_ideas_to_source_rejects_invented_entries() -> None:
    source = [_idea("alpha"), _idea("beta")]
    candidates = [_idea("alpha"), _idea("invented")]

    kept, rejects = _restrict_ideas_to_source(candidate_ideas=candidates, source_ideas=source)
    assert len(kept) == 1
    assert kept[0].name == "alpha"
    assert any("synthesis_invented_strategy" in r for r in rejects)


def test_restrict_ideas_to_source_uses_original_source_params() -> None:
    source = [_idea("alpha")]
    mutated = _idea("alpha")
    mutated.params["breakout_lookback"] = 200

    kept, rejects = _restrict_ideas_to_source(candidate_ideas=[mutated], source_ideas=source)
    assert rejects == []
    assert len(kept) == 1
    assert kept[0].params["breakout_lookback"] == 50
