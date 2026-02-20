from __future__ import annotations

from research.generator.adapter import ideas_to_candidate_drafts
from research.generator.schema import ParsedIdea


def test_ideas_to_candidate_drafts_accepts_valid_family_params() -> None:
    ideas = [
        ParsedIdea(
            family="momentum_breakout",
            name="momo",
            description="desc",
            params={
                "breakout_lookback": [20, 50, 100],
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
            source_lane="grok_generate",
        )
    ]

    drafts, rejects = ideas_to_candidate_drafts(
        ideas=ideas,
        requested_timeframes=["1h", "5m"],
        rules_version="v1",
        dataset_key="BTC_BITMEX_PERP_1M",
    )
    assert rejects == []
    assert len(drafts) == 1
    assert drafts[0].spec.family == "momentum_breakout"
    assert drafts[0].spec.timeframe == "1h"


def test_ideas_to_candidate_drafts_rejects_invalid_param_range() -> None:
    ideas = [
        ParsedIdea(
            family="momentum_breakout",
            name="bad",
            description="desc",
            params={
                "breakout_lookback": -5,
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
            source_lane="grok_generate",
        )
    ]

    drafts, rejects = ideas_to_candidate_drafts(
        ideas=ideas,
        requested_timeframes=["1h"],
        rules_version="v1",
        dataset_key="BTC_BITMEX_PERP_1M",
    )
    assert drafts == []
    assert any("param_out_of_range" in r for r in rejects)
