from __future__ import annotations

from research.engine.types import CandidateSpec
from research.generator.adapter import CandidateDraft
from research.generator.prefilter import apply_prefilter
from research.generator.schema import ParsedIdea


def _draft(strategy_id: str, params: dict, confidence: float) -> CandidateDraft:
    return CandidateDraft(
        spec=CandidateSpec(
            strategy_id=strategy_id,
            family="momentum_breakout",
            timeframe="1h",
            params=params,
            rules_version="v1",
            dataset_key="BTC_BITMEX_PERP_1M",
        ),
        idea=ParsedIdea(
            family="momentum_breakout",
            name=strategy_id,
            description="desc",
            params=params,
            expected_turnover="medium",
            confidence=confidence,
            why_it_passes_gates="reason",
            suggested_timeframes=["1h"],
            source_model="x-ai/grok-4.1-fast",
            source_lane="grok_generate",
        ),
    )


def test_prefilter_removes_duplicates_and_high_overlap() -> None:
    d1 = _draft("a", {"breakout_lookback": 20, "atr_lookback": 14}, confidence=9.0)
    d2 = _draft("a", {"breakout_lookback": 20, "atr_lookback": 14}, confidence=8.0)
    d3 = _draft("b", {"breakout_lookback": 20, "atr_lookback": 14}, confidence=7.0)
    d4 = _draft("c", {"breakout_lookback": 100, "atr_lookback": 20}, confidence=6.0)

    out = apply_prefilter(
        drafts=[d1, d2, d3, d4],
        max_count=5,
        diversity_param_overlap_max=0.7,
    )

    ids = [x.spec.strategy_id for x in out.selected]
    assert "a" in ids
    assert "c" in ids
    assert "b" not in ids
    assert any("duplicate_strategy_id" in x or "high_param_overlap" in x for x in out.rejected)
