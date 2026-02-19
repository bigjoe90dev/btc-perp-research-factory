from __future__ import annotations

from research.strategies.generator import generate_candidates


def test_candidate_generation_is_deterministic() -> None:
    a = generate_candidates(
        families=["momentum_breakout"],
        timeframes=["5m"],
        count=5,
        rules_version="v1",
        dataset_key="BTC_HYPERLIQUID_PERP_1M",
    )
    b = generate_candidates(
        families=["momentum_breakout"],
        timeframes=["5m"],
        count=5,
        rules_version="v1",
        dataset_key="BTC_HYPERLIQUID_PERP_1M",
    )
    assert [x.strategy_id for x in a] == [x.strategy_id for x in b]
