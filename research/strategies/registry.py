from __future__ import annotations

from typing import Any

from .base import Strategy
from .families.liquidation_reversal import LiquidationReversalStrategy
from .families.meanrev_vwap import MeanReversionVWAPStrategy
from .families.momentum_breakout import MomentumBreakoutStrategy
from .families.trend_ma_regime import TrendMARegimeStrategy
from .families.volatility_expansion import VolatilityExpansionStrategy

REGISTRY = {
    "liquidation_reversal": LiquidationReversalStrategy,
    "meanrev_vwap": MeanReversionVWAPStrategy,
    "momentum_breakout": MomentumBreakoutStrategy,
    "trend_ma_regime": TrendMARegimeStrategy,
    "volatility_expansion": VolatilityExpansionStrategy,
}


def build_strategy(family: str, params: dict[str, Any]) -> Strategy:
    if family not in REGISTRY:
        raise ValueError(f"Unknown strategy family: {family}")
    return REGISTRY[family](params=params)


def available_families() -> list[str]:
    return sorted(REGISTRY.keys())
