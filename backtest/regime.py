"""
Market regime detection.

Classifies market bars into 4 regimes using ADX + ATR:
- trending_high_vol
- trending_low_vol
- ranging_high_vol
- ranging_low_vol

Tracks strategy performance per regime to ensure the strategy
isn't dependent on a single market condition.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REGIMES = [
    "trending_high_vol",
    "trending_low_vol",
    "ranging_high_vol",
    "ranging_low_vol",
]


@dataclass
class RegimeStats:
    regime: str
    bar_count: int = 0
    bar_pct: float = 0.0
    trade_count: int = 0
    avg_pnl: float = 0.0
    win_rate: float = 0.0
    total_pnl: float = 0.0


@dataclass
class RegimeReport:
    stats: Dict[str, RegimeStats] = field(default_factory=dict)
    dominant_regime: str = ""
    single_regime_dependent: bool = False

    def summary(self) -> str:
        lines = ["=== Market Regime Analysis ==="]
        for r in REGIMES:
            s = self.stats.get(r)
            if s and s.bar_count > 0:
                lines.append(
                    f"  {r:25s} | bars={s.bar_count:>6} ({s.bar_pct:>5.1f}%) | "
                    f"trades={s.trade_count:>4} | wr={s.win_rate:.0%} | pnl=${s.total_pnl:>+.2f}"
                )
        if self.single_regime_dependent:
            lines.append(f"\n  WARNING: Strategy appears dependent on '{self.dominant_regime}'")
        return "\n".join(lines)


def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                  period: int = 14) -> np.ndarray:
    """Average True Range."""
    n = len(highs)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1]),
        )
    # Simple moving average of TR
    atr = np.full(n, np.nan)
    for i in range(period - 1, n):
        atr[i] = np.mean(tr[i - period + 1: i + 1])
    return atr


def calculate_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                  period: int = 14) -> np.ndarray:
    """Average Directional Index (simplified Wilder's method)."""
    n = len(highs)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        up = highs[i] - highs[i-1]
        down = lows[i-1] - lows[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1]),
        )

    # Smoothed averages (simple MA for simplicity)
    adx = np.full(n, np.nan)
    for i in range(period - 1, n):
        sl = slice(i - period + 1, i + 1)
        atr_val = np.mean(tr[sl])
        if atr_val <= 0:
            continue
        plus_di = 100 * np.mean(plus_dm[sl]) / atr_val
        minus_di = 100 * np.mean(minus_dm[sl]) / atr_val
        di_sum = plus_di + minus_di
        if di_sum > 0:
            dx = 100 * abs(plus_di - minus_di) / di_sum
        else:
            dx = 0
        adx[i] = dx

    # Smooth ADX itself
    smooth_adx = np.full(n, np.nan)
    for i in range(2 * period - 2, n):
        sl = slice(i - period + 1, i + 1)
        vals = adx[sl]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            smooth_adx[i] = np.mean(valid)

    return smooth_adx


def classify_regime(adx_val: float, atr_val: float, median_atr: float,
                    adx_threshold: float = 25.0) -> str:
    """Classify a single bar into one of 4 regimes."""
    trending = adx_val > adx_threshold
    high_vol = atr_val > median_atr

    if trending and high_vol:
        return "trending_high_vol"
    elif trending:
        return "trending_low_vol"
    elif high_vol:
        return "ranging_high_vol"
    else:
        return "ranging_low_vol"


def label_regimes(df: pd.DataFrame, period: int = 14,
                  adx_threshold: float = 25.0) -> pd.Series:
    """
    Add regime labels to a DataFrame of quotes.

    Expects columns: mid_h/mid_l/mid_c (candles) or bid/ask (ticks).

    Returns a Series of regime strings aligned with df index.
    """
    if {"mid_h", "mid_l", "mid_c"}.issubset(df.columns):
        highs = df["mid_h"].values.astype(float)
        lows = df["mid_l"].values.astype(float)
        closes = df["mid_c"].values.astype(float)
    else:
        mid = ((df["bid"] + df["ask"]) / 2).values.astype(float)
        window = max(period, 5)
        mid_series = pd.Series(mid)
        highs = mid_series.rolling(window, min_periods=1).max().values
        lows = mid_series.rolling(window, min_periods=1).min().values
        closes = mid

    atr = calculate_atr(highs, lows, closes, period)
    adx = calculate_adx(highs, lows, closes, period)

    # Median ATR for threshold (use rolling 100-bar window)
    atr_series = pd.Series(atr)
    median_atr = atr_series.rolling(100, min_periods=period).median().values

    labels = []
    for i in range(len(df)):
        if np.isnan(adx[i]) or np.isnan(atr[i]) or np.isnan(median_atr[i]):
            labels.append("unknown")
        else:
            labels.append(classify_regime(adx[i], atr[i], median_atr[i], adx_threshold))

    return pd.Series(labels, index=df.index)


def regime_performance(trade_pnls: List[float],
                       trade_regimes: List[str]) -> RegimeReport:
    """
    Compute performance metrics broken down by market regime.

    Args:
        trade_pnls: P&L for each trade
        trade_regimes: Regime at the time of each trade entry
    """
    report = RegimeReport()

    # Initialize all regimes
    for r in REGIMES:
        report.stats[r] = RegimeStats(regime=r)

    total_trades = len(trade_pnls)

    for pnl, regime in zip(trade_pnls, trade_regimes):
        if regime not in report.stats:
            report.stats[regime] = RegimeStats(regime=regime)
        s = report.stats[regime]
        s.trade_count += 1
        s.total_pnl += pnl
        if pnl > 0:
            s.bar_count += 1  # reusing bar_count as win count for regime trades

    # Finalize stats
    for r, s in report.stats.items():
        if s.trade_count > 0:
            s.avg_pnl = s.total_pnl / s.trade_count
            s.win_rate = s.bar_count / s.trade_count  # bar_count = wins here
            s.bar_pct = (s.trade_count / total_trades * 100) if total_trades > 0 else 0

    # Check single-regime dependency
    if total_trades > 10:
        profitable_regimes = [
            r for r, s in report.stats.items()
            if s.trade_count >= 3 and s.total_pnl > 0
        ]
        all_regimes_with_trades = [
            r for r, s in report.stats.items() if s.trade_count >= 3
        ]

        if len(all_regimes_with_trades) >= 2 and len(profitable_regimes) <= 1:
            report.single_regime_dependent = True
            if profitable_regimes:
                report.dominant_regime = profitable_regimes[0]

    return report
