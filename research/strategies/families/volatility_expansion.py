from __future__ import annotations

from typing import Any

import numpy as np

from ..base import Strategy, StrategyContext, StrategySignal


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int) -> float:
    if len(high) < lookback + 1:
        return 0.0
    h = high[-(lookback + 1) :]
    l = low[-(lookback + 1) :]
    c = close[-(lookback + 1) :]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr))


class VolatilityExpansionStrategy(Strategy):
    family = "volatility_expansion"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.range_lookback = int(params.get("range_lookback", 30))
        self.expansion_mult = float(params.get("expansion_mult", 2.0))
        self.confirm_break_pct = float(params.get("confirm_break_pct", 0.0004))
        self.atr_lookback = int(params.get("atr_lookback", 20))
        self.atr_stop_mult = float(params.get("atr_stop_mult", 1.8))
        self.time_stop_bars = int(params.get("time_stop_bars", 24))

        self._last_side = 0
        self._bars_in_pos = 0
        self._peak: float | None = None
        self._trough: float | None = None

    def warmup_bars(self) -> int:
        return max(self.range_lookback + 3, self.atr_lookback + 3)

    def _sync_state(self, current_side: int, close_px: float) -> None:
        if current_side != self._last_side:
            self._bars_in_pos = 0
            self._peak = close_px if current_side > 0 else None
            self._trough = close_px if current_side < 0 else None
        else:
            if current_side != 0:
                self._bars_in_pos += 1
                if current_side > 0:
                    self._peak = close_px if self._peak is None else max(self._peak, close_px)
                else:
                    self._trough = close_px if self._trough is None else min(self._trough, close_px)
        self._last_side = current_side

    def on_bar(self, ctx: StrategyContext, current_side: int) -> StrategySignal:
        close_hist = ctx.history("close")
        high_hist = ctx.history("high")
        low_hist = ctx.history("low")

        close_px = float(close_hist[-1])
        self._sync_state(current_side=current_side, close_px=close_px)

        if len(close_hist) < self.warmup_bars():
            return StrategySignal(target_side=current_side, reason="warmup")

        ranges = high_hist - low_hist
        cur_range = float(ranges[-1])
        avg_range = float(np.mean(ranges[-(self.range_lookback + 1) : -1]))
        prev_high = float(np.max(high_hist[-(self.range_lookback + 1) : -1]))
        prev_low = float(np.min(low_hist[-(self.range_lookback + 1) : -1]))
        atr_val = _atr(high_hist, low_hist, close_hist, self.atr_lookback)

        expansion = cur_range > self.expansion_mult * max(avg_range, 1e-12)
        breakout_up = close_px > prev_high * (1.0 + self.confirm_break_pct)
        breakout_dn = close_px < prev_low * (1.0 - self.confirm_break_pct)

        if current_side == 0:
            if expansion and breakout_up:
                return StrategySignal(target_side=1, reason="vol_expand_long")
            if expansion and breakout_dn:
                return StrategySignal(target_side=-1, reason="vol_expand_short")
            return StrategySignal(target_side=0, reason="flat")

        if self._bars_in_pos >= self.time_stop_bars:
            return StrategySignal(target_side=0, reason="time_stop")

        if current_side > 0 and self._peak is not None:
            stop = self._peak - self.atr_stop_mult * atr_val
            if close_px < stop:
                return StrategySignal(target_side=0, reason="atr_stop_long")
            if breakout_dn:
                return StrategySignal(target_side=0, reason="opposite_break")
        if current_side < 0 and self._trough is not None:
            stop = self._trough + self.atr_stop_mult * atr_val
            if close_px > stop:
                return StrategySignal(target_side=0, reason="atr_stop_short")
            if breakout_up:
                return StrategySignal(target_side=0, reason="opposite_break")

        return StrategySignal(target_side=current_side, reason="hold")
