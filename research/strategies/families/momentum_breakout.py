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


class MomentumBreakoutStrategy(Strategy):
    family = "momentum_breakout"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.lookback = int(params.get("breakout_lookback", 20))
        self.atr_lookback = int(params.get("atr_lookback", 14))
        self.atr_min_pct = float(params.get("atr_min_pct", 0.0008))
        self.time_stop_bars = int(params.get("time_stop_bars", 36))
        self.trailing_stop_atr = float(params.get("trailing_stop_atr", 2.0))

        self._last_side = 0
        self._bars_in_pos = 0
        self._peak = None
        self._trough = None

    def warmup_bars(self) -> int:
        return max(self.lookback + 2, self.atr_lookback + 2)

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

        atr_val = _atr(high_hist, low_hist, close_hist, self.atr_lookback)
        atr_pct = (atr_val / close_px) if close_px > 0 else 0.0

        prev_high = float(np.max(high_hist[-(self.lookback + 1) : -1]))
        prev_low = float(np.min(low_hist[-(self.lookback + 1) : -1]))

        # Flat -> breakout entries with ATR activity filter.
        if current_side == 0:
            if atr_pct < self.atr_min_pct:
                return StrategySignal(target_side=0, reason="atr_filter")
            if close_px > prev_high:
                return StrategySignal(target_side=1, reason="breakout_up")
            if close_px < prev_low:
                return StrategySignal(target_side=-1, reason="breakout_down")
            return StrategySignal(target_side=0, reason="flat")

        # Position management: time stop, trailing stop, opposite breakout.
        if self._bars_in_pos >= self.time_stop_bars:
            return StrategySignal(target_side=0, reason="time_stop")

        if current_side > 0 and self._peak is not None:
            trail = self._peak - self.trailing_stop_atr * atr_val
            if close_px < trail:
                return StrategySignal(target_side=0, reason="trail_stop_long")
            if close_px < prev_low:
                return StrategySignal(target_side=0, reason="breakdown_exit_long")

        if current_side < 0 and self._trough is not None:
            trail = self._trough + self.trailing_stop_atr * atr_val
            if close_px > trail:
                return StrategySignal(target_side=0, reason="trail_stop_short")
            if close_px > prev_high:
                return StrategySignal(target_side=0, reason="breakout_exit_short")

        return StrategySignal(target_side=current_side, reason="hold")
