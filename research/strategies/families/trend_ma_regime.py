from __future__ import annotations

from typing import Any

import numpy as np

from ..base import Strategy, StrategyContext, StrategySignal


def _sma(arr: np.ndarray, n: int) -> float:
    if len(arr) < n or n <= 0:
        return 0.0
    return float(np.mean(arr[-n:]))


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int) -> float:
    if len(high) < lookback + 1:
        return 0.0
    h = high[-(lookback + 1) :]
    l = low[-(lookback + 1) :]
    c = close[-(lookback + 1) :]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr))


class TrendMARegimeStrategy(Strategy):
    family = "trend_ma_regime"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.fast_ma = int(params.get("fast_ma", 20))
        self.slow_ma = int(params.get("slow_ma", 80))
        self.atr_lookback = int(params.get("atr_lookback", 20))
        self.atr_stop_mult = float(params.get("atr_stop_mult", 2.0))
        self.trend_buffer_pct = float(params.get("trend_buffer_pct", 0.0005))
        self.time_stop_bars = int(params.get("time_stop_bars", 96))

        self._last_side = 0
        self._bars_in_pos = 0
        self._peak: float | None = None
        self._trough: float | None = None

    def warmup_bars(self) -> int:
        return max(self.slow_ma + 2, self.atr_lookback + 2)

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

        fast = _sma(close_hist, self.fast_ma)
        slow = _sma(close_hist, self.slow_ma)
        atr_val = _atr(high_hist, low_hist, close_hist, self.atr_lookback)
        buffer_px = abs(close_px) * self.trend_buffer_pct

        trend_side = 0
        if fast > (slow + buffer_px):
            trend_side = 1
        elif fast < (slow - buffer_px):
            trend_side = -1

        if current_side == 0:
            if trend_side != 0:
                return StrategySignal(target_side=trend_side, reason="ma_trend_entry")
            return StrategySignal(target_side=0, reason="flat")

        if self._bars_in_pos >= self.time_stop_bars:
            return StrategySignal(target_side=0, reason="time_stop")

        if current_side > 0 and self._peak is not None:
            stop = self._peak - self.atr_stop_mult * atr_val
            if close_px < stop:
                return StrategySignal(target_side=0, reason="atr_stop_long")
        if current_side < 0 and self._trough is not None:
            stop = self._trough + self.atr_stop_mult * atr_val
            if close_px > stop:
                return StrategySignal(target_side=0, reason="atr_stop_short")

        if current_side == 1 and trend_side <= 0:
            return StrategySignal(target_side=0, reason="ma_cross_exit_long")
        if current_side == -1 and trend_side >= 0:
            return StrategySignal(target_side=0, reason="ma_cross_exit_short")

        return StrategySignal(target_side=current_side, reason="hold")
