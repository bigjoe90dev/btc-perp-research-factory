from __future__ import annotations

from typing import Any

import numpy as np

from ..base import Strategy, StrategyContext, StrategySignal


def _rolling_vwap(close: np.ndarray, volume: np.ndarray, n: int) -> float:
    if len(close) < n or n <= 0:
        return 0.0
    c = close[-n:]
    v = volume[-n:]
    v_sum = float(np.sum(v))
    if v_sum <= 0:
        return float(np.mean(c))
    return float(np.sum(c * v) / v_sum)


class MeanReversionVWAPStrategy(Strategy):
    family = "meanrev_vwap"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.vwap_lookback = int(params.get("vwap_lookback", 48))
        self.entry_dev_pct = float(params.get("entry_dev_pct", 0.002))
        self.exit_dev_pct = float(params.get("exit_dev_pct", 0.0007))
        self.vol_lookback = int(params.get("vol_lookback", 36))
        self.max_vol_pct = float(params.get("max_vol_pct", 0.0045))
        self.time_stop_bars = int(params.get("time_stop_bars", 48))

        self._last_side = 0
        self._bars_in_pos = 0

    def warmup_bars(self) -> int:
        return max(self.vwap_lookback + 2, self.vol_lookback + 2)

    def _sync_state(self, current_side: int) -> None:
        if current_side != self._last_side:
            self._bars_in_pos = 0
        elif current_side != 0:
            self._bars_in_pos += 1
        self._last_side = current_side

    def on_bar(self, ctx: StrategyContext, current_side: int) -> StrategySignal:
        close_hist = ctx.history("close")
        vol_hist = ctx.history("volume")

        self._sync_state(current_side=current_side)

        if len(close_hist) < self.warmup_bars():
            return StrategySignal(target_side=current_side, reason="warmup")

        close_px = float(close_hist[-1])
        vwap = _rolling_vwap(close_hist, vol_hist, self.vwap_lookback)
        if vwap <= 0:
            return StrategySignal(target_side=current_side, reason="invalid_vwap")

        dev = (close_px - vwap) / vwap
        ret = np.diff(np.log(np.maximum(close_hist[-(self.vol_lookback + 1) :], 1e-12)))
        vol = float(np.std(ret)) if len(ret) > 1 else 0.0

        if current_side == 0:
            if vol > self.max_vol_pct:
                return StrategySignal(target_side=0, reason="vol_filter")
            if dev >= self.entry_dev_pct:
                return StrategySignal(target_side=-1, reason="meanrev_short_entry")
            if dev <= -self.entry_dev_pct:
                return StrategySignal(target_side=1, reason="meanrev_long_entry")
            return StrategySignal(target_side=0, reason="flat")

        if self._bars_in_pos >= self.time_stop_bars:
            return StrategySignal(target_side=0, reason="time_stop")

        if abs(dev) <= self.exit_dev_pct:
            return StrategySignal(target_side=0, reason="revert_exit")

        if current_side > 0 and dev > self.entry_dev_pct * 1.25:
            return StrategySignal(target_side=0, reason="adverse_exit_long")
        if current_side < 0 and dev < -self.entry_dev_pct * 1.25:
            return StrategySignal(target_side=0, reason="adverse_exit_short")

        return StrategySignal(target_side=current_side, reason="hold")
