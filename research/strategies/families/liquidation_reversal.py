from __future__ import annotations

from typing import Any

import numpy as np

from ..base import Strategy, StrategyContext, StrategySignal


class LiquidationReversalStrategy(Strategy):
    family = "liquidation_reversal"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.range_lookback = int(params.get("range_lookback", 24))
        self.volume_lookback = int(params.get("volume_lookback", 24))
        self.range_mult = float(params.get("range_mult", 2.0))
        self.volume_mult = float(params.get("volume_mult", 2.0))
        self.reclaim_frac = float(params.get("reclaim_frac", 0.65))
        self.time_stop_bars = int(params.get("time_stop_bars", 12))
        self.adverse_stop_pct = float(params.get("adverse_stop_pct", 0.008))

        self._last_side = 0
        self._bars_in_pos = 0
        self._pending_side = 0
        self._pending_confirm_level: float | None = None
        self._pending_expiry_idx = -1

    def warmup_bars(self) -> int:
        return max(self.range_lookback + 2, self.volume_lookback + 2)

    def _sync_state(self, current_side: int) -> None:
        if current_side != self._last_side:
            self._bars_in_pos = 0
        elif current_side != 0:
            self._bars_in_pos += 1
        self._last_side = current_side

    def on_bar(self, ctx: StrategyContext, current_side: int) -> StrategySignal:
        open_hist = ctx.history("open")
        high_hist = ctx.history("high")
        low_hist = ctx.history("low")
        close_hist = ctx.history("close")
        vol_hist = ctx.history("volume")

        self._sync_state(current_side=current_side)

        if len(close_hist) < self.warmup_bars():
            return StrategySignal(target_side=current_side, reason="warmup")

        i = ctx.idx
        close_px = float(close_hist[-1])
        high_px = float(high_hist[-1])
        low_px = float(low_hist[-1])
        open_px = float(open_hist[-1])
        range_px = max(high_px - low_px, 1e-12)

        ranges = high_hist - low_hist
        avg_range = float(np.mean(ranges[-(self.range_lookback + 1) : -1]))
        avg_volume = float(np.mean(vol_hist[-(self.volume_lookback + 1) : -1]))
        vol_now = float(vol_hist[-1])

        large_range = range_px >= self.range_mult * max(avg_range, 1e-12)
        vol_spike = vol_now >= self.volume_mult * max(avg_volume, 1e-12)
        close_near_high = (close_px - low_px) / range_px >= self.reclaim_frac
        close_near_low = (high_px - close_px) / range_px >= self.reclaim_frac

        flush_down = large_range and vol_spike and close_near_high and close_px <= open_px
        flush_up = large_range and vol_spike and close_near_low and close_px >= open_px

        if current_side == 0:
            if self._pending_side != 0 and i <= self._pending_expiry_idx and self._pending_confirm_level is not None:
                if self._pending_side > 0 and close_px > self._pending_confirm_level:
                    self._pending_side = 0
                    self._pending_confirm_level = None
                    return StrategySignal(target_side=1, reason="flush_reversal_confirm_long")
                if self._pending_side < 0 and close_px < self._pending_confirm_level:
                    self._pending_side = 0
                    self._pending_confirm_level = None
                    return StrategySignal(target_side=-1, reason="flush_reversal_confirm_short")

            if flush_down:
                self._pending_side = 1
                self._pending_confirm_level = high_px
                self._pending_expiry_idx = i + 2
                return StrategySignal(target_side=0, reason="flush_down_pending")
            if flush_up:
                self._pending_side = -1
                self._pending_confirm_level = low_px
                self._pending_expiry_idx = i + 2
                return StrategySignal(target_side=0, reason="flush_up_pending")

            if i > self._pending_expiry_idx:
                self._pending_side = 0
                self._pending_confirm_level = None
            return StrategySignal(target_side=0, reason="flat")

        if self._bars_in_pos >= self.time_stop_bars:
            return StrategySignal(target_side=0, reason="time_stop")

        if current_side > 0:
            adverse = (close_px / max(open_px, 1e-12)) - 1.0
            if adverse <= -self.adverse_stop_pct:
                return StrategySignal(target_side=0, reason="adverse_exit_long")
            if flush_up:
                return StrategySignal(target_side=0, reason="opposite_flush")

        if current_side < 0:
            adverse = (open_px / max(close_px, 1e-12)) - 1.0
            if adverse <= -self.adverse_stop_pct:
                return StrategySignal(target_side=0, reason="adverse_exit_short")
            if flush_down:
                return StrategySignal(target_side=0, reason="opposite_flush")

        return StrategySignal(target_side=current_side, reason="hold")
